"""Golden fixtures for tsecon-breaks: multiple structural breaks (Bai-Perron).

VALIDATION STRATEGY
===================
Every number this file writes is produced by an INDEPENDENT reference path --
brute-force enumeration plus `numpy.linalg.lstsq` segment regressions, or a
DOCUMENTED published formula evaluated with plain numpy/scipy.  Nothing here
imports tsecon, so reproducing these numbers in Rust is a genuine
cross-implementation check, never circular.  All data are derived from seeded
`numpy.random.default_rng` DGPs -- nothing is a redistributed dataset.

QUANTITIES AND THEIR REFERENCES
-------------------------------
1. GLOBAL PARTITIONS (the key design).  The Bai-Perron (1998, Econometrica
   66(1), 47-78) estimator finds, for each number of breaks m, the partition
   of 0..T-1 into m+1 contiguous segments (each of length >= h =
   ceil(trim*T)) minimizing the total OLS sum of squared residuals, with
   every regression coefficient switching at each break (pure structural
   change).  The crate implements the dynamic program of Bai-Perron (2003,
   J. Applied Econometrics 18(1), 1-22).  THIS FILE computes the exact
   optimum by BRUTE-FORCE ENUMERATION over all admissible break placements
   (itertools.combinations) with each segment's SSR from numpy.linalg.lstsq
   -- an entirely different algorithmic path.  The DP must reproduce the
   brute force EXACTLY: same minimal SSR (to float tolerance) and the same
   break dates.  Break dates are stored 0-indexed as the LAST observation of
   each regime.

2. sup-F(l+1 | l) SEQUENTIAL STATISTICS (Bai-Perron 1998 Section 5.3; 2003
   Section 3).  Given the l-break global minimizers, each of the l+1
   segments with at least 2h observations is tested for one additional
   break, in the classical homoskedastic Wald form used to tabulate the
   published critical values:

       F_i = (T_i - 2q) * (SSR_i - min_split SSR) / min_split SSR,
       supF(l+1|l) = max_i F_i,

   where T_i is the segment length, q the number of regressors, and the
   split point within segment [s, e] ranges over tau in [s+h-1, e-h] so both
   sub-segments have >= h observations (segments with T_i < 2h contribute 0,
   as in Perron's own implementation).  supF(1|0) is the same statistic on
   the full sample.  The 5% critical values c(q, l+1) are the published
   Bai-Perron values (1998 Table II; expanded in Bai-Perron 2003b,
   Econometrics Journal 6(1), 72-78), transcribed below from the plaintext
   tables distributed inside Perron et al.'s own R package `mbreaks`
   (CRAN, R/supF_next/cv_{1..5}.csv; file index = trimming 0.05/0.10/0.15/
   0.20/0.25; the 5% block is rows 11-20, q = 1..10, columns l+1 = 1..10).
   The selected number of breaks is the largest n such that supF(l+1|l)
   rejects at 5% for every l < n, stopping at the first non-rejection.

3. ANDREWS/QUANDT sup-F (Andrews 1993, Econometrica 61(4), 821-856) with
   Hansen (1997, JBES 15(1), 60-67) approximate asymptotic p-values.  The
   statistic is the maximum over candidate dates d in [h-1, T-h-1] of the
   Wald-form Chow statistic

       F(d) = (T - 2q) * (SSR_0 - SSR_1(d) - SSR_2(d)) / (SSR_1(d) + SSR_2(d)),

   exactly the statistic strucchange's `Fstats` computes and the one
   Hansen's response surfaces are calibrated to.  The p-value approximation
   is Hansen's published response surface: for each (q, tau) row the
   surface gives (b0, b1, v) with  p ~ P(chi2_v > b0 + b1*x)  and linear
   interpolation across the tau grid 0.49, 0.47, ..., 0.01 (tau = h/T is
   the effective symmetric trimming).  The coefficients are transcribed
   below from strucchange (CRAN, R/pvalue.Fstats.R, object `sc.beta.sup`,
   rows k = 1..10), which packages Hansen's published GAUSS tables; the
   interpolation logic is transcribed from `pvalue.Fstats`.

4. BREAK-DATE CONFIDENCE INTERVALS (Bai 1997, Review of Economics and
   Statistics 79(4), 551-563), homogeneous shrinking-break asymptotics: with
   common regressor second moment Q = X'X/T and common error variance
   s2 = SSR_m / (T - (m+1) q) across regimes, and delta_i the coefficient
   change at break i,

       L_i = delta_i' Q delta_i / s2,
       L_i * (That_i - T0_i)  ==>  argmax_s { W(s) - |s|/2 },

   where W is a two-sided standard Wiener process.  The argmax cdf is the
   known closed form (Bai 1997, Appendix B; Yao 1987), for x >= 0:

       G(x) = 1 + sqrt(x/(2 pi)) exp(-x/8)
                + (3/2) exp(x) Phi(-(3/2) sqrt(x))
                - ((x+5)/2) Phi(-sqrt(x)/2),          G(-x) = 1 - G(x).

   The level-a two-sided critical value c_a solves 2 G(c) - 1 = a, and the
   CI is [That - ceil(c_a/L) - 1, That + ceil(c_a/L) + 1] clipped to the
   sample, following Bai-Perron's implementation convention.  This file
   verifies the transcription against the two published anchors
   P(|xi| <= 7.7) ~ 0.90 and P(|xi| <= 11.03) ~ 0.95 before writing
   anything.

Regenerate with:  .venv/bin/python fixtures/generate_tsecon-breaks_fixtures.py
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.stats import chi2, norm

OUT = os.path.join(os.path.dirname(__file__), "tsecon-breaks.json")

# ---------------------------------------------------------------------------
# Published tables (transcribed; sources cited in the module docstring).
# HANSEN_SUP: 250 rows = 10 regressor counts (k = 1..10) x 25 tau rows
# (tau = 0.49, 0.47, ..., 0.01); each row is (b0, b1, chi-square df).
# BP_SEQ_CV_5PCT[trim][q-1][l] = 5% critical value of supF(l+1|l).
# ---------------------------------------------------------------------------
# fmt: off
HANSEN_SUP = [
    (-0.0648467, 0.99671156, 1.2573241),
    (-0.15305667, 0.96757645, 1.43651315),
    (-0.23071675, 0.95825013, 1.56531261),
    (-0.29247661, 0.95715782, 1.68851898),
    (-0.35706588, 0.95237505, 1.77889686),
    (-0.40514715, 0.96055795, 1.89761322),
    (-0.45248386, 0.95810999, 1.98370422),
    (-0.50458517, 0.96149538, 2.07086519),
    (-0.55009159, 0.96684273, 2.17244338),
    (-0.60241623, 0.96518399, 2.23509053),
    (-0.64308746, 0.97354902, 2.33722161),
    (-0.68833676, 0.97829106, 2.42776524),
    (-0.73136161, 0.9820687, 2.50833918),
    (-0.78078198, 0.99475319, 2.6209551),
    (-0.82894221, 1.00023473, 2.70286643),
    (-0.88430133, 1.00162529, 2.77835798),
    (-0.9499265, 1.00384268, 2.84325795),
    (-0.98828874, 1.01604712, 2.96752918),
    (-1.04892713, 1.02570939, 3.08442682),
    (-1.11527783, 1.03526033, 3.19770394),
    (-1.14992644, 1.06643436, 3.44739573),
    (-1.24010871, 1.06929468, 3.54993605),
    (-1.38522113, 1.06808543, 3.61667211),
    (-1.48296473, 1.1283969, 4.07202473),
    (-1.78738608, 1.17220925, 4.4726701),
    (-0.03933112, 0.99525784, 2.44239332),
    (-0.14650989, 0.9938654, 2.78837829),
    (-0.28765583, 0.98139359, 2.93724618),
    (-0.35528752, 0.9942739, 3.18187129),
    (-0.49538961, 0.96923268, 3.19484388),
    (-0.58780544, 0.97339124, 3.32763253),
    (-0.66900489, 0.97444133, 3.44170617),
    (-0.77825803, 0.96678414, 3.48652761),
    (-0.87260977, 0.97626434, 3.61709836),
    (-0.96782445, 0.97636983, 3.6779502),
    (-1.07358977, 0.97407744, 3.71945379),
    (-1.12531079, 0.99897105, 3.95114475),
    (-1.16041007, 1.01629918, 4.14932643),
    (-1.2975797, 1.015268, 4.17103882),
    (-1.36743697, 1.03004743, 4.3223804),
    (-1.43012785, 1.05139461, 4.53836755),
    (-1.53311747, 1.05280599, 4.61175851),
    (-1.65273586, 1.05932633, 4.69340428),
    (-1.79350728, 1.04936541, 4.68380985),
    (-1.9117747, 1.0588459, 4.8073354),
    (-2.02659329, 1.07200268, 4.97772701),
    (-2.18499383, 1.08674781, 5.14231666),
    (-2.3765858, 1.11342895, 5.40268693),
    (-2.51267262, 1.15305027, 5.88573096),
    (-3.05559851, 1.17629253, 6.11067794),
    (-0.1250856, 0.98151496, 3.43036267),
    (-0.27323021, 0.98324873, 3.83943122),
    (-0.43148524, 0.98336371, 4.07662685),
    (-0.48172245, 1.00450689, 4.44796533),
    (-0.6898932, 0.99692978, 4.49665381),
    (-0.7928579, 1.00417536, 4.69772537),
    (-0.85066146, 1.01852003, 4.95584784),
    (-1.06133796, 1.00719185, 4.90812703),
    (-1.11977435, 1.03046019, 5.20741484),
    (-1.20203015, 1.03885443, 5.3693925),
    (-1.28736105, 1.04597379, 5.52416853),
    (-1.45874969, 1.03549948, 5.4888527),
    (-1.60889159, 1.03374376, 5.50958245),
    (-1.64346161, 1.06767681, 5.89426861),
    (-1.68334797, 1.09676432, 6.23634969),
    (-1.81974576, 1.10216977, 6.34122715),
    (-2.02648892, 1.0962988, 6.29077373),
    (-2.04820499, 1.13491785, 6.75208304),
    (-2.3230492, 1.12030935, 6.59886028),
    (-2.59516448, 1.09122076, 6.32294425),
    (-2.79776102, 1.09109337, 6.35576003),
    (-3.03865047, 1.09037434, 6.37212114),
    (-3.31299004, 1.10476754, 6.5282289),
    (-3.65667229, 1.13355758, 6.84167661),
    (-4.09174097, 1.21034051, 7.79013953),
    (-0.40798205, 0.90528031, 3.93318971),
    (-0.58214849, 0.91505724, 4.41094611),
    (-0.67464799, 0.9364688, 4.85980413),
    (-0.85920669, 0.93433918, 5.01958587),
    (-0.99797889, 0.9411933, 5.22477986),
    (-1.10019761, 0.95132377, 5.46557096),
    (-1.24048639, 0.96334287, 5.6723934),
    (-1.44519417, 0.96603109, 5.73132847),
    (-1.43104108, 1.00890357, 6.29613628),
    (-1.52887344, 1.02148881, 6.51227891),
    (-1.66695722, 1.02616925, 6.63448592),
    (-1.79016819, 1.03765517, 6.82686641),
    (-1.91159633, 1.04392701, 6.95318158),
    (-1.93474222, 1.071198, 7.36997641),
    (-1.94039709, 1.10122554, 7.82328663),
    (-2.14388035, 1.09891616, 7.83079173),
    (-2.34856811, 1.10457466, 7.89044965),
    (-2.52186324, 1.10727065, 7.95561422),
    (-2.81381108, 1.10513873, 7.90116455),
    (-3.02543477, 1.12012324, 8.08624226),
    (-3.3338615, 1.12405358, 8.08884313),
    (-3.70012182, 1.12268437, 8.00223819),
    (-4.08017759, 1.14201041, 8.18314656),
    (-4.47426188, 1.17607899, 8.60611855),
    (-5.32601074, 1.21390643, 8.90496643),
    (-0.11619422, 1.02457601, 5.80252297),
    (-0.46546717, 1.01129094, 6.12710758),
    (-0.74391826, 1.00172704, 6.30619245),
    (-1.02744491, 0.9890919, 6.35729244),
    (-1.26392205, 0.98620432, 6.4591257),
    (-1.42145334, 0.9972466, 6.71491425),
    (-1.65407989, 0.99253425, 6.740473),
    (-1.81536469, 1.0044353, 6.95402735),
    (-2.02334533, 1.00046922, 6.98677984),
    (-2.09938834, 1.01518764, 7.27464455),
    (-2.20567987, 1.02275886, 7.47265174),
    (-2.44943836, 1.02127435, 7.46129265),
    (-2.63257362, 1.01887636, 7.46907351),
    (-2.81161264, 1.02006744, 7.52690766),
    (-2.98179286, 1.0305527, 7.67224919),
    (-3.13342287, 1.0446574, 7.90265755),
    (-3.30649507, 1.05369295, 8.05397638),
    (-3.4646422, 1.06781143, 8.27206632),
    (-3.54117909, 1.10989892, 8.90945304),
    (-3.53272742, 1.15870693, 9.72776829),
    (-3.80907119, 1.17183781, 9.91294592),
    (-4.20676069, 1.17153767, 9.84654688),
    (-4.84259892, 1.14601974, 9.30420469),
    (-5.47685207, 1.15819167, 9.30236461),
    (-6.39138173, 1.17642784, 9.41930982),
    (-0.0687869, 1.03775792, 7.09045346),
    (-0.33471313, 1.03320405, 7.62520471),
    (-0.58645046, 1.03391895, 7.96566986),
    (-0.84305517, 1.03864019, 8.22807265),
    (-1.17255237, 1.02620388, 8.19864669),
    (-1.46491217, 1.01747206, 8.192869),
    (-1.55238215, 1.02887917, 8.53383279),
    (-1.79236992, 1.02880434, 8.59456107),
    (-1.78127768, 1.06533575, 9.2685801),
    (-2.21251422, 1.04598986, 8.91322396),
    (-2.47288843, 1.04669836, 8.92953537),
    (-2.61217463, 1.05733536, 9.17320934),
    (-2.93562547, 1.04956126, 9.00301305),
    (-3.34428329, 1.03025578, 8.63825734),
    (-3.48448133, 1.04393869, 8.87565157),
    (-3.64732289, 1.05837582, 9.13002628),
    (-3.79479594, 1.07938759, 9.47650111),
    (-4.04745714, 1.08339951, 9.51466537),
    (-4.13484904, 1.11380916, 10.08288929),
    (-4.27469148, 1.13359417, 10.47680321),
    (-4.61491387, 1.14810477, 10.64305003),
    (-4.91271109, 1.18479646, 11.1806159),
    (-5.37456783, 1.1930588, 11.24873153),
    (-6.01333062, 1.22140239, 11.55114952),
    (-7.08021496, 1.25536339, 11.84388128),
    (-0.53689878, 0.95219603, 7.1028369),
    (-0.58562073, 0.98400121, 8.16310859),
    (-0.95896123, 0.97640322, 8.31297248),
    (-1.31359234, 0.97126847, 8.39458871),
    (-1.55985878, 0.97651275, 8.6116192),
    (-1.74442449, 0.99363695, 8.97007085),
    (-2.0175967, 0.99545856, 9.05632019),
    (-2.20599731, 1.00681619, 9.31295267),
    (-2.25783227, 1.03037828, 9.83449349),
    (-2.53419037, 1.02237747, 9.74010484),
    (-2.83535247, 1.01736263, 9.65255439),
    (-2.94857734, 1.03701081, 10.0582618),
    (-3.23125462, 1.04527054, 10.13194554),
    (-3.35534682, 1.07043687, 10.59763704),
    (-3.46151576, 1.08475765, 10.91888223),
    (-3.87863069, 1.07699402, 10.68834341),
    (-4.23908351, 1.07739268, 10.59992445),
    (-4.4191379, 1.10217153, 11.01503763),
    (-4.82113958, 1.10489462, 10.94912199),
    (-4.95712189, 1.14759912, 11.72034341),
    (-5.33535348, 1.15489501, 11.78096338),
    (-5.94327236, 1.14993703, 11.48022849),
    (-6.2144914, 1.2117616, 12.57656495),
    (-6.95870114, 1.21291275, 12.41746897),
    (-8.48837563, 1.16895303, 11.0568314),
    (-0.06587378, 1.00259864, 9.00416324),
    (-0.58124366, 0.98811229, 9.29588314),
    (-0.53482076, 1.02737978, 10.38413127),
    (-0.70899479, 1.03875734, 10.86726887),
    (-1.07383099, 1.03945526, 10.98596388),
    (-1.05381577, 1.0784185, 11.87560902),
    (-1.27585106, 1.10005486, 12.3098159),
    (-1.68931537, 1.09777938, 12.24252384),
    (-2.18809932, 1.08874332, 12.00013658),
    (-2.50101866, 1.08718514, 11.97891812),
    (-2.85948167, 1.08365161, 11.88283066),
    (-3.35512723, 1.06078274, 11.38799144),
    (-3.65167655, 1.06322641, 11.38822596),
    (-4.01035186, 1.06373417, 11.32180872),
    (-4.24709439, 1.07465019, 11.50213594),
    (-4.50345226, 1.09341658, 11.80959613),
    (-4.9099655, 1.08551877, 11.57319597),
    (-5.36063931, 1.07646817, 11.25922919),
    (-5.73370206, 1.07605364, 11.18469963),
    (-6.09245707, 1.07280436, 11.07807286),
    (-6.46768896, 1.0801085, 11.14611701),
    (-6.84622409, 1.09621271, 11.40693419),
    (-7.23578885, 1.12575792, 11.92603449),
    (-7.80794405, 1.15036691, 12.32590546),
    (-9.19862689, 1.17118238, 12.18766263),
    (-0.8377844, 0.93113614, 8.5883869),
    (-1.18727357, 0.93903334, 9.24954183),
    (-1.67892814, 0.93065103, 9.29685978),
    (-2.13655356, 0.92934556, 9.33656379),
    (-2.40733753, 0.94488795, 9.70552238),
    (-2.37013442, 0.97862192, 10.57272004),
    (-2.49480314, 0.99988795, 11.11202863),
    (-2.82625963, 0.99845236, 11.11899557),
    (-3.17119868, 0.99862288, 11.12620845),
    (-3.5636537, 0.99017086, 10.91577546),
    (-3.88321378, 0.9946165, 10.95270011),
    (-4.0313254, 1.01955132, 11.46852117),
    (-4.38155427, 1.01232498, 11.27876602),
    (-4.50039414, 1.03871207, 11.86056231),
    (-4.87091786, 1.03085863, 11.63335876),
    (-5.1383982, 1.03917618, 11.78048327),
    (-5.39812039, 1.05170079, 11.99793995),
    (-5.43335672, 1.10431373, 13.09302596),
    (-6.18441355, 1.06499129, 12.01561699),
    (-6.60044603, 1.07700965, 12.12034223),
    (-7.11669991, 1.0808924, 12.02669094),
    (-7.46755944, 1.09541073, 12.30726854),
    (-8.07312922, 1.1095527, 12.39994474),
    (-8.86841479, 1.11196659, 12.21815495),
    (-10.22371148, 1.13939491, 12.28861527),
    (-0.12732608, 1.01972827, 11.23288885),
    (-0.77782321, 0.9982764, 11.39255875),
    (-1.24808905, 0.9999548, 11.66130981),
    (-1.76904345, 0.98955787, 11.58597271),
    (-2.09486459, 0.99858018, 11.87831304),
    (-2.30685545, 1.01743721, 12.36966797),
    (-2.70509677, 1.02589592, 12.51217351),
    (-2.92069286, 1.04951504, 13.04128558),
    (-3.1997618, 1.06492822, 13.38535841),
    (-3.6337934, 1.05272485, 13.0950741),
    (-4.01274834, 1.05200434, 13.0322117),
    (-4.50166071, 1.03350523, 12.57591029),
    (-4.97277337, 1.01177319, 12.03313903),
    (-5.21536918, 1.03214392, 12.43310006),
    (-5.65195302, 1.01761463, 12.01632623),
    (-6.06457328, 1.01623184, 11.88077482),
    (-6.0595936, 1.06276209, 12.97932137),
    (-6.46819929, 1.060171, 12.81597962),
    (-6.87347962, 1.06428574, 12.80920657),
    (-7.25454462, 1.07640874, 12.97479282),
    (-7.76257167, 1.07583608, 12.81308974),
    (-8.18205476, 1.09754669, 13.20385908),
    (-8.84260296, 1.10727034, 13.19094338),
    (-9.55384281, 1.13929623, 13.69566019),
    (-11.00985153, 1.1430948, 13.27442097),
]

BP_SEQ_CV_5PCT = {
    0.05: [
        [9.63, 11.14, 12.16, 12.83, 13.45, 14.05, 14.29, 14.5, 14.69, 14.88],
        [12.89, 14.5, 15.42, 16.16, 16.61, 17.02, 17.27, 17.55, 17.76, 17.97],
        [15.37, 17.15, 17.97, 18.72, 19.23, 19.59, 19.94, 20.31, 21.05, 21.2],
        [17.6, 19.33, 20.22, 20.75, 21.15, 21.55, 21.9, 22.27, 22.63, 22.83],
        [19.5, 21.43, 22.57, 23.33, 23.9, 24.34, 24.62, 25.14, 25.34, 25.51],
        [21.59, 23.72, 24.66, 25.29, 25.89, 26.36, 26.84, 27.1, 27.26, 27.4],
        [23.5, 25.17, 26.34, 27.19, 27.96, 28.25, 28.64, 28.84, 28.97, 29.14],
        [25.22, 27.18, 28.21, 28.99, 29.54, 30.05, 30.45, 30.79, 31.29, 31.75],
        [27.08, 29.1, 30.24, 30.99, 31.48, 32.46, 32.71, 32.89, 33.15, 33.43],
        [28.49, 30.65, 31.9, 32.83, 33.57, 34.27, 34.53, 35.01, 35.33, 35.65],
    ],
    0.10: [
        [9.1, 10.55, 11.36, 12.35, 12.97, 13.45, 13.88, 14.12, 14.45, 14.51],
        [12.25, 13.83, 14.73, 15.46, 16.13, 16.55, 16.82, 17.07, 17.34, 17.58],
        [14.6, 16.53, 17.43, 17.98, 18.61, 19.02, 19.25, 19.61, 19.94, 20.35],
        [16.76, 18.56, 19.53, 20.24, 20.72, 21.13, 21.55, 21.83, 22.08, 22.4],
        [18.68, 20.57, 21.6, 22.55, 23.0, 23.63, 24.13, 24.48, 24.82, 25.14],
        [20.76, 23.01, 24.14, 24.77, 25.48, 25.89, 26.25, 26.77, 26.96, 27.14],
        [22.62, 24.64, 25.57, 26.54, 27.04, 27.51, 28.14, 28.44, 28.74, 28.87],
        [24.34, 26.42, 27.66, 28.25, 28.99, 29.34, 29.86, 30.29, 30.5, 30.68],
        [26.2, 28.23, 29.44, 30.31, 30.77, 31.35, 31.91, 32.6, 32.71, 32.86],
        [27.64, 29.78, 31.02, 31.9, 32.71, 33.32, 33.95, 34.29, 34.52, 34.81],
    ],
    0.15: [
        [8.58, 10.13, 11.14, 11.83, 12.25, 12.66, 13.08, 13.35, 13.75, 13.89],
        [11.47, 12.95, 14.03, 14.85, 15.29, 15.8, 16.16, 16.44, 16.77, 16.84],
        [13.98, 15.72, 16.83, 17.61, 18.14, 18.74, 19.09, 19.41, 19.68, 19.77],
        [16.19, 18.11, 18.93, 19.64, 20.19, 20.54, 21.21, 21.42, 21.72, 21.97],
        [18.23, 19.91, 20.99, 21.71, 22.37, 22.77, 23.15, 23.42, 24.04, 24.42],
        [20.08, 22.11, 23.04, 23.77, 24.43, 24.75, 24.96, 25.22, 25.61, 25.93],
        [21.87, 24.17, 25.13, 26.03, 26.65, 27.06, 27.37, 27.9, 28.18, 28.36],
        [23.7, 25.75, 26.81, 27.65, 28.48, 28.8, 29.08, 29.3, 29.5, 29.69],
        [25.65, 27.66, 28.91, 29.67, 30.52, 30.96, 31.48, 31.77, 31.94, 32.33],
        [27.03, 29.24, 30.45, 31.45, 32.12, 32.5, 32.84, 33.12, 33.22, 33.85],
    ],
    0.20: [
        [8.22, 9.71, 10.66, 11.34, 11.93, 12.3, 12.68, 12.92, 13.21, 13.61],
        [10.98, 12.55, 13.46, 14.22, 14.78, 15.37, 15.81, 16.13, 16.44, 16.69],
        [13.47, 15.25, 16.36, 17.08, 17.51, 18.08, 18.44, 18.89, 19.01, 19.35],
        [15.67, 17.61, 18.54, 19.21, 19.8, 20.22, 20.53, 21.06, 21.31, 21.55],
        [17.66, 19.5, 20.63, 21.4, 21.72, 22.19, 22.72, 23.01, 23.24, 23.67],
        [19.55, 21.44, 22.64, 23.19, 23.75, 24.28, 24.46, 24.75, 24.96, 25.02],
        [21.33, 23.31, 24.75, 25.38, 26.1, 26.47, 26.87, 27.15, 27.37, 27.74],
        [23.19, 25.23, 26.39, 27.19, 27.63, 28.09, 28.49, 28.7, 28.83, 29.02],
        [24.91, 26.92, 28.1, 28.93, 29.64, 30.29, 30.87, 31.09, 31.39, 31.67],
        [26.38, 28.56, 29.62, 30.48, 31.23, 31.96, 32.2, 32.38, 32.72, 32.9],
    ],
    0.25: [
        [7.86, 9.29, 10.12, 10.93, 11.37, 11.82, 12.2, 12.65, 12.79, 13.09],
        [10.55, 12.19, 12.97, 13.84, 14.32, 14.92, 15.28, 15.48, 15.87, 16.34],
        [13.04, 14.65, 15.6, 16.51, 17.08, 17.39, 17.76, 18.08, 18.32, 18.72],
        [15.19, 17.0, 18.1, 18.72, 19.14, 19.63, 20.1, 20.5, 20.98, 21.23],
        [17.12, 18.94, 20.02, 20.81, 21.45, 21.72, 22.1, 22.69, 22.98, 23.15],
        [18.97, 20.89, 21.92, 22.66, 23.09, 23.42, 23.96, 24.28, 24.46, 24.75],
        [20.75, 22.78, 24.24, 24.93, 25.66, 26.03, 26.28, 26.56, 26.87, 27.21],
        [22.56, 24.54, 25.71, 26.5, 27.01, 27.51, 27.74, 28.09, 28.48, 28.7],
        [24.18, 26.28, 27.42, 28.27, 29.03, 29.67, 30.34, 30.79, 30.93, 31.13],
        [25.77, 27.75, 29.18, 30.02, 30.83, 31.4, 31.92, 32.2, 32.38, 32.72],
    ],
}
# fmt: on

# ---------------------------------------------------------------------------
# Reference implementations (numpy/scipy only).
# ---------------------------------------------------------------------------


def seg_ssr(y: np.ndarray, X: np.ndarray) -> float:
    """OLS sum of squared residuals of y on X via numpy.linalg.lstsq."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float(r @ r)


def ssr_table(y: np.ndarray, X: np.ndarray, h: int) -> dict:
    """SSR of every admissible segment [i, j] (inclusive, length >= h)."""
    T = len(y)
    tbl = {}
    for i in range(T - h + 1):
        for j in range(i + h - 1, T):
            tbl[(i, j)] = seg_ssr(y[i : j + 1], X[i : j + 1])
    return tbl


def brute_force(T: int, h: int, m: int, tbl: dict):
    """Exact optimal m-break partition by enumeration.

    Break dates are the LAST index of each regime (0-indexed).  Admissible:
    d_1 >= h-1, d_{i+1} - d_i >= h, and T - 1 - d_m >= h.
    """
    best_ssr, best_dates = math.inf, None
    for dates in itertools.combinations(range(h - 1, T - h), m):
        prev = -1
        ok = True
        for d in dates:
            if d - prev < h:
                ok = False
                break
            prev = d
        if not ok:
            continue
        bounds = [-1] + list(dates) + [T - 1]
        ssr = sum(tbl[(bounds[i] + 1, bounds[i + 1])] for i in range(m + 1))
        if ssr < best_ssr:
            best_ssr, best_dates = ssr, dates
    return best_ssr, list(best_dates)


def supf_full(y: np.ndarray, X: np.ndarray, h: int):
    """Andrews sup-F: Wald-form Chow path over the trimmed candidate range."""
    T, q = X.shape
    ssr0 = seg_ssr(y, X)
    dates, path = [], []
    for d in range(h - 1, T - h):
        s = seg_ssr(y[: d + 1], X[: d + 1]) + seg_ssr(y[d + 1 :], X[d + 1 :])
        path.append((T - 2 * q) * (ssr0 - s) / s)
        dates.append(d)
    k = int(np.argmax(path))
    return ssr0, dates, path, dates[k], path[k]


def hansen_supf_pvalue(stat: float, q: int, tau: float) -> float:
    """Hansen (1997) approximate p-value, transcribed from pvalue.Fstats."""
    rows = HANSEN_SUP[(q - 1) * 25 : q * 25]
    pp = [float(chi2.sf(max(b0 + b1 * stat, 0.0), df)) for b0, b1, df in rows]
    if tau == 0.5:
        p = float(chi2.sf(stat, q))
    elif tau <= 0.01:
        p = pp[24]
    elif tau >= 0.49:
        p = ((0.5 - tau) * pp[0] + (tau - 0.49) * float(chi2.sf(stat, q))) * 100.0
    else:
        taua = (0.51 - tau) * 50.0
        t1 = int(math.floor(taua))
        p = (t1 + 1 - taua) * pp[t1 - 1] + (taua - t1) * pp[t1]
    return float(min(max(p, 0.0), 1.0))


def supf_seq_stat(T: int, q: int, h: int, dates: list, tbl: dict) -> float:
    """supF(l+1 | l) given the l-break null partition (global minimizers)."""
    bounds = [0] + [d + 1 for d in dates] + [T]
    best = 0.0
    for s, e in zip(bounds[:-1], bounds[1:]):  # segment rows s..e-1
        n_i = e - s
        if n_i < 2 * h:
            continue
        ssr_i = tbl[(s, e - 1)]
        smin = min(tbl[(s, t)] + tbl[(t + 1, e - 1)] for t in range(s + h - 1, e - h))
        best = max(best, (n_i - 2 * q) * (ssr_i - smin) / smin)
    return best


def select_n_breaks(seq_stats: list, q: int, trim: float) -> int:
    """Sequential selection at 5%: stop at the first non-rejection."""
    cvs = BP_SEQ_CV_5PCT[round(trim, 2)][q - 1]
    n = 0
    for l, st in enumerate(seq_stats):
        if st > cvs[l]:
            n = l + 1
        else:
            break
    return n


def regime_fits(y: np.ndarray, X: np.ndarray, dates: list):
    """Per-regime OLS coefficients with nonrobust standard errors."""
    T, q = X.shape
    bounds = [0] + [d + 1 for d in dates] + [T]
    out = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        Xi, yi = X[s:e], y[s:e]
        b, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
        r = yi - Xi @ b
        ssr = float(r @ r)
        s2 = ssr / (e - s - q)
        cov = s2 * np.linalg.inv(Xi.T @ Xi)
        out.append(
            {
                "start": s,
                "end": e - 1,
                "params": [float(v) for v in b],
                "se": [float(v) for v in np.sqrt(np.diag(cov))],
                "ssr": ssr,
            }
        )
    return out


def argmax_cdf(x: float) -> float:
    """Bai (1997) closed-form cdf of argmax_s { W(s) - |s|/2 }."""
    if x < 0.0:
        return 1.0 - argmax_cdf(-x)
    return float(
        1.0
        + math.sqrt(x / (2.0 * math.pi)) * math.exp(-x / 8.0)
        + 1.5 * math.exp(x) * norm.cdf(-1.5 * math.sqrt(x))
        - 0.5 * (x + 5.0) * norm.cdf(-0.5 * math.sqrt(x))
    )


# Verify the transcription against the two published anchors BEFORE writing.
assert abs((2.0 * argmax_cdf(7.7) - 1.0) - 0.90) < 5e-4
assert abs((2.0 * argmax_cdf(11.03) - 1.0) - 0.95) < 5e-4

CRIT90 = float(brentq(lambda c: 2.0 * argmax_cdf(c) - 1.0 - 0.90, 1e-9, 100.0))
CRIT95 = float(brentq(lambda c: 2.0 * argmax_cdf(c) - 1.0 - 0.95, 1e-9, 100.0))


def break_cis(y: np.ndarray, X: np.ndarray, dates: list, regimes: list, ssr_m: float):
    """Bai (1997) homogeneous-case break-date confidence intervals."""
    T, q = X.shape
    Q = (X.T @ X) / T
    s2 = ssr_m / (T - (len(dates) + 1) * q)
    out = []
    for i, d in enumerate(dates):
        delta = np.array(regimes[i + 1]["params"]) - np.array(regimes[i]["params"])
        L = float(delta @ Q @ delta) / s2
        hw90 = math.ceil(CRIT90 / L)
        hw95 = math.ceil(CRIT95 / L)
        out.append(
            {
                "date": d,
                "scale": L,
                "lower90": int(max(0, d - hw90 - 1)),
                "upper90": int(min(T - 1, d + hw90 + 1)),
                "lower95": int(max(0, d - hw95 - 1)),
                "upper95": int(min(T - 1, d + hw95 + 1)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Case builders.
# ---------------------------------------------------------------------------


def dp_case(name, y, X, trim, m_list):
    """Brute-force exact optimal partitions for m in m_list."""
    T, q = X.shape
    h = math.ceil(trim * T)
    tbl = ssr_table(y, X, h)
    per_m = []
    for m in m_list:
        ssr, dates = brute_force(T, h, m, tbl)
        per_m.append({"m": m, "ssr": ssr, "dates": dates})
    return {
        "name": name,
        "y": [float(v) for v in y],
        "x": [[float(v) for v in col] for col in X.T],
        "trim": trim,
        "h": h,
        "ssr0": tbl[(0, T - 1)],
        "optima": per_m,
    }


def bai_perron_case(name, y, X, trim, max_breaks):
    """Full Bai-Perron reference output for the golden test."""
    T, q = X.shape
    h = math.ceil(trim * T)
    tbl = ssr_table(y, X, h)
    ssr_path = [tbl[(0, T - 1)]]
    dates_by_m = []
    for m in range(1, max_breaks + 1):
        ssr, dates = brute_force(T, h, m, tbl)
        ssr_path.append(ssr)
        dates_by_m.append(dates)
    seq = []
    for l in range(max_breaks):
        null_dates = [] if l == 0 else dates_by_m[l - 1]
        seq.append(supf_seq_stat(T, q, h, null_dates, tbl))
    n_breaks = select_n_breaks(seq, q, trim)
    sel_dates = [] if n_breaks == 0 else dates_by_m[n_breaks - 1]
    regimes = regime_fits(y, X, sel_dates)
    cis = break_cis(y, X, sel_dates, regimes, ssr_path[n_breaks]) if n_breaks else []
    return {
        "name": name,
        "y": [float(v) for v in y],
        "x": [[float(v) for v in col] for col in X.T],
        "trim": trim,
        "max_breaks": max_breaks,
        "h": h,
        "ssr_path": ssr_path,
        "break_dates_by_m": dates_by_m,
        "sup_f_seq": seq,
        "sup_f_crit": list(BP_SEQ_CV_5PCT[round(trim, 2)][q - 1][:max_breaks]),
        "n_breaks": n_breaks,
        "break_dates": sel_dates,
        "regimes": regimes,
        "ci": cis,
    }


def supf_case(name, y, X, trim):
    T, q = X.shape
    h = math.ceil(trim * T)
    ssr0, dates, path, bdate, stat = supf_full(y, X, h)
    tau = h / T
    return {
        "name": name,
        "y": [float(v) for v in y],
        "x": [[float(v) for v in col] for col in X.T],
        "trim": trim,
        "h": h,
        "tau": tau,
        "ssr0": ssr0,
        "dates": dates,
        "f_path": path,
        "break_date": bdate,
        "stat": stat,
        "p_value": hansen_supf_pvalue(stat, q, tau),
    }


def main():
    fixture = {
        "_meta": {
            "generator": "fixtures/generate_tsecon-breaks_fixtures.py",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "references": [
                "Bai & Perron (1998), Econometrica 66(1), 47-78",
                "Bai & Perron (2003), J. Applied Econometrics 18(1), 1-22",
                "Bai & Perron (2003b), Econometrics Journal 6(1), 72-78 "
                "(critical values, via CRAN mbreaks R/supF_next)",
                "Bai (1997), Review of Economics and Statistics 79(4), 551-563",
                "Andrews (1993), Econometrica 61(4), 821-856",
                "Hansen (1997), JBES 15(1), 60-67 "
                "(response surfaces, via CRAN strucchange sc.beta.sup)",
            ],
        }
    }

    # ---- DP == brute force cases (small T; exhaustive enumeration) --------
    dp_cases = []

    rng = np.random.default_rng(20260721)
    T = 30
    y = rng.standard_normal(T)
    y[14:] += 2.0  # one mean shift
    X = np.ones((T, 1))
    dp_cases.append(dp_case("mean_shift_T30", y, X, 0.2, [1, 2]))

    rng = np.random.default_rng(7)
    T = 40
    x1 = rng.standard_normal(T)
    beta = np.where(np.arange(T) < 22, 1.5, -0.5)
    y = 0.3 + beta * x1 + 0.7 * rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x1])
    dp_cases.append(dp_case("slope_break_T40_q2", y, X, 0.15, [1, 2]))

    rng = np.random.default_rng(99)
    T = 36
    y = rng.standard_normal(T)  # no break at all
    X = np.ones((T, 1))
    dp_cases.append(dp_case("no_break_T36", y, X, 0.25, [1, 2]))

    rng = np.random.default_rng(1234)
    T = 50
    x1 = 0.5 * np.sin(0.3 * np.arange(T)) + rng.standard_normal(T)
    mu = np.where(np.arange(T) < 17, 0.0, np.where(np.arange(T) < 33, 2.5, -1.0))
    y = mu + 0.8 * x1 + 0.6 * rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x1])
    dp_cases.append(dp_case("two_breaks_T50_q2", y, X, 0.10, [1, 2, 3]))

    fixture["dp_cases"] = dp_cases

    # ---- Full bai_perron cases -------------------------------------------
    rng = np.random.default_rng(31415)
    T = 120
    x1 = rng.standard_normal(T)
    seg = np.digitize(np.arange(T), [40, 80])  # regimes end at 39, 79
    alpha = np.array([0.0, 2.0, -1.0])[seg]
    beta = np.array([1.0, -1.0, 0.5])[seg]
    y = alpha + beta * x1 + 0.8 * rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x1])
    fixture["bai_perron_case"] = bai_perron_case("two_breaks_T120_q2", y, X, 0.15, 3)

    rng = np.random.default_rng(2718)
    T = 90
    y = rng.standard_normal(T)  # pure noise: expect 0 breaks
    X = np.ones((T, 1))
    fixture["bai_perron_null_case"] = bai_perron_case("null_T90_q1", y, X, 0.15, 2)

    # ---- Andrews sup-F cases ---------------------------------------------
    sup_cases = []

    rng = np.random.default_rng(555)
    T = 100
    y = rng.standard_normal(T)
    y[42:] += 1.2
    X = np.ones((T, 1))
    sup_cases.append(supf_case("mean_shift_T100_q1_trim15", y, X, 0.15))

    rng = np.random.default_rng(8080)
    T = 150
    x1 = rng.standard_normal(T)
    y = 0.5 - 0.3 * x1 + rng.standard_normal(T)  # stable: high p-value
    X = np.column_stack([np.ones(T), x1])
    sup_cases.append(supf_case("stable_T150_q2_trim10", y, X, 0.10))

    rng = np.random.default_rng(4242)
    T = 120
    x1 = rng.standard_normal(T)
    x2 = 0.4 * np.cos(0.2 * np.arange(T)) + rng.standard_normal(T)
    b1 = np.where(np.arange(T) < 60, 0.5, 1.8)
    y = 0.2 + b1 * x1 - 0.6 * x2 + 0.9 * rng.standard_normal(T)
    X = np.column_stack([np.ones(T), x1, x2])
    sup_cases.append(supf_case("slope_break_T120_q3_trim15", y, X, 0.15))

    fixture["sup_f_cases"] = sup_cases

    # ---- Hansen p-value unit cases (exercise every interpolation branch) --
    hansen_cases = []
    for stat, q, tau in [
        (8.58, 1, 0.15),
        (12.0, 2, 0.15),
        (25.0, 5, 0.15),
        (10.0, 2, 0.12),
        (5.0, 1, 0.5),
        (3.0, 1, 0.495),
        (15.0, 3, 0.005),
        (0.5, 1, 0.15),
    ]:
        hansen_cases.append(
            {"stat": stat, "q": q, "tau": tau, "p": hansen_supf_pvalue(stat, q, tau)}
        )
    fixture["hansen_pvalue_cases"] = hansen_cases

    # ---- Bai argmax cdf and two-sided critical values --------------------
    xs = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 7.7, 11.03, 20.0, -1.0, -7.7]
    fixture["argmax_cdf"] = {
        "x": xs,
        "cdf": [argmax_cdf(v) for v in xs],
        "crit90": CRIT90,
        "crit95": CRIT95,
    }

    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    print(
        "bai_perron_case: n_breaks =",
        fixture["bai_perron_case"]["n_breaks"],
        "dates =",
        fixture["bai_perron_case"]["break_dates"],
        "| null case n_breaks =",
        fixture["bai_perron_null_case"]["n_breaks"],
    )
    print("crit90 =", CRIT90, "crit95 =", CRIT95)


if __name__ == "__main__":
    main()
