import json, sys, time, warnings
import numpy as np
import tsecon
warnings.simplefilter("ignore")
_rng = np.random.default_rng(12345)


def _ar1(T, seed=0, phi=0.5):
    r = np.random.default_rng(seed)
    e = r.standard_normal(T)
    y = np.empty(T)
    prev = 0.0
    for t in range(T):
        prev = phi * prev + e[t]
        y[t] = prev
    return y


def _varsim(T, k, seed=7):
    r = np.random.default_rng(seed)
    A = 0.5 * np.eye(k) + 0.1 * (np.ones((k, k)) - np.eye(k))
    X = np.zeros((T, k))
    for t in range(1, T):
        X[t] = X[t - 1] @ A.T + r.standard_normal(k)
    return X


def _garch(T, seed=3):
    r = np.random.default_rng(seed)
    z = r.standard_normal(T)
    y = np.empty(T)
    s2 = 1.0
    for t in range(T):
        y[t] = np.sqrt(s2) * z[t]
        s2 = 0.05 + 0.08 * y[t] ** 2 + 0.88 * s2
    return y


T = 200
Y = _ar1(T)
Y2 = _ar1(T, 1)
Y3 = _ar1(T, 2)
RW = np.cumsum(np.random.default_rng(5).standard_normal(T))
POS = 10.0 + np.abs(_ar1(T, 9)) + np.arange(T) * 0.05
SEAS = 10.0 + np.sin(2 * np.pi * np.arange(96) / 12) + 0.3 * np.random.default_rng(4).standard_normal(96) + 0.02 * np.arange(96)
X2 = _varsim(T, 2)
X3 = _varsim(T, 3)
_r = np.random.default_rng(4)
XD = np.column_stack([np.ones(T), _r.standard_normal((T, 2))])
YD = XD @ np.array([0.5, 1.0, -0.5]) + np.random.default_rng(104).standard_normal(T)
XNC = _r.standard_normal((T, 3))
YNC = XNC @ np.array([1.0, 0.0, -0.5]) + _r.standard_normal(T)
_x = np.cumsum(np.random.default_rng(11).standard_normal(T))
C2 = np.column_stack([1.5 * _x + np.random.default_rng(12).standard_normal(T), _x])
CY = C2[:, 0]
CX = C2[:, 1:]
G = _garch(500)
G2 = np.column_stack([_garch(500, 3), _garch(500, 4)])
G3 = np.column_stack([_garch(500, 3), _garch(500, 4), _garch(500, 6)])
# panel N x T
PN, PT = 6, 40
POUT = np.array([_ar1(PT, i) for i in range(PN)]) + np.arange(PN)[:, None]
PREG = np.array([np.array([_ar1(PT, 10 + i) for i in range(PN)])])
PSHOCK = _ar1(PT, 99)
_tr = np.zeros((PN, PT)); _tr[0, 20:] = 1; _tr[1, 25:] = 1; _tr[2, 30:] = 1
PTREAT = _tr
YS = [_ar1(60, i) for i in range(5)]
XS = [np.random.default_rng(30 + i).standard_normal((60, 2)) for i in range(5)]
ENT = [_varsim(80, 2, seed=40 + i) for i in range(4)]
# curves / yields
MAT = np.array([3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 84.0, 120.0])
def _ns_load(m, lam=0.0609):
    a = lam * m
    l1 = (1 - np.exp(-a)) / a
    return np.column_stack([np.ones_like(m), l1, l1 - np.exp(-a)])
_F = np.column_stack([5 + np.cumsum(0.1 * np.random.default_rng(21).standard_normal(120)),
                      -2 + np.cumsum(0.1 * np.random.default_rng(22).standard_normal(120)),
                      1 + np.cumsum(0.1 * np.random.default_rng(23).standard_normal(120))])
YC = _F @ _ns_load(MAT).T + 0.02 * np.random.default_rng(24).standard_normal((120, len(MAT)))
YC1 = YC[-1]
MATI = np.arange(1, 13)
_Fd = np.column_stack([0.05 + np.cumsum(0.001 * np.random.default_rng(31).standard_normal(150)),
                       -0.01 + np.cumsum(0.001 * np.random.default_rng(32).standard_normal(150)),
                       0.005 + np.cumsum(0.001 * np.random.default_rng(33).standard_normal(150))])
YLD = _Fd @ _ns_load(MATI.astype(float), 0.3).T + 0.0002 * np.random.default_rng(34).standard_normal((150, 12))
CURVES = np.diff(YC, axis=0)  # 119 x 8
CY_OUT = _ar1(119, 77)
DELTA = np.ones(8) * 0.1
# copula u
_z = np.random.default_rng(50).multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=300)
import math as _math
U = 0.5 * (1.0 + np.vectorize(_math.erf)(_z / np.sqrt(2.0)))
# losses
LOSS = np.abs(np.random.default_rng(51).standard_normal((200, 4))) + np.array([0.0, 0.05, 0.1, 0.2])
BENCH = np.abs(np.random.default_rng(52).standard_normal(200))
MODELS = np.abs(np.random.default_rng(53).standard_normal((200, 3))) + 0.05
# intraday
IR = 0.001 * np.random.default_rng(60).standard_normal(390)
HI = 101 + np.abs(np.random.default_rng(61).standard_normal(20))
LO = 99 - np.abs(np.random.default_rng(62).standard_normal(20))
CHAINS = np.array([_ar1(400, i, 0.3) for i in range(4)])
# dfm panel
_f = _ar1(80, 70, 0.7)
DFMP = np.column_stack([_f * (0.5 + 0.2 * i) + 0.6 * np.random.default_rng(71 + i).standard_normal(80) for i in range(5)])
DFMR = DFMP.copy(); DFMR[-1, 2:] = np.nan
DFMR2 = DFMP.copy(); DFMR2[-1, 2:] = np.nan; DFMR2[-2, 4] = np.nan
DFMOLD = DFMP.copy(); DFMOLD[-1, 1:] = np.nan
DFMNEW = DFMP.copy(); DFMNEW[-1, 3:] = np.nan
# midas
YM = _ar1(120, 80)
HF = np.random.default_rng(81).standard_normal((120, 6))
# binary
_lat = 0.8 * XD[:, 1] + np.random.default_rng(90).standard_normal(T)
YBIN = (_lat > 0.5).astype(float)
# threshold
SETY = np.zeros(300)
_e = np.random.default_rng(91).standard_normal(300)
for _t in range(1, 300):
    SETY[_t] = (0.6 if SETY[_t - 1] <= 0 else -0.3) * SETY[_t - 1] + _e[_t]
TV = _varsim(300, 2, seed=92)
REG = (np.arange(T) >= T // 2).astype(int)
PROXY = (X3[:, 0] - 0.5 * np.roll(X3[:, 0], 1)) + 0.5 * np.random.default_rng(93).standard_normal(T)
RX = np.random.default_rng(94).standard_normal(T)
_xp = np.zeros(T)
for _t in range(1, T):
    _xp[_t] = 0.95 * _xp[_t - 1] + np.random.default_rng(95 + _t).standard_normal()
XP = _xp
RP = 0.05 * np.r_[0.0, XP[:-1]] + RX
HIGHDIM_X = np.random.default_rng(96).standard_normal((T, 10))
HIGHDIM_Y = HIGHDIM_X[:, 0] - 0.5 * HIGHDIM_X[:, 3] + np.random.default_rng(97).standard_normal(T)
HD_D = HIGHDIM_X[:, 1] + np.random.default_rng(98).standard_normal(T)
GRP = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
VAR_FC = -1.65 * np.ones(T)
HITS = (np.random.default_rng(99).random(T) < 0.05).astype(float)
# survey panel
SURV = np.random.default_rng(100).standard_normal((40, 8))
ERR = _ar1(80, 101)
REV = _ar1(80, 102)
FE_REG = np.column_stack([np.ones(80), _ar1(80, 103)])
# DSGE (simple 1 predetermined + 1 jump; saddle path)
DSGE_A = np.eye(2)
DSGE_B = np.array([[0.9, 0.1], [0.2, 1.5]])
DSGE_C = np.array([[1.0], [0.0]])
SR3 = [(0, 0, 0, "+"), (1, 0, 0, "+")]


def wv(A, idx, v):
    B = np.array(A, dtype=float, copy=True); B[idx] = v; return B
