"""Independent reference implementations from the published formulas.

Doornik & Hansen (2008), OBES 70 s1, 927-939, section 3 & 4:
  centre, S = (1/n) sum (x - xbar)(x - xbar)', V = diag(S)^{-1/2}, C = V S V,
  C = H L H', y_i = H L^{-1/2} H' V (x_i - xbar)
  per column: sqrt(b1), b2 sample skewness and (raw) kurtosis.
  z1: D'Agostino (1970) skewness transform; z2: Shenton-Bowman gamma + Wilson-Hilferty.
"""
import numpy as np
from scipy import stats


def dagostino_z1(sb1, n):
    beta = 3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1.0 + np.sqrt(2.0 * (beta - 1.0))
    delta = 1.0 / np.sqrt(np.log(np.sqrt(w2)))
    y = sb1 * np.sqrt((w2 - 1.0) / 2.0 * (n + 1) * (n + 3) / (6.0 * (n - 2)))
    return delta * np.log(y + np.sqrt(y * y + 1.0))


def wh_z2(b1, b2, n):
    d = (n - 3) * (n + 1) * (n * n + 15 * n - 4)
    a = (n - 2) * (n + 5) * (n + 7) * (n * n + 27 * n - 70) / (6.0 * d)
    c = (n - 7) * (n + 5) * (n + 7) * (n * n + 2 * n - 5) / (6.0 * d)
    k = (n + 5) * (n + 7) * (n ** 3 + 37 * n * n + 11 * n - 313) / (12.0 * d)
    alpha = a + b1 * c
    chi = (b2 - 1.0 - b1) * 2.0 * k
    return (np.cbrt(chi / (2.0 * alpha)) - 1.0 + 1.0 / (9.0 * alpha)) * np.sqrt(9.0 * alpha)


def doornik_hansen(X):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    V = np.diag(1.0 / np.sqrt(np.diag(S)))
    C = V @ S @ V
    lam, H = np.linalg.eigh(C)
    Y = Xc @ V @ H @ np.diag(lam ** -0.5) @ H.T  # rows y_i'
    m2 = (Y ** 2).mean(0)
    m3 = (Y ** 3).mean(0)
    m4 = (Y ** 4).mean(0)
    sb1 = m3 / m2 ** 1.5
    b2 = m4 / m2 ** 2
    b1 = sb1 ** 2
    z1 = dagostino_z1(sb1, n)
    z2 = wh_z2(b1, b2, n)
    Ep = float(z1 @ z1 + z2 @ z2)
    return dict(
        doornik_hansen=Ep,
        doornik_hansen_pvalue=stats.chi2.sf(Ep, 2 * k),
        doornik_hansen_skewness=float(z1 @ z1),
        doornik_hansen_skewness_pvalue=stats.chi2.sf(z1 @ z1, k),
        doornik_hansen_kurtosis=float(z2 @ z2),
        doornik_hansen_kurtosis_pvalue=stats.chi2.sf(z2 @ z2, k),
        doornik_hansen_z_skewness=z1,
        doornik_hansen_z_kurtosis=z2,
        doornik_hansen_df=2 * k,
    )


def mjb(X, root="chol"):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    if root == "chol":
        Pinv = np.linalg.inv(np.linalg.cholesky(S))
    else:
        lam, H = np.linalg.eigh(S)
        Pinv = H @ np.diag(lam ** -0.5) @ H.T
    w = Pinv @ Xc.T
    b1 = (w ** 3).sum(1) / n
    b2 = (w ** 4).sum(1) / n - 3
    ls = n * b1 @ b1 / 6
    lk = n * b2 @ b2 / 24
    return dict(skew=ls, kurt=lk, jb=ls + lk, b1=b1, b2=b2,
                jb_p=stats.chi2.sf(ls + lk, 2 * k), skew_p=stats.chi2.sf(ls, k), kurt_p=stats.chi2.sf(lk, k))
