import numpy as np
import pandas as pd
import numpy.linalg as la


W = pd.read_excel('W.xlsx').to_numpy()
X = pd.read_excel('X.xlsx').to_numpy()
Y = pd.read_excel('Y.xlsx').to_numpy().flatten()
sigmae = pd.read_excel('Sigma_e.xlsx').to_numpy()


def f_func(x):
    c = 0
    D = len(Y)
    A1 = np.identity(D) - x[1] * W
    C = A1 @ A1.T
    invC = la.inv(C)
    G = x[0] * invC + sigmae
    val_propios = la.eigvals(G)
    logdetG = np.log(val_propios).sum()
    invG = la.inv(G)
    XT_invG = X.T @ invG
    beta = la.inv(XT_invG @ X) @ XT_invG @ Y
    y5 = Y - X @ beta
    y6 = la.solve(G, y5)
    y1 = c - 1 / 2 * logdetG - 1 / 2 * np.dot(y5, y6)  # Función objetivo máximo
    y = -y1
    return y


def f_grad(x):
    # La gradiente de una función Func, no opcional
    D = len(Y)
    A1 = np.identity(D) - x[1] * W
    C = A1 @ A1.T
    inv_C = la.inv(C)
    G = x[0] * inv_C + sigmae
    C_G = C @ G
    invG = la.inv(G)
    XGX = X.T @ invG @ X
    z1 = la.solve(G, Y)
    z2 = X.T @ z1
    beta = la.solve(XGX, z2)
    Y2 = Y - X @ beta
    Y1 = la.solve(G @ C_G, Y2)
    invCG = la.inv(C_G)
    y1 = -0.5 * np.trace(invCG) + 0.5 * np.dot(Y2, Y1)
    M = 2 * x[1] * W @ W.T - (W + W.T)
    R3 = la.solve((G @ C), Y2)
    R2 = M @ R3
    R1 = la.solve(C_G, R2)
    y2 = 0.5 * x[0] * (np.trace((invCG @ M) @ inv_C) - np.dot(Y2, R1))
    # La gradiente del modelo LML es
    y = np.array([-y1, -y2])
    return y


def g_func(x):
    # La función Funcg(x,no) da como resultado un vector y que está evaluado en
    # las restricciones de gi(x), y 'no' es la opción a escoger
    g1 = -x[0] - 1
    g2 = x[0] - 2.5
    g3 = -x[1]
    g4 = x[1] - 0.03
    y = np.array([g1, g2, g3, g4])
    return y


def g_grad(x):
    g1 = [-1, 0]
    g2 = [1, 0]
    g3 = [0, -1]
    g4 = [0, 1]
    y = np.c_[g1, g2, g3, g4]
    return y
