import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def draw_contourf(domain_x: np.ndarray, domain_y: np.ndarray, z_points: np.ndarray, x_opt: list[float], name: str, POINTS=50):
    plt.contourf(domain_x, domain_y, z_points, POINTS)
    plt.title(name)
    # plt.plot(*x_opt, 'o', c='yellow')
    # plt.text(*x_opt, "Óptimo")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()


x1, x2, x3 = sp.symbols("x1 x2 x3")
f = -2 * sp.np.exp(-(x1 - 1)**2 - (x2 - 1)**2)
print(sp.latex(f))
print(f)
df = sp.Matrix([f]).jacobian((x1, x2)).T
print(df)
d2f = sp.hessian(f, (x1, x2))
print(d2f)



