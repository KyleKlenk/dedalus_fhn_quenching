import numpy as np
from scipy import interpolate


def load_and_interp_auto_sol(filename, xx):
    sol = np.loadtxt(filename)
    x, y = sol[:, 0], sol[:, 1:]
    f = interpolate.PchipInterpolator(x, y, axis=0)
    yy = f(xx)

    u = []
    for n in range(yy.shape[1]):
        u.append(yy[:, n])

    return np.array(u)


def dilate_and_scale_sol(u, amp):
    U = np.copy(u)
    U[0] = U[0] * amp
    return U


def tanhpert(x, u, us, xs, s, ind=0):
    U = np.copy(u)
    pert = us / 4.0
    pert = pert * (1.0 + np.sign(x - s + xs / 2))  # rise H(x-s+xs/2)
    pert = pert * (1.0 - np.sign(x - s - xs / 2))  # fall H(x-s-xs/2)
    # guarantees L1(pert) = xs*us
    U[ind] += pert
    return U


def gaussianpert(x, u, us, xs, s, ind=0):
    U = np.copy(u)
    pert = us * np.exp(-(((x - s) / xs) ** 2.0)) / np.sqrt(np.pi)
    # guarantees L1(pert) = xs*us
    U[ind] += pert
    return U
