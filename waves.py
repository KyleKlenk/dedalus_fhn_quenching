"""
Usage:
waves.py [options]

Options:
--N=<N>                       number of modes               [default:         1024]
--L=<L>                       domain truncation             [default:         300.]
--dealias=<dealias>           dealias factor                [default:            2]
--index=<index>               initial solution index        [default:           11]
--tol=<tol>										solution tolerance            [default:        3e-15]
--autodir=<autodir>						AUTO outputs directory
--soldir=<soldir>             solution directory
"""

from docopt import docopt
import h5py

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from dedalus import public as de

from fhn import *
from fhn_params import *
import pert

args = docopt(__doc__)
N = int(args["--N"])
da = max([int(args["--dealias"]), int(2048 / N)])
index = int(args["--index"])  # index=11: fast waves, index=9: slow waves
tol = float(args["--tol"])
autodir = args["--autodir"]
soldir = args["--soldir"]

nvar = 2

# define model pameters
p = fhn_params(f"{autodir}/_label", index=index)

# define initial domain size
L0 = p["L"]

# define truncated domain size
p["L"] = float(args["--L"])

# change q
p["q"] = 1.0

# Set up domain
(z_basis, domain, z) = gendomain(N, p["L"], da)

# ancillary field
q = domain.new_field(name="q")
q.set_scales(da)

# define L^1 norm using q so that if q = w*v, then L^1(q) == L^2(w*v)**2
int_op = z_basis.Integrate(q)


def scaleeigenfunctions(w, v, u, ind=0):
    if ind == 1:
        nm = np.argmax(np.abs(v[0, :]))
        v = v / v[0, nm]
        w = w / w[0, nm]
    elif ind == 2:
        q["g"] = np.copy(u[0, :])
        q["g"] = np.copy(z_basis.Differentiate(q).evaluate()["g"])
        nm = np.argmax(q["g"])
        v = v * q["g"][nm] / v[0, nm]
        w = w / w[0, nm]
    else:
        print(f"ind = {ind}, don" "t know how to deal with it.")

    return (w, v)


def orthonormalize(w, v):

    q["g"] = np.copy(w[0, :] * v[0, :] + w[1, :] * v[1, :])
    pnorm = np.sqrt(int_op.evaluate()["g"][0])  # sqrt(L^2(w*v)**2)
    w = w / pnorm
    v = v / 1.0
    return (w, v, pnorm)


# load critical state with N modes
normalizedoffsetgrid = (z + L0 / 2) / L0
auto_state = pert.load_and_interp_auto_sol(
    f"{autodir}/_solution_{index+1}", normalizedoffsetgrid
)

# gammas to solve for
gg = np.array([0.025, 0.02, 0.01, 0.001])

wave = np.copy(auto_state)

for n, g in enumerate(gg):

    dirname = f"{soldir}/{n}"

    while p["gamma"] > g:

        p["gamma"] = np.max(np.array([p["gamma"] / 2.0, g]))

        # save directory
        print("gamma = {1:2.16f}, saving things to {0}.".format(dirname, p["gamma"]))

        # BVP
        (wave, p) = fhn_bvp(
            wave,
            p,
            N // 2,
            dealias=2 * da,
            tol=np.sqrt(tol),
            maxiter=128,
            savestring=dirname,
        )

    if n == 0 and p["gamma"] < g:
        print("Increasing state gamma from {0:2.6f} to {1:2.6f}.".format(p["gamma"], g))
        p["gamma"] = g
        # BVP
        (wave, p) = fhn_bvp(
            wave,
            p,
            N // 2,
            dealias=2 * da,
            tol=np.sqrt(tol),
            maxiter=128,
            savestring=dirname,
        )

    # BVP
    (wave, p) = fhn_bvp(wave, p, N, dealias=da, tol=tol, maxiter=32, savestring=dirname)

    # plot BVP solution
    plt.figure(figsize=(6, 3))
    plt.plot(z, wave[0], "-", label=r"$\hat{u}_1(z)$")
    plt.plot(z, wave[1], "-", label=r"$\hat{u}_2(z)$")
    plt.title(r"$c={0:+2.16f}$".format(p["c"]))
    plt.legend(loc=0)
    plt.xlabel(r"$z$")
    plt.savefig(f"{dirname}/U.svg", bbox_inches="tight")
    plt.close()

    # plot distance from rest state
    plt.figure(figsize=(6, 3))
    plt.plot(
        z,
        np.sqrt(wave[0] ** 2 + wave[1] ** 2),
        "-k",
        label=r"$\|\mathbf{u}(z)-\mathbf{u}_0\|_2$",
    )
    plt.legend(loc=0)
    plt.xlabel(r"$z$")
    plt.yscale("log")
    plt.savefig(f"{dirname}/celldist.svg", bbox_inches="tight")
    plt.close()

    # save params to text file
    np.savetxt(
        f"{dirname}/p",
        np.array([p["alpha"], p["beta"], p["gamma"], p["c"], p["L"], 1.0]),
    )

    # save data to text file
    np.savetxt(
        f"{dirname}/U", np.array([z, wave[0], wave[1], wave[2]]).T, fmt="%+2.16f"
    )

    # target eigenvalue
    te = 5 * p["beta"] / 4 + 3 * np.sqrt(5 * p["gamma"] * p["alpha"]) / 2
    te = np.linspace(-te, te, 5)[::-1]

    # number of eigenvalues
    ne = 1

    # EVP
    (V, sigma) = fhn_evp(
        wave,
        p,
        N,
        dealias=da,
        tol=5e-13,
        neig=ne,
        avp=False,
        targeteig=te,
        savestring=dirname,
    )

    # AVP
    (W, amgis) = fhn_evp(
        wave,
        p,
        N,
        dealias=da,
        tol=5e-13,
        neig=ne,
        avp=True,
        targeteig=te,
        savestring=dirname,
    )

    # plot eigenvalues
    plt.figure(figsize=(6, 3))
    plt.plot(
        np.real(sigma), np.imag(sigma), "ok", markerfacecolor="none", label=r"$\sigma$"
    )
    plt.plot(np.real(amgis), np.imag(amgis), ".k", label=r"$\bar{\sigma}$")
    plt.legend(loc=0)
    plt.xscale("symlog")
    plt.yscale("symlog")
    plt.xlabel(r"Re$(\sigma)$")
    plt.ylabel(r"Im$(\sigma)$")
    plt.savefig(f"{dirname}/spectrum.svg", bbox_inches="tight")
    plt.close()

    # save eigenvalues to text file
    np.savetxt(f"{dirname}/sigma", np.array(sigma).T, fmt=["%+2.16f%+2.16fi"])

    for ns, s in enumerate(sigma):
        if np.real(s) >= -1e-3:
            plt.figure(figsize=(6, 3))
            plt.plot(z, V[ns][0], "-", label=r"$\hat{V}_1(z)$")
            plt.plot(z, V[ns][1], "-", label=r"$\hat{V}_2(z)$")
            plt.title(r"$\sigma={0:+2.16f}$".format(s))
            plt.legend(loc=0)
            yl = plt.ylim()
            yl = np.max(np.abs(np.array(yl)))
            plt.ylim([-yl, yl])
            plt.yticks([])
            plt.xlabel(r"$z$")
            plt.savefig(f"{dirname}/V{ns+1}.svg", bbox_inches="tight")
            plt.close()

            # save eigenfunctions to text file
            np.savetxt(
                f"{dirname}/V{ns+1}",
                np.real(np.array([z, V[ns][0], V[ns][1]]).T),
                fmt=["%+2.16f", "%+2.16f", "%+2.16f"],
            )

    # save eigenvalues to text file
    np.savetxt(f"{dirname}/amgis", np.array(amgis).T, fmt=["%+2.16f%+2.16fi"])

    for ns, s in enumerate(amgis):
        if np.real(s) >= -1e-3:
            plt.figure(figsize=(6, 3))
            plt.plot(z, W[ns][0], "-", label=r"$\hat{W}_1(z)$")
            plt.plot(z, W[ns][1], "-", label=r"$\hat{W}_2(z)$")
            plt.title(r"$\sigma^*={0:+2.16f}$".format(s))
            plt.legend(loc=0)
            yl = plt.ylim()
            yl = np.max(np.abs(np.array(yl)))
            plt.ylim([-yl, yl])
            plt.yticks([])
            plt.xlabel(r"$z$")
            plt.savefig(f"{dirname}/W{ns+1}.svg", bbox_inches="tight")
            plt.close()

            # save eigenfunctions to text file
            np.savetxt(
                f"{dirname}/W{ns+1}",
                np.real(np.array([z, W[ns][0], W[ns][1]]).T),
                fmt=["%+2.16f", "%+2.16f", "%+2.16f"],
            )
