"""
Usage:
sweep.py [options]
	
Options:
--savedir=<savedir>		Save directory
--Ufile=<Ufile>				fast wave file	
--Pfile=<Pfile>				fast wave params
--ufile=<ufile>				slow wave file	
--pfile=<pfile>				slow wave params
--N=<N>  							number of modes  	[default: 4096]
--dealias=<dealias>  	dealias factor 		[default: 2]
--dt=<dt>  						max time-step 		[default: 0.015625]
--theta=<theta>  			theta							[default: 0.0]
--tol=<tol>  					tol 							[default: 1e-06]
"""

from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline

from dedalus import public as de
import pathlib
import h5py

from fhn import *
from fhn_params import *
import pert

args = docopt(__doc__)

savedir = args["--savedir"]
N = int(args["--N"])
dealias = int(args["--dealias"])
dt = float(args["--dt"])
theta = float(args["--theta"])
tol = float(args["--tol"])
Ufile = args["--Ufile"]
Pfile = args["--Pfile"]
ufile = args["--ufile"]
pfile = args["--pfile"]

nvar = 2

# path
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

# load model parameters
p = np.loadtxt(pfile)
p = dict(alpha=p[0], beta=p[1], gamma=p[2], c=p[3], L=p[4], D1=p[5])
P = np.loadtxt(Pfile)
P = dict(alpha=P[0], beta=P[1], gamma=P[2], c=P[3], L=P[4], D1=P[5])

# define ranges for x and t, x \in interval, t \in [0,T]
L = P["L"]
T = L / (2 * P["c"])

# get appropriate bound
usmin = -1.0  # initial min (will adapt)
usmax = +0.0  # initial max (will adapt)
usmin0 = -1000.0  # ultimate min
usmax0 = +0.0  # ultimate max
ltol = np.sqrt(tol)

# stable wave
ur = np.loadtxt(Ufile).T

# unstable wave
uc = np.loadtxt(ufile).T

# grid
(z_basis, domain, z) = gendomain(N, p["L"], dealias)

# make ancillary field variable for norm calculatons
du = domain.new_field(name="du")

# L1-norm of uc

uc = [InterpolatedUnivariateSpline(uc[0][:], uc[m][:], k=3, ext=3)(z) for m in range(1, 3)]
du.set_scales(len(uc[0]) / N, keep_data=True)
du["g"] = np.copy(np.abs(uc[0]))
nc = (du.integrate("z"))["g"][0]

# grid
(z_basis, domain, z) = gendomain(N, P["L"], dealias)

# make ancillary field variable for norm calculatons
du = domain.new_field(name="du")

# L1-norm of ur
ur = [InterpolatedUnivariateSpline(ur[0], ur[m], k=3, ext=3)(z) for m in range(1, 3)]
du.set_scales(len(ur[0]) / N, keep_data=True)
du["g"] = np.copy(np.abs(ur[0]))
nr = (du.integrate("z"))["g"][0]

# truncate the fields at N modes
for n in range(len(ur)):
    du.set_scales(len(ur[n]) / N, keep_data=True)
    du["g"] = np.copy(ur[n])
    du.set_scales(dealias, keep_data=True)
    ur[n] = np.copy(du["g"])

# print the L1 norms
print(f"|û|_1 = {nc}, |ǔ|_1 = {nr}")


# pert
def pertprofile(x, u, extent, strength, s):
    return pert.tanhpert(x, u, extent, strength, s)


# size of DNS sweep
NDNS = 129

# create arrays for DNS sweep
xs = np.logspace(np.log10(2 * L / N), np.log10(2 * L), NDNS)[::-1]
us = np.zeros_like(xs)
th = theta * np.ones_like(xs)  # theta = constant; if theta varies then idk
ex = np.zeros_like(xs)
sr = np.zeros_like(xs)

# define scaling search factor
dA = 1.1

# make sure us is properly agnostic
us = np.nan * us

# filename
perdir = f"{savedir}/run/"

for n in range(0, len(xs)):
    # plot (A, phi)
    Afig = plt.figure()

    # compute extent (only changes for each xs)
    dU = pertprofile(
        z, [np.zeros(N * dealias), np.zeros(N * dealias)], 1.0, xs[n], th[n]
    )
    du.set_scales(len(dU[0]) / N, keep_data=True)
    du["g"] = np.copy(np.abs(dU[0]))
    ex[n] = (du.integrate("z"))["g"][0]

    # slow, accurate function
    def f(
        A, skippable=True, T=T, tol=tol, color="k", ms=5, N=N, dealias=dealias, dt=dt
    ):

        # check that perturbation decays to rest_state
        init_state = pertprofile(z, ur, A, xs[n], th[n])

        if np.abs(A) < tol:
            skippable = False

        # check evolution
        sol = fhn_ivp(
            init_state,
            P,
            T,
            dt,
            N,
            dealias=dealias,
            tol=tol,
            savestring=perdir,
            skippable=skippable,
        )

        # check psi(T) = norm(u(T,x), 1)
        with h5py.File(
            f"{perdir}/IVP/analysis/analysis_s1/analysis_s1_p0.h5", mode="r"
        ) as file:
            pp = np.ravel(file["tasks"]["psi"])[-1]

        # print some stuff
        print(f"{perdir}: A = {A}, f(A) = {pp-nc}")

        # plot (A, psi)
        plt.figure(Afig.number)
        plt.plot(A, pp - nc, ".{0}".format(color), markersize=ms)
        plt.xlabel(r"$A$")
        plt.ylabel(r"$\psi_{T}(A)$")
        plt.savefig(f"./A.svg", bbox_inches="tight")
        plt.savefig(f"{perdir}/A.svg", bbox_inches="tight")

        # compare psi(T) = norm(u(T,x), 1) to norm(uc(x),1)
        return pp - nc

    # solve the fast problem approximately to accelerate the slow problem solution
    try:
        us[n] = brentq(f, usmin, usmax, xtol=tol)
        dA = np.max([xs[n + 1] / xs[n], xs[n] / xs[n + 1]])
        usmin = us[n] * dA  # we expect that us increases with n (xs decreases)
        usmax = us[n] / dA  # and we wish to keep usmin < us[n] < usmax
        sr[n] = np.abs(us[n])
    except:
        print("=" * 130)
        print(
            "\n\tInitial root-finding failed, falling back to exhaustive search with looser initial tolerances.\n"
        )
        try:
            us[n] = brentq(f, usmin0, usmax0, xtol=tol)
            dA = np.max([xs[n + 1] / xs[n], xs[n] / xs[n + 1]])
            usmin = us[n] * dA
            usmax = us[n] / dA
            sr[n] = np.abs(us[n])
        except:
            us[n] = np.nan
            pass
        # close Afig
        plt.figure(Afig.number)
        plt.close()
    plt.close("all")

# save results
with h5py.File(f"{savedir}/crit_dns.h5", "w") as file:
    file.create_dataset("xs", data=xs)
    file.create_dataset("us", data=us)
    file.create_dataset("th", data=th)
    file.create_dataset("ex", data=ex)
    file.create_dataset("sr", data=sr)
