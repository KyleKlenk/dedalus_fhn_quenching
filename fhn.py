"""
1D Fitzhugh-Nagumo

This script should be ran serially (because it is 1D), and creates a space-time
plot of the computed solution.

"""

import numpy as np
import matplotlib.pyplot as plt

import scipy as sp

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

import h5py
import pathlib

import logging

logger = logging.getLogger(__name__)

from fhn_params import *


def projectionBCs(p, left=False):
    # Using projection boundary conditions
    u0 = np.array([0.0, 0.0, 0.0])
    f11 = -3 * u0[0] ** 2 - p["beta"] + 2 * (1 + p["beta"]) * u0[0]
    f12 = -p["q"] * (u0[1] ** (p["q"] - 1.0))
    f21 = p["alpha"] * p["gamma"]
    f22 = -p["gamma"]
    c = p["c"]
    projector = True
    prnt = False
    if not left:
        J = [[0, 0, 1], [-f21 / c, -f22 / c, 0], [-f11, -f12, -c]]
    elif left:
        J = [[0, 0, 1], [f12 / c, f22 / c, 0], [-f11, -f21, c]]
    J = np.array(J)
    if prnt:
        print(f"right? {not left}. left? {left}.")
        print("J:")
        print(J)
    S, B = sp.linalg.eig(J, right=True, left=False)
    if prnt:
        print("S:")
        print(S)
        print("B:")
        print(B)
    if projector:
        B = sp.linalg.inv(B)
        B = B.T
        if prnt:
            print("B^{-T}:")
            print(B)
    for n, s in enumerate(S):
        if np.isreal(np.real_if_close(s)):
            B[:, n] = np.real(B[:, n])
        elif np.imag(s) > 0.0:
            B[:, n] = np.real(B[:, n])
        elif np.imag(s) < 0.0:
            B[:, n] = np.imag(B[:, n])
            B[:, n] = B[:, n] / np.linalg.norm(B[:, n])
    if prnt:
        print("Q:")
        print(B)
    Lp = np.real(B[:, np.real(S) > 0.0].T)  # Lp is E_u
    Lm = np.real(B[:, np.real(S) <= 0.0].T)  # Lm is E_s
    return (Lp, Lm)


def fhn_bvp(
    state,
    p,
    N,
    dealias=1,
    tol=1e-13,
    maxiter=128,
    damped=True,
    savestring="",
    consc=True,
):

    # Set up domain
    (z_basis, domain, z) = gendomain(N, p["L"], dealias)

    # Setup problem
    varz = ["u", "v", "uz", "c"]

    problem = de.NLBVP(domain, variables=varz, ncc_cutoff=tol)

    problem.meta[:]["z"]["dirichlet"] = True
    if not consc:
        problem.meta["c"]["z"]["constant"] = True

    # precompute the BCs
    Lp, Lm = projectionBCs(p, left=False)

    # parameters
    c0 = p.pop("c")
    for par in p.keys():
        problem.parameters[par] = p[par]

    problem.substitutions["f1(u,v)"] = "u*(u-beta)*(1.0-u) - v**q"
    problem.substitutions["f2(u,v)"] = "gamma*(alpha*u - v)"

    problem.add_equation("dz(uz) = -(c*uz + f1(u,v))/D1")
    problem.add_equation("dz(v) = -f2(u,v)/c")
    problem.add_equation("dz(u) - uz = 0")
    if consc:
        problem.add_equation("dz(c) = 0")

    # add BCs using Lp,Lm
    problem.add_bc("interp(uz,z=0.0) = 0")
    problem.add_bc(f"{Lm[0,0]}*left(u)  + {Lm[0,1]}*left(v)  + {Lm[0,2]}*left(uz)  = 0")
    problem.add_bc(f"{Lp[0,0]}*right(u) + {Lp[0,1]}*right(v) + {Lp[0,2]}*right(uz) = 0")
    problem.add_bc(f"{Lp[1,0]}*right(u) + {Lp[1,1]}*right(v) + {Lp[1,2]}*right(uz) = 0")

    # build solver
    solver = problem.build_solver()

    # Initial conditions
    u = solver.state["u"]
    v = solver.state["v"]
    c = solver.state["c"]
    uz = solver.state["uz"]

    u.set_scales(dealias, keep_data=True)
    v.set_scales(dealias, keep_data=True)
    c.set_scales(dealias, keep_data=True)
    uz.set_scales(dealias, keep_data=True)

    u["g"] = np.copy(state[0])
    v["g"] = np.copy(state[1])
    c["g"] = c0
    u.differentiate(0, out=uz)

    u.set_scales(dealias, keep_data=True)
    v.set_scales(dealias, keep_data=True)
    c.set_scales(dealias, keep_data=True)
    uz.set_scales(dealias, keep_data=True)

    # Initial perturbation
    pert = solver.perturbations.data
    pert.fill(1.0 + tol)

    u0 = np.copy(u["g"])
    v0 = np.copy(v["g"])
    uz0 = np.copy(uz["g"])

    # path
    pathlib.Path("./{0}/".format(savestring)).mkdir(parents=True, exist_ok=True)

    # Solve system iteratively
    while solver.iteration < maxiter and np.max(np.abs(pert)) > tol:
        if damped:
            # This is stupid, but effective:
            solver.newton_iteration(
                damping=((1 + solver.iteration) / (1 + maxiter)) ** 0.5
            )
        else:
            solver.newton_iteration()
        logger.info(
            "Iteration: {1:2d}, perturbation L∞-norm: {0:2.16f}".format(
                np.max(np.abs(pert)), solver.iteration
            )
        )
        logger.info("-" * 50)

        if not np.all(np.isnan(u["g"])):

            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3, 5))
            fig.suptitle(
                r"Iteration {0}, $L^\infty = {1:+2.16f}$".format(
                    solver.iteration, np.max(np.abs(pert))
                )
            )
            ax[0].plot(z, u["g"], "-", label=r"$u_1$")
            ax[0].plot(z, v["g"], "-", label=r"$u_2$")
            ax[0].plot(z, c["g"], "-", label=r"$c$")
            ax[0].plot(z, uz["g"], "-", label=r"$\partial_zu_1$")
            ax[0].legend(loc=0)
            ax[1].plot(z, u["g"] - u0, "-", label=r"$\delta u_1$")
            ax[1].plot(z, v["g"] - v0, "-", label=r"$\delta u_2$")
            ax[1].plot(z, c["g"] - c0, "-", label=r"$\delta c$")
            ax[1].plot(z, uz["g"] - uz0, "-", label=r"$\delta \partial_zu_1$")
            ax[1].legend(loc=0)
            plt.xlabel(r"$z$")
            plt.savefig("./{0}/newton.svg".format(savestring), bbox_inches="tight")
            plt.savefig("./newton.svg".format(savestring), bbox_inches="tight")
            plt.close()

            u0 = np.copy(u["g"])
            v0 = np.copy(v["g"])
            uz0 = np.copy(uz["g"])
            c0 = np.copy(c["g"])
        else:
            u["g"] = np.copy(u0)
            v["g"] = np.copy(v0)
            uz["g"] = np.copy(uz0)
            c["g"] = np.copy(c0)
            print("nan err")
            break

    # copy state vector
    state = np.array([np.copy(u["g"]), np.copy(v["g"]), np.copy(uz["g"])])
    p["c"] = np.mean(c["g"])

    with h5py.File("./{0}/BVP.h5".format(savestring), mode="w") as file:
        file.create_dataset("z", data=z)
        file.create_dataset("u", data=u["g"])
        file.create_dataset("v", data=v["g"])
        file.create_dataset("uz", data=uz["g"])
        file.create_dataset("N", data=N)
        file.create_dataset("dealias", data=dealias)
        for par in p.keys():
            file.create_dataset("{0}".format(par), data=p[par])

    return (state, p)


def fhn_evp(
    state,
    p,
    N,
    dealias=1,
    tol=1e-10,
    neig=16,
    avp=False,
    targeteig=[0.0],
    savestring="",
):

    # Set up domain
    (z_basis, domain, z) = gendomain(N, p["L"], dealias, complex=True)

    # Problem
    problem = de.EVP(
        domain, variables=["U", "Uz", "V"], eigenvalue="sigma", tolerance=tol
    )

    problem.meta[:]["z"]["dirichlet"] = True

    # precompute the BCs
    if not avp:
        Lp, Lm = projectionBCs(p, left=False)
        fname = "EVP"
    else:
        Lp, Lm = projectionBCs(p, left=True)
        fname = "AVP"

    # set values and names from parameters dict
    for par in p.keys():
        problem.parameters[par] = p[par]

    # add nonlinear solution as NCC in equations
    ncc = domain.new_field(name="u")
    ncc.set_scales(np.size(state[0]) / N, keep_data=True)
    ncc["g"] = np.copy(state[0])
    ncc.set_scales(dealias, keep_data=True)
    problem.parameters["u"] = ncc
    ncc = domain.new_field(name="v")
    ncc.set_scales(np.size(state[0]) / N, keep_data=True)
    ncc["g"] = np.copy(state[1])
    ncc.set_scales(dealias, keep_data=True)
    problem.parameters["v"] = ncc

    # define local terms
    problem.substitutions["f11"] = "-3*u**2 - beta + 2*(1 + beta)*u"
    problem.substitutions["f12"] = "-q*(v**(q-1))"
    problem.substitutions["f21"] = "alpha*gamma"
    problem.substitutions["f22"] = "-gamma"

    if not avp:  # s*U - dF(u0)*U = 0
        problem.add_equation("sigma*U - (D1*dz(Uz) + c*Uz + f11*U + f12*V) = 0")
        problem.add_equation("sigma*V - (c*dz(V) + f21*U + f22*V) = 0")
        problem.add_equation("dz(U) - Uz = 0")
    elif avp:  # s*U - dF'(u0)*U = 0
        problem.add_equation("sigma*U - (D1*dz(Uz) - c*Uz + f11*U + f21*V) = 0")
        problem.add_equation("sigma*V - (-c*dz(V) + f12*U + f22*V) = 0")
        problem.add_equation("dz(U) - Uz = 0")

    # add BCs
    if not avp:
        problem.add_bc(
            f"{Lm[0,0]}*left(U)  + {Lm[0,1]}*left(V)  + {Lm[0,2]}*left(Uz)  = 0"
        )
        problem.add_bc(
            f"{Lp[0,0]}*right(U) + {Lp[0,1]}*right(V) + {Lp[0,2]}*right(Uz) = 0"
        )
        problem.add_bc(
            f"{Lp[1,0]}*right(U) + {Lp[1,1]}*right(V) + {Lp[1,2]}*right(Uz) = 0"
        )
    elif avp:
        problem.add_bc(
            f"{Lm[0,0]}*left(U)  + {Lm[0,1]}*left(V)  + {Lm[0,2]}*left(Uz)  = 0"
        )
        problem.add_bc(
            f"{Lm[1,0]}*left(U)  + {Lm[1,1]}*left(V)  + {Lm[1,2]}*left(Uz)  = 0"
        )
        problem.add_bc(
            f"{Lp[0,0]}*right(U) + {Lp[0,1]}*right(V) + {Lp[0,2]}*right(Uz) = 0"
        )

    # build solver
    solver = problem.build_solver()

    # solve eigensystem
    eigenvectors = []
    eigenvalues = []

    for te in targeteig:
        try:
            # solver solve
            solver.solve_sparse(
                solver.pencils[0], neig, te, maxiter=2**13, ncv=2**7, tol=tol
            )

            # sort by eigenvalue
            indx = solver.eigenvalues.argsort()[::-1]
            indx = indx[np.isfinite(solver.eigenvalues[indx])]

            if len(indx) > 0:

                print(f"Computed {len(indx)} eigenvalues near {te}...")

                for n in range(len(indx)):

                    # for the n-th mode
                    solver.set_state(indx[n])

                    # save mode amplitudes and eigenvalues to file
                    eigenvalues.append(solver.eigenvalues[indx[n]])

                    solver.state["U"].set_scales(dealias, keep_data=True)
                    solver.state["V"].set_scales(dealias, keep_data=True)

                    eigenvectors.append(
                        [
                            np.copy(solver.state["U"]["g"]),
                            np.copy(solver.state["V"]["g"]),
                        ]
                    )

            else:
                print(f"Eigenmode computation near {te} failed.")
        except:
            print(f"Eigenmode computation near {te} failed.")
            pass
    # after all computations are done, organize modes by largest real part
    indx = np.argsort(np.real(eigenvalues))[::-1]

    eigenvalues = np.array(eigenvalues)[indx]

    eigenvectors = np.array(eigenvectors)[indx, :, :]

    # filter the modes to exclude ~duplicates~
    indx = ~(np.triu(np.abs(eigenvalues[:, None] - eigenvalues) <= tol, 1)).any(0)

    eigenvalues = np.array(eigenvalues)[indx]

    eigenvectors = np.array(eigenvectors)[indx, :, :]

    # write to HDF5 file
    with h5py.File(f"./{savestring}/{fname}.h5", mode="w") as file:

        file.create_dataset("z", data=z)

        file.create_dataset("eigenvalues", data=eigenvalues)

        file.create_dataset("eigenvectors", data=eigenvectors)

    return (eigenvectors, eigenvalues)


def fhn_ivp(
    state,
    p,
    T,
    dt,
    N,
    dealias=1,
    tol=1e-10,
    savestring="",
    ur=[0.0, 0.0, 0.0],
    skippable=True,
):

    # Set up domain
    (z_basis, domain, z) = gendomain(N, p["L"], dealias)

    # compute time-steps for [0,T]
    Nt = np.ceil(T / dt)
    dt = T / Nt

    # Problem
    problem = de.IVP(domain, variables=["u", "uz", "v"], ncc_cutoff=tol)

    problem.meta[:]["z"]["dirichlet"] = True

    for par in p.keys():
        problem.parameters[par] = p[par]

    problem.substitutions["f1(u,v)"] = "u*(u-beta)*(1-u) - v"
    problem.substitutions["f2(u,v)"] = "gamma*(alpha*u - v)"

    problem.add_equation("dt(u) - D1*dz(uz) = f1(u,v)")
    problem.add_equation("dz(u) - uz = 0")
    problem.add_equation("dt(v) = f2(u,v)")

    problem.add_bc(" left(u) - right(u) = 0")
    problem.add_bc(" left(uz) - right(uz) = 0")

    # Build solver
    solver = problem.build_solver(de.timesteppers.SBDF4)
    solver.stop_sim_time = T
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    # Initial conditions
    u = solver.state["u"]
    v = solver.state["v"]
    uz = solver.state["uz"]

    u.set_scales(dealias, keep_data=True)
    v.set_scales(dealias, keep_data=True)
    uz.set_scales(dealias, keep_data=True)

    # assign initial conditions
    u["g"] = np.copy(state[0])
    v["g"] = np.copy(state[1])
    u.differentiate(0, out=uz)

    u.set_scales(dealias, keep_data=True)
    v.set_scales(dealias, keep_data=True)
    uz.set_scales(dealias, keep_data=True)

    # initial L1-norm of u
    psi0 = (u.integrate("z"))["g"][0]
    solver.evaluator.vars["psi0"] = psi0

    # path
    pathlib.Path("{0}/IVP/".format(savestring)).mkdir(parents=True, exist_ok=True)

    # Snapshots
    snapdir = "{0}/IVP/snapshots/".format(savestring)
    snapshots = solver.evaluator.add_file_handler(snapdir, sim_dt=1.0, mode="overwrite")
    snapshots.add_system(solver.state)

    # Analysis
    analdir = "{0}/IVP/analysis/".format(savestring)
    analysis = solver.evaluator.add_file_handler(analdir, iter=1, mode="overwrite")
    analysis.add_task("integ(abs(u-{0}),'z')".format(ur[0]), name="psi")

    # turn on time-step shortening if np.max(np.abs(u['g'])) > some threshold
    if np.max(np.abs(u["g"])) > 1.0:
        minnn = -1 - 2 * np.log(np.max(np.abs(u["g"])))
    else:
        minnn = 0

    # Main loop
    logger.debug("t = {0:.5E}, dt= {1:.5E}".format(solver.sim_time, dt))
    while solver.ok:
        nn = np.min(
            [np.max([int(np.log2(solver.sim_time / 100.0 + 2.0**minnn)), minnn]), 0]
        )
        solver.step(dt * 2.0**nn)

        if solver.iteration % 2**13 == 0:
            logger.debug(
                "t = {0:.5E}, dt= {1:.5E}".format(solver.sim_time, dt * 2.0**nn)
            )

        if np.any(np.isnan(u["g"])):
            logger.info(
                "t = {0:.5E}, dt= {1:.5E}, isnan(u) = {2}".format(
                    solver.sim_time, dt * 2.0**nn, np.any(np.isnan(u["g"]))
                )
            )
            break

        if skippable and (
            np.max(np.abs(u["g"])) < 1e-06 and np.max(np.abs(v["g"])) < 1e-06
        ):
            logger.info(
                "t = {0:.5E}, dt= {1:.5E}, Skipping...".format(
                    solver.sim_time, dt * 2.0**nn
                )
            )
            break

    # make sure T is the final time, for skippable things
    T = solver.sim_time

    # copy state vector
    state = np.array([np.copy(u["g"]), np.copy(v["g"])])

    with h5py.File(
        "{0}/IVP/analysis/analysis_s1/analysis_s1_p0.h5".format(savestring), mode="r"
    ) as file:
        t = np.ravel(file["scales"]["sim_time"])
        psi = np.ravel(file["tasks"]["psi"])
        plt.figure()
        plt.plot(t, psi, label=r"$\psi(t)$")
        plt.xlabel(r"$t$")
        plt.legend(loc=0, edgecolor=(1, 1, 1), facecolor=(1, 1, 1), framealpha=0.9)
        plt.gcf().set_size_inches(1.618 * 3, 3)
        plt.savefig("{0}/IVP/psi.svg".format(savestring), bbox_inches="tight")
        # plt.savefig('./psi.svg'.format(savestring),bbox_inches='tight')
        plt.close()
    # Create u(t,x) spacetime plot
    with h5py.File(
        "{0}/IVP/snapshots/snapshots_s1/snapshots_s1_p0.h5".format(savestring), mode="r"
    ) as file:
        t = np.ravel(file["scales"]["sim_time"])
        z = np.ravel(file["scales"]["z"]["1.0"])
        u = file["tasks"]["u"]
        xmesh, ymesh = quad_mesh(x=z, y=t)
        plt.figure()
        plt.pcolormesh(
            xmesh, ymesh, u, cmap="seismic", rasterized=True, vmin=-1.0, vmax=+1.0
        )
        plt.axis(pad_limits(xmesh, ymesh))
        plt.colorbar()
        plt.xlabel(r"$z$")
        plt.ylabel(r"$t$")
        plt.savefig(
            "{0}/IVP/dynamics.svg".format(savestring), bbox_inches="tight", dpi=300
        )
        # plt.savefig('./dynamics.svg'.format(savestring),bbox_inches='tight',dpi=300)
        plt.close()
    # Create u(t,x) spacetime plot
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, constrained_layout=True)
    with h5py.File(
        "{0}/IVP/snapshots/snapshots_s1/snapshots_s1_p0.h5".format(savestring), mode="r"
    ) as file:
        t = np.ravel(file["scales"]["sim_time"])
        z = np.ravel(file["scales"]["z"]["1.0"])
        u = np.array(file["tasks"]["u"])
        xmesh, ymesh = quad_mesh(x=t, y=z)
        im = axs[0].pcolormesh(
            xmesh, ymesh, u.T, cmap="seismic", rasterized=True, vmin=-1.0, vmax=+1.0
        )
        # plt.axis(pad_limits(xmesh, ymesh))
        plt.colorbar(im, ax=axs[:-1], label=r"$u_1(t,z)$", location="right")
    with h5py.File(
        "{0}/IVP/analysis/analysis_s1/analysis_s1_p0.h5".format(savestring), mode="r"
    ) as file:
        t = np.ravel(file["scales"]["sim_time"])
        psi = np.ravel(file["tasks"]["psi"])
        axs[1].plot(t, psi, label=r"$\psi(t)$")
        axs[1].legend(loc=0, edgecolor=(1, 1, 1), facecolor=(1, 1, 1), framealpha=0.9)
        axs[1].set_xlim([t[0], t[-1]])
        axs[1].set_yscale("log")
        axs[1].set_xlabel(r"$t$")
        axs[0].set_ylabel(r"$z$")
    plt.savefig("./complete.svg".format(savestring), bbox_inches="tight", dpi=300)
    plt.savefig("{0}/IVP/complete.svg".format(savestring), bbox_inches="tight", dpi=300)
    plt.close()

    return state
