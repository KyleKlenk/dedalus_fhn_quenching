"""

This is a convenience code to make the different parameter sets for Karma callable.
Options are:
  thesis
  A file from Auto bifurcation diagram (requires an index to select from within file, default 1)

"""

import numpy as np
from dedalus import public as de


def gendomain(N, L, dealias, complex=False):

    z_basis = de.Chebyshev("z", N, interval=(-7 * L / 8, +1 * L / 8), dealias=dealias)

    # Compound basis seems like a good idea -- resolve central region densely and resolve relaxation sparsely, but the implementation seems to have an issue where some discontinuities in (u,v,c) arise at the interior subdomains if the number of modes in each adjacent domain do not match (maybe even if they do match?), so until resolved, use a single domain.
    #
    # zb1 = de.Chebyshev('z1',3*N//8, interval=(-7*L/8,-2*L/8), dealias=dealias)
    # zb2 = de.Chebyshev('z2',5*N//8, interval=(-2*L/8,+1*L/8 ), dealias=dealias)
    # z_basis = de.Compound('z', (zb1, zb2), dealias=dealias)
    #

    if complex:
        dtype = np.complex128
    else:
        dtype = np.float64

    domain = de.Domain([z_basis], dtype)
    z = domain.grid(0, scales=dealias)

    return (z_basis, domain, z)


def fhn_params(setname, index=1):

    p = np.loadtxt(setname)
    p = p[index - 1, :]
    par = dict(
        alpha=p[1], beta=p[2], gamma=p[3], c=p[10], L=p[11], J=p[12], D1=p[15], D2=p[16]
    )

    return par
