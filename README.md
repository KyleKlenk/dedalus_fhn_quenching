# dedalus_fhn_quenching
Code for the investigation of quenching of stable pulses in FitzHugh-Nagumo using the Dedalus Project.

Instructions:
- install [dedalus](https://github.com/DedalusProject/dedalus)  (version 2.2207) in an environment (conda recommended -- see `environment.lock.yml`). Environment may be instantiated using 
```sh
conda env create -f environment.lock.yml
```
- activate the conda environment with `conda activate dedalus2`
- run `waves.sh` to generate the input data for the stable and unstable pulses; this involves the solution[^1][^2] of a nonlinear boundary value problem (NLBVP) for $(u,c)$:

$$
	D u'' + c u' + f(u) = 0, B_l(u, u') = 0, B_r(u, u') = 0.
$$

This code then solves the left (AVP) and right (EVP) eigenproblems,

$$ 
	v \sigma = L v, w^\dagger \sigma^* = L^\dagger w^\dagger,
$$

for the leading eigenmodes (e.g., $Re(\sigma) > -Re(\sigma_1) $).

whose boundary conditions are inherited from the NLBVP formulation of $B_l(u,u')$ and $B_r(u,u')$.[^3]
- run `quench.sh` which solves the quenching problem for a selected stable pulse and (presently hard-coded) family of perturbations. This uses a bisection root-finding procedure to solve for the root of the function,

$$
	f(U_q) = \psi(\check{u} + U_q \check{X}(x-\theta; x_s)) - \hat{\psi},
$$

for $0 > U_q \geq U_q^{max}$, where $|U_q^{max}| \gg 1$ is 'large', yielding $(x_s, \theta, U_q)$ tuples.

The corresponding linear theory calculation code is available [here](https://github.com/cmarcotte/linear_prediction/tree/main).

[^1]: This is technically implemented as a multi-fidelity continuation; first the input wave is continued in $\gamma$ with half as many modes and doubled dealiasing for intermediate $\gamma$s, then refined when $\gamma$ is in the predefined set of interesting values to yield a resolved wave. 
[^2]: This is a substantial calculation at the defaults (in part due to an inefficiency with handling constant fields in dedalus 2); if running for curiousity, I recommend setting `N=1024` and `L=1000` in `waves.sh` for `index=10` -- unfortunately only the tolerance can be tweaked for `index=11` and still successfully find the solution (e.g. `tol=3e-7`).
[^3]: The eigenproblems are solved as each interesting $\gamma$ value yields a fully resolved wave.
