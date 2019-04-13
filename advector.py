"""Particle trajectory computation module.

In this module, we compute the path of the particles on the equilibrium state
as functions of the poloidal angle theta.

This is done using the integrability of the system, using the three invariant
quantities
    - mu
    - energy = A/2 vpar**2 + mu B + Z Phi(psi)
    - ltor = A R vpar - Z psi

The idea is to compute the trajectories as function of the poloidal angle
theta, ie. solving the following system of equations:
    E = A/2 vpar(theta)**2 + mu B(psi(theta), theta) + Z Phi(psi(theta))
    P = A R(psi(theta), theta) vpar(theta) - Z psi(theta)

This is done point-by-point by Picard iteration on the solved formulas
    psi(theta) = A R(psi(theta), theta) vpar(theta) - P/Z
    vpar(theta) = sqrt(2/A) sqrt(E - mu B(psi(theta), theta) - Z Phi(psi(theta))

This method is computationally cheap and converges relatively quickly.

Future directions:
    - Newton iteration?
"""

import numpy as np
from scipy.interpolate import make_interp_spline as interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
plt.ion()

mpl.rcParams['figure.figsize'] = 10, 5

class ParticleAdvector:
    def __init__(self, grid, pot):
        self._grid = grid
        self._pot = interp1d(grid.psi.squeeze(), pot)

    def compute_trajectory(self):
        """
        This method computes the path of a trapped or passing particle.

        More specifically, we want to compute the path `psi(theta)`
        of the particle, parametrized by:
        - the average position `psi` ;
        - the parallel velocity `vpar` ;
        - the magnetic momentum `mu`.
        """
        g = self._grid
        A = g.A; Z = g.Z; R0 = g.R0
        P = self._pot

        sign = g.sign

        vpar = g.vpar
        psi  = g.psi
        r    = g.radius
        Rred = 1 + r/R0

        ener = g.vpar**2 + 2/A / Rred * g.mu + 2*Z/A * P(psi)
        ltor = psi - A/Z * R0 * Rred * vpar

        shape = np.broadcast(psi, g.theta, g.mu, vpar).shape

        np.who(locals())

        for _ in range(3):
            vpar_old, psi_old = vpar, psi

            r = g.radius_at(psi)
            Rred = 1 + r/R0 * np.cos(g.theta)

            if False:
                err = np.zeros(shape + (2,))
                err[..., 0] = - ltor + psi - A/Z * R0 * Rred * vpar
                err[..., 1] = - ener + g.vpar**2 + 2/A * g.mu / Rred + 2*Z/A * P(psi)

                jac = np.zeros(shape + (2, 2))
                jac[..., 0, 0] = 1 - A/Z * np.cos(g.theta) * vpar
                jac[..., 0, 1] = - A/Z * R0 * Rred
                jac[..., 1, 0] = - 2/A/R0 * g.mu * (np.cos(g.theta) / Rred**2)\
                                 + 2*Z/A * P(psi, nu=1)
                jac[..., 1, 1] = 2 * g.vpar

                diff = np.linalg.solve(jac, err)
                psi  -= diff[..., 0]
                vpar -= diff[..., 1]

            else:
                vpar2 = ener - 2/A * g.mu / Rred - 2*Z/A * P(psi)
                vpar2 = np.clip(vpar2, 0, None)
                vpar = sign * np.sqrt(vpar2)
                psi = ltor + A/Z * R0 * Rred * vpar

                # Anchor at slowest position.
                # This allows to have the same anchor the two sides of a
                # trapped particle, while controlling the error for passing
                # particles.
                avg_psi = np.take_along_axis(
                    psi,
                    np.argmin(abs(vpar), axis=0)[np.newaxis],
                    axis=0,
                )
                psi -= avg_psi
                psi += g.psi

            if np.allclose(vpar_old, vpar) and np.allclose(psi_old, psi):
                break

        assert np.all(np.isfinite(psi))
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(vpar))
        self._psi = psi
        self._r = r
        self._vpar = vpar
        self._Rred = Rred

        # Compute trapped particles
        trapped = np.any(vpar == 0, axis=0, keepdims=True) & g.mu.astype(bool)
        if g.vpar.squeeze().ndim == 2:
            assert np.all(trapped[..., 0] == trapped[..., 1])
            trapped = trapped[..., 0]
        self._trapped = trapped.squeeze()

    def compute_freq(self):
        g = self._grid
        q = g.qprofile_at(self._r)
        theta = self._grid.theta.squeeze()

        vtheta  = self._vpar / (q * g.R0 * self._Rred)
        vtheta += self._pot(self._psi, nu=1)
        # FIXME Add vD.gradtheta

        ifreq = 1/vtheta
        ifreq[~np.isfinite(ifreq)] = 0
        spline = interp1d(theta, ifreq)
        self._ifreq = spline
        self._int_time = spline.antiderivative()

        vphi  = self._vpar / (g.R0 * self._Rred)
        # FIXME Add vE.gradphi
        # FIXME Add vD.gradphi

        self._vphi = vphi
        spline = interp1d(theta, vphi * ifreq)
        self._int_phi = spline.antiderivative()

        # Compute full-bounce quantities
        bounce_time = self._int_time(theta[-1]) - self._int_time(theta[0])
        bounce_phi  = self._int_phi (theta[-1]) - self._int_time(theta[0])

        # Get positive frequencies
        bounce_sign  = np.sign(bounce_time)
        bounce_phi  *= bounce_sign
        bounce_time *= bounce_sign

        # Add the mirror part for trapped particles
        trapped = self._trapped
        bounce_time[trapped, :] = bounce_time[trapped, :].sum(axis=-1, keepdims=True)
        bounce_phi [trapped, :] = bounce_phi [trapped, :].sum(axis=-1, keepdims=True)

        self._bounce_time = bounce_time
        self._bounce_phi  = bounce_phi

        # Compute liveness of particles
        self._living = interp1d(theta, self._vpar != 0, k=1)

    @property
    def ifreq(self):
        return self._ifreq

    @property
    def psi_path(self):
        return self._psi

    @property
    def time_path(self):
        return self._int_time

    @property
    def bounce_time(self):
        return self._bounce_time

    @property
    def phi_path(self):
        return self._int_phi

    def living_path(self, theta):
        return self._living(theta)

    @property
    def bounce_phi(self):
        return self._bounce_phi

    @property
    def trapped(self):
        return self._trapped

def main():
    A = Z = 1
    R0 = 900

    # Build grid
    rg = np.linspace(100, 150, 18)
    qq = 1.3 + 0*np.linspace(0, 1, rg.size)**2

    theta = np.linspace(- np.pi, np.pi, 41)
    vpar = np.multiply.outer(np.linspace(.1, 4, 12), [1, -1])
    mu = np.linspace(0, 1, 8)

    grid = Grid(1, 1, 900, rg, qq, theta, vpar, mu)
    np.who(grid.__dict__)

    # Advect particles
    pot = np.zeros_like(rg)
    adv = ParticleAdvector(grid, pot)
    adv.compute_trajectory()
    np.who(adv.__dict__)

    plt.figure()
    plt.subplot(121)
    plt.plot(theta, adv._r[:, 16].reshape(theta.size, -1))
    plt.subplot(122)
    plt.plot(theta, adv._vpar[:, 16].reshape(theta.size, -1))
    plt.show()

    # Compute trajectory timing
    adv.compute_freq()
    np.who(adv.__dict__)

    time = adv.time_path(grid.theta.squeeze())
    phi  = adv.phi_path(grid.theta.squeeze())
    plt.figure()
    plt.subplot(121)
    plt.plot(
        theta,
        time[:, 16].reshape(theta.size, -1),
    )
    plt.subplot(122)
    plt.plot(
        theta,
        phi[:, 16].reshape(theta.size, -1),
    )
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        plt.show(block=True)
