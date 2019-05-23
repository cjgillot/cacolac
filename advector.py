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
import scipy.optimize
from scipy.interpolate import make_interp_spline as interp1d

class Energy:
    def __init__(self, grid, shape, pot, ener, ltor):
        self._grid = grid
        self._shape = shape
        self._pot = pot
        self._ener = ener
        self._ltor = ltor
        self._maxerr = None
        self._ZAR = grid.Z/grid.A/grid.R0

    def set_maxerr(self, maxerr):
        self._maxerr = maxerr

    def value(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)

        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)
        U = self._ZAR * (self._ltor - psi)
        P = self._pot(psi)
        E = g.A/2 * U**2 / Rred**2 + g.mu / Rred + g.Z * P - self._ener
        E = E.ravel()
        if self._maxerr is not None:
            maxerr = self._maxerr
            E = E.clip(-maxerr, maxerr)
        return E

    def dpsi(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)

        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)
        dRr_dy = g.radius_at(psi, nu=1)/g.R0 * np.cos(theta)

        U = self._ZAR * (self._ltor - psi)
        dU_dy = - self._ZAR
        dP_dy = self._pot(psi, nu=1)
        dE_dy = (
            g.A * U * dU_dy/Rred**2
            - g.A * U**2 * dRr_dy/Rred**3
            - g.mu * dRr_dy/Rred**2
            + g.Z * dP_dy
        )
        return dE_dy.ravel()

    def dtheta(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)

        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)
        dRr_dj = - g.radius_at(psi)/g.R0 * np.sin(theta)

        U = self._ZAR * (self._ltor - psi)
        dE_dj = (
            - g.A * U**2 * dRr_dj/Rred**3
            - g.mu * dRr_dj/Rred**2
        )
        return dE_dj.ravel()

    def dltor(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)

        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)

        U = self._ZAR * (self._ltor - psi)
        dU_dl = self._ZAR
        dE_dl = g.A * U * dU_dl/Rred**2
        return dE_dl.ravel()

    def dpsidpsi(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)

        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)
        dRr_dy = g.radius_at(psi, nu=1)/g.R0 * np.cos(theta)
        d2Rr_dydy = g.radius_at(psi, nu=2)/g.R0 * np.cos(theta)

        U = self._ZAR * (self._ltor - psi)
        dU_dy = - self._ZAR
        d2P_dydy = self._pot(psi, nu=2)
        d2E_dydy = (
            g.A * dU_dy**2 /Rred**2
            - 2 * g.A * U * dU_dy * dRr_dy/Rred**3
            + 3 * g.A * U**2 * dRr_dy**2/Rred**4
            - g.A * U**2 * d2Rr_dydy/Rred**3
            + 2 * g.mu * dRr_dy**2/Rred**3
            - g.mu * d2Rr_dydy/Rred**2
            + g.Z * d2P_dydy
        )
        return d2E_dydy.ravel()

    def dpsidtheta(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)

        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)
        dRr_dy = g.radius_at(psi, nu=1)/g.R0 * np.cos(theta)
        dRr_dj = - g.radius_at(psi)/g.R0 * np.sin(theta)
        d2Rr_dydj = - g.radius_at(psi, nu=1)/g.R0 * np.sin(theta)

        U = self._ZAR * (self._ltor - psi)
        dU_dy = - self._ZAR
        d2E_dydj = (
            - 2 * g.A * U * dU_dy * dRr_dj/Rred**3
            + 3 * g.A * U**2 * dRr_dy * dRr_dj/Rred**4
            - g.A * U**2 * d2Rr_dydj/Rred**3
            + 2 * g.mu * dRr_dy/Rred**3
            - g.mu * d2Rr_dydj/Rred**2
        )
        return d2E_dydj.ravel()

class Ptheta:
    def __init__(self, grid, shape, pot, ener, ltor):
        self._grid = grid
        self._shape = shape
        self._pot = pot
        self._ener = ener
        self._ltor = ltor
        self._maxerr = None
        self._ZAR = grid.Z/grid.A/grid.R0

    def set_maxerr(self, maxerr):
        self._maxerr = maxerr

    def value(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)
        raise NotImplementedError()

        r = g.radius_at(psi)
        q = g.qprofile_at(psi)
        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)

        U = self._ZAR * (ltor - psi)
        H = A/g.R0 * r**2/(q * Rred)

        P = U * H - Z * q
        return Ptheta.ravel()

    def dpsi(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)

        r = g.radius_at(psi)
        q = g.qprofile_at(psi)
        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)
        dRr_dy = g.radius_at(psi, nu=1)/g.R0 * np.cos(theta)

        U = self._ZAR * (self._ltor - psi)
        dU_dy = - self._ZAR

        H = g.A/g.R0 * r**2/(q * Rred)
        dH_dy = (
            2 * g.radius_at(psi, nu=1)
            - r * g.qprofile_at(psi, nu=1)/q
            - r * dRr_dy/Rred
        ) * g.A/g.R0 * r/(q * Rred)

        dPtheta_dy = dU_dy * H + U * dH_dy - g.Z * q
        return dPtheta_dy.ravel()

    def dltor(self, psi, theta):
        g = self._grid
        psi = psi.reshape(self._shape)

        r = g.radius_at(psi)
        q = g.qprofile_at(psi)
        Rred = 1 + g.radius_at(psi)/g.R0 * np.cos(theta)

        U = self._ZAR * (self._ltor - psi)
        dU_dl = self._ZAR

        H = g.A/g.R0 * r**2/(q * Rred)

        dPtheta_dl = dU_dl * H
        return dPtheta_dl.ravel()

class ParticleAdvector:
    def __init__(self, grid, pot):
        self._grid = grid
        self._pot = interp1d(grid.psi.squeeze(), pot)

    def compute_invariants(self):
        g = self._grid
        A = g.A; Z = g.Z; R0 = g.R0
        shape = np.broadcast(g.psi, g.vpar, g.mu).shape

        Rred = 1 + g.radius/R0
        ener = A/2 * g.vpar**2 + g.mu/Rred + Z * self._pot(g.psi)
        del Rred

        Rred  = 1 + g.radius/R0 * np.cos(g.theta)
        vp2   = ener - g.mu/Rred - Z * self._pot(g.psi)
        vpar0 = g.sign * np.sqrt(2/A) * np.sqrt(vp2.clip(0, None))

        trapped = vp2.min(axis=0) < 0
        assert np.all(trapped[..., 0] == trapped[..., 1])

        ltor = np.empty(shape[1:])
        ltor[:] = g.psi.squeeze(axis=0)
        ltor[~trapped] += A/Z * R0 * np.mean(Rred * vpar0, axis=0)[~trapped]

        self._ener = ener
        self._ltor = ltor
        self._trapped = trapped[..., 0]

    def compute_bounce_point(self):
        """
        This method computes position of the banana tip for trapped particles.

        More specifically, we want to compute the position `psi, theta`
        of the particle, parametrized by:
        - the average position `psi` ;
        - the parallel velocity `vpar` ;
        - the magnetic momentum `mu` ;
        and defined as `E = E_0` and `dE_dy = 0`.
        """
        g = self._grid
        shape = np.broadcast(g.psi, g.vpar, g.mu).shape[1:]

        ener = self._ener.squeeze(axis=0)
        ltor = self._ltor
        psi  = np.empty(shape)
        tht  = np.empty(shape)

        # Initial guess for banana tip: where vpar==0
        psi[:] = g.psi.squeeze(axis=0)
        tht[:] = np.arccos(
            g.R0 / g.radius_at(g.psi) * (
                g.mu / (ener - g.Z * self._pot(g.psi)) - 1
            )
        ).squeeze(axis=0)
        np.negative(tht[..., 1], out=tht[..., 1])

        # Compare two definitions of trapped particles
        trapped = np.isfinite(tht).all(axis=-1)
        assert not np.any(trapped ^ self._trapped)

        computer = Energy(
            self._grid, psi.shape,
            self._pot, ener, ltor,
        )

        # Compute banana tip by Newton iteration
        for _ in range(10):
            val      = computer.value     (psi, tht).reshape(shape)[trapped]
            dpsi     = computer.dpsi      (psi, tht).reshape(shape)[trapped]

            if np.all(abs(val) < 1e-6) and np.all(abs(dpsi) < 1e-6):
                break

            dtht     = computer.dtheta    (psi, tht).reshape(shape)[trapped]
            dpsidpsi = computer.dpsidpsi  (psi, tht).reshape(shape)[trapped]
            dpsidtht = computer.dpsidtheta(psi, tht).reshape(shape)[trapped]

            # Error and jacobian
            vec = np.empty((*val.shape, 2))
            jac = np.empty((*val.shape, 2, 2))

            vec[..., 0] = val
            vec[..., 1] = dpsi

            jac[..., 0, 0] = dpsi
            jac[..., 0, 1] = dtht
            jac[..., 1, 0] = dpsidpsi
            jac[..., 1, 1] = dpsidtht

            # Update
            delta = np.linalg.solve(jac, vec)
            assert np.all(np.isfinite(delta))

            psi[trapped, :] -= delta[..., 0]
            tht[trapped, :] -= delta[..., 1]

            if np.all(abs(delta) < 1e-6):
                break

        self._banana_psi   = psi[trapped]
        self._banana_theta = tht[trapped]

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
        theta = np.empty((2*g.theta.size-1,) + g.theta.shape[1:])
        theta[0::2] = g.theta
        theta[1::2] = .5 * (g.theta[1:] + g.theta[:-1])
        shape = np.broadcast(g.psi, theta, g.vpar, g.mu).shape

        ener = self._ener
        ltor = self._ltor

        Rred  = 1 + g.radius/R0 * np.cos(theta)
        vp2   = ener - g.mu/Rred - Z * self._pot(g.psi)
        vpar0 = g.sign * np.sqrt(2/A) * np.sqrt(vp2.clip(0, None))
        psi0  = ltor - A/Z * R0 * Rred * vpar0
        del Rred, vp2

        computer = Energy(
            self._grid, shape,
            self._pot, self._ener, self._ltor
        )

        maxerr = np.inf
        maxerr = abs(computer.value(psi0, theta)).max()

        computer.set_maxerr(maxerr)

        psi = scipy.optimize.zeros.newton(
            func=computer.value, fprime=computer.dpsi,
            x0=psi0.copy().ravel(),
            args=(theta,),
            maxiter=10,
        )
        solerr  = np.logical_not(abs(computer.value(psi, theta)) < 1e-5 * maxerr)
        solerr |= psi < 0
        psi[solerr] = psi0.ravel()[solerr]
        psi = psi.reshape(shape)
        r = g.radius_at(psi)
        Rred = 1 + r/R0 * np.cos(theta)
        vpar = (ltor - psi) * Z/A/R0 / Rred

        assert np.all(np.isfinite(psi))
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(vpar))
        self._psi  = psi [0::2]
        self._r    = r   [0::2]
        self._vpar = vpar[0::2]
        self._Rred = Rred[0::2]
        self._mid_psi  = psi [1::2]
        self._mid_r    = r   [1::2]
        self._mid_vpar = vpar[1::2]
        self._mid_Rred = Rred[1::2]

    def compute_time(self):
        g = self._grid
        A = g.A; Z = g.Z; R0 = g.R0
        theta = g.theta
        theta = .5 * (theta[1:] + theta[:-1])
        shape = np.broadcast(g.psi, theta, g.vpar, g.mu).shape

        ltor = self._ltor
        psi  = self._mid_psi
        r    = self._mid_r

        energy = Energy(
            self._grid, shape,
            self._pot, self._ener, self._ltor
        )
        dE_dy = energy.dpsi(self._mid_psi, theta)
        dE_dl = energy.dltor(self._mid_psi, theta)

        ptheta = Ptheta(
            self._grid, shape,
            self._pot, self._ener, self._ltor
        )
        dP_dy = ptheta.dpsi(self._mid_psi, theta)
        dP_dl = ptheta.dltor(self._mid_psi, theta)

        dt_dtheta = dP_dy / dE_dy
        dphi_dtheta = (dP_dy * dE_dl / dE_dy - dP_dl) / g.Z

        dt_dtheta   = dt_dtheta  .reshape(shape)
        dphi_dtheta = dphi_dtheta.reshape(shape)

        self._dt_dtheta, self._time = self._regularize_trapped(
            theta, dt_dtheta
        )
        self._dphi_dtheta, self._phi = self._regularize_trapped(
            theta, dphi_dtheta
        )

        self._bounce_time = self._time[-1] - self._time[0]
        self._bounce_phi  = self._phi [-1] - self._phi [0]

    def _regularize_trapped(self, theta, dt_dtheta):
        """Trapped trajectories need some regularisation near the banana tips.
        """
        g = self._grid
        theta   = theta.squeeze()
        trapped = self._trapped
        dtheta  = np.diff(theta).mean()

        # Load banana tips
        uptip = self._banana_theta[..., 0]
        dwtip = self._banana_theta[..., 1]

        # Restrict to trapped support
        dt_dtheta[:, trapped] *= np.less.outer(theta, uptip)[..., np.newaxis]
        dt_dtheta[:, trapped] *= np.greater.outer(theta, dwtip)[..., np.newaxis]

        # Weight the last point for the square-root divergence
        #   2 = \int_0^1 dx/sqrt(x)
        uptip_idx = theta.searchsorted(uptip, 'left')
        dwtip_idx = theta.searchsorted(dwtip, 'right')
        uptip = uptip - theta[uptip_idx-1]
        dwtip = theta[dwtip_idx  ] - dwtip

        dt_dtheta[uptip_idx, trapped].T[:] *= 2/dtheta * uptip
        dt_dtheta[dwtip_idx, trapped].T[:] *= 2/dtheta * dwtip

        # Integrate time series
        theta     = g.theta.squeeze()
        time      = np.zeros((theta.size, *dt_dtheta.shape[1:]))
        time[1:]  = np.cumsum(dt_dtheta, axis=0)
        time[1:] *= np.diff(theta)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        time     -= time[theta.searchsorted(0)]

        return dt_dtheta, time

    def compute_freq(self):
        g = self._grid
        q = g.qprofile_at(self._r)
        theta = self._grid.theta.squeeze()

        vtheta  = self._vpar / (q * g.R0 * self._Rred)
        vtheta += self._pot(self._psi, nu=1) * self._Rred / q
        # FIXME Add vD.gradtheta

        ifreq = 1/vtheta
        ifreq[~np.isfinite(ifreq)] = 0
        spline = interp1d(theta, ifreq)
        self._ifreq = spline
        self._int_time = spline.antiderivative()

        vphi  = self._vpar / (g.R0 * self._Rred)
        # FIXME Add vE.gradphi

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
    def vpar_path(self):
        return self._vpar

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
    from grid import Grid

    # General parameters
    A = Z = 1
    R0 = 900

    # Build grid
    rg = np.linspace(100, 150, 18)
    qq = 1.3 + 0*np.linspace(0, 1, rg.size)**2

    theta = np.linspace(- np.pi, np.pi, 141)
    vpar = np.multiply.outer(np.linspace(.1, 4, 12), [1, -1])
    mu = np.linspace(0, 1, 8)

    grid = Grid(A, Z, R0, rg, qq, theta, vpar, mu)
    np.who(grid.__dict__)

    # Advect particles
    pot = np.zeros_like(rg)
    adv = ParticleAdvector(grid, pot)
    adv.compute_invariants()
    adv.compute_trajectory()
    np.who(adv.__dict__)

    plt.figure()
    plt.subplot(121)
    plt.plot(theta, adv.psi_path[:, 16].reshape(theta.size, -1))
    plt.subplot(122)
    plt.plot(theta, adv.vpar_path[:, 16].reshape(theta.size, -1))
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
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.ion()

    mpl.rcParams['figure.figsize'] = 10, 5

    try:
        main()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        plt.show(block=True)
