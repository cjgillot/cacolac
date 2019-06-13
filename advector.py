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

In all the following, the variable `s_psi` (or `s`) denotes the sqrt of psi.
"""

import warnings

import numpy as np
import scipy.optimize
import scipy.integrate
from scipy.interpolate import make_interp_spline as interp1d

def double_mesh(array, *, axis):
    assert axis == 0
    shap = list(array.shape)
    shap[axis] *= 2
    shap[axis] -= 1
    ret  = np.empty(shap, dtype=array.dtype)
    ret[0::2]  = array
    ret[1::2]  = array[1:  ]
    ret[1::2] += array[ :-1]
    ret[1::2] /= 2
    return ret

class Energy:
    """This class encapsulates the computation of the particle energy
    (hamiltonian) in function of:
    - mu
    - ltor
    - psi
    - theta

    We also provide derivatives wrt. the variables to solve equations.
    """
    def __init__(self, grid, pot, ltor):
        """Constructor."""
        self._grid = grid
        self._pot = pot
        self._ltor = ltor
        self._ZAR = grid.Z/grid.A/grid.R0

    def value(self, psi, theta):
        """Compute energy."""
        g = self._grid

        Rred = g.Rred_at(psi, theta)
        U = self._ZAR * (self._ltor - psi)
        P = self._pot(psi)
        E = g.A/2 * U**2 / Rred**2 + g.mu / Rred + g.Z * P

        return E

    def ds(self, psi, theta):
        """Derivative wrt. psi."""
        g = self._grid

        s = np.sqrt(psi)
        Rred = g.Rred_at(psi, theta)
        dRr_ds = g.Rred_at(psi, theta, ds=1)

        U = self._ZAR * (self._ltor - psi)
        dU_ds = - 2 * self._ZAR * s
        dP_ds = 2 * s * self._pot(psi, nu=1)
        dE_ds = (
            g.A * U * dU_ds/Rred**2
            - g.A * U**2 * dRr_ds/Rred**3
            - g.mu * dRr_ds/Rred**2
            + g.Z * dP_ds
        )
        return dE_ds

    def dtheta(self, psi, theta):
        """Derivative wrt. theta."""
        g = self._grid

        Rred = g.Rred_at(psi, theta)
        dRr_dj = g.Rred_at(psi, theta, dj=1)

        U = self._ZAR * (self._ltor - psi)
        dE_dj = (
            - g.A * U**2 * dRr_dj/Rred**3
            - g.mu * dRr_dj/Rred**2
        )
        return dE_dj

    def dltor(self, psi, theta):
        """Derivative wrt. ltor."""
        g = self._grid

        Rred = g.Rred_at(psi, theta)

        U = self._ZAR * (self._ltor - psi)
        dU_dl = self._ZAR
        dE_dl = g.A * U * dU_dl/Rred**2
        return dE_dl

    def dsds(self, psi, theta):
        """Second derivative in psi."""
        g = self._grid

        s = np.sqrt(psi)
        Rred = g.Rred_at(psi, theta)
        dRr_ds = g.Rred_at(psi, theta, ds=1)
        d2Rr_dsds = g.Rred_at(psi, theta, ds=2)

        U = self._ZAR * (self._ltor - psi)
        dU_ds = - 2 * s * self._ZAR
        d2U_dsds = - 2 * self._ZAR
        d2P_dsds = 4 * psi * self._pot(psi, nu=2) + 2 * self._pot(psi, nu=1)
        d2E_dsds = (
            g.A * dU_ds**2 /Rred**2
            + g.A * U * d2U_dsds /Rred**2
            - 2 * g.A * U * dU_ds * dRr_ds/Rred**3
            + 3 * g.A * U**2 * dRr_ds**2/Rred**4
            - g.A * U**2 * d2Rr_dsds/Rred**3
            + 2 * g.mu * dRr_ds**2/Rred**3
            - g.mu * d2Rr_dsds/Rred**2
            + g.Z * d2P_dsds
        )
        return d2E_dsds

    def dsdtheta(self, psi, theta):
        """Crossed derivative in sqrt(psi) & theta."""
        g = self._grid

        s = np.sqrt(psi)
        Rred = g.Rred_at(psi, theta)
        dRr_ds = g.Rred_at(psi, theta, ds=1)
        dRr_dj = g.Rred_at(psi, theta, dj=1)
        d2Rr_dsdj = g.Rred_at(psi, theta, ds=1, dj=1)

        U = self._ZAR * (self._ltor - psi)
        dU_ds = - 2 * s * self._ZAR
        d2E_dsdj = (
            - 2 * g.A * U * dU_ds * dRr_dj/Rred**3
            + 3 * g.A * U**2 * dRr_ds * dRr_dj/Rred**4
            - g.A * U**2 * d2Rr_dsdj/Rred**3
            + 2 * g.mu * dRr_ds * dRr_dj/Rred**3
            - g.mu * d2Rr_dsdj/Rred**2
        )
        return d2E_dsdj

class Ptheta:
    """This class encapsulates the computation of the particle poloidal
    angular momentum in function of:
    - mu
    - ltor
    - psi
    - theta

    We also provide derivatives wrt. the variables to solve equations.
    """
    def __init__(self, grid, ltor):
        """Constructor."""
        self._grid = grid
        self._ltor = ltor
        self._ZAR = grid.Z/grid.A/grid.R0

    def _Ypol(self, psi, theta):
        """psi_pol = \int q R_0/R dpsi

        Since dpsi/dr = r/q, we get
            psi_pol(r) = \int r R_0/R dr
                       = r/a - ln(1 + a r)/a^2
            where a = cos(theta)/R_0
        """
        g = self._grid
        r = g.radius_at(psi)
        a = np.cos(theta)/g.R0
        ra = r * a
        Ypol  = np.log(1 + ra) - ra
        Ypol /= - a**2
        mask = abs(np.cos(theta)) < 1e-6
        mask = mask.squeeze()
        Ypol[mask] = r[mask]**2 / 2
        return Ypol

    def value(self, psi, theta):
        """Derivative wrt. psi."""
        g = self._grid

        r = g.radius_at(psi)
        q = g.qprofile_at(psi)
        Rred = g.Rred_at(psi, theta)

        U = self._ZAR * (self._ltor - psi)
        H = g.A/g.R0 * r**2/(q * Rred)
        Y = self._Ypol(psi, theta)

        return U * H - g.Z * Y

    def ds(self, psi, theta):
        """Derivative wrt. psi."""
        g = self._grid

        s = np.sqrt(psi)
        r = g.radius_at(psi)
        q = g.qprofile_at(psi)

        Rred = g.Rred_at(psi, theta)
        dRr_ds = g.Rred_at(psi, theta, ds=1)

        U = self._ZAR * (self._ltor - psi)
        dU_ds = - 2 * s * self._ZAR

        H = g.A/g.R0 * r**2/(q * Rred)
        dH_ds = (
            2 * g.radius_at(psi, ds=1)
            - r * g.qprofile_at(psi, ds=1)/q
            - r * dRr_ds/Rred
        ) * g.A/g.R0 * r/(q * Rred)

        dPtheta_ds = dU_ds * H + U * dH_ds - 2 * g.Z * s * q
        return dPtheta_ds

    def dltor(self, psi, theta):
        """Derivative wrt. ltor."""
        g = self._grid

        r = g.radius_at(psi)
        q = g.qprofile_at(psi)
        Rred = g.Rred_at(psi, theta)

        dU_dl = self._ZAR

        H = g.A/g.R0 * r**2/(q * Rred)

        dPtheta_dl = dU_dl * H
        return dPtheta_dl

class ParticleAdvector:
    """Compute trajectory of the particles parametrized by the poloidal angle.
    """
    def __init__(self, grid, pot):
        """Constructor."""
        self._grid = grid
        self._pot = interp1d(grid.psi.squeeze(), pot)

    def compute_invariants(self):
        """Initialize energy, toroidal momentum and trappedness.

        The toroidal momentum is defined to be independent of the velocity sign
        for trapped particles. This allows to model full-bounce in one go.
        """
        g = self._grid
        A = g.A; Z = g.Z; R0 = g.R0
        shape = np.broadcast(g.psi, g.vpar, g.mu).shape

        Rred = g.Rred_LFS
        ener = A/2 * g.vpar**2 + g.mu/Rred + Z * self._pot(g.psi)
        del Rred

        # We do not use a fancy trick here to have ltor closer to psi.
        # Doing so would mess up the computation of the velocity,
        # which is required everywhere in Energy and Ptheta.
        # FIXME Find a way for `g.psi` to be a grid in `psibar = <psi(theta)>`
        # instead of a grid in `ltor`.
        ltor = g.psi

        self._ener = ener.squeeze(axis=0)
        self._ltor = ltor.squeeze(axis=0)

    def _lowest_energy(self):
        """
        This method computes the lowest energy of a particle to pass
        high-field-side, parametrized by:
        - the average position `psi` ;
        - the parallel velocity `vpar` ;
        - the magnetic momentum `mu`.

        It is defined as the level lines of the energy as a function of mu and
        ltor. Those are found by Newton iteration.
        """
        g = self._grid

        # Search array shape, and mask of searched points.
        shape = np.broadcast(self._ltor, g.mu).shape
        theta = np.pi
        psi0  = np.broadcast_to(g.psi, shape)

        # Compute energy and psi-derivative
        computer = Energy(self._grid, self._pot, self._ltor)
        def fval(s_psi):
            psi = np.square(s_psi)
            psi = psi.reshape(shape)
            ret = computer.value(psi, theta)
            return ret.sum()
        def fprime(s_psi):
            psi = np.square(s_psi)
            psi = psi.reshape(shape)
            ret = computer.ds(psi, theta)
            ret += 1e3 * np.clip(psi, None, 0)
            ret += 1e3 * np.clip(psi - g.psimax, 0, None)
            return ret.ravel()

        # Find level line
        res = scipy.optimize.minimize(
            fun=fval, jac=fprime,
            x0=np.sqrt(psi0).ravel(),
            bounds=scipy.optimize.Bounds(
                np.sqrt(g.psimin)+1e-8,
                np.sqrt(g.psimax)
            ),
            method='L-BFGS-B',
            tol=1e-18,
        )
        res_x = res.x
        assert np.all(np.isfinite(res_x))
        np.square(res_x, out=res_x)

        # Minimal energy
        psi_min = res_x.reshape(shape).squeeze(axis=0)
        ener_min = computer.value(psi_min, theta).squeeze(axis=0)

        return psi_min, ener_min

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

        # The last axis (sign) is used to find the point for negative theta.
        shape = np.broadcast(g.psi, g.vpar, g.mu).shape[1:]

        # Compute particle trappedness
        psi_min, ener_min = self._lowest_energy()
        trapped = self._ener < ener_min
        assert np.all(trapped[..., 0] == trapped[..., 1])
        self._trapped = trapped[..., 0]

        # Initial guess for banana tip: where vpar==0
        psi  = np.empty(shape)
        tht  = np.empty(shape)

        with warnings.catch_warnings():
            # We rely on `nan`s to find trapped and passing particles
            warnings.simplefilter('ignore', RuntimeWarning)
            psi[:] = g.psi.squeeze(axis=0)
            tht[:] = np.arccos(
                g.R0 / g.radius_at(g.psi) * (
                    g.mu / (self._ener - g.Z * self._pot(g.psi)) - 1
                )
            ).squeeze(axis=0)
            np.negative(tht[..., 1], out=tht[..., 1])

        # Compare two definitions of trapped particles
        trapped  = np.isfinite(tht).all(axis=-1)
        # Minimal energy definition is more restrictive
        trapped &= self._trapped

        computer = Energy(
            self._grid,
            self._pot, self._ltor,
        )
        assert not np.any(trapped ^ self._trapped)

        # Sanitize input for solver
        tht[~trapped] = 0

        ener = self._ener[trapped, 0][:, np.newaxis]
        shape = np.r_[1, trapped.shape, 2]

        relevant    = np.ones(shape[1:], dtype=bool)
        relevant[:] = trapped[..., np.newaxis]

        def sel(a):
            a = np.broadcast_to(a, shape)
            return a

        # Compute banana tip by Newton iteration
        for _ in range(10):
            np.clip(psi, 0, None, out=psi)
            assert np.all(np.isfinite(psi))
            assert np.all(np.isfinite(tht))

            val   = sel(computer.value(psi, tht))[:, trapped]
            val  -= ener
            dpsi  = sel(computer.ds   (psi, tht))[:, trapped]

            relevant[trapped] &= (
                (abs(val) > 1e-12) |
                (abs(dpsi) > 1e-12)
            ).squeeze(axis=0)

            if not np.any(relevant):
                break

            val      = val [:, relevant[trapped]]
            dpsi     = dpsi[:, relevant[trapped]]
            dtht     = sel(computer.dtheta    (psi, tht))[:, relevant]
            dpsidpsi = sel(computer.dsds      (psi, tht))[:, relevant]
            dpsidtht = sel(computer.dsdtheta  (psi, tht))[:, relevant]

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

            # Convert in psi coordinate
            delta[..., 0] *= 2 * np.sqrt(psi[relevant])

            psi[relevant] -= delta[..., 0].squeeze(axis=0)
            tht[relevant] -= delta[..., 1].squeeze(axis=0)

            if np.all(abs(delta) < 1e-6):
                break

        else:
            warnings.warn(
                "Failed to converge in 10 iterations.",
                RuntimeWarning,
            )
            pass

        # Wrap between -pi and pi
        tht += np.pi
        tht %= 2 * np.pi
        tht -= np.pi

        self._banana_psi   = psi[trapped]
        self._banana_theta = tht[trapped]

    def _init_trajectory(self):
        """Simple initialisation of the particle trajectory:
            - circular for passing particles
            - ltor +- sqrt(E - mu B) for trapped particles
        """
        g = self._grid

        shape   = np.broadcast(g.Rred, self._ener, self._ltor).shape
        psi0    = np.empty(shape)
        psi0[:] = g.psi
        psi0[:, self._trapped] = self._banana_psi.mean(axis=-1)[:, np.newaxis]

        # Parallel velocity and banana width
        Rred  = g.Rred_at(psi0, g.theta)
        vp2   = self._ener - g.mu/Rred - g.Z * self._pot(psi0)
        vpar0 = g.sign * np.sqrt(2/g.A * vp2.clip(0, None))
        dRv0  = Rred * vpar0
        dRv0 *= self.living_path(g.theta.squeeze())[..., np.newaxis]
        psi0 -= g.A/g.Z * g.R0 * dRv0

        # Compute passage of the central point r=0
        ener_r0 = (.5 * g.Z**2/g.A/g.R0**2) * self._ltor**2 + g.mu + g.Z * self._pot(0)
        self._dead = np.any(psi0 <= 0, axis=0) & (self._ener > ener_r0)
        np.clip(psi0, 0, None, out=psi0)

        return psi0

    def compute_trajectory(self):
        """
        This method computes the path of a trapped or passing particle.

        More specifically, we want to compute the path `psi(theta)`
        of the particle, parametrized by:
        - the average position `psi` ;
        - the parallel velocity `vpar` ;
        - the magnetic momentum `mu`.

        It is defined as the level lines of the energy as a function of mu,
        ltor and theta. Those are found by Newton iteration.
        """
        g = self._grid

        psi0  = self._init_trajectory()

        # Use a twice refined grid for path computation.
        # Regular points are used for trajectory,
        # staggered points are used for velocities.
        theta = double_mesh(g.theta, axis=0)
        psi0  = double_mesh(psi0, axis=0)

        # Search array shape, and mask of searched points.
        shape  = np.broadcast(g.psi, theta, g.vpar, g.mu).shape
        dead   = ~self.living_path(theta.squeeze())
        dead   = dead[..., np.newaxis] | self._dead

        # Compute energy and psi-derivative
        computer = Energy(self._grid, self._pot, self._ltor)
        def fval(s_psi):
            psi = np.square(s_psi)
            psi = psi.reshape(shape)
            ret = computer.value(psi, theta)
            ret-= self._ener
            ret[dead] = 0
            return ret.ravel()
        def fprime(s_psi):
            psi = np.square(s_psi)
            psi = psi.reshape(shape)
            ret = computer.ds(psi, theta)
            assert np.all(np.isfinite(ret))
            return ret.ravel()
        def fsecond(s_psi):
            psi = np.square(s_psi)
            psi = psi.reshape(shape)
            ret = computer.dsds(psi, theta)
            return ret.ravel()

        # Find level line
        assert np.all(psi0 >= 0)
        s_psi = scipy.optimize.zeros.newton(
            func=fval, fprime=fprime, fprime2=fsecond,
            x0=np.sqrt(psi0).ravel(),
            maxiter=10, tol=1e-6,
        )
        psi = np.square(s_psi)
        psi = psi.reshape(shape)
        assert np.all(psi >= 0)
        assert np.all(np.isfinite(psi))

        # Compute trajectory
        Rred = g.Rred_at(psi, theta)
        vpar = (self._ltor - psi) * g.Z/g.A/g.R0 / Rred
        assert np.all(np.isfinite(vpar))

        # Save path
        self._psi     = psi [0::2]
        self._vpar    = vpar[0::2]
        self._mid_psi = psi [1::2]

    def compute_displacement(self):
        """Compute time and toroidal displacement in function of theta.

        The velocities are computed from the Lagrangian
            L(ltor, psi, theta, dt, dphi) =
                Ptheta(ltor, psi, theta)
                + Z * ltor * dphi
                - energy(ltor, psi, theta) * dt
        """
        g = self._grid
        theta = g.theta
        theta = .5 * (theta[1:] + theta[:-1])

        # Energy computation
        energy = Energy(self._grid, self._pot, self._ltor)
        dE_ds = energy.ds(self._mid_psi, theta)
        dE_dl = energy.dltor(self._mid_psi, theta)

        # Poloidal momentum computation
        ptheta = Ptheta(self._grid, self._ltor)
        dP_ds = ptheta.ds(self._mid_psi, theta)
        dP_dl = ptheta.dltor(self._mid_psi, theta)

        # Time displacement
        dt_dtheta = dP_ds / dE_ds
        dt_dtheta = self._regularize_trapped(theta, dt_dtheta)
        self._dt_dtheta = dt_dtheta
        self._time = self._integrate_theta(dt_dtheta)

        # Toroidal displacement
        dphi_dtheta = (dE_dl * dt_dtheta - dP_dl) / g.Z
        self._dphi_dtheta = dphi_dtheta
        self._phi = self._integrate_theta(dphi_dtheta)

    def compute_precession(self):
        """Compute full-bounce quantities from displacements."""
        # Compute full-poloidal quantity
        self._bounce_time = self._time[-1] - self._time[0]
        self._bounce_phi  = self._phi [-1] - self._phi [0]

        # Do not forget return path for bananas
        trapped = self._trapped
        self._bounce_time[trapped] *= 2
        self._bounce_phi [trapped] *= 2

        # Change sign for positive times
        sign = np.sign(self._bounce_time)
        self._bounce_time *= sign
        self._bounce_phi  *= sign

    def compute_ballooning(self):
        g = self._grid
        epsilon = g.radius_at(g.psi)/g.R0
        frac = np.sqrt((1 - epsilon) / (1 + epsilon))

        thetastar  = np.tan(g.theta/2)
        thetastar  = frac * thetastar
        thetastar  = np.arctan(thetastar)
        thetastar *= 2

        q  = g.qprofile_at(g.psi)
        q /= np.sqrt(1 - epsilon**2)

        self._trans = self._phi - q * thetastar

    def compute_canon(self):
        """Compute ballooning momentum."""
        g = self._grid
        theta = g.theta
        assert theta.size & 1 == 1
        assert np.allclose(np.diff(theta), 2*np.pi/(theta.size-1))

        L = self.living_path(theta.squeeze())[..., np.newaxis]

        # Poloidal momentum computation
        ptheta = Ptheta(self._grid, self._ltor)
        P  = ptheta.value(self._psi, theta)
        P *= L

        J2  = scipy.integrate.simps(P, dx=1, axis=0)
        J2 /= L.sum(axis=0)
        J2[self._trapped, 0] -= J2[self._trapped, 1]
        J2[self._trapped, 1]  = J2[self._trapped, 0]

        self._banana_momentum = J2
        self._banana_angle = 2 * np.pi * self._time / self._bounce_time

    def _regularize_trapped(self, theta, dt_dtheta):
        """Trapped trajectories need some regularisation near the banana tips.
        """
        g = self._grid
        theta   = theta.squeeze()
        trapped = self._trapped
        dtheta  = np.diff(theta).mean()
        assert np.allclose(np.diff(theta), dtheta)

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
        dt_dtheta[uptip_idx-1, trapped].T[:] *= 2/dtheta * uptip
        dt_dtheta[dwtip_idx  , trapped].T[:] *= 2/dtheta * dwtip

        return dt_dtheta

    def _integrate_theta(self, dt_dtheta):
        g       = self._grid
        theta   = g.theta.squeeze()
        dtheta  = np.diff(theta).mean()
        assert np.allclose(np.diff(theta), dtheta)

        # Integrate time series
        time      = np.zeros((theta.size, *dt_dtheta.shape[1:]))
        time[1:]  = np.cumsum(dt_dtheta, axis=0)
        time[1:] *= dtheta
        time     -= time[theta.searchsorted(0)]

        return time

    @property
    def ifreq_path(self):
        return self._dt_dtheta

    def living_path(self, theta):
        uptip   = self._banana_theta[..., 0]
        dwtip   = self._banana_theta[..., 1]
        trapped = self._trapped
        living  = np.empty(theta.shape + trapped.shape, dtype=bool)
        living[:]          = ~trapped
        living[:, trapped] = (
            np.less.outer(theta, uptip) &
            np.greater.outer(theta, dwtip)
        )
        return living

    @property
    def psi_path(self):
        return self._psi

    @property
    def vpar_path(self):
        return self._vpar

    @property
    def time_path(self):
        return self._time

    @property
    def phi_path(self):
        return self._phi

    @property
    def bounce_time(self):
        return self._bounce_time

    @property
    def bounce_phi(self):
        return self._bounce_phi

    @property
    def energy(self):
        return self._ener

    @property
    def ltor(self):
        return self._ltor

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
    np.who(adv.__dict__)

    adv.compute_bounce_point()
    np.who(adv.__dict__)

    plt.figure()
    plt.suptitle('Banana tip')
    plt.subplot(121)
    plt.gca().set_title('psi')
    plt.plot(adv._banana_psi[:, 0])
    plt.xlabel('trapped index')
    plt.subplot(122)
    plt.gca().set_title('theta')
    plt.plot(adv._banana_theta[:, 0])
    plt.xlabel('trapped index')
    plt.show()

    adv.compute_trajectory()
    np.who(adv.__dict__)

    plt.figure()
    plt.suptitle('Trajectory')
    plt.subplot(121)
    plt.gca().set_title('psi')
    plt.plot(theta, adv.psi_path[:, 3].reshape(theta.size, -1))
    plt.subplot(122)
    plt.gca().set_title('vpar')
    plt.plot(theta, adv.vpar_path[:, 3].reshape(theta.size, -1))
    plt.show()

    # Compute trajectory timing
    adv.compute_displacement()
    adv.compute_precession()
    adv.compute_ballooning()
    np.who(adv.__dict__)

    living = np.where(
        adv.living_path(theta)[..., np.newaxis],
        1, np.nan
    )
    time = adv.time_path * living
    phi  = adv._trans  * living

    plt.figure()
    plt.suptitle('Displacement')
    plt.subplot(121)
    plt.gca().set_title('time')
    plt.plot(
        theta,
        time[:, 16].reshape(theta.size, -1),
    )
    plt.subplot(122)
    plt.gca().set_title('phi')
    plt.plot(
        theta,
        phi[:, 16].reshape(theta.size, -1),
    )
    plt.show()

    adv.compute_canon()
    np.who(adv.__dict__)

    idx = np.argsort(vpar.T.ravel())

    plt.figure()
    plt.subplot(121)
    plt.plot(
        vpar.T.ravel()[idx],
        adv._banana_momentum.T.reshape(vpar.size, -1)[idx],
    )
    plt.subplot(122)
    plt.plot(
        vpar.T.ravel()[idx],
        (
            adv._banana_momentum
            + grid.Z * (
                grid.qprofile * adv._ltor
                + rg[0]**2/2
            )
        ).T.reshape(vpar.size, -1)[idx],
    )
    plt.show()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('error')

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.ion()

    mpl.rcParams['figure.figsize'] = 10, 5

    import cProfile, pstats
    pr = cProfile.Profile()

    try:
        pr.enable()
        main()
        pr.disable()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        ps = pstats.Stats(pr).sort_stats('cumulative')
        ps.print_stats(10)

        plt.show(block=True)
