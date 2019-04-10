
import importlib

import numpy as np
import scipy.interpolate as scip
from scipy.interpolate import make_interp_spline as interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline

mpl.rcParams['figure.figsize'] = 10, 5

def meshgrid(*xi):
    xi = [np.asanyarray(_) for _ in xi]
    ns = [_.ndim for _ in xi]
    ll = len(ns)
    for i, n in enumerate(ns):
        sl = (Ellipsis,) + n * (None,)
        for j in range(i):
            xi[j] = xi[j][sl]
        sl = n * (None,) + (Ellipsis,)
        for j in range(i+1, ll):
            xi[j] = xi[j][sl]
    return xi

class Grid:
    def __init__(self, As, Zs, R0, rg, qq, theta, vpar, mu):
        self._As = As
        self._Zs = Zs
        self._R0 = R0

        psi = interp1d(rg, rg/qq).antiderivative()
        psi = psi(rg)
        theta, psi, mu, vpar, = meshgrid(theta, psi, mu, vpar)
        rg = rg.reshape(psi.shape)
        Rred = 1 + rg/R0 * np.cos(theta)

        self._r = rg
        self._y = psi
        self._theta = theta
        self._ltor = - Zs * psi + As * R0 * Rred * vpar
        self._vpar = vpar
        self._ener = As/2 * vpar**2 + mu / Rred
        if vpar.ndim == 2:
            self._spar = np.asarray([1, -1]).reshape(
                (vpar.ndim - 1) * (1,) + (2,)
            )
        else:
            self._spar = np.sign(vpar)
        self._mu = mu
        self._q = qq[..., np.newaxis, np.newaxis, np.newaxis]

        self._r_at = interp1d(psi.squeeze(), rg.squeeze())
        self._q_at = interp1d(psi.squeeze(), qq.squeeze())

    @property
    def A(self):
        return self._As

    @property
    def R0(self):
        return self._R0

    @property
    def Z(self):
        return self._Zs

    @property
    def radius(self):
        return self._r

    @property
    def qprofile(self):
        return self._q

    @property
    def psi(self):
        return self._y

    @property
    def theta(self):
        return self._theta

    @property
    def ltor(self):
        return self._ltor

    @property
    def vpar(self):
        return self._vpar

    @property
    def energy(self):
        return self._ener

    @property
    def mu(self):
        return self._mu

    @property
    def sign(self):
        return self._spar

    def radius_at(self, y):
        return self._r_at(y)

    def qprofile_at(self, y):
        return self._q_at(y)

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
            print('LOOP', _)
            vpar_old, psi_old = vpar, psi

            r = g.radius_at(psi)
            print('LOOP r', _)
            Rred = 1 + r/R0 * np.cos(g.theta)
            print('LOOP R', _)

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
                print('LOOP v', _)
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

#             try:
#                 plt.subplot(121)
#                 plt.plot(g.theta.squeeze(), psi[50].reshape(256, -1))
# #                 plt.axhline(g.psi[50], c='black')
#                 plt.legend(ener[50].ravel())
#                 plt.subplot(122)
#                 plt.plot(g.theta.squeeze(), vpar[50].reshape(256, -1))
#                 plt.legend(g.mu.ravel())
#                 plt.show()
#             except:
#                 pass

#             print(
#                 np.linalg.norm(vpar_old - vpar) / np.linalg.norm(vpar),
#                 np.linalg.norm(psi_old - psi) / np.linalg.norm(psi),
#             )
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

        freq  = self._vpar / (q * g.R0 * self._Rred)
        freq += self._pot(self._psi, nu=1)
        # FIXME Add vD.gradtheta

        self._freq = freq

        ifreq = 1/freq
        ifreq[~np.isfinite(ifreq)] = 0
        self._ifreq = interp1d(theta, ifreq)
        self._int_time = self._ifreq.antiderivative()

        vphi  = self._vpar / self._Rred / g.R0
        # FIXME Add vE.gradphi
        # FIXME Add vD.gradphi

        self._vphi = vphi
        spline = interp1d(theta, vphi * ifreq)
        self._int_phi = spline.antiderivative()

        # Compute full-bounce quantities
        bounce_time = self._int_time(np.pi)
        bounce_phi  = self._int_phi (np.pi)

        # Get positive frequencies
        bounce_sign  = np.sign(bounce_time)
        bounce_phi  *= bounce_sign
        bounce_time *= bounce_sign

        # Add the mirror part for trapped particles
        trapped = self._trapped
        bounce_time[trapped, :] += bounce_time[trapped, :].sum(axis=-1, keepdims=True)
        bounce_phi [trapped, :] += bounce_phi [trapped, :].sum(axis=-1, keepdims=True)

        self._bounce_time = bounce_time
        self._bounce_phi  = bounce_phi

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

    @property
    def bounce_phi(self):
        return self._bounce_phi

    @property
    def trapped(self):
        return self._trapped

class make_interp_kernel:
    def __init__(
        self,
        psi_grid, psi,
        theta_grid, theta,
        with_deriv=False,
        gyroavg=False,
    ):
        psi, theta = self._adjust_points(psi, theta)
        self.psi_size, self.psi_slice = self._choose_slice(psi_grid, psi)
        self.theta_size, self.theta_slice = self._choose_slice(theta_grid, theta)

        print('INTERP', (
            self.psi_size,
            self.theta_size,
            *psi.shape
        ))
        self.val_ker = np.zeros((
            self.psi_size,
            self.theta_size,
            *psi.shape
        ))
        if with_deriv:
            self.dpsi_ker = np.zeros_like(self.val_ker)
            self.dtheta_ker = np.zeros_like(self.val_ker)

        psi_loc, theta_loc = np.meshgrid(
            psi_grid[self.psi_slice],
            theta_grid[self.theta_slice],
            indexing='ij', sparse=True,
        )

#         np.who(locals())

        # Gaussian interpolation in psi
        dpsi = np.diff(psi_loc.squeeze()).mean()
        psi_dist = np.subtract.outer(psi_loc, psi)
        gauss = np.exp(- psi_dist**2 / dpsi**2)

        self.val_ker[:] = gauss
        if with_deriv:
            self.dtheta_ker[:] = gauss
            self.dpsi_ker[:] = gauss
            self.dpsi_ker[:] *= - 2 * psi_dist / dpsi**2

        # Gaussian interpolation in theta
        dtheta = np.diff(theta_loc.squeeze()).mean()
        theta_dist = np.subtract.outer(theta_loc, theta)
        sinc = np.exp(- theta_dist**2 / dtheta**2)

        self.val_ker *= sinc
        if with_deriv:
            self.dpsi_ker *= sinc
            self.dtheta_ker *= sinc
            self.dtheta_ker *= - 2 * theta_dist / dtheta**2

    def _adjust_points(self, psi, theta):
        psi = np.asanyarray(psi); n = psi.ndim
        theta = np.asanyarray(theta); m = theta.ndim
        theta = theta[(Ellipsis,) + (n - m) * (None,)]
        return psi, theta

    def _choose_slice(self, grid, values):
        i_min = grid.searchsorted(values.min(), 'left') - 1
        i_max = grid.searchsorted(values.max(), 'right') + 1
        i_min = max(i_min, 0)
        i_max = min(i_max, grid.size)

        return i_max - i_min, np.s_[i_min:i_max]

class KernelElementComputer:
    def __init__(self, grid, computer, adv, theta1, gyroavg):
        self._grid      = grid
        self._computer  = computer
        self._adv       = adv
        self._gyroavg   = gyroavg

        self._omega, self._ntor = np.meshgrid(
            computer._omega, computer._ntor,
            indexing='ij', sparse=True,
        )
        self._theta1    = theta1
        self._psi_path  = interp1d(grid.theta.squeeze(), adv.psi_path)
        self._phi_path  = adv.phi_path
        self._time_path = adv.time_path

    def precompute(self):
        c = self._computer
        a = self._adv

        # Interpolation path at past position
        psi1 = self._psi_path (self._theta1)
        phi1 = self._phi_path (self._theta1)
        tim1 = self._time_path(self._theta1)
        self._phi1 = phi1
        self._tim1 = tim1

        # Interpolate past position
        self._past_kernel = make_interp_kernel(
            c._psi, psi1,
            c._theta, self._theta1,
            with_deriv=True,
            gyroavg=self._gyroavg,
        )

        # Fourier-space time-periodicity
        bounce_dist = (
            + 1j * np.multiply.outer(self._omega, a.bounce_time)
            + 1j * np.multiply.outer(self._ntor,  a.bounce_phi)
        )
        bounce_warp = - np.expm1(1j * bounce_dist)
        np.reciprocal(bounce_warp, out=bounce_warp)
        #assert np.all(np.isfinite(warp))

        # Distribution function
        bounce_warp *= c._FonT
        #assert np.all(np.isfinite(warp))

        # Jacobian for theta integral
        bounce_warp *= a.ifreq(self._theta1)

        self._bounce_warp = bounce_warp

    def compute_passing(self, output):
        c = self._computer
        a = self._adv

        # Interpolation path at present position
        theta2 = c._theta
        psi2 = self._psi_path (theta2)
        phi2 = self._phi_path (theta2) - self._phi1
        tim2 = self._time_path(theta2) - self._tim1

        # Adjust in case the causality is wrong
        uncausal = tim2 < 0
        tim2 += uncausal * a.bounce_time
        phi2 += uncausal * a.bounce_phi
        assert np.all(tim2 >= 0)
        del uncausal

        # Compute relevant particles
        mask = np.any(self._past_kernel.val_ker, axis=(0, 1))

        # Fourier-space displacement
        dist = (
            - np.multiply.outer(self._omega, tim2)
            + np.multiply.outer(self._ntor,  phi2)
        )[:, :, :, mask]
        warp = np.exp(1j * dist)
        #assert np.all(np.isfinite(warp))

        # Common part
        warp *= self._bounce_warp[:, :, np.newaxis, mask]
        #assert np.all(np.isfinite(warp))

        # Interpolate present point
        present_kernel = make_interp_kernel(
            c._psi, psi2[:, mask],
            c._theta, theta2,
            with_deriv=False,
            gyroavg=self._gyroavg,
        )

        # Sum everything
        self.add_contribution(present_kernel, warp, mask, output)

    def add_contribution(self, present_kernel, warp, mask, output):
        c = self._computer
        past_kernel = self._past_kernel

        # Contribution
        print('ES')
        source = np.zeros((
            self._omega.size, self._ntor.size,
            past_kernel.psi_size, present_kernel.psi_size,
            past_kernel.theta_size, present_kernel.theta_size,
        ), dtype=np.complex128)
        np.who(locals())

        # Entropic frequency
        freq_star = c._freq_star

#         print('ES 0')
#         source += np.einsum(
#             'wn...p,p,yhp,zj...p->wnyzhj',
#             warp, freq_star[0][mask],
#             past_kernel.dpsi_ker[:, :, mask],
#             present_kernel.val_ker,
#             optimize=True,
#         )

#         print('ES 1')
#         source += np.einsum(
#             'wn...p,p,yhp,zj...p->wnyzhj',
#             warp, freq_star[1][mask],
#             past_kernel.dtheta_ker[:, :, mask],
#             present_kernel.val_ker,
#             optimize=True,
#         )

        print('ES 2')
        source += np.einsum(
            'wn...p,p,n,yhp,zj...p->wnyzhj',
            warp, freq_star[2][mask],
            1j * self._ntor.squeeze(),
            past_kernel.val_ker[:, :, mask],
            present_kernel.val_ker,
            optimize=True,
        )
        #assert np.all(np.isfinite(source))

        print('ACC')
        output[
            :, :,
            past_kernel.psi_slice, present_kernel.psi_slice,
            past_kernel.theta_slice, present_kernel.theta_slice
        ] += source

class KernelComputer:
    def __init__(self, grid, Neq, Teq, omega, ntor, psi=None, theta=None, Veq=None):
        self._grid = grid

        self._omega = np.atleast_1d(omega)
        if psi is None:
            psi = grid.psi.squeeze()
        self._psi   = np.atleast_1d(psi)
        if theta is None:
            theta = grid.theta.squeeze()
        self._theta = np.atleast_1d(theta)
        self._ntor  = np.atleast_1d(ntor)

        sl = (Ellipsis,) + (grid.energy.ndim - 2) * (None,)
        self._Neq = Neq[sl]
        self._Teq = Teq[sl]
        if Veq is not None:
            self._Veq = Veq[sl]
        else:
            self._Veq = None

        self.compute_distribution()

        # Warning: Large matrix
        self._output = np.zeros((
            omega.size, ntor.size,
            psi.size, theta.size,
            psi.size, theta.size,
        ), dtype=np.complex128)

    def compute_distribution(self):
        g = self._grid
        Neq = self._Neq
        Teq = self._Teq
        Veq = self._Veq

        # Distribution function
        Feq = np.zeros(np.broadcast(g.psi, g.vpar, g.mu).shape[1:])
        print(Feq.shape, g.psi.shape)
        Feq *= np.exp(- g.energy.squeeze()/Teq)
        if Veq is not None:
            Feq *= np.exp(g.ltor * Veq/Teq)
        Feq *= Neq * Teq**1.5
        self._FonT = Feq/Teq

        # Entropic frequency vector `Teq {ln Feq, X}`
        lnF = interp1d(g.psi.squeeze(), Feq)
        self._freq_star = [
            np.zeros_like(Feq),
            np.zeros_like(Feq),
            lnF(g.psi.squeeze(), nu=1)/Feq,
        ]

    def compute(self, adv, gyroavg=False):
        # Loop over the past position
        for theta1 in self._theta:
            kec = KernelElementComputer(self._grid, self, adv, theta1, gyroavg)

            # Precompute effects of past particle
            kec.precompute()

            # Add passing particles
            kec.compute_passing(self._output)

            return

def main():
    A = Z = 1
    R0 = 900

    rg = np.linspace(100, 150, 32)
    qq = 1 + 0*np.linspace(0, 1, rg.size)**2

    theta = np.linspace(0, 2 * np.pi, 128)
    vpar = np.multiply.outer(np.linspace(0, 1, 24), [1, -1])
    mu = np.linspace(0, 1, 16)

    grid = Grid(1, 1, 900, rg, qq, theta, vpar, mu)
    np.who(grid.__dict__)

    plt.plot(grid.radius.squeeze(), grid.psi.squeeze())
    plt.show()

    pot = np.zeros_like(rg)

    adv = ParticleAdvector(grid, pot)

    adv.compute_trajectory()

    np.who(adv.__dict__)

    plt.subplot(121)
    plt.plot(theta, adv._r[:, 16].reshape(theta.size, -1))
    plt.subplot(122)
    plt.plot(theta, adv._vpar[:, 16].reshape(theta.size, -1))
    plt.show()

    plt.pcolormesh(
#         vpar[:, 0],
#         rg,
        np.any(adv.trapped[:, :, :, 0] ^ adv.trapped[:, :, :, 1], axis=1)
    )
    plt.colorbar()
    plt.show()
    return

    adv.compute_freq()

    np.who(adv.__dict__)

    time = adv._int_time(grid.theta.squeeze())
    phi  = adv._int_phi(grid.theta.squeeze())
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

    kern = KernelComputer(
        grid,
        psi=grid.psi.squeeze()[2:15],
        theta=theta[2:21],
        Neq=np.ones_like(rg),
        Teq=np.ones_like(rg),
#         Veq=np.zeros_like(rg),
        omega=np.ones(1),
        ntor=np.arange(10),
    )
    kern.compute(adv)

    out = kern._output
    print(out.shape)
    print(abs(out).min(), abs(out).max())

    plt.pcolormesh(
        abs(out[0, 5].transpose(0, 2, 1, 3).reshape(13*19, 13*19)),
#         norm=mpl.colors.LogNorm(),
    )
    plt.colorbar()

# %load_ext line_profiler

# %prun main()
# %lprun -f KernelComputer.compute main()
main()
