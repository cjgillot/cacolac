
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
            self._spar = np.asarray([1, -1])[(None,) * (vpar.ndim - 1)]
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
        bounce_time = self._int_time(2 * np.pi)
        bounce_phi  = self._int_phi (2 * np.pi)

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
        self.with_deriv = with_deriv
        self.gyroavg = gyroavg

        psi, theta = self._adjust_points(psi, theta)
        self.psi_size, self.psi_slice = self._choose_slice(psi_grid, psi)
        self.theta_size, self.theta_slice = self._choose_slice(theta_grid, theta)

        # Allocate memory
        shape = (self.psi_size, self.theta_size, *psi.shape)
        self.val_ker = np.ones(shape)
        if with_deriv:
            self.dpsi_ker = np.zeros(shape)
            self.dtheta_ker = np.zeros(shape)

        # Compute local interpolation grid
        psi_loc, theta_loc = np.meshgrid(
            psi_grid[self.psi_slice],
            theta_grid[self.theta_slice],
            indexing='ij', sparse=True,
        )
        self._compute_psi(psi, psi_loc, theta_loc)
        self._compute_theta(theta, psi_loc, theta_loc)

        # Print statistics on sparsity
        print('SPARSITY',
              self.val_ker.size,
              self.val_ker.astype(bool).sum()
             )

    def _compute_psi(self, psi, psi_loc, theta_loc):
        # Linear interpolation in psi
        dpsi = np.diff(psi_loc.squeeze()).mean()
        psi_dist = np.subtract.outer(psi_loc, psi)

        gauss = 1 - abs(psi_dist / dpsi)
        np.clip(gauss, 0, None, out=gauss)

        dgauss  = - np.sign(psi_dist) / dpsi
        dgauss *= abs(psi_dist) < dpsi

        self.val_ker *= gauss
        if self.with_deriv:
            self.dpsi_ker   *= dgauss
            self.dtheta_ker *= gauss

    def _compute_theta(self, theta, psi_loc, theta_loc):
        # Linear interpolation in theta
        dtheta = np.diff(theta_loc.squeeze()).mean()
        theta_dist = np.subtract.outer(theta_loc, theta)

        sinc = 1 - abs(theta_dist / dtheta)
        np.clip(sinc, 0, None, out=sinc)

        dsinc  = - np.sign(theta_dist) / dtheta
        dsinc *= abs(theta_dist) < dtheta

        self.val_ker *= sinc
        if self.with_deriv:
            self.dpsi_ker   *= sinc
            self.dtheta_ker *= dsinc

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

def ravel_4(array):
    shape = array.shape[:-4] + (-1,)
    return np.reshape(array, shape)

class KernelElementComputer:
    """Helper class handling computation state for a kernel element.

    This class is designed for vectorised computation of the positions of
    the present particles.
    """
    def __init__(self, grid, computer, adv, gyroavg):
        self._grid      = grid
        self._computer  = computer
        self._adv       = adv
        self._gyroavg   = gyroavg

        self._omega, self._ntor = np.meshgrid(
            computer._omega, computer._ntor,
            indexing='ij', sparse=True,
        )
        self._psi_path  = interp1d(grid.theta.squeeze(), adv.psi_path)
        self._phi_path  = adv.phi_path
        self._time_path = adv.time_path

    def precompute(self):
        """This method computes the position-independent quantities."""
        c = self._computer
        a = self._adv

        # Fourier-space time-periodicity
        bounce_dist = (
            + 1j * np.multiply.outer(self._omega, a.bounce_time)
            + 1j * np.multiply.outer(self._ntor,  a.bounce_phi)
        )
        bounce_warp = - np.expm1(1j * bounce_dist)
        np.reciprocal(bounce_warp, out=bounce_warp)
        assert np.all(np.isfinite(bounce_warp))

        # Distribution function
        bounce_warp *= c._FonT
        assert np.all(np.isfinite(bounce_warp))

        self._bounce_warp = bounce_warp

    def interpolate(self, theta):
        c = self._computer
        a = self._adv

        theta = np.atleast_1d(theta)

        # Interpolation path at past position
        psi = self._psi_path (theta)
        self._phi = self._phi_path (theta)
        self._tim = self._time_path(theta)
        self._tht = theta

        # Interpolate past position
        interp = make_interp_kernel(
            c._psi, psi,
            c._theta, theta,
            with_deriv=True,
            gyroavg=self._gyroavg,
        )

        self._interpolator = interp

    def compute_past(self):
        """This method computes the effect on the past particle at `theta1`.

        It contains:
        - the coordinates of the past point,
        - the interpolation kernel.
        """
        c = self._computer
        a = self._adv

        # Interpolate past position
        interpolator = self._interpolator

        # Contribution
        print('PP')
        warp = np.zeros(
            self._ntor.shape[1:] +
            (interpolator.psi_size, interpolator.theta_size,) +
            self._phi.shape
        , dtype=np.complex128)

        # Entropic frequency
        freq_star = c._freq_star

#         print('PP 0')
#         warp += freq_star[0] * interpolator.dpsi_ker

#         print('PP 1')
#         warp += freq_star[1] * interpolator.dtheta_ker

        print('PP 2')
        warp += np.multiply.outer(
            1j * self._ntor.squeeze(),
            freq_star[2] *
            interpolator.val_ker
        )

        # Jacobian for theta integral
        warp *= a.ifreq(self._tht)

        # Verify computation
        np.who(locals())
        assert np.all(np.isfinite(warp))

        # Position-independent quantities
        self._past_warp = warp

    def compute_passing(self, output):
        """This method computes the effect on the past particle at `theta2`.

        It contains:
        - the coordinates of the past point,
        - the interpolation kernel,
        - the displacement warping.
        """
        c = self._computer
        a = self._adv

        # Interpolate present point
        present_kernel = self._interpolator.val_ker
        present_kernel = ravel_4(present_kernel)

        # Compute warping
        warp = self._warp_present(self._phi, self._tim)

        # Assemble
        self._add_contribution(present_kernel, warp, output=output)

    def compute_trapped(self, output):
        """Compute contribution of trapped particle when the trajectory
        between the two points involves a u-turn.

        It contains:
        - the coordinates of the past point,
        - the interpolation kernel,
        - the displacement warping.

        This function is still fragile.
        """
        c = self._computer
        a = self._adv

        # Interpolation path at present position
        swap = np.s_[..., [1,0]]
        phi2 = self._phi[swap]
        tim2 = self._tim[swap]

        # Only select trapped particles
        mask = np.repeat(a.trapped[..., np.newaxis], 2, axis=-1)

        # Interpolate present point
        present_kernel = self._interpolator.val_ker
        present_kernel = present_kernel[..., mask]

        # The computation of the elapsed time is done as follows:
        # Consider the trajectory
        #               theta1  theta=0
        #          /-------o__t1__+
        #          v              +
        #          \--------------+==t2==o
        # The time to go from point 1 to point 2 is decomposed as follows:
        #  - from theta1 to theta=0 (-t1);
        #  - from theta=0 to theta=0 on the other branch;
        #  - from theta=0 to theta2 (+t2).

        # Add half-bounce to use the theta=0 crossing as a reference
        tim2 -= .5 * a.bounce_time
        phi2 -= .5 * a.bounce_phi

        # Compute warping
        warp = self._warp_present(phi2, tim2, mask)

        # Assemble
        self._add_contribution(present_kernel, warp, mask, output=output)

    def _warp_present(self, phi2, tim2, mask=None):
        """Compute the displacement warping between the past and present particle."""
        c = self._computer
        a = self._adv

        phi2 = phi2[np.newaxis, :] - self._phi[:, np.newaxis]
        tim2 = tim2[np.newaxis, :] - self._tim[:, np.newaxis]
        bwrp = self._bounce_warp[..., np.newaxis, np.newaxis, :, :, :, :]

        # Adjust in case the causality is wrong
        while True:
            uncausal = tim2 < 0
            if not np.any(uncausal):
                break
            tim2 += uncausal * a.bounce_time
            phi2 += uncausal * a.bounce_phi
            del uncausal
        assert np.all(tim2 >= 0)
        assert np.all(tim2 <= a.bounce_time)

        # Compute relevant particles
        if mask is not None:
            tim2 = tim2[..., mask]
            phi2 = phi2[..., mask]
            bwrp = bwrp[..., mask]
        else:
            tim2 = ravel_4(tim2)
            phi2 = ravel_4(phi2)
            bwrp = ravel_4(bwrp)

        # Fourier-space displacement
        dist = (
            - np.multiply.outer(self._omega, tim2)
            + np.multiply.outer(self._ntor,  phi2)
        )
        warp = np.exp(1j * dist)
        assert np.all(np.isfinite(warp))

        # Periodicity effect
        warp *= bwrp

        return warp

    def _add_contribution(
        self, present_kernel, present_warp,
        mask=None, *, output
    ):
        c = self._computer

        interpolator = self._interpolator
        past_warp = self._past_warp

        # Compute relevant subarray
        if mask is not None:
            past_warp = past_warp[..., mask]
        else:
            past_warp = ravel_4(past_warp)

        np.who(locals())
        np.who(self.__dict__)
        np.who(interpolator.__dict__)

        # Sum all the things!
        print('ES')
        source = np.einsum(
            # w=omega, n=ntor, p=particles,
            # uv=theta vectorisation,
            # yz=both psi, hj=both theta
            'wnuvp,nyhup,zjvp->wnyzhj',
            present_warp, past_warp,
            present_kernel,
            optimize=True,
        )
        assert np.all(np.isfinite(source))
        np.who(locals())

        print('ACC')
        output[
            :, :,
            interpolator.psi_slice, interpolator.psi_slice,
            interpolator.theta_slice, interpolator.theta_slice
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
        lnFeq = np.zeros(np.broadcast(g.psi, g.vpar, g.mu).shape[1:])
        lnFeq -= - g.energy.mean(axis=0).squeeze()/Teq
        if Veq is not None:
            lnFeq += g.ltor.mean(axis=0) * Veq/Teq
        lnFeq += np.log(Neq)
        lnFeq -= 1.5 * np.log(Teq)

        FonT  = np.exp(lnFeq)
        FonT /= Teq
        self._FonT = FonT

        # Entropic frequency vector `Teq {ln Feq, X}`
        lnF = interp1d(g.psi.squeeze(), lnFeq)
        self._freq_star = [
            np.zeros_like(lnFeq),
            np.zeros_like(lnFeq),
            lnF(g.psi.squeeze(), nu=1),
        ]
        for _ in self._freq_star:
            assert np.all(np.isfinite(_))

    def compute(self, adv, gyroavg=False):
        kec = KernelElementComputer(self._grid, self, adv, gyroavg)

        # Particle-independent effects
        kec.precompute()

        # Compute interpolation matrix
        kec.interpolate(self._theta[:5])

        # Effects of past particle
        kec.compute_past()

        # Add passing particles
        kec.compute_passing(output=self._output)

        # Add trapped particles
        kec.compute_trapped(output=self._output)

def main():
    A = Z = 1
    R0 = 900

    # Build grid
    rg = np.linspace(100, 150, 32)
    qq = 1 + 0*np.linspace(0, 1, rg.size)**2

    theta = np.linspace(0, 2 * np.pi, 128)
    vpar = np.multiply.outer(np.linspace(0, 1, 24), [1, -1])
    mu = np.linspace(0, 1, 16)

    grid = Grid(1, 1, 900, rg, qq, theta, vpar, mu)
    np.who(grid.__dict__)

    plt.plot(grid.radius.squeeze(), grid.psi.squeeze())
    plt.show()

    # Advect particles
    pot = np.zeros_like(rg)
    adv = ParticleAdvector(grid, pot)
    adv.compute_trajectory()
    np.who(adv.__dict__)

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
        omega=np.asarray([1e-3 + 1e-4j]),
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
