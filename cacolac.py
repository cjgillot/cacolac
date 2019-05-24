"""Cacolac is a linear global gyrokinetic solver.

The method is outlined below:
    - we compute the path of the particles on the equilibrium state as
    functions of the poloidal angle theta
        t(theta), psi(theta), vpar(theta)
    - we compute the variational kernel by integrating against those
    trajectories.

The variational problem is written as follows:
    S = \\int d{psi(0), theta(0), phi(0)}
        Phi(0, psi(0), theta(0), phi(0))*
        \\int d{vpar(0), mu} Feq(psi*, vpar, mu)
        \\int_0^{-oo} dt
        (W* grad) Phi(t, psi(t), theta(t), phi(t)
      + Boussinesq term

avec W* = { ln Feq, X } Poisson bracket.

We recast this integral as an integral on the angle theta, and go to Fourier
space in time and toroidal angle:
    S = \\sum_n \\int dw
        \\int Feq{psi*, energy, mu}
        \\int_0^2 pi dtheta1
        \\int_-oo^+oo dtheta2 * dt/dtheta(theta2)
        Phi(psi(theta1), theta1)*
        (W* grad) Phi(psi(theta2), theta2)
        exp(+ 1j n (phi(theta2) - phi(theta1)))
        exp(- 1j w (time(theta2) - time(theta1)))
        1{time(theta2) <= time(theta1)}

The infinite integral on theta2 comes from the infinite integral on
time, to take into account multiple passages in the periodic trajectory.
The Heaviside function corresponds to causality.

We integrate the first one out by explicitly using the bounce time \\tau_b and
bounce precession \\phi_b.

        \\int_-oo^+oo dtheta2
        exp(+ 1j n (phi(theta2) - phi(theta1)))
        exp(- 1j w (time(theta2) - time(theta1)))
        1{time(theta2) <= time(theta1)}
        ..(theta2)
      = \\sum_N \\int dtheta
        exp(+ 1j n (phi(theta2)  - phi(theta1)  - N \\phi_b))
        exp(- 1j w (time(theta2) - time(theta1) - N \\tau_b))
        1{time(theta2) - N \\tau_b <= time(theta1)}
        ..(theta2)
      = \\int dtheta
        exp(+ 1j n (phi(theta2)  - phi(theta1) ))
        exp(- 1j w (time(theta2) - time(theta1)))
        1{time(theta2) <= time(theta1)}
        1/[1 - exp(- 1j n \\phi_b + 1j w \\tau_b)
        ..(theta2)

With the integral on theta2 spanning only a 2 pi interval.
The result is therefore,

    S = \\sum_n \\int dw
        \\int Feq{psi*, energy, mu}
        \\oint dtheta1 dtheta2
        Phi(psi(theta1), theta1)*
        (W* grad) Phi(psi(theta2), theta2) * dt/dtheta(theta2)
        exp(+ 1j n (phi(theta2)  - phi(theta1) ))
        exp(- 1j w (time(theta2) - time(theta1)))
        1/[1 - exp(- 1j n \\phi_b + 1j w \\tau_b)
        1{time(theta2) <= time(theta1)}
"""

import importlib

import numpy as np
import scipy.interpolate as scip
from scipy.interpolate import make_interp_spline as interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline

mpl.rcParams['figure.figsize'] = 10, 5

from .grid import Grid
from .advector import ParticleAdvector
from .fem import DenseP1Basis

def staggered(array, *, axis):
    assert axis == 0
    ret = array[1:] + array[:-1]
    ret/= 2
    return ret

def ravel_4(array):
    shape = array.shape[:-4] + (-1,)
    return np.reshape(array, shape)

class KernelElementComputer:
    """Helper class handling computation state for a kernel element.

    This class is designed for vectorised computation of the positions of
    the present particles.
    """
    def __init__(self, grid, computer, adv):
        self._grid      = grid
        self._computer  = computer
        self._adv       = adv

        self._omega, self._ntor = np.meshgrid(
            computer._omega, computer._ntor,
            indexing='ij', sparse=True,
        )

        theta = grid.theta.squeeze()
        self._psi_path   = interp1d(theta, adv.psi_path)
        self._phi_path   = interp1d(theta, adv.phi_path)
        self._time_path  = interp1d(theta, adv.time_path)

        theta = staggered(theta, axis=0)
        self._ifreq_path = interp1d(theta, adv.ifreq_path)

    def precompute(self):
        """This method computes the position-independent quantities."""
        c = self._computer
        a = self._adv

        # Fourier-space time-periodicity
        bounce_dist = (
            + 1j * np.multiply.outer(self._omega, a.bounce_time)
            - 1j * np.multiply.outer(self._ntor,  a.bounce_phi)
        )
        bounce_warp = np.expm1(1j * bounce_dist)
        np.negative(bounce_warp, out=bounce_warp)
        np.reciprocal(bounce_warp, out=bounce_warp)
        assert np.all(np.isfinite(bounce_warp))

        self._bounce_warp = bounce_warp

    def interpolate(self, theta, fe_basis):
        c = self._computer
        a = self._adv

        theta = np.atleast_1d(theta)

        # Interpolation path at past position
        psi = self._psi_path (theta)
        self._phi = self._phi_path (theta)
        self._phi.flags.writeable = False
        self._tim = self._time_path(theta)
        self._tim-= self._time_path(0)
        self._tim.flags.writeable = False
        self._tht = theta
        self._tht.flags.writeable = False

        assert np.all(abs(self._tim) < a.bounce_time)

        self._living = a.living_path(theta) != 0

        # Interpolate past position
        interp = fe_basis.interpolate(
            psi, theta,
            mask=self._living[..., np.newaxis],
            with_deriv=True,
        )

        self._interpolator = interp

    def compute_past(self):
        """This method computes the effect on the past particle at `theta1`.

        It contains:
        - the coordinates of the past point,
        - the interpolation kernel,
        - the action of the equilibrium distribution function.

        Basically, what we are trying to compute can be written as
            W* grad Phi * dt/dtheta
        with W* = grad ln Feq
             grad Phi = (dr, dtheta/r, 1j n) Phi
             dt/dtheta given by `adv`
        """
        c = self._computer
        a = self._adv

        # Interpolate past position
        interpolator = self._interpolator

        # Contribution
        print('PP')
        warp = np.zeros(
            (self._ntor.size, *interpolator.val_data.shape)
        , dtype=np.complex128)

        # Entropic frequency
        freq_star = c._freq_star

#         print('PP 0')
#         warp += freq_star[0] * interpolator.dps_data

#         print('PP 1')
#         warp += freq_star[1] * interpolator.dth_data

        print('PP 2')
        warp += np.multiply.outer(
            1j * self._ntor.squeeze(),
            freq_star[2] *
            interpolator.val_data
        )

        # Distribution function
        warp *= c._FonT

        # Verify computation
        warp  = ravel_4(warp)
        assert np.all(np.isfinite(warp))

        # Warp to reference position
        warp_w, warp_n = \
                self._warp_half(self._phi, self._tim, +1)

        # Jacobian for theta integral
        warp_n *= ravel_4(self._ifreq_path(self._tht))

        # Position-independent quantities
        self._past_warp = warp
        self._past_warp_w = warp_w
        self._past_warp_n = warp_n

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
        present_kernel = self._interpolator.val_data
        present_kernel = ravel_4(present_kernel) # Flatten particle axes

        # Easy part
        present_warp = self._warp_half(self._phi, self._tim, -1)

        # Compute warping
        causal_warp = self._causality(self._tim)

        # Assemble
        self._add_contribution(
            present_kernel, present_warp,
            causal_warp,
            output=output,
        )

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

        # Only select trapped particles
        mask = np.repeat(a.trapped[..., np.newaxis], 2, axis=-1)
        # Select the other side of the bounce
        # as the symmetric particle wrt. velocity sign.
        swap = np.s_[..., ::-1]

        # Interpolate present point
        present_kernel = self._interpolator.val_data
        present_kernel = present_kernel[swap][..., mask]

        # Interpolation path at present position
        phi1 = self._phi[swap].copy()
        tim1 = self._tim[swap].copy()

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
        tim1 += .5 * a.bounce_time
        phi1 += .5 * a.bounce_phi

        # Easy part
        present_warp = self._warp_half(phi1, tim1, -1, mask)

        # Compute warping
        causal_warp = self._causality(tim1, a.trapped)

        # Assemble
        self._add_contribution(
            present_kernel, present_warp,
            causal_warp, mask=mask,
            output=output,
        )

    def _warp_half(self, phi, tim, sign, mask=None):
        """Compute the displacement warping between the past and present particle."""
        c = self._computer
        a = self._adv

        # Compute relevant particles
        if mask is not None:
            tim = tim[..., mask]
            phi = phi[..., mask]
        else:
            tim = ravel_4(tim)
            phi = ravel_4(phi)

        # Fourier-space displacement
        warp_w = np.exp(- sign * 1j * np.multiply.outer(c._omega, tim))
        warp_n = np.exp(+ sign * 1j * np.multiply.outer(c._ntor,  phi))
        assert np.all(np.isfinite(warp_w))
        assert np.all(np.isfinite(warp_n))

        return warp_w, warp_n

    def _causality(self, tim1, mask=None):
        """Compute the displacement warping between the past and present particle."""
        c = self._computer
        a = self._adv

        # Compute shifts
        dtim = tim1[np.newaxis, :] - self._tim[:, np.newaxis]
        del tim1

        # Find non-living particles
        both_live = self._living
        if mask is not None:
            both_live = both_live & mask
        both_live = both_live[np.newaxis, :] & both_live[:, np.newaxis]

        # Remove non-living particles from count
        dtim[~both_live] = 0

        # Sanity check for unhandled cases
        assert np.all(dtim < a.bounce_time)
        assert np.all(dtim >= -a.bounce_time)

        # Compute the numbre of elements to add/remove
        count = dtim < 0

        # Compute relevant particles
        if mask is not None:
            count = count[..., mask]
        else:
            count = ravel_4(count)

        return count

    def _add_contribution(
        self, present_kernel,
        present_warp, causal_count,
        *, output, mask=None,
    ):
        """Integrate on the two particles trajectories to yield the required
        function.
        """
        raise NotImplementedError()

class DenseKernelElementComputer(KernelElementComputer):
    def _add_contribution(
        self, present_kernel,
        present_warp, causal_count,
        *, output, mask=None,
    ):
        """Integrate on the two particles trajectories to yield the required
        function.
        """
        c = self._computer

        # Compute relevant subarray
        past_warp   = self._past_warp
        past_warp_n = self._past_warp_n
        past_warp_w = self._past_warp_w
        if mask is not None:
            past_warp   = self._past_warp  [..., mask.ravel()]
            past_warp_n = self._past_warp_n[..., mask.ravel()]
            past_warp_w = self._past_warp_w[..., mask.ravel()]
            bounce_warp = self._bounce_warp[..., mask]
        else:
            bounce_warp = ravel_4(self._bounce_warp)

        present_warp_w, present_warp_n = present_warp

        # Sum all the things!
        source = np.einsum(
            # w=omega, n=ntor, p=particles,
            # uv=theta vectorisation,
            # yz=both psi, hj=both theta
            'wnp,naup,wup,nup,wvp,nvp,bvp->wnab',
            bounce_warp,
            past_warp, past_warp_w, past_warp_n,
            present_warp_w, present_warp_n,
            present_kernel,
            optimize=True,
        )
        assert np.all(np.isfinite(source))

        # Remove uncausal elements
        if causal_count.any():
            source -= np.einsum(
                # w=omega, n=ntor, p=particles,
                # uv=theta vectorisation,
                # yz=both psi, hj=both theta
                'naup,wup,nup,wvp,nvp,bvp,uvp->wnab',
                past_warp,
                past_warp_w, past_warp_n,
                present_warp_w, present_warp_n,
                present_kernel,
                causal_count,
                optimize=True,
            )
            assert np.all(np.isfinite(source))

        print('ACC')
        interpolator = self._interpolator
        source = source.reshape(
            *source.shape[:2],
            *interpolator.shape[:2],
            *interpolator.shape[:2],
        )
        output[
            :, :,
            interpolator.psi_slice, interpolator.theta_slice,
            interpolator.psi_slice, interpolator.theta_slice,
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

        sl = (Ellipsis,) + (grid.mu.ndim - 2) * (None,)
        self._Neq = Neq[sl]
        self._Teq = Teq[sl]
        if Veq is not None:
            self._Veq = Veq[sl]
        else:
            self._Veq = None

        # Warning: Large matrix
        self._output = np.zeros((
            omega.size, ntor.size,
            psi.size, theta.size,
            psi.size, theta.size,
        ), dtype=np.complex128)

    def compute_distribution(self, adv):
        g = self._grid
        Neq = self._Neq
        Teq = self._Teq
        Veq = self._Veq

        # Distribution function
        lnFeq = np.zeros(np.broadcast(g.psi, g.vpar, g.mu).shape[1:])
        lnFeq -= - adv.energy.mean(axis=0).squeeze()/Teq
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
        feb = DenseP1Basis(self._psi, self._theta, gyroavg)
        kec = DenseKernelElementComputer(self._grid, self, adv)

        # Particle-independent effects
        kec.precompute()

        # Compute interpolation matrix
        kec.interpolate(self._theta, feb)

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
    adv.compute_invariants()
    adv.compute_bounce_point()
    adv.compute_trajectory()
    adv.compute_displacement()
    adv.compute_precession()
    np.who(adv.__dict__)

    # Compute kernel
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
    kern.compute_distribution(adv)
    kern.compute(adv)

    out = kern._output
    print(out.shape)
    print(abs(out).min(), abs(out).max())

    plt.pcolormesh(
        abs(out[0, -1].reshape(13*19, 13*19)),
#         norm=mpl.colors.LogNorm(),
    )
    plt.colorbar()

# %load_ext line_profiler

# %prun main()
# %lprun -f KernelComputer.compute main()
main()
