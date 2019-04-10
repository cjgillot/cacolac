
# coding: utf-8

# In[1]:


import importlib

import numpy as np
import scipy.interpolate as scip
from scipy.interpolate import make_interp_spline as interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

mpl.rcParams['figure.figsize'] = 10, 5


# In[2]:


class Grid:
    def __init__(self, As, Zs, R0, rg, qq, theta, vpar, mu):
        self._As = As
        self._Zs = Zs
        self._R0 = R0

        psi = interp1d(rg, rg/qq).antiderivative()
        psi = psi(rg)
        theta, psi, vpar, mu = np.meshgrid(
            theta, psi, vpar, mu,
            indexing='ij', sparse=True,
        )
        rg = rg[np.newaxis, :, np.newaxis, np.newaxis]
        Rred = 1 + rg/R0 * np.cos(theta)
        ltor = - Zs * psi + As * R0 * Rred * vpar

        self._r = rg
        self._y = psi
        self._theta = theta
        self._ltor = ltor
        self._vpar = vpar
        self._ener = vpar**2 + mu
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


# In[49]:


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

        ener = g.vpar**2 + 2/A * g.mu + 2*Z/A * P(psi)
        ltor = psi - A/Z * R0 * Rred * vpar

        shape = np.broadcast(psi, g.theta, g.mu, vpar).shape

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
                jac[..., 1, 0] = - 2/A/R0 * g.mu * (np.cos(g.theta) / Rred**2)                                 + 2*Z/A * P(psi, nu=1)
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

                trapped = np.any(vpar == 0, axis=1, keepdims=True)

                # Remove average for passing particles
                avg_psi = psi.mean(axis=1, keepdims=True) - g.psi
                psi -= (~trapped) * avg_psi

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

        self._psi = psi
        self._r = r
        self._vpar = vpar
        self._Rred = Rred

        self._trapped = np.any(vpar == 0, axis=1, keepdims=True)

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
        self._ifreq = ifreq
        spline = interp1d(theta, ifreq)
        self._int_time = spline.antiderivative()

        vphi  = self._vpar / self._Rred / g.R0
        # FIXME Add vE.gradphi
        # FIXME Add vD.gradphi

        self._vphi = vphi
        spline = interp1d(theta, vphi * ifreq)
        self._int_phi = spline.antiderivative()

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
    def phi_path(self):
        return self._int_phi


# In[112]:


class make_interp_kernel:
    def __init__(
        self,
        psi_grid, psi,
        theta_grid, theta,
        with_deriv=False,
        gyroavg=False,
    ):
        if np.ndim(theta) != 0:
            theta = theta[:, np.newaxis, np.newaxis, np.newaxis]
        else:
            theta = np.atleast_3d(theta)
        self.psi_size, self.psi_slice = self._choose_slice(psi_grid, psi)
        self.theta_size, self.theta_slice = self._choose_slice(theta_grid, theta)

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
        np.who(locals())
        np.who(self.__dict__)

        self.val_ker *= sinc
        if with_deriv:
            self.dpsi_ker *= sinc
            self.dtheta_ker *= sinc
            self.dtheta_ker *= - 2 * theta_dist / dtheta**2

    def _choose_slice(self, grid, values):
        i_min = grid.searchsorted(values.min(), 'left') - 1
        i_max = grid.searchsorted(values.max(), 'right') + 1
        i_min = max(i_min, 0)
        i_max = min(i_max, grid.size)

        return i_max - i_min, np.s_[i_min:i_max]


# In[177]:


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

        self._Neq = Neq[:, np.newaxis, np.newaxis]
        self._Teq = Teq[:, np.newaxis, np.newaxis]
        if Veq is not None:
            self._Veq = Veq[:, np.newaxis, np.newaxis]
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
        Feq = np.zeros((
            g.psi.size, g.vpar.size, g.mu.size,
        ))
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
        g = self._grid
        psi_grid = self._psi
        theta_grid = self._theta

        # Background and entropic frequency
        FonT = self._FonT
        freq_star = self._freq_star

        # Interpolate path
        psi_path  = interp1d(g.theta.squeeze(), adv.psi_path)
        time_path = adv.time_path
        phi_path  = adv.phi_path

        # Prepare grid
        omega, ntor = np.meshgrid(
            self._omega, self._ntor,
            indexing='ij', sparse=True,
        )

        # Loop over the past position
        for j1 in range(3):#theta_grid.size):
            theta1 = theta_grid[j1]
            psi1 = psi_path(theta1)
            tim1 = time_path(theta1)
            phi1 = phi_path(theta1)

            # Interpolation
            past_kernel = make_interp_kernel(
                psi_grid, psi1,
                theta_grid, theta1,
                with_deriv=True,
                gyroavg=gyroavg,
            )

            # Loop over the present position
            if True:
                print('THETA', j1)
                theta2 = theta_grid.squeeze()
            #for j2 in range(theta_grid.size):
            #    print('THETA', j1, j2)
            #    theta2 = theta_grid[j2]
                psi2       = psi_path(theta2)
                time_shift = time_path(theta2) - tim1
                phi_shift  = phi_path (theta2) - phi1

                # Interpolation
                present_kernel = make_interp_kernel(
                    psi_grid, psi2,
                    theta_grid, theta2,
                    with_deriv=False,
                    gyroavg=gyroavg,
                )
                np.who(locals())

                # Fourier-space displacement
                warp  = (
                    - np.multiply.outer(omega, time_shift)
                    + np.multiply.outer(ntor, phi_shift)
                )
                warp  = np.exp(1j * warp)
                warp *= adv.ifreq[j1]

                #warp = warp[np.newaxis, np.newaxis, :] * (time_shift <= 0)
                warp[:, :, time_shift > 0] = 0
                print('WARP')

                # Contribution
                source = np.zeros((
                    omega.size, ntor.size,
                    past_kernel.psi_size, present_kernel.psi_size,
                    past_kernel.theta_size, present_kernel.theta_size,
                ), dtype=np.complex128)
                print('ES')

                #source += np.einsum(
                #    'wn...lvm,lvm,lvm,yhlvm,zj...lvm->wnyzhj',
                #    warp, FonT, freq_star[0],
                #    past_kernel.dpsi_ker, present_kernel.val_ker,
                #    optimize='optimal',
                #)
                #print('ES 0')

                #source += np.einsum(
                #    'wn...lvm,lvm,lvm,yhlvm,zj...lvm->wnyzhj',
                #    warp, FonT, freq_star[1],
                #    past_kernel.dtheta_ker, present_kernel.val_ker,
                #    optimize='optimal',
                #)
                #print('ES 1')

                source += np.einsum(
                    'wn...lvm,lvm,lvm,n,yhlvm,zj...lvm->wnyzhj',
                    warp, FonT, freq_star[2], 1j * ntor.squeeze(),
                    past_kernel.val_ker, present_kernel.val_ker,
                    optimize='optimal',
                )
                print('ES 2')

                self._output[
                    :, :,
                    past_kernel.psi_slice, present_kernel.psi_slice,
                    past_kernel.theta_slice, present_kernel.theta_slice
                ] += source
                #raise NotImplementedError()


# In[178]:


def main():
    A = Z = 1
    R0 = 900

    rg = np.linspace(100, 150, 32)
    qq = 1 + 0*np.linspace(0, 1, rg.size)**2

    theta = np.linspace(0, 2 * np.pi, 64)
    vpar = np.linspace(-1, 1, 24)
    mu = np.linspace(0, 1, 16)

    grid = Grid(1, 1, 900, rg, qq, theta, vpar, mu)
    np.who(grid.__dict__)

    #plt.plot(grid.radius.squeeze(), grid.psi.squeeze())
    #plt.show()

    pot = np.zeros_like(rg)

    adv = ParticleAdvector(grid, pot)

    adv.compute_trajectory()

    np.who(adv.__dict__)

    #plt.subplot(121)
    #plt.plot(theta, adv._r[16].reshape(theta.size, -1))
    #plt.subplot(122)
    #plt.plot(theta, adv._vpar[16].reshape(theta.size, -1))
    #plt.show()

    adv.compute_freq()

    np.who(adv.__dict__)

    time = adv._int_time(grid.theta.squeeze())
    phi  = adv._int_phi(grid.theta.squeeze())
    #plt.subplot(121)
    #plt.plot(
    #    theta,
    #    time[16, :].reshape(theta.size, -1),
    #)
    #plt.subplot(122)
    #plt.plot(
    #    theta,
    #    phi[16, :].reshape(theta.size, -1),
    #)
    #plt.show()

    kern = KernelComputer(
        grid,
        psi=rg[2:15],
        theta=theta[2:21],
        Neq=np.ones_like(rg),
        Teq=np.ones_like(rg),
#         Veq=np.zeros_like(rg),
        omega=1e-4 * np.ones(1),
        ntor=np.arange(10),
    )
    kern.compute(adv)


# In[179]:


# %%prun

main()


# In[ ]:


