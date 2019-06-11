"""Grid class for cacolac."""

import numpy as np
from scipy.interpolate import make_interp_spline as interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
plt.ion()

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

        # Sanity checks
        assert np.all(np.diff(theta) > 0)
        assert np.isclose(theta[0], -np.pi)
        assert np.isclose(theta[-1], np.pi)
        assert np.isclose(theta[theta.searchsorted(0)], 0)

        psi = interp1d(rg, rg/qq).antiderivative()
        psi = psi(rg)
        theta, psi, mu, vpar, = meshgrid(theta, psi, mu, vpar)
        rg = rg.reshape(psi.shape)

        self._r = rg
        self._y = psi
        self._theta = theta
        self._vpar = vpar
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
    def vpar(self):
        return self._vpar

    @property
    def mu(self):
        return self._mu

    @property
    def sign(self):
        return self._spar

    @property
    def Rred(self):
        return 1 + self._r/self._R0 * np.cos(self._theta)

    @property
    def Rred_LFS(self):
        return 1 + self._r/self._R0

    def Rred_at(self, psi, theta, dy=0, dj=0):
        if dj & 1 == 0:
            t = np.cos(theta)
        else:
            t = - np.sin(theta)
        if dj & 2 == 1:
            t *= -1
        r = self._r_at(psi, nu=dy)
        Rred = r/self._R0 * t
        if dy == 0 and dj == 0:
            Rred += 1
        return Rred

    def radius_at(self, y, nu=0):
        return self._r_at(y, nu=nu)

    def qprofile_at(self, y, nu=0):
        return self._q_at(y, nu=nu)

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

    plt.figure()
    plt.plot(grid.radius.squeeze(), grid.psi.squeeze())
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        plt.show(block=True)
