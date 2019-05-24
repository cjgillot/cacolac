import time

import numpy as np
import scipy
import scipy.interpolate
import scipy.special as scsp

import matplotlib as mpl
import matplotlib.pyplot as plt

class coord_system:
    def advance(self, jgrid, vgrid, distrib, times):
        raise NotImplementedError
    def compute_energy(self, jgrid, pgrid):
        raise NotImplementedError
    def from_jv(self, jgrid, vgrid):
        raise NotImplementedError
    def into_jv(self, jgrid, vgrid):
        raise NotImplementedError

def system_theta_vpar(coord_system):
    """Simulate pendulum in j/v coordinates.
    The hamiltonian writes:
        H = v^2/2 + 1 - cos(j)
    The equations of motion write:
        dj/dt = v
        dv/dt = - sin(j)
    """
    def advance(self, jgrid, vgrid, distrib, times):
        jgrid_pad = np.pad(
            jgrid.squeeze(), pad_width=5, mode='linear_ramp',
            end_values=(
                jgrid[ 0, 0] - 5 * (jgrid[1, 0] - jgrid[0, 0]),
                jgrid[-1, 0] + 5 * (jgrid[1, 0] - jgrid[0, 0]),
            )
        )

        yield distrib
        for dt in np.diff(times):
            jfoot = jgrid + .5 * vgrid * dt
            vfoot = vgrid - np.sin(jfoot) * dt
            jfoot = jfoot + .5 * vgrid * dt

            # Wrap angle
            jfoot[jfoot > np.pi] -= 2 * np.pi
            jfoot[jfoot <-np.pi] += 2 * np.pi
            assert np.all(jfoot <= np.pi)
            assert np.all(jfoot >=-np.pi)

            distrib = np.pad(distrib, pad_width=((5,5),(0,0)), mode='wrap')
            distrib = scipy.interpolate.RectBivariateSpline(
                jgrid_pad, vgrid.squeeze(), distrib
            )(jfoot, vfoot, grid=False)
            yield distrib

    def compute_energy(self, jgrid, vgrid):
        return .5 * vgrid**2 - np.cos(jgrid)

    def from_jv(self, jgrid, vgrid):
        return jgrid, vgrid
    def into_jv(self, jgrid, vgrid):
        return jgrid, vgrid

def system_theta_ener(coord_system):
    """Simulate pendulum in angle/energy coordinates.
    The angular velocity writes:
        - if v < 1:
            w = pi/8 K(v)
        - if v > 1:
            w = pi sqrt(v)/4 K(v)
    with KK elliptic function.
    """
    def advance(self, jgrid, vgrid, distrib, times):
        yield distrib
        fft_distrib = np.fft.rfft(distrib.T, axis=1)
        n = np.fft.rfftfreq(jgrid.size, jgrid[1] - jgrid[0])

        for dt in np.diff(times):
            w = np.select(
                [vgrid < 1, vgrid > 1],
                [.5 * np.pi / scsp.ellipk(vgrid),
                 np.pi * np.sqrt(vgrid) / scsp.ellipk(1/vgrid)
                ],
                default=0
            )
            fft_distrib *= np.exp(1j * n * w.T * dt)
            yield np.fft.irfft(fft_distrib, axis=1).T

    def compute_energy(self, jgrid, vgrid):
        return vgrid

    def pulsation_canon(self, jgrid, vgrid):
        return np.select(
            [vgrid < 1, vgrid > 1],
            [.5 * np.pi / scsp.ellipk(vgrid),
             np.pi * np.sqrt(vgrid) / scsp.ellipk(1/vgrid)
            ],
            default=0
        )

    def into_jv(self, jgrid, vgrid):
        sv = np.sqrt(vgrid)
        jgrid = jgrid / pulsation_canon(jgrid, vgrid)
        return np.select(
            [vgrid < 1, vgrid > 1, True],
            [2 * np.arcsin(sv * scsp.ellipj(jgrid, vgrid)[0]),
             2 * np.arcsin(scsp.ellipj(sv * jgrid, 1/vgrid)[0]),
             2 * np.arcsin(np.tanh(jgrid))
            ]
        ), np.select(
            [vgrid < 1, vgrid > 1, True],
            [2 * sv * scsp.ellipj(jgrid, vgrid)[1],
             2 * sv * scsp.ellipj(sv * jgrid, 1/vgrid)[2],
             2 / np.cosh(jgrid)
            ]
        )

def system_canon(coord_system):
    """Simulate pendulum in canonical coordinates.
    The angular velocity writes:
        - if v < 1:
            w = pi/8 K(v)
        - if v > 1:
            w = pi sqrt(v)/4 K(v)
    with KK elliptic function.
    """
    def advance(self, jgrid, vgrid, distrib, times):
        yield distrib
        fft_distrib = np.fft.rfft(distrib.T, axis=1)
        n = np.fft.rfftfreq(jgrid.size, jgrid[1] - jgrid[0])

        for dt in np.diff(times):
            w = np.select(
                [vgrid < 1, vgrid > 1],
                [.5 * np.pi / scsp.ellipk(vgrid),
                 np.pi * np.sqrt(vgrid) / scsp.ellipk(1/vgrid)
                ],
                default=0
            )
            fft_distrib *= np.exp(1j * n * w.T * dt)
            yield np.fft.irfft(fft_distrib, axis=1).T

    def compute_energy(self, jgrid, vgrid):
        return vgrid

    def pulsation_canon(self, jgrid, vgrid):
        return np.select(
            [vgrid < 1, vgrid > 1],
            [.5 * np.pi / scsp.ellipk(vgrid),
             np.pi * np.sqrt(vgrid) / scsp.ellipk(1/vgrid)
            ],
            default=0
        )

    def into_jv(self, jgrid, vgrid):
        sv = np.sqrt(vgrid)
        jgrid = jgrid / pulsation_canon(jgrid, vgrid)
        return np.select(
            [vgrid < 1, vgrid > 1, True],
            [2 * np.arcsin(sv * scsp.ellipj(jgrid, vgrid)[0]),
             2 * np.arcsin(scsp.ellipj(sv * jgrid, 1/vgrid)[0]),
             2 * np.arcsin(np.tanh(jgrid))
            ]
        ), np.select(
            [vgrid < 1, vgrid > 1, True],
            [2 * sv * scsp.ellipj(jgrid, vgrid)[1],
             2 * sv * scsp.ellipj(sv * jgrid, 1/vgrid)[2],
             2 / np.cosh(jgrid)
            ]
        )

def main(times):
    energy = np.zeros((times.size, 2))

    stv = system_theta_vpar()
    stk = system_theta_ener()

    # Initialize distribution function
    vgrid = np.linspace(-3., 3., 100)
    jgrid = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    jgrid, vgrid = np.meshgrid(jgrid, vgrid, indexing='ij', sparse=True)
    energy_map = stv.compute_energy(jgrid, vgrid)

    distrib    = np.zeros((jgrid.size, vgrid.size))
    distrib[:] = np.exp(- .5 * energy_map)
    distrib[:]*= 1 + .1 * np.cos(5 * jgrid)

    stv.energy_map = energy_map
    stv.distrib    = distrib

    # Same for canonical
    vgrid = np.linspace(0, 3., 100)
    jgrid = np.linspace(-np.pi, np.pi, 100, endpoint=False)
    jgrid, vgrid = np.meshgrid(jgrid, vgrid, indexing='ij', sparse=True)
    energy_map = system.compute_energy(jgrid, vgrid)

    distrib    = np.zeros((jgrid.size, vgrid.size))
    distrib[:] = np.exp(-.5 * energy_map)
    distrib[:]*= 1 + .1 * np.cos(5 * theta_canon(jgrid, vgrid))

    stk.energy_map = energy_map
    stk.distrib    = distrib

    for ii, sys in enumerate([stv, stk]):
        for it, distrib in enumerate(sys.advance(sys.distrib, jgrid, vgrid, times)):
            energy[it, ii] = np.sum(sys.energy_map * sys.distrib)

        plt.figure()
        plt.pcolormesh(vgrid.squeeze(), jgrid.squeeze(), distrib,
                       norm=mpl.colors.LogNorm())
        plt.contour(vgrid.squeeze(), jgrid.squeeze(), energy_map,
                    levels=[1.], cmap='Reds')
        plt.show(block=False)

        if sys is not stv:
            plt.figure()
            jloc, vloc = sys.into_jv(jgrid, vgrid[vgrid<1])
            plt.pcolormesh(jloc, vloc,
                           distrib[:, vgrid[0]<1],
                           norm=mpl.colors.LogNorm())
            jloc, vloc = sys.into_jv(jgrid, vgrid[vgrid>1])
            plt.pcolormesh(jloc, vloc,
                           distrib[:, vgrid[0]>1],
                           norm=mpl.colors.LogNorm())
            jloc, vloc = sys.into_jv(jgrid, vgrid[vgrid=1])
            plt.plot(jloc, vloc, c='r')
            plt.show(block=False)

    # Plot!
    plt.figure()
    plt.plot(times, abs(energy / energy[0]))
    plt.yscale('log')
    plt.show(block=True)

plt.ion()
main(np.arange(0, 70, .1))
