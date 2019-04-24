import numpy as np

class InterpolationMatrix: pass

class FEBasis:
    def size(self):
        raise NotImplementedError()

    def interpolate(self, psi, theta, with_deriv=False, gyroavg=False):
        raise NotImplementedError()

    def _adjust_points(self, psi, theta):
        psi = np.asanyarray(psi); n = psi.ndim
        theta = np.asanyarray(theta); m = theta.ndim
        theta = theta[(Ellipsis,) + (n - m) * (None,)]
        return psi, theta

class P1Basis(FEBasis):
    def __init__(self, psi_grid, theta_grid, gyroavg=False):
        self._psi_grid = psi_grid
        self._theta_grid = theta_grid
        self._gyroavg = gyroavg

    def size(self):
        return self._psi_grid.size * self._theta_grid.size

    def _compute_linear(self, grid, point):
        # Linear interpolation in theta
        sample = np.diff(grid.squeeze()).mean()
        dist   = np.subtract.outer(grid, point)

        mask = abs(dist) < sample
#         dist = dist[mask]

        val = 1 - abs(dist / sample)
        val*= mask
        der = - np.sign(dist) / sample
        der*= mask

        return val, der, mask

class DenseP1Basis(P1Basis):
    def interpolate(self, psi, theta, mask, with_deriv=False):
        psi, theta = self._adjust_points(psi, theta)

        # Compute coefficients
        val_psi, der_psi, _ = self._compute_linear(self._psi_grid, psi)
        val_tht, der_tht, _ = self._compute_linear(self._theta_grid, theta)

        # Remove disabled cells
        val_psi *= mask
        der_psi *= mask

        ret = InterpolationMatrix()
        ret.psi_size, ret.psi_slice = self._choose_slice(self._psi_grid, psi)
        ret.theta_size, ret.theta_slice = self._choose_slice(self._theta_grid, theta)

        shape = ret.psi_size, ret.theta_size, *psi.shape
        reshape = ret.psi_size * ret.theta_size, *psi.shape
        ret.shape = shape

        # Compute data matrices
        sl_psi = np.s_[ret.psi_slice, np.newaxis]
        sl_tht = np.s_[np.newaxis, ret.theta_slice]
        ret.val_data     = np.empty(shape, dtype=float)
        ret.val_data[:]  = val_psi[sl_psi]
        ret.val_data[:] *= val_tht[sl_tht]
        ret.val_data     = ret.val_data.reshape(reshape)

        if with_deriv:
            ret.dps_data = np.empty(shape, dtype=float)
            ret.dps_data[:]  = der_psi[sl_psi]
            ret.dps_data[:] *= val_tht[sl_tht]
            ret.dps_data     = ret.dps_data.reshape(reshape)
            ret.dth_data = np.empty(shape, dtype=float)
            ret.dth_data[:]  = val_psi[sl_psi]
            ret.dth_data[:] *= der_tht[sl_tht]
            ret.dth_data     = ret.dth_data.reshape(reshape)

        # Print statistics on sparsity
        print('SPARSITY', np.count_nonzero(ret.val_data) / ret.val_data.size)

        return ret

    def _choose_slice(self, grid, values):
        i_min = grid.searchsorted(values.min(), 'left') - 1
        i_max = grid.searchsorted(values.max(), 'right') + 1
        i_min = max(i_min, 0)
        i_max = min(i_max, grid.size)

        return i_max - i_min, np.s_[i_min:i_max]

class SparseP1Basis(P1Basis):
    def interpolate(self, psi, theta, mask, with_deriv=False):
        # Compute coefficients
        val_psi, der_psi, mask_psi = self._compute_linear(self._psi_grid, psi)
        val_tht, der_tht, mask_tht = self._compute_linear(self._theta_grid, theta)

        mask_psi, mask_tht = self._adjust_points(mask_psi, mask_tht)

        # Find non-zero values
        mask = (
            mask_psi[:, np.newaxis, :] &
            mask_tht[np.newaxis, :, :]
        ) & mask
        print('SPARSITY', mask.sum() / mask.size)
        block = mask.shape[-4:]
        mask  = mask.any(axis=(-4, -3, -2, -1))
        del mask_psi, mask_tht

        np.who(locals())

        # Compute positions of the non-zero values
        idx_psi, idx_tht, idx_vec = mask.nonzero()
        indices = idx_vec

        nvals = indices.size
        #ncols = np.product(mask.shape[2:])
        nrows = np.product(mask.shape[:2])
        block = mask.shape[3:]

        mask = mask.reshape(*mask.shape[:2], -1)

        # Dummy object for returning
        ret = InterpolationMatrix()

        ret.nvals      = nvals
        ret.block      = block

        ret.indices    = indices
        ret.idx_psi    = idx_psi
        ret.idx_theta  = idx_tht
        ret.idx_vector = idx_vec

        ret.indptr     = np.zeros(1+nrows, dtype=np.int32)
        ret.indptr[1:] = np.cumsum(mask.sum(axis=-1).ravel())

        # Compute data matrices
        shape = nvals, *block
        sl_psi = np.s_[idx_psi, idx_vec, :, :, :, :]
        sl_tht = np.s_[idx_tht, idx_vec] + len(block) * (np.newaxis,)
        ret.val_data     = np.empty(shape, dtype=float)
        ret.val_data[:]  = val_psi[sl_psi]
        ret.val_data[:] *= val_tht[sl_tht]

        if with_deriv:
            ret.dps_data = np.empty(shape, dtype=float)
            ret.dps_data[:]  = der_psi[sl_psi]
            ret.dps_data[:] *= val_tht[sl_tht]
            ret.dth_data = np.empty(shape, dtype=float)
            ret.dth_data[:]  = val_psi[sl_psi]
            ret.dth_data[:] *= der_tht[sl_tht]

        return ret
