import numpy as np

class ConvolveOps:
    def __init__(self, x, filter_):
        self.in_dim = x.shape[0]
        self.filter_size = filter_.shape[0]
        self.c = filter_.shape[2]
        self.out_dim = self.in_dim - self.filter_size + 1

        self.feature_map = np.zeros((self.out_dim, self.out_dim, self.c))

        self.x = x
        self.filter = filter_

    def convolve(self, optimize=True):
        """Perform convolution between the input image and the filter.
        Args:
            optimize(boolean): choose to use the optimized convolution op or not.
            the brute force operation can be used for debugging.
        """
        if optimize:
            return self._convolve_optimize()
        else:
            return self._convolve_brute_force()

    def _convolve_brute_force(self):
        """Brute force convolution operation, assuming that stride=1, padding=0."""
        for depth_slice in range(self.c):
            for i in range(self.out_dim):
                for j in range(self.out_dim):
                    self.feature_map[i][j][depth_slice] = np.sum(
                                np.multiply(self.x[i:i+self.filter_size, j:j+self.filter_size], 
                                            self.filter[:, :, depth_slice])
                                )
        
        return self.feature_map
    
    def _convolve_optimize(self):
        """Optimized convolution operation using matrix multiplication.
        Achieved a speedup of ~10 times.
        """
        x_col = self._image_to_col()
        filter_row = self._filter_to_row()

        # Convolve, resulting in shape (#filters, #receptive_fields)
        self.feature_map = np.dot(filter_row, x_col)

        # Reshape the result into (out_dim, out_dim, c), no channel at this point
        self.feature_map = self.feature_map.T.reshape((self.out_dim, self.out_dim, self.c))

        return self.feature_map
    
    def _image_to_col(self):
        """Stretch the local regions in the input image into columns.
        Original dimension of x: (d, d)
        Stretched dimension of x: (#weights_in_one_filter, #receptive_fields)
                i.e., (filter_size * filter_size, out_dim * out_dim)
                where c is the number of channels
        """
        x_col = np.zeros((self.filter_size * self.filter_size, 
                          self.out_dim * self.out_dim))
        index = 0
        for i in range(self.out_dim):
            for j in range(self.out_dim):
                x_col[:, index] = self.x[i:i+self.filter_size, j:j+self.filter_size].flatten()
                index += 1

        return x_col
    
    def _filter_to_row(self):
        """Stretch the weights in the filter into rows.
        Original dimension of filter: (filter_size, filter_size)
        Stretched dimention of filter: (#filters, #weights_in_one_filter)
                i.e., (c, filter_size * filter_size)
                where c is the number of channels
        """
        filter_row = np.zeros((self.c, self.filter_size * self.filter_size))
        for i in range(self.c):
            filter_row[i] = self.filter[:, :, i].T.flatten()
        
        return filter_row
