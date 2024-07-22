import numpy as np


class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=(0, 1, 2))
            self.std = data.std(axis=(0, 1, 2))
            self.min_val = data.min()
            self.max_val = data.max()
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0
            self.min_val = np.inf
            self.max_val = 0

    def update(self, data):
        """
        data: ndarray
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=(0, 1, 2))
            newstd = data.std(axis=(0, 1, 2))
            new_min_val = data.min()
            new_max_val = data.max()

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std = (
                m / (m + n) * self.std**2
                + n / (m + n) * newstd**2
                + m * n / (m + n) ** 2 * (tmp - newmean) ** 2
            )
            self.std = np.sqrt(self.std)
            self.nobservations += n

            if self.min_val > new_min_val:
                self.min_val = new_min_val
            if self.max_val < new_max_val:
                self.max_val = new_max_val
