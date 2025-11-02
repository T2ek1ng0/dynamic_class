import numpy as np


class Gaussian_noise:
    def __init__(self, begin_percent, end_percent, begin_std=None, end_std=None, mean=0):
        #self.begin_std = begin_std
        #self.end_std = end_std
        #self.mean = mean
        self.begin_percent = begin_percent
        self.end_percent = end_percent

    def make_noise(self, curr_fes: int, max_fes, ps, cost_fes=0):  # noise_matrix: (batch_size,ps)
        batch_size = cost_fes // ps
        fes = np.arange(curr_fes + 1, curr_fes + 1 + ps * batch_size).reshape(batch_size, ps)  # (batch_size,ps)
        total_std = self.begin_std + (self.end_std - self.begin_std) * fes / max_fes
        noise = np.random.normal(self.mean, total_std)
        return noise[0] if batch_size == 1 else noise

    def make_pnoise(self, curr_fes: int, max_fes, ps, cost_fes=0):
        batch_size = cost_fes // ps
        fes = np.arange(curr_fes + 1, curr_fes + 1 + ps * batch_size).reshape(batch_size, ps)  # (batch_size,ps)
        p = self.begin_percent + (self.end_percent - self.begin_percent) * fes / max_fes
        return p[0] if batch_size == 1 else p




