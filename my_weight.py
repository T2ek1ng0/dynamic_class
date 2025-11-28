import numpy as np
from typing import Union

class Sub_Problem_Weight:
    def __init__(self, n_problem, max_fes, rates: Union[list, np.ndarray, float]):
        if isinstance(rates, list):
            rates = np.array(rates, dtype=float)
        elif isinstance(rates, float):
            if not (0 < rates <= 1.0):
                raise ValueError(f"Invalid rate value {rates}, must be in (0, 1].")
            temp_rates = np.arange(rates, 1.0 + 1e-8, rates)
            rates = np.round(np.clip(temp_rates, None, 1.0), 4)
        elif not isinstance(rates, np.ndarray):
            raise TypeError("rates must be list, np.ndarray, or float")
        if isinstance(rates, np.ndarray):
            if rates[-1] < 1.0 - 1e-8:
                rates = np.append(rates, 1.0)
        self.n_problem = n_problem
        self.rates = rates
        self.maxfes = max_fes
        self.pos = np.random.randint(self.n_problem)
        self.fes_thresholds = np.sort(self.rates * self.maxfes)
        self.fes_process = -1
        self.change_flag = 0

    def cal_weight(self, ps: int, curr_fes: int, cost_fes, change_flag=False):  # weight_matrix: (batch_size,ps,n_problem)
        batch_size = cost_fes // ps
        assert batch_size >= 1
        total_fes = np.arange(curr_fes + 1, curr_fes + 1 + ps * batch_size)  # (batch_size*ps,)
        total_process = np.searchsorted(self.fes_thresholds, total_fes, side='right') - 1  # (batch_size*ps,)
        old_process = np.r_[self.fes_process, total_process[:-1]]
        update_mask = (total_process > old_process).astype(int)  # (batch_size*ps,)
        if np.sum(update_mask) > 0:
            self.change_flag = 1
        pos_matrix = ((self.pos + np.cumsum(update_mask)) % self.n_problem).reshape(batch_size, ps)
        weight_matrix = np.zeros((batch_size, ps, self.n_problem))
        weight_matrix[np.arange(batch_size)[:, None], np.arange(ps), pos_matrix] = 1

        if change_flag:
            self.pos = int(pos_matrix[-1][-1])
            self.fes_process = int(total_process[-1])
        return weight_matrix[0] if batch_size == 1 else weight_matrix













