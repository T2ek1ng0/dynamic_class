import numpy as np
import torch
from my_weight import Sub_Problem_Weight
from my_noise import Gaussian_noise
from my_dynamic_class import *
from basic_problem import *
from torch.utils.data import Dataset

class Dynamic_Dataset(Dataset):
    def __init__(self, data, batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)

    @staticmethod
    def get_datasets(upperbound,
                    dim=10,
                    shifted=True,
                    rotated=True,
                    biased=True,
                    version='numpy',
                    train_batch_size=1,
                    test_batch_size=1,
                    difficulty=None,
                    user_train_list=None,
                    user_test_list=None,
                    instance_seed=3849):

        if instance_seed > 0:
            np.random.seed(instance_seed)
            torch.manual_seed(instance_seed)

        func_list = ["Sphere", "Schwefel12", "Ellipsoidal", "Ellipsoidal_high_cond", "Bent_cigar",
                     "Discus", "Dif_powers", "Rosenbrock", "Ackley", "Weierstrass", "Griewank", "Rastrigin",
                     "Buche_Rastrigin", "Mod_Schwefel", "Katsuura", "Grie_rosen", "Escaffer6", "Happycat",
                     "Hgbat", "Lunacek_bi_Rastrigin", "Zakharov", "Levy", "Scaffer_F7", "Step_Rastrigin",
                     "Linear_Slope", "Attractive_Sector", "Step_Ellipsoidal", "Sharp_Ridge", "Rastrigin_F15",
                     "Schwefel", "Gallagher101", "Gallagher21"]
        func_idx = [i for i in range(32)]  # 0-31
        my_noise = Gaussian_noise(begin_percent=0.01, end_percent=0.05)
        assert upperbound >= 5., f'Argument upperbound must be at least 5, but got {upperbound}.'
        ub = upperbound
        lb = -upperbound
        instance_noise = []  # 32
        for id in func_idx:
            if shifted:
                shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
            else:
                shift = np.zeros(dim)
            if rotated:
                H = rotate_gen(dim)
            else:
                H = np.eye(dim)
            if biased:
                bias = np.random.randint(1, 26) * 100
            else:
                bias = 0
            instance = eval(func_list[id])(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
            instance_noise.append(Dynamic_Problem(max_fes=20000, problem_list=instance, noise=my_noise))

        instance_weight = []  # 12*2+12*3+12*4+12*5
        for i in range(48):
            sample_num = i // 12 + 2
            sample_idx = np.random.choice(func_idx, sample_num, replace=True)
            instance_list = []
            for id in sample_idx:
                if shifted:
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                else:
                    shift = np.zeros(dim)
                if rotated:
                    H = rotate_gen(dim)
                else:
                    H = np.eye(dim)
                if biased:
                    bias = np.random.randint(1, 26) * 100
                else:
                    bias = 0
                instance = eval(func_list[id])(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
                instance_list.append(instance)
            instance_weight.append(Dynamic_Problem(max_fes=20000,
                                                   problem_list=instance_list,
                                                   population_weight=Sub_Problem_Weight(len(instance_list), 20000, rates=1 / len(instance_list))))

        instance_composition = []  # 12*2+12*3+12*4+12*5
        for i in range(48):
            sample_num = i // 12 + 2
            sample_idx = np.random.choice(func_idx, sample_num, replace=True)
            instance_list = []
            for id in sample_idx:
                if shifted:
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                else:
                    shift = np.zeros(dim)
                if rotated:
                    H = rotate_gen(dim)
                else:
                    H = np.eye(dim)
                if biased:
                    bias = np.random.randint(1, 26) * 100
                else:
                    bias = 0
                instance = eval(func_list[id])(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
                instance_list.append(instance)
            instance_composition.append(Dynamic_Problem(max_fes=20000,
                                                   problem_list=instance_list,
                                                   population_weight=Sub_Problem_Weight(len(instance_list), 20000, rates=1 / len(instance_list)),
                                                   noise=my_noise))

        instance_list = instance_noise + instance_weight + instance_composition
        indices = np.arange(len(instance_list))
        train_idx = np.random.choice(indices, 64, replace=False)
        test_idx = np.setdiff1d(indices, train_idx)
        train_set = [instance_list[i] for i in train_idx]
        test_set = [instance_list[i] for i in test_idx]

        return Dynamic_Dataset(train_set, train_batch_size), Dynamic_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Dynamic_Dataset'):
        return Dynamic_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)



