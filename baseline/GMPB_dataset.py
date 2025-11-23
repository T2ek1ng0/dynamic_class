import numpy as np
import torch
from torch.utils.data import Dataset
from dynamic_class.baseline.GMPB import *

class GMPB_Dataset(Dataset):
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
    def get_datasets(RunNumber=31,
                     dim=5,
                     PeakNumber=10,
                     ChangeFrequency=5000,
                     ShiftSeverity=1,
                     EnvironmentNumber=100,
                     HeightSeverity=7,
                     WidthSeverity=1,
                     AngleSeverity=math.pi / 9,
                     TauSeverity=0.2,
                     EtaSeverity=10,
                     version='numpy',
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty=None,
                     user_train_list=None,
                     user_test_list=None,
                     instance_seed=3849):

        test_set = []
        for i in range(RunNumber):
            np.random.seed(i + 1)
            test_set.append(GMPB(dim=dim,
                                 PeakNumber=PeakNumber,
                                 ChangeFrequency=ChangeFrequency,
                                 ShiftSeverity=ShiftSeverity,
                                 EnvironmentNumber=EnvironmentNumber,
                                 HeightSeverity=HeightSeverity,
                                 WidthSeverity=WidthSeverity,
                                 AngleSeverity=AngleSeverity,
                                 TauSeverity=TauSeverity,
                                 EtaSeverity=EtaSeverity))
        train_set = []
        indices = np.random.choice(RunNumber, RunNumber // 3, replace=False)
        for i in indices:
            np.random.seed(i + 1)
            train_set.append(GMPB(dim=dim,
                                  PeakNumber=PeakNumber,
                                  ChangeFrequency=ChangeFrequency,
                                  ShiftSeverity=ShiftSeverity,
                                  EnvironmentNumber=EnvironmentNumber,
                                  HeightSeverity=HeightSeverity,
                                  WidthSeverity=WidthSeverity,
                                  AngleSeverity=AngleSeverity,
                                  TauSeverity=TauSeverity,
                                  EtaSeverity=EtaSeverity))

        if instance_seed > 0:
            np.random.seed(instance_seed)
            torch.manual_seed(instance_seed)

        return GMPB_Dataset(train_set, train_batch_size), GMPB_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'GMPB_Dataset'):
        return GMPB_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)