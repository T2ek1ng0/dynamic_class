from dynamic_class.baseline.PSPSO import PSPSO
from dynamic_class.baseline.DPCPSO import DPCPSO
from dynamic_class.baseline.SPSO_AP_AD import SPSO_AP_AD
from dynamic_class.baseline.ACFPSO import ACFPSO
from dynamic_class.baseline.APCPSO import APCPSO
from dynamic_class.baseline.GMPB import GMPB
from dynamic_class.my_config import Config
import numpy as np
import math
import random

config = {
    'train_problem': 'GMPB',
    'train_batch_size': 8,
    'train_parallel_mode': 'dummy',  # dummy/subproc/ray/ray-subproc
    'max_epoch': 20,  # 100
    'train_mode': 'multi',
    'test_problem': 'GMPB',  # specify the problem set you want to benchmark
    'test_batch_size': 8,
    'test_difficulty': 'easy',  # this is a train-test split mode
    'test_parallel_mode': 'Serial',  # 'Full', 'Baseline_Problem', 'Problem_Testrun', 'Batch', 'Serial'
    'baselines': {
        'Pspso': {'optimizer': APCPSO},
    },
}
RunNumber = 10
config = Config(config)
OfflineError = []
for i in range(RunNumber):
    run_counter = i + 1
    np.random.seed(run_counter)
    random.seed(run_counter)
    problem = GMPB(dim=5,
                 PeakNumber=10,
                 ChangeFrequency=2500,
                 ShiftSeverity=1,
                 EnvironmentNumber=10,
                 HeightSeverity=7,
                 WidthSeverity=1,
                 AngleSeverity=math.pi / 9,
                 TauSeverity=0.2,
                 EtaSeverity=10)
    #np.random.seed(None)
    #random.seed(None)
    opt = APCPSO(config)
    res = opt.run_episode(problem)
    OfflineError.append(res['current_error'])
    print(f"Run {run_counter}/{RunNumber}, current error: {res['current_error']:.2f}")
print(f"E_o: mean={np.mean(OfflineError)}, median={np.median(OfflineError)}, std={np.std(OfflineError)}")
