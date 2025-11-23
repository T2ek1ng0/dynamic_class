from dynamic_class.my_utils import construct_problem_set
from dynamic_class.my_config import Config
from dynamic_class.my_tester import Tester, get_baseline
from dynamic_class.gleet_optimizer import GLEET_Optimizer
from dynamic_class.baseline.PSPSO import PSPSO

# specify your configuration
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
    'test_run': 1,
    'baselines': {
        # Other baselines to compare
        'PSPSO': {'optimizer': PSPSO},
    },
}
config = Config(config)
# construct dataset
config, datasets = construct_problem_set(config)
# initialize all baselines to compare (yours + others)
baselines, config = get_baseline(config)
# initialize tester
tester = Tester(config, baselines, datasets)
# test
if __name__ == '__main__':
    tester.test()