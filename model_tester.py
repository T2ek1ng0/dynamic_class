from my_utils import construct_problem_set
from my_config import Config
#from metaevobox import get_baseline
from my_tester import Tester, get_baseline
from gleet_optimizer import GLEET_Optimizer
from fixedact_nbnc_optimizer import basic_nbnc_Optimizer
from my_dynamic_dataset import Dynamic_Dataset

# specify your configuration
config = {
    'train_problem': 'dynamic-problem',
    'train_batch_size': 8,
    'train_parallel_mode': 'dummy',  # dummy/subproc/ray/ray-subproc
    'max_epoch': 20,  # 100
    'train_mode': 'multi',
    'test_problem': 'dynamic-problem',  # specify the problem set you want to benchmark
    'test_batch_size': 8,
    'test_difficulty': 'easy',  # this is a train-test split mode
    'test_parallel_mode': 'Serial',  # 'Full', 'Baseline_Problem', 'Problem_Testrun', 'Batch', 'Serial'
    'baselines': {
        # your MetaBBO
        'NBNC': {
            'agent': 'GLEET',
            'optimizer': GLEET_Optimizer,
            'model_load_path': r"agent_model\train\GLEET\20251102T125655_dynamic-problem_easy\checkpoint-0.pkl",
        },

        # Other baselines to compare
        #'Fixed_parameter': {'optimizer': basic_nbnc_Optimizer},
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
