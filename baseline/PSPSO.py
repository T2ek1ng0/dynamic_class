"""
Signorelli, Federico, and Anil Yaman.
        "A Perturbation and Speciation-Based Algorithm for Dynamic Optimization Uninformed of Change."
        arXiv:2505.11634 (2025).
paper: https://arxiv.org/abs/2505.11634
source: https://github.com/FreddyDeWatersir/PSPSO
GMPB是最大化优化，如果做最小化优化的话有几个argmax和>要改
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy

class PSPSO(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.w = 0.6
        self.c1 = 2.83
        self.c2 = 2.83
        self.SwarmNumber = 10
        self.SwarmSize = 7
        self.ConvergenceLimit = 0.01
        self.DiversityDegree = 0.7
        self.PerturbationFactor = 0.025
        self.initPopulationSize = self.SwarmSize * self.SwarmNumber

        self.avg_dist = 0

    def __str__(self):
        return "PSPSO"

    def initialize_swarm(self, problem):
        self.ub = problem.ub
        self.lb = problem.lb
        self.dim = problem.dim
        self.ConvergenceLimit = 0.01 * np.sqrt(self.dim)
        self.PerturbationRange = self.PerturbationFactor * (self.ub - self.lb)
        init_swarm = self.lb + (self.ub - self.lb) * np.random.rand(self.initPopulationSize, self.dim)
        self.pop = self.sub_population_generator(init_swarm, problem)
        self.SwarmNumber = len(self.pop)

    def sub_population_generator(self, init_swarm, problem):
        swarm_size = init_swarm.shape[0]
        fitness = problem.eval(init_swarm)
        if problem.avg_dist:
            self.avg_dist += problem.avg_dist
        # Sort particles by their fitness in descending order
        sortIndex = np.argsort(fitness)[::-1]
        # Form clusters
        clusters = []
        used = np.zeros(swarm_size, dtype=bool)
        for i in range(swarm_size):
            if not used[sortIndex[i]]:
                curr_cluster = [sortIndex[i]]
                used[sortIndex[i]] = True
                xa = init_swarm[sortIndex[i]].reshape(1, -1)  # (1,dim)
                available = np.where(~used)[0]
                if available.size > 0:
                    xb = init_swarm[available]
                    distance = cdist(xa, xb)[0]
                    closest_pos = np.argsort(distance)
                else:
                    closest_pos = []
                num_to_add = min(self.SwarmSize - 1, len(closest_pos))
                for j in range(num_to_add):
                    curr_cluster.append(available[closest_pos[j]])
                    used[available[closest_pos[j]]] = True
                clusters.append(curr_cluster)
        population = {
            'X': None,
            'Velocity': None,
            'FitnessValue': None,
            'PbestPosition': None,
            'IsConverged': None,
            'PbestValue': None,
            'GbestValue': None,
            'GbestID': None,
            'GbestPosition': None,
            'Center': None,
            'InitRadius': None,
            'CurrentRadius': None
        }
        Swarm = [copy.deepcopy(population) for _ in range(len(clusters))]
        for i in range(len(clusters)):
            clusterIndices = clusters[i]
            Swarm[i]['X'] = init_swarm[clusterIndices, :]
            Swarm[i]['Velocity'] = -(self.ub - self.lb) / 4 + (2 * (self.ub - self.lb) / 4) * np.random.rand(len(Swarm[i]['X']), self.dim)
            Swarm[i]['FitnessValue'] = fitness[clusterIndices]
            Swarm[i]['PbestPosition'] = Swarm[i]['X'].copy()
            Swarm[i]['IsConverged'] = 0
            Swarm[i]['Center'] = np.mean(Swarm[i]['X'], axis=0)
            Swarm[i]['InitRadius'] = np.mean(cdist(Swarm[i]['X'], Swarm[i]['Center'].reshape(1, -1))[:, 0])
            Swarm[i]['CurrentRadius'] = Swarm[i]['InitRadius']
            if not problem.RecentChange:
                Swarm[i]['PbestValue'] = Swarm[i]['FitnessValue'].copy()
                Swarm[i]['GbestID'] = np.argmax(Swarm[i]['PbestValue'])
                Swarm[i]['GbestValue'] = Swarm[i]['PbestValue'][Swarm[i]['GbestID']]
                Swarm[i]['GbestPosition'] = Swarm[i]['PbestPosition'][Swarm[i]['GbestID'], :].copy()
            else:
                Swarm[i]['FitnessValue'] = -np.inf * np.ones(Swarm[i]['X'].shape[0])
                Swarm[i]['PbestValue'] = Swarm[i]['FitnessValue'].copy()
                Swarm[i]['GbestID'] = np.argmax(Swarm[i]['PbestValue'])
                Swarm[i]['GbestValue'] = Swarm[i]['PbestValue'][Swarm[i]['GbestID']]
                Swarm[i]['GbestPosition'] = Swarm[i]['PbestPosition'][Swarm[i]['GbestID'], :].copy()
        return Swarm

    def iterative_components(self, problem):
        # Sub-swarm movements
        for i in range(self.SwarmNumber):
            if not self.pop[i]['IsConverged']:
                r1 = np.random.rand(*self.pop[i]['X'].shape)
                r2 = np.random.rand(*self.pop[i]['X'].shape)
                self.pop[i]['Velocity'] = self.w * (self.pop[i]['Velocity'] + self.c1 * r1 * (self.pop[i]['PbestPosition'] - self.pop[i]['X']) + self.c2 * r2 * (self.pop[i]['GbestPosition'] - self.pop[i]['X']))
                self.pop[i]['X'] += self.pop[i]['Velocity']
                clip_mask = (self.pop[i]['X'] > self.ub) | (self.pop[i]['X'] < self.lb)
                self.pop[i]['X'] = np.clip(self.pop[i]['X'], self.lb, self.ub)
                self.pop[i]['Velocity'][clip_mask] = 0
                tmp = problem.eval(self.pop[i]['X'])
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
                if problem.RecentChange:
                    return
                self.pop[i]['FitnessValue'] = tmp
                update_mask = self.pop[i]['FitnessValue'] > self.pop[i]['PbestValue']
                self.pop[i]['PbestValue'][update_mask] = tmp[update_mask]
                self.pop[i]['PbestPosition'][update_mask] = self.pop[i]['X'][update_mask]
                BestPbestValue = np.max(self.pop[i]['PbestValue'])
                BestPbestID = np.argmax(self.pop[i]['PbestValue'])
                if BestPbestValue > self.pop[i]['GbestValue']:
                    self.pop[i]['GbestValue'] = BestPbestValue
                    self.pop[i]['GbestPosition'] = self.pop[i]['PbestPosition'][BestPbestID]
        # Update swarm center
        for i in range(self.SwarmNumber):
            if not self.pop[i]['IsConverged']:
                self.pop[i]['Center'] = np.mean(self.pop[i]['PbestPosition'], axis=0)
        # Check overlapping and remove worst subpopulation
        valid_idx = [k for k, p in enumerate(self.pop) if p['X'].shape[0] > 0 and not p['IsConverged']]
        G = np.array([self.pop[k]['GbestPosition'] for k in valid_idx])
        dist_matrix = squareform(pdist(G))
        delete_idx = []
        for i in range(len(valid_idx)):
            for j in range(i+1, len(valid_idx)):
                idx_i = valid_idx[i]
                idx_j = valid_idx[j]
                if dist_matrix[i, j] < min(self.pop[idx_i]['InitRadius'], self.pop[idx_j]['InitRadius']):
                    if self.pop[idx_i]['GbestValue'] > self.pop[idx_j]['GbestValue']:
                        delete_idx.append(idx_j)
                    else:
                        delete_idx.append(idx_i)
        delete_idx = sorted(set(delete_idx), reverse=True)
        for idx in delete_idx:
            del self.pop[idx]
        self.SwarmNumber = len(self.pop)
        # Random Subpop Perturbation: 随机挑选一个子群对其Pbest重评估，并在速度上加入扰动
        idx = np.random.randint(0, self.SwarmNumber)
        self.pop[idx]['PbestValue'] = problem.eval(self.pop[idx]['PbestPosition'])
        if problem.avg_dist:
            self.avg_dist += problem.avg_dist
        BestPbestID = np.argmax(self.pop[idx]['PbestValue'])
        self.pop[idx]['GbestValue'] = self.pop[idx]['PbestValue'][BestPbestID]
        self.pop[idx]['GbestPosition'] = self.pop[idx]['PbestPosition'][BestPbestID]
        num_particles, dim = self.pop[idx]['X'].shape
        perturb = -self.PerturbationRange + 2 * self.PerturbationRange * np.random.rand(num_particles, dim)
        self.pop[idx]['Velocity'] += perturb
        # Update Current Radius
        for i in range(self.SwarmNumber):
            if not self.pop[i]['IsConverged']:
                self.pop[i]['CurrentRadius'] = np.mean(cdist(self.pop[i]['PbestPosition'], self.pop[i]['Center'].reshape(1, -1))[:, 0])
        # Convergence Detection and Deactivation
        AnyConverged = 0
        converged_list = []
        BestID = np.argmax([p['GbestValue'] for p in self.pop])
        for i in range(self.SwarmNumber):
            if self.pop[i]['CurrentRadius'] < self.ConvergenceLimit and i != BestID:
                self.pop[i]['IsConverged'] = True
                AnyConverged += 1
                converged_list.append(i)
        # Diversity Check and Mechanism
        SurvivedParticles = 0
        for i in range(self.SwarmNumber):
            if not self.pop[i]['IsConverged']:
                SurvivedParticles += self.pop[i]['X'].shape[0]
        SaveBestPosition = np.zeros((AnyConverged, self.dim))
        for i, idx in enumerate(converged_list):
            SaveBestPosition[i] = self.pop[idx]['GbestPosition']
        if SurvivedParticles < self.initPopulationSize * self.DiversityDegree:
            for i in reversed(converged_list):
                del self.pop[i]
        NumAddParticles = max(self.initPopulationSize - SurvivedParticles - len(converged_list), 0)
        AddParticles_X = self.lb + (self.ub - self.lb) * np.random.rand(NumAddParticles, self.dim)
        AddParticles_X = np.concatenate((AddParticles_X, SaveBestPosition), axis=0)
        if AddParticles_X.shape[0]:
            add_swarm = self.sub_population_generator(AddParticles_X, problem)
            if problem.RecentChange:
                return
            self.pop += add_swarm
            self.SwarmNumber = len(self.pop)

    def run_episode(self, problem):
        self.initialize_swarm(problem)
        while problem.fes < problem.maxfes:
            self.iterative_components(problem)
            if problem.RecentChange == 1:
                problem.reset_RecentChange()
                #print(f"Environment number:{problem.Environmentcounter}")

        gbest_list = []
        for i in range(self.SwarmNumber):
            gbest_list.append(self.pop[i]['GbestValue'])
        result = {'cost': gbest_list, 'fes': problem.fes, 'avg_dist': self.avg_dist}
        if hasattr(problem, 'CurrentError'):
            err = problem.CurrentError
            offlineerror = np.nanmean(err)
            result['current_error'] = offlineerror
        return result













