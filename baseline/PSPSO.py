"""
Signorelli, Federico, and Anil Yaman.
        "A Perturbation and Speciation-Based Algorithm for Dynamic Optimization Uninformed of Change."
        arXiv:2505.11634 (2025).
paper: https://arxiv.org/abs/2505.11634
source: https://github.com/FreddyDeWatersir/PSPSO
最大化优化，如果做最小化的话eval值取反
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
                self.pop[i]['PbestPosition'][update_mask] = self.pop[i]['X'][update_mask].copy()
                BestPbestValue = np.max(self.pop[i]['PbestValue'])
                BestPbestID = np.argmax(self.pop[i]['PbestValue'])
                if BestPbestValue > self.pop[i]['GbestValue']:
                    self.pop[i]['GbestValue'] = BestPbestValue
                    self.pop[i]['GbestPosition'] = self.pop[i]['PbestPosition'][BestPbestID].copy()
        # Update swarm center
        for i in range(self.SwarmNumber):
            if not self.pop[i]['IsConverged']:
                self.pop[i]['Center'] = np.mean(self.pop[i]['PbestPosition'], axis=0)
        # Check overlapping and remove worst subpopulation
        idx = np.inf
        while idx != -1:
            idx = -1
            i = 0
            while i < self.SwarmNumber:
                if self.pop[i]['X'].shape[0] == 0 or self.pop[i]['IsConverged']:
                    i += 1
                    continue
                j = i + 1
                while j < self.SwarmNumber:
                    if self.pop[j]['X'].shape[0] == 0 or self.pop[j]['IsConverged']:
                        j += 1
                        continue
                    dist = np.linalg.norm(self.pop[i]['GbestPosition'] - self.pop[j]['GbestPosition'])
                    if dist < self.pop[i]['InitRadius'] and dist < self.pop[j]['InitRadius']:
                        if self.pop[i]['GbestValue'] > self.pop[j]['GbestValue']:
                            self.pop.pop(j)
                        else:
                            self.pop.pop(i)
                        self.SwarmNumber -= 1
                        idx = i
                        break
                    j += 1
                if idx != -1:
                    break
                i += 1
        # Random Subpop Perturbation: 随机挑选一个子群对其Pbest重评估，并在速度上加入扰动
        idx = np.random.randint(0, self.SwarmNumber)
        self.pop[idx]['PbestValue'] = problem.eval(self.pop[idx]['PbestPosition'])
        if problem.avg_dist:
            self.avg_dist += problem.avg_dist
        BestPbestID = np.argmax(self.pop[idx]['PbestValue'])
        self.pop[idx]['GbestValue'] = self.pop[idx]['PbestValue'][BestPbestID]
        self.pop[idx]['GbestPosition'] = self.pop[idx]['PbestPosition'][BestPbestID].copy()
        num_particles, dim = self.pop[idx]['X'].shape
        perturb = -self.PerturbationRange + 2 * self.PerturbationRange * np.random.rand(num_particles, dim)
        self.pop[idx]['Velocity'] += perturb
        # Update Current Radius
        for i in range(self.SwarmNumber):
            if not self.pop[i]['IsConverged']:
                self.pop[i]['CurrentRadius'] = np.mean(cdist(self.pop[i]['PbestPosition'], self.pop[i]['Center'].reshape(1, -1))[:, 0])
        # Convergence Detection and Deactivation
        AnyConverged = 0
        BestID = np.argmax([p['GbestValue'] for p in self.pop])
        for i in range(self.SwarmNumber):
            if self.pop[i]['CurrentRadius'] < self.ConvergenceLimit and i != BestID:
                self.pop[i]['IsConverged'] = True
                AnyConverged += 1
        # Diversity Check and Mechanism
        SurvivedParticles = 0
        for i in range(self.SwarmNumber):
            if not self.pop[i]['IsConverged']:
                SurvivedParticles += self.pop[i]['X'].shape[0]
        SaveBestPosition = []
        count = 0
        if SurvivedParticles < self.initPopulationSize * self.DiversityDegree:
            while AnyConverged:
                i = 0
                while i < self.SwarmNumber:
                    if self.pop[i]['IsConverged']:
                        SaveBestPosition.append(self.pop[i]['GbestPosition'].copy())
                        count += 1
                        self.pop.pop(i)
                        self.SwarmNumber -= 1
                        break
                    i += 1
                AnyConverged -= 1
            SaveBestPosition = np.array(SaveBestPosition)
            if SaveBestPosition.ndim == 1 and SaveBestPosition.size == 0:
                SaveBestPosition = np.empty((0, self.dim))
            elif SaveBestPosition.ndim == 1 and SaveBestPosition.size > 0:
                SaveBestPosition = SaveBestPosition.reshape(-1, self.dim)
            NumAddParticles = max(self.initPopulationSize - SurvivedParticles - count, 0)
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













