"""
Liu, Yuanchao, et al.
        "An affinity propagation clustering based particle swarm optimizer for dynamic optimization."
        Knowledge-Based Systems 195 (2020): 105711.
paper: https://www.sciencedirect.com/science/article/pii/S0950705120301362
source: https://github.com/EvoMindLab/EDOLAB/tree/main/Algorithm/APCPSO
最大化优化，如果做最小化的话eval值取反
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy
from sklearn.cluster import AffinityPropagation

class APCPSO(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.ps = 100
        self.omega_max = 0.6
        self.omega_min = 0.3
        self.c1 = 2.05
        self.c2 = 2.05
        self.MaxSubPopIterations = 50
        self.StagnationThreshold = 15
        self.ConvergenceThreshold = 0.1

        self.avg_dist = 0

    def __str__(self):
        return "APCPSO"

    def initialize_swarm(self, problem):
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        init_pop = self.lb + (self.ub - self.lb) * np.random.rand(self.ps, self.dim)
        self.pop = self.sub_population_generator(init_pop, problem)
        self.SwarmNumber = len(self.pop)
        self.ExclusionRadius = 0.5 * ((self.ub - self.lb) / (self.SwarmNumber ** (1 / self.dim)))

    def sub_population_generator(self, init_pop, problem):
        Subpops, Centers = self.AffinityPropagationClustering(init_pop)
        population = {
            'X': None,
            'Velocity': None,
            'FitnessValue': None,
            'PbestPosition': None,
            'IsConverged': None,
            'StagnationCounter': None,
            'IsStagnated': None,
            'PbestValue': None,
            'GbestValue': None,
            'GbestID': None,
            'GbestPosition': None,
            'Center': None,
            'InitRadius': None,
            'CurrentRadius': None,
            'IsExcluded': None
        }
        Swarm = [copy.deepcopy(population) for _ in range(len(Subpops))]
        for k in range(len(Subpops)):
            Swarm[k]['X'] = Subpops[k]
            Swarm[k]['Velocity'] = -5 + 10 * np.random.rand(Swarm[k]['X'].shape[0], self.dim)
            Swarm[k]['FitnessValue'] = problem.eval(Swarm[k]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            Swarm[k]['PbestPosition'] = Swarm[k]['X'].copy()
            Swarm[k]['PbestValue'] = Swarm[k]['FitnessValue'].copy()
            Swarm[k]['IsConverged'] = 0
            Swarm[k]['IsStagnated'] = 0
            Swarm[k]['IsExcluded'] = 0
            Swarm[k]['StagnationCounter'] = np.zeros(Swarm[k]['X'].shape[0])
            best_idx = np.argmax(Swarm[k]['PbestValue'])
            Swarm[k]['GbestPosition'] = Swarm[k]['PbestPosition'][best_idx].copy()
            Swarm[k]['GbestValue'] = Swarm[k]['PbestValue'][best_idx].copy()
            Swarm[k]['GbestID'] = best_idx
        return Swarm

    '''
    def AffinityPropagationClustering(self, pop):
        # Clustering Function (Modified for Affinity Propagation)
        lamb = 0.5
        max_iter = 200
        stable_iter = 50
        N = pop.shape[0]
        s = -cdist(pop, pop, metric='sqeuclidean')
        Med = np.median(s)
        for k in range(N):
            s[k, k] = Med
        r = np.zeros((N, N))
        a = np.zeros((N, N))
        iter = 0
        converged = False
        last_exemplars = []
        exemplar_indices = []
        stable_count = 0
        while iter < max_iter and not converged:
            iter += 1
            r_prev = r.copy()
            a_prev = a.copy()
            r_new = np.zeros((N, N))
            for i in range(N):
                as_i = a_prev[i, :] + s[i, :]
                for k in range(N):
                    if k == 1:
                        mask = np.arange(1, N)
                    elif k == N - 1:
                        mask = np.arange(0, N - 1)
                    else:
                        mask = np.concatenate((np.arange(0, k - 1), np.arange(k, N)))
                    if mask.size:
                        max_val = np.max(as_i[mask])
                    else:
                        max_val = -np.inf
                    r_new[i, k] = s[i, k] - max_val
            r = lamb * r_prev + (1 - lamb) * r_new
            a_new = np.zeros((N, N))
            sum_contrib = np.sum(np.maximum(0, r), axis=0) - np.maximum(0, np.diag(r))
            for k in range(N):
                sum_contrib_k = sum_contrib[k]
                r_kk = r[k, k]
                for i in range(N):
                    if i == k:
                        a_new[i, k] = sum_contrib_k
                    else:
                        temp = r_kk + sum_contrib_k - np.maximum(0, r[i, k])
                        a_new[i, k] = min(0, temp)
            a = lamb * a_prev + (1 - lamb) * a_new
            Su = a + r
            exemplar_indices = np.argmax(Su, axis=1)
            current_exemplars = np.unique(exemplar_indices)
            current_sorted = np.sort(current_exemplars)
            last_sorted = np.sort(last_exemplars)
            if np.array_equal(current_sorted, last_sorted):
                stable_count += 1
                if stable_count >= stable_iter:
                    converged = True
            else:
                stable_count = 0
                last_exemplars = current_exemplars
        ClusterID, CentersIdx = np.unique(exemplar_indices, return_inverse=True)
        SubPopulations = [pop[exemplar_indices == k, :] for k in np.unique(ClusterID)]
        Centers = pop[ClusterID, :]
        if not Centers.size:
            SubPopulations = [pop]
            Centers = np.mean(pop, axis=0)
        return SubPopulations, Centers
    '''

    def AffinityPropagationClustering(self, pop):
        ap = AffinityPropagation(affinity='euclidean', damping=0.5, max_iter=200, convergence_iter=50)
        ap.fit(pop)
        labels = ap.labels_
        unique_labels = np.unique(labels)
        SubPops = [pop[labels == k] for k in unique_labels]
        Centers = ap.cluster_centers_
        return SubPops, Centers

    def iterative_components(self, problem):
        # PSO Local Search Update for Each Sub-Population
        for i in range(self.SwarmNumber):
            if 'localIter' not in self.pop[i] or self.pop[i]['localIter'] is None:
                self.pop[i]['localIter'] = 1
            else:
                self.pop[i]['localIter'] += 1
            omega_current = self.omega_max - (self.omega_max - self.omega_min) * self.pop[i]['localIter'] / self.MaxSubPopIterations
            r1 = np.random.rand(self.pop[i]['X'].shape[0], self.dim)
            r2 = np.random.rand(self.pop[i]['X'].shape[0], self.dim)
            self.pop[i]['Velocity'] = omega_current * self.pop[i]['Velocity'] + self.c1 * r1 * (self.pop[i]['PbestPosition'] - self.pop[i]['X']) + self.c2 * r2 * (self.pop[i]['GbestPosition'] - self.pop[i]['X'])
            self.pop[i]['X'] += self.pop[i]['Velocity']
            mask = (self.pop[i]['X'] < self.lb) | (self.pop[i]['X'] > self.ub)
            self.pop[i]['X'] = np.clip(self.pop[i]['X'], self.lb, self.ub)
            self.pop[i]['Velocity'][mask] = 0
            fitness = problem.eval(self.pop[i]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            self.pop[i]['Fitness'] = fitness
            mask = self.pop[i]['Fitness'] > self.pop[i]['PbestValue']
            self.pop[i]['PbestValue'][mask] = self.pop[i]['Fitness'][mask].copy()
            self.pop[i]['PbestPosition'][mask] = self.pop[i]['X'][mask].copy()
            BestPbestID = np.argmax(self.pop[i]['PbestValue'])
            if self.pop[i]['PbestValue'][BestPbestID] > self.pop[i]['GbestValue']:
                self.pop[i]['GbestValue'] = self.pop[i]['PbestValue'][BestPbestID].copy()
                self.pop[i]['GbestPosition'] = self.pop[i]['PbestPosition'][BestPbestID].copy()
                self.pop[i]['GbestID'] = BestPbestID
        # Exclusion Mechanism: Prevent multiple sub-populations from exploring the same peak
        rex = (self.ub - self.lb) / (2 * self.SwarmNumber ** (1 / self.dim))
        for i in range(self.SwarmNumber - 1):
            for j in range(i + 1, self.SwarmNumber):
                if self.pop[i]['IsExcluded'] or self.pop[j]['IsExcluded']:
                    continue
                dist = np.linalg.norm(self.pop[i]['GbestPosition'] - self.pop[j]['GbestPosition'])
                if dist < rex:
                    if self.pop[i]['GbestValue'] > self.pop[j]['GbestValue']:
                        self.pop[i] = self.merge_SubPopulation(self.pop[i], self.pop[j])
                        self.pop[j]['IsExcluded'] = 1
                    else:
                        self.pop[j] = self.merge_SubPopulation(self.pop[i], self.pop[j])
                        self.pop[i]['IsExcluded'] = 1
        for i in range(self.SwarmNumber):
            if self.pop[i]['IsExcluded']:
                self.pop[i]['X'] = self.lb + (self.ub - self.lb) * np.random.rand(self.pop[i]['X'].shape[0], self.dim)
                self.pop[i]['Velocity'] = -5 + 10 * np.random.rand(self.pop[i]['X'].shape[0], self.dim)
                self.pop[i]['FitnessValue'] = problem.eval(self.pop[i]['X'])
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
                if problem.RecentChange:
                    return
                self.pop[i]['PbestPosition'] = self.pop[i]['X'].copy()
                self.pop[i]['PbestValue'] = self.pop[i]['FitnessValue'].copy()
                self.pop[i]['IsConverged'] = 0
                self.pop[i]['IsStagnated'] = 0
                self.pop[i]['IsExcluded'] = 0
                self.pop[i]['StagnationCounter'] = np.zeros(self.pop[i]['X'].shape[0])
                best_idx = np.argmax(self.pop[i]['PbestValue'])
                self.pop[i]['GbestPosition'] = self.pop[i]['PbestPosition'][best_idx].copy()
                self.pop[i]['GbestValue'] = self.pop[i]['PbestValue'][best_idx].copy()
                self.pop[i]['GbestID'] = best_idx
        # Convergence Detection: Check if sub-populations have converged
        for i in range(self.SwarmNumber):
            center = np.mean(self.pop[i]['PbestPosition'], axis=0)
            radius = np.mean(np.linalg.norm(self.pop[i]['PbestPosition'] - center, axis=1))
            if radius < self.ConvergenceThreshold:
                self.pop[i]['IsConverged'] = 1
            else:
                self.pop[i]['IsConverged'] = 0
        Gbest_fitness = np.array([p['GbestValue'] for p in self.pop[:self.SwarmNumber]])
        converge_count = np.sum([p['IsConverged'] for p in self.pop[:self.SwarmNumber]])
        if converge_count == self.SwarmNumber:
            worst_idx = np.argmin(Gbest_fitness)
            self.pop[worst_idx]['X'] = self.lb + (self.ub - self.lb) * np.random.rand(self.pop[worst_idx]['X'].shape[0], self.dim)
            self.pop[worst_idx]['Velocity'] = -5 + 10 * np.random.rand(self.pop[worst_idx]['X'].shape[0], self.dim)
            self.pop[worst_idx]['FitnessValue'] = problem.eval(self.pop[worst_idx]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            self.pop[worst_idx]['PbestPosition'] = self.pop[worst_idx]['X'].copy()
            self.pop[worst_idx]['PbestValue'] = self.pop[worst_idx]['FitnessValue'].copy()
            self.pop[worst_idx]['IsConverged'] = 0
            self.pop[worst_idx]['IsStagnated'] = 0
            self.pop[worst_idx]['IsExcluded'] = 0
            self.pop[worst_idx]['StagnationCounter'] = np.zeros(self.pop[worst_idx]['X'].shape[0])
            best_idx = np.argmax(self.pop[worst_idx]['PbestValue'])
            self.pop[worst_idx]['GbestPosition'] = self.pop[worst_idx]['PbestPosition'][best_idx].copy()
            self.pop[worst_idx]['GbestValue'] = self.pop[worst_idx]['PbestValue'][best_idx].copy()
            self.pop[worst_idx]['GbestID'] = best_idx
        if problem.RecentChange:
            return

    def merge_SubPopulation(self, pop1, pop2):
        mergedPop = {
            'X': np.concatenate((pop1['X'], pop2['X']), axis=0),
            'Velocity': np.concatenate((pop1['Velocity'], pop2['Velocity']), axis=0),
            'Shifts': None,
            'FitnessValue': np.concatenate((pop1['FitnessValue'], pop2['FitnessValue']), axis=0),
            'PbestPosition': np.concatenate((pop1['PbestPosition'], pop2['PbestPosition']), axis=0),
            'IsConverged': 0,
            'StagnationCounter': None,
            'IsStagnated': 0,
            'PbestValue': np.concatenate((pop1['PbestValue'], pop2['PbestValue']), axis=0),
            'GbestValue': None,
            'GbestID': None,
            'GbestPosition': None,
            'Center': None,
            'InitRadius': None,
            'CurrentRadius': None,
            'IsExcluded': 0,
            'localIter': max(pop1['localIter'], pop2['localIter'])
        }
        sortedIdx = np.argsort(mergedPop['FitnessValue'])[::-1]
        k = pop2['X'].shape[0]
        bestIdx = sortedIdx[:len(sortedIdx) - k]
        mergedPop['X'] = mergedPop['X'][bestIdx]
        mergedPop['FitnessValue'] = mergedPop['FitnessValue'][bestIdx]
        mergedPop['PbestValue'] = mergedPop['PbestValue'][bestIdx]
        mergedPop['PbestPosition'] = mergedPop['PbestPosition'][bestIdx]
        mergedPop['Velocity'] = mergedPop['Velocity'][bestIdx]
        mergedPop['StagnationCounter'] = np.zeros(mergedPop['X'].shape[0])
        pbest_idx = np.argmax(mergedPop['PbestValue'])
        mergedPop['GbestPosition'] = mergedPop['PbestPosition'][pbest_idx]
        mergedPop['GbestValue'] = mergedPop['PbestValue'][pbest_idx]
        mergedPop['GbestID'] = pbest_idx
        return mergedPop

    def change_reaction(self, problem):
        GbestValues_before = []
        for i in range(self.SwarmNumber):
            GbestValues_before.append(self.pop[i]['GbestValue'])
        SortIndex_before = np.argsort(GbestValues_before)[::-1]
        Swarm_Sort_before = []
        for i in range(self.SwarmNumber):
            Swarm_Sort_before.append(self.pop[SortIndex_before[i]])
        GbestValues = np.zeros(self.SwarmNumber)
        for i in range(self.SwarmNumber):
            self.pop[i]['GbestValue'] = problem.eval(self.pop[i]['GbestPosition']).item()
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            GbestValues[i] = self.pop[i]['GbestValue']
        SortIndex = np.argsort(GbestValues)[::-1]
        Swarm_Sort = []
        for i in range(self.SwarmNumber):
            Swarm_Sort.append(self.pop[SortIndex[i]])
        self.pop = Swarm_Sort
        # 1. Optimal Particle Calibration
        new_X = np.zeros((self.ps, self.dim))
        new_fitness = np.zeros(self.ps)
        for i in range(self.SwarmNumber):
            tryBestPositon = self.pop[i]['GbestPosition'].copy()
            tryBestValue = self.pop[i]['GbestValue']
            # Particle calibration: Adjust the historical best in each dimension
            for j in range(self.dim):
                tempTryPosition = self.pop[i]['GbestPosition'].copy()
                tempTryPosition[j] += np.random.rand()  # small increment
                tempTryPosition = np.clip(tempTryPosition, self.lb, self.ub)
                tempTryFitness = problem.eval(tempTryPosition)
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
                if tempTryFitness > tryBestValue:
                    tryBestPositon = tempTryPosition
                    tryBestValue = tempTryFitness.item()
                else:
                    tempTryPosition[j] -= np.random.rand()
                    tempTryPosition = np.clip(tempTryPosition, self.lb, self.ub)
                    tempTryFitness = problem.eval(tempTryPosition)
                    if problem.avg_dist:
                        self.avg_dist += problem.avg_dist
                    if tempTryFitness > tryBestValue:
                        tryBestPositon = tempTryPosition
                        tryBestValue = tempTryFitness.item()
            new_X[i, :] = tryBestPositon
            new_fitness[i] = tryBestValue
        # 2. Diversity Maintenance
        N = self.ps
        n = self.SwarmNumber
        new_X[n:, :] = self.lb + (self.ub - self.lb) * np.random.rand(N - n, self.dim)
        self.pop = self.sub_population_generator(new_X, problem)
        self.SwarmNumber = len(self.pop)
        self.ExclusionRadius = 0.5 * ((self.ub - self.lb) / (self.SwarmNumber ** (1 / self.dim)))

    def run_episode(self, problem):
        self.initialize_swarm(problem)
        while problem.fes < problem.maxfes:
            self.iterative_components(problem)
            if problem.RecentChange == 1:
                problem.reset_RecentChange()
                self.change_reaction(problem)
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

