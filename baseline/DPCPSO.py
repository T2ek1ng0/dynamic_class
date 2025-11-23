"""
Li, Fei, et al.
       "A fast density peak clustering based particle swarm optimizer for dynamic optimization."
       Expert Systems with Applications 236 (2024): 121254.
paper: https://www.sciencedirect.com/science/article/pii/S0957417423017566
source: https://github.com/EvoMindLab/EDOLAB/tree/main/Algorithm/DPCPSO
GMPB是最大化优化，如果做最小化优化的话有几个argmax和>要改
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy

class DPCPSO(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.ps = 100
        self.omega_max = 0.5
        self.omega_min = 0.2
        self.c1 = 2.05
        self.c2 = 2.05
        self.MaxSubPopIterations = 50
        self.StagnationThreshold = 15
        self.ConvergenceThreshold = 0.1
        self.QuantumNumber = 5

        self.avg_dist= 0

    def __str__(self):
        return "DPCPSO"

    def initialize_swarm(self, problem):
        self.ub = problem.ub
        self.lb = problem.lb
        self.dim = problem.dim
        init_swarm = self.lb + (self.ub - self.lb) * np.random.rand(self.ps, self.dim)
        self.pop = self.sub_population_generator(init_swarm, problem)
        self.SwarmNumber = len(self.pop)
        self.ExclusionRadius = 0.5 * ((self.ub - self.lb) / (self.SwarmNumber ** (1 / self.dim)))



    def DensityPeakClustering(self, population, PeakNumber=10):
        # Stage 1: Compute density and distance
        N = population.shape[0]
        dist = squareform(pdist(population, metric='sqeuclidean'))
        np.fill_diagonal(dist, np.inf)
        # Calculate dc based on percentiles of distances
        sda = np.sort(dist.ravel())
        percent = 2
        position = int(round(sda.size * percent / 100)) - 1
        dc = sda[position]
        # Gaussian kernel density calculation
        rho_matrix = np.exp(-(dist / dc) ** 2)
        np.fill_diagonal(rho_matrix, 0)
        rho = rho_matrix.sum(axis=1)
        # Compute relative distance delta
        maxd = np.max(dist[np.isfinite(dist)])
        ordrho = np.argsort(rho)[::-1]
        rho_sorted = rho[ordrho]
        # Initialize delta and nneigh arrays
        delta = np.full(N, maxd)
        nneigh = np.zeros(N, dtype=int)
        for i in range(1, N):
            for j in range(i):
                if dist[ordrho[i], ordrho[j]] < delta[ordrho[i]]:
                    delta[ordrho[i]] = dist[ordrho[i], ordrho[j]]
                    nneigh[ordrho[i]] = ordrho[j]
        delta[ordrho[0]] = np.max(delta)  # Assign delta value to the first (highest density) point
        # Compute gamma (density * relative distance)
        gamma = rho * delta
        gamma_idx = np.argsort(gamma)[::-1]
        sort_gamma = gamma[gamma_idx]
        gamma_threshold = -6.077 * N * np.log(PeakNumber) + 50.49 * N
        CenterIdx = []
        for i in range(len(sort_gamma)):
            if sort_gamma[i] > gamma_threshold:
                CenterIdx.append(gamma_idx[i])
            if len(CenterIdx) >= 30:
                break
        # Stage 3: Assign samples to the nearest centers
        if len(CenterIdx) == 0:
            SubPopulations = [population]
            Centers = np.mean(population, axis=0, keepdims=True)
            return SubPopulations, Centers
        Centers = population[CenterIdx, :]
        distances = cdist(population, Centers)
        clusterIndices = np.argmin(distances, axis=1)
        numClusters = len(CenterIdx)
        SubPopulations = []
        for i in range(numClusters):
            idx = np.where(clusterIndices == i)[0]
            SubPopulations.append(population[idx, :])
        validClusters = [i for i, s in enumerate(SubPopulations) if s.shape[0] >= 3]
        SubPopulations = [SubPopulations[i] for i in validClusters]
        Centers = Centers[validClusters, :]
        return SubPopulations, Centers

    def sub_population_generator(self, init_swarm, problem):
        SubPops, Centers = self.DensityPeakClustering(init_swarm, problem.PeakNumber)
        population = {
            'X': None,
            'Velocity': None,
            'Shifts': None,
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
        Swarm = [copy.deepcopy(population) for _ in range(len(SubPops))]
        for i in range(len(SubPops)):
            current_pop = SubPops[i]
            current_fitness = None
            if current_pop.shape[0] > 5:
                current_fitness = problem.eval(current_pop)
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
                sorted_indices = np.argsort(current_fitness)[::-1]
                selected_indices = sorted_indices[:5]
                current_pop = current_pop[selected_indices, :]
                current_fitness = problem.eval(current_pop)
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist

            Swarm[i]['X'] = current_pop.copy()
            Swarm[i]['Velocity'] = -4 + 8 * np.random.rand(current_pop.shape[0], self.dim)
            if current_fitness is not None:
                Swarm[i]['FitnessValue'] = current_fitness
                current_fitness = None
            else:
                Swarm[i]['FitnessValue'] = problem.eval(Swarm[i]['X'])
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
            Swarm[i]['PbestPosition'] = current_pop.copy()
            Swarm[i]['PbestValue'] = Swarm[i]['FitnessValue'].copy()
            Swarm[i]['IsConverged'] = 0
            Swarm[i]['IsStagnated'] = 0
            Swarm[i]['IsExcluded'] = 0
            Swarm[i]['localIter'] = 0
            Swarm[i]['StagnationCounter'] = np.zeros(current_pop.shape[0])
            best_idx = np.argmax(Swarm[i]['PbestValue'])
            Swarm[i]['GbestID'] = best_idx
            Swarm[i]['GbestPosition'] = Swarm[i]['PbestPosition'][best_idx, :]
            Swarm[i]['GbestValue'] = Swarm[i]['PbestValue'][best_idx]
        return Swarm

    def iterative_components(self, problem):
        # Iterative optimization process for DPCPSO including PSO local search, stagnation detection, exclusion, and convergence detection.
        # PSO Local Search Update for Each Sub-Population
        for i in range(self.SwarmNumber):
            # Update the local iteration counter for this sub-population
            if 'localIter' not in self.pop[i] or self.pop[i]['localIter'] is None:
                self.pop[i]['localIter'] = 1
            else:
                self.pop[i]['localIter'] += 1
            omega_current = self.omega_max - (self.omega_max - self.omega_min) * (self.pop[i]['localIter'] / self.MaxSubPopIterations)
            num_particles = self.pop[i]['X'].shape[0]
            r1 = np.random.rand(num_particles, self.dim)
            r2 = np.random.rand(num_particles, self.dim)
            self.pop[i]['Velocity'] = omega_current * self.pop[i]['Velocity'] + self.c1 * r1 * (self.pop[i]['PbestPosition'] - self.pop[i]['X']) + self.c2 * r2 * (self.pop[i]['GbestPosition'] - self.pop[i]['X'])
            self.pop[i]['X'] += self.pop[i]['Velocity']
            clip_mask = (self.pop[i]['X'] > self.ub) | (self.pop[i]['X'] < self.lb)
            self.pop[i]['X'] = np.clip(self.pop[i]['X'], self.lb, self.ub)
            self.pop[i]['Velocity'][clip_mask] = 0
            fitness = problem.eval(self.pop[i]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            self.pop[i]['FitnessValue'] = fitness.copy()
            update_mask = self.pop[i]['FitnessValue'] > self.pop[i]['PbestValue']
            self.pop[i]['PbestValue'][update_mask] = fitness[update_mask].copy()
            self.pop[i]['PbestPosition'][update_mask] = self.pop[i]['X'][update_mask].copy()
            self.pop[i]['StagnationCounter'][update_mask] = 0
            self.pop[i]['StagnationCounter'][~update_mask] += 1
            BestPbestID = np.argmax(self.pop[i]['PbestValue'])
            if self.pop[i]['PbestValue'][BestPbestID] > self.pop[i]['GbestValue']:
                self.pop[i]['GbestValue'] = self.pop[i]['PbestValue'][BestPbestID]
                self.pop[i]['GbestPosition'] = self.pop[i]['PbestPosition'][BestPbestID]
                self.pop[i]['GbestID'] = BestPbestID
            reintialize_mask = self.pop[i]['StagnationCounter'] >= self.StagnationThreshold
            num_reinit = reintialize_mask.sum()
            self.pop[i]['X'][reintialize_mask] = self.lb + (self.ub - self.lb) * np.random.rand(num_reinit, self.dim)
            self.pop[i]['Velocity'][reintialize_mask] = -4 + 8 * np.random.rand(num_reinit, self.dim)
            self.pop[i]['StagnationCounter'][reintialize_mask] = 0
            self.pop[i]['FitnessValue'][reintialize_mask] = problem.eval(self.pop[i]['X'][reintialize_mask])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            self.pop[i]['PbestValue'][reintialize_mask] = self.pop[i]['FitnessValue'][reintialize_mask].copy()
            self.pop[i]['PbestPosition'][reintialize_mask] = self.pop[i]['X'][reintialize_mask].copy()
        # Exclusion Mechanism: Prevent multiple sub-populations from exploring the same peak
        toExclude = []
        for i in range(self.SwarmNumber - 1):
            for j in range(i + 1, self.SwarmNumber):
                distance = np.linalg.norm(self.pop[i]['GbestPosition'] - self.pop[j]['GbestPosition'])
                if distance < self.ExclusionRadius:
                    if self.pop[i]['GbestValue'] > self.pop[j]['GbestValue']:
                        self.pop[i] = self.merge_SubPopulation(self.pop[i], self.pop[j])
                        self.pop[j]['IsExcluded'] = 1
                        toExclude.append(j)
                    else:
                        self.pop[j] = self.merge_SubPopulation(self.pop[i], self.pop[j])
                        self.pop[i]['IsExcluded'] = 1
                        toExclude.append(i)
        for i in range(self.SwarmNumber):
            if self.pop[i]['IsExcluded'] == 1:
                self.pop[i]['X'] = self.lb + (self.ub - self.lb) * np.random.rand(self.pop[i]['X'].shape[0], self.dim)
                self.pop[i]['Velocity'] = -4 + 8 * np.random.rand(self.pop[i]['X'].shape[0], self.dim)
                self.pop[i]['Shifts'] = None
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
                self.pop[i]['localIter'] = 0
                self.pop[i]['StagnationCounter'] = np.zeros(self.pop[i]['X'].shape[0])
                best_idx = np.argmax(self.pop[i]['PbestValue'])
                self.pop[i]['GbestValue'] = self.pop[i]['PbestValue'][best_idx]
                self.pop[i]['GbestPosition'] = self.pop[i]['PbestPosition'][best_idx].copy()
                self.pop[i]['GbestID'] = best_idx
        # Convergence Detection: Check if sub-populations have converged
        for i in range(self.SwarmNumber):
            center = np.mean(self.pop[i]['PbestPosition'], axis=0)
            distances = np.linalg.norm(self.pop[i]['PbestPosition'] - center, axis=1)
            radius = np.mean(distances)
            if radius < self.ConvergenceThreshold:
                self.pop[i]['IsConverged'] = 1
            else:
                self.pop[i]['IsConverged'] = 0
        converge_count = 0
        Gbest_fitness = np.zeros(self.SwarmNumber)
        for i in range(self.SwarmNumber):
            Gbest_fitness[i] = self.pop[i]['GbestValue']
            if self.pop[i]['IsConverged'] == 1:
                converge_count += 1
        if converge_count == self.SwarmNumber:
            worst_idx = np.argmin(Gbest_fitness)
            self.pop[worst_idx]['X'] = self.lb + (self.ub - self.lb) * np.random.rand(self.pop[worst_idx]['X'].shape[0], self.dim)
            self.pop[worst_idx]['Velocity'] = -4 + 8 * np.random.rand(self.pop[worst_idx]['X'].shape[0], self.dim)
            self.pop[worst_idx]['Shifts'] = None
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
            self.pop[worst_idx]['localIter'] = 0
            self.pop[worst_idx]['StagnationCounter'] = np.zeros(self.pop[worst_idx]['X'].shape[0])
            best_idx = np.argmax(self.pop[worst_idx]['PbestValue'])
            self.pop[worst_idx]['GbestValue'] = self.pop[worst_idx]['PbestValue'][best_idx]
            self.pop[worst_idx]['GbestPosition'] = self.pop[worst_idx]['PbestPosition'][best_idx].copy()
            self.pop[worst_idx]['GbestID'] = best_idx

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
            'localIter': min(pop1['localIter'], pop2['localIter'])
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
        GbestValues_before = [p['GbestValue'] for p in self.pop]
        SortIndex_before = np.argsort(GbestValues_before)[::-1]
        Swarm_Sort_before = [self.pop[i] for i in SortIndex_before]
        self.pop = Swarm_Sort_before
        GbestValues = np.zeros(self.SwarmNumber)
        for i in range(self.SwarmNumber):
            self.pop[i]['GbestValue'] = problem.eval(self.pop[i]['GbestPosition'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            GbestValues[i] = self.pop[i]['GbestValue']
        SortIndex = np.argsort(GbestValues)[::-1]
        Swarm_Sort = [self.pop[i] for i in SortIndex]
        self.pop = Swarm_Sort
        # 1. Optimal Particle Calibration
        new_X = np.zeros((self.ps, self.dim))
        new_fitness = np.zeros(self.ps)
        for i in range(self.SwarmNumber):
            tryBestPosition = self.pop[i]['GbestPosition'].copy()
            tryBestValue = self.pop[i]['GbestValue']
            # Particle calibration: Adjust the historical best in each dimension
            for j in range(self.dim):
                temptryPosition = self.pop[i]['GbestPosition'].copy()
                temptryPosition[j] += 0.0001
                temptryPosition = np.clip(temptryPosition, self.lb, self.ub)
                temptryFitness = problem.eval(temptryPosition)
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
                if temptryFitness > self.pop[i]['GbestValue']:
                    a = 1  # Adjust in the positive direction
                else:
                    a = -1  # Adjust in the negative direction
                tryPosition = np.tile(self.pop[i]['GbestPosition'].copy(), (10, 1))
                try_idx = np.arange(10)
                tryPosition[try_idx, j] += a * 0.1 * (try_idx + 1)
                tryPosition = np.clip(tryPosition, self.lb, self.ub)
                alltryFitness = problem.eval(tryPosition)
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
                bestIdx = np.argmax(alltryFitness)
                if alltryFitness[bestIdx] > tryBestValue:
                    tryBestPosition = tryPosition[bestIdx, :]
                    tryBestValue = alltryFitness[bestIdx]
            new_X[i, :] = tryBestPosition
            new_fitness[i] = tryBestValue
        # 2. Diversity Maintenance
        # Re-initialize N - n particles randomly to maintain diversity
        N = self.ps
        n = self.SwarmNumber
        new_X[n:, :] = self.lb + (self.ub - self.lb) * np.random.rand(N - n, self.dim)
        Swarm = self.sub_population_generator(new_X, problem)
        self.pop = Swarm
        self.SwarmNumber = len(self.pop)
        self.ExclusionRadius = 0.5 * ((self.ub - self.lb) / (self.SwarmNumber ** (1 / self.dim)))

    def run_episode(self, problem):
        self.initialize_swarm(problem)
        while problem.fes < problem.maxfes:
            self.iterative_components(problem)
            if problem.RecentChange == 1:
                problem.reset_RecentChange()
                self.change_reaction(problem)
                #print(f"Environment number: {problem.Environmentcounter}")

        gbest_list = []
        for i in range(self.SwarmNumber):
            gbest_list.append(self.pop[i]['GbestValue'])
        result = {'cost': gbest_list, 'fes': problem.fes, 'avg_dist': self.avg_dist}
        return result

















