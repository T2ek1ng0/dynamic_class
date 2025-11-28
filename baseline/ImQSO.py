"""
Javidan Kazemi Kordestani et al.,
        "A note on the exclusion operator in multi-swarm PSO algorithms for dynamic environments"
        Connection Science, pp. 1-25, 2019.
paper: https://www.tandfonline.com/doi/full/10.1080/09540091.2019.1700912
source: https://github.com/EvoMindLab/EDOLAB/blob/main/Algorithm/ImQSO
最大化优化，如果做最小化的话eval值取反
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy

class ImQSO(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.ps = 5
        self.w = 0.729843788
        self.c1 = 2.05
        self.c2 = 2.05
        self.QuantumNumber = 5
        self.SwarmNumber = 10
        self.alpha = 1
        self.avg_dist = 0

    def __str__(self):
        return "ImQSO"

    def initialize_swarm(self, problem):
        self.ub = problem.ub
        self.lb = problem.lb
        self.dim = problem.dim
        self.DiversityPlus = 1
        self.ShiftSeverity = 1
        self.QuantumRadius = 1
        self.ExclusionLimit = 0.5 * ((self.ub - self.lb) / (self.SwarmNumber ** (1 / self.dim)))
        self.ConvergenceLimit = self.ExclusionLimit
        self.pop = []
        for i in range(self.SwarmNumber):
            self.pop.append(self.sub_population_generator(problem))

    def sub_population_generator(self, problem):
        random_x = self.lb + (self.ub - self.lb) * np.random.rand(self.ps, self.dim)
        population = {
            'X': random_x.copy(),
            'Velocity': np.zeros((self.ps, self.dim)),
            'FitnessValue': problem.eval(random_x),
            'PbestPosition': random_x.copy(),
            'IsConverged': 0,
            'PbestValue': None,
            'BestValue': None,
            'GbestID': None,
            'BestPosition': None,
            'Gbest_past_environment': np.full(self.dim, np.nan),
            'Shifts': []
        }
        if problem.avg_dist:
            self.avg_dist += problem.avg_dist
        if not problem.RecentChange:
            population['PbestValue'] = population['FitnessValue'].copy()
            gbestidx = np.argmax(population['PbestValue'])
            population['BestPosition'] = population['PbestPosition'][gbestidx].copy()
            population['BestValue'] = population['FitnessValue'][gbestidx].copy()
            population['GbestID'] = gbestidx
        else:
            population['FitnessValue'] = np.full(self.ps, -np.inf)
            population['PbestValue'] = population['FitnessValue'].copy()
            gbestidx = np.argmax(population['PbestValue'])
            population['BestPosition'] = population['PbestPosition'][gbestidx].copy()
            population['BestValue'] = population['FitnessValue'][gbestidx].copy()
            population['GbestID'] = gbestidx
        return population

    def iterative_components(self, problem):
        # Sub-swarm movement
        for i in range(self.SwarmNumber):
            r1 = np.random.rand(self.ps, self.dim)
            r2 = np.random.rand(self.ps, self.dim)
            self.pop[i]['Velocity'] = self.w * (self.pop[i]['Velocity'] + (self.c1 * r1 * (self.pop[i]['PbestPosition'] - self.pop[i]['X'])) + (self.c2 * r2 * (self.pop[i]['BestPosition'] - self.pop[i]['X'])))
            self.pop[i]['X'] += self.pop[i]['Velocity']
            clip_mask = (self.pop[i]['X'] < self.lb) | (self.pop[i]['X'] > self.ub)
            self.pop[i]['X'] = np.clip(self.pop[i]['X'], self.lb, self.ub)
            self.pop[i]['Velocity'][clip_mask] = 0
            tmp = problem.eval(self.pop[i]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            self.pop[i]['FitnessValue'] = tmp
            update_mask = self.pop[i]['FitnessValue'] > self.pop[i]['PbestValue']
            self.pop[i]['PbestValue'][update_mask] = self.pop[i]['FitnessValue'][update_mask].copy()
            self.pop[i]['PbestPosition'][update_mask] = self.pop[i]['X'][update_mask].copy()
            bestpbestid = np.argmax(self.pop[i]['PbestValue'])
            if self.pop[i]['PbestValue'][bestpbestid] > self.pop[i]['BestValue']:
                self.pop[i]['BestPosition'] = self.pop[i]['PbestPosition'][bestpbestid].copy()
                self.pop[i]['BestValue'] = self.pop[i]['PbestValue'][bestpbestid].copy()
            for j in range(self.QuantumNumber):
                QuantumPosition = self.pop[i]['BestPosition'] + (2 * np.random.rand(self.dim) - 1) * self.QuantumRadius
                QuantumFitnessValue = problem.eval(QuantumPosition)
                if problem.RecentChange:
                    return
                if QuantumFitnessValue > self.pop[i]['BestValue']:
                    self.pop[i]['BestValue'] = QuantumFitnessValue
                    self.pop[i]['BestPosition'] = QuantumPosition.copy()
        # Exclusion
        for i in range(self.SwarmNumber - 1):
            for j in range(i + 1, self.SwarmNumber):
                if np.linalg.norm(self.pop[i]['BestPosition'] - self.pop[j]['BestPosition']) < self.ExclusionLimit:
                    Exclusion_Probability = ((self.ExclusionLimit - np.linalg.norm(self.pop[i]['BestPosition'] - self.pop[j]['BestPosition'])) / self.ExclusionLimit) ** self.alpha
                    r = np.random.rand()
                    if r < Exclusion_Probability:
                        if self.pop[i]['BestValue'] < self.pop[j]['BestValue']:
                            self.pop[i] = self.sub_population_generator(problem)
                            if problem.RecentChange:
                                return
                        else:
                            self.pop[j] = self.sub_population_generator(problem)
                            if problem.RecentChange:
                                return
        # Anti Convergence
        IsAllConverged = 0
        WorstSwarmValue = np.inf
        WorstSwarmIndex = None
        for i in range(self.SwarmNumber):
            Radius = 0
            for j in range(self.ps):
                for k in range(self.ps):
                    Radius = max(Radius,np.max(np.abs(self.pop[i]['X'][j, :] - self.pop[i]['X'][k, :])))
            if Radius < self.ConvergenceLimit:
                self.pop[i]['IsConverged'] = 1
            else:
                self.pop[i]['IsConverged'] = 0
            IsAllConverged += self.pop[i]['IsConverged']
            if self.pop[i]['BestValue'] < WorstSwarmValue:
                WorstSwarmValue = self.pop[i]['BestValue']
                WorstSwarmIndex = i
        if IsAllConverged == self.SwarmNumber:
            self.pop[WorstSwarmIndex] = self.sub_population_generator(problem)
            if problem.RecentChange:
                return

    def change_reaction(self, problem):
        # Updating Shift Severity
        dummy = np.full((self.SwarmNumber, problem.EnvironmentNumber), np.nan)
        for j in range(self.SwarmNumber):
            if np.all(~np.isnan(self.pop[j]['Gbest_past_environment'])):
                self.pop[j]['Shifts'].append(np.linalg.norm(self.pop[j]['Gbest_past_environment'] - self.pop[j]['BestPosition']))
            dummy[j, : len(self.pop[j]['Shifts'])] = self.pop[j]['Shifts'].copy()
        dummy = dummy[~np.isnan(dummy)].flatten()
        if dummy.size:
            self.ShiftSeverity = np.mean(dummy)
        self.QuantumRadius = self.ShiftSeverity
        # Updating memory
        for j in range(self.SwarmNumber):
            self.pop[j]['PbestValue'] = problem.eval(self.pop[j]['PbestPosition'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            self.pop[j]['Gbest_past_environment'] = self.pop[j]['BestPosition'].copy()
            bestpbestid = np.argmax(self.pop[j]['PbestValue'])
            self.pop[j]['BestValue'] = self.pop[j]['PbestValue'][bestpbestid].copy()
            self.pop[j]['BestPosition'] = self.pop[j]['PbestPosition'][bestpbestid].copy()
            self.pop[j]['IsConverged'] = 0

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
            gbest_list.append(self.pop[i]['BestValue'])
        result = {'cost': gbest_list, 'fes': problem.fes, 'avg_dist': self.avg_dist}
        if hasattr(problem, 'CurrentError'):
            err = problem.CurrentError
            offlineerror = np.nanmean(err)
            result['current_error'] = offlineerror
        return result

