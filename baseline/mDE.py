"""
Danial Yazdani et al.,
        "Scaling up dynamic optimization problems: A divide-and-conquer approach"
        IEEE Transactions on Evolutionary Computation, vol. 24(1), pp. 1 - 15, 2019.
paper: https://ieeexplore.ieee.org/document/8657680
source: https://github.com/EvoMindLab/EDOLAB/blob/main/Algorithm/mDE
GMPB是最大化优化，如果做最小化优化的话给eval值加负号?
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy

class mDE(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.ps = 5
        self.DiversityPlus = 1
        self.ShiftSeverity = 1
        self.SwarmNumber = 1
        self.FreeSwarmID = 0
        self.avg_dist = 0

    def __str__(self):
        return "mDE"

    def initialize_swarm(self, problem):
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.ExclusionLimit = 0.5 * ((self.ub - self.lb) / (self.SwarmNumber ** (1 / self.dim)))
        self.ConvergenceLimit = self.ExclusionLimit
        self.pop = []
        for i in range(self.SwarmNumber):
            self.pop.append(self.sub_population_generator(problem))

    def sub_population_generator(self, problem):
        random_x = self.lb + (self.ub - self.lb) * np.random.rand(self.ps, self.dim)
        pop = {
            'X': random_x.copy(),
            'Shifts': [],
            'FitnessValue': problem.eval(random_x),
            'BestPosition': None,
            'BestFitness': None,
            'BestID': None,
            'Gbest_past_environment': np.full((self.dim,), np.nan),
            'OffspringPosition': np.full((self.ps, self.dim), np.nan),
            'OffspringFitness': np.full(self.ps, np.nan),
            'Donor': np.full((self.ps, self.dim), np.nan),
            'Cr': np.ones(self.ps) * 0.9,
            'F': np.ones(self.ps) * 0.5,
            'F_lb': 0.1,
            'F_ub': 0.9,
            'r1': 0.1,
            'r2': 0.1
        }
        if problem.avg_dist:
            self.avg_dist += problem.avg_dist
        if not problem.RecentChange:
            bestid = np.argmax(pop['FitnessValue'])
            pop['BestPosition'] = random_x[bestid].copy()
            pop['BestFitness'] = pop['FitnessValue'][bestid].copy()
            pop['BestID'] = bestid
        else:
            pop['FitnessValue'] = np.full(self.ps, -np.inf)
            bestid = np.argmax(pop['FitnessValue'])
            pop['BestPosition'] = random_x[bestid].copy()
            pop['BestFitness'] = pop['FitnessValue'][bestid].copy()
            pop['BestID'] = bestid
        return pop

    def iterative_components(self, problem):
        TmpSwarmNum = self.SwarmNumber
        # Sub-population movement
        for i in range(self.SwarmNumber):
            self.pop[i]['BestID'] = np.argmax(self.pop[i]['FitnessValue'])
            self.pop[i]['BestPosition'] = self.pop[i]['X'][self.pop[i]['BestID']].copy()
            self.pop[i]['BestFitness'] = self.pop[i]['FitnessValue'][self.pop[i]['BestID']].copy()
            self.pop[i]['F'] = np.random.rand(self.ps)
            self.pop[i]['Cr'] = np.random.rand(self.ps)
            # Mutation
            for j in range(self.ps):
                R = np.random.permutation(self.ps)
                R = R[R != j]
                self.pop[i]['Donor'][j, :] = self.pop[i]['BestPosition'] + self.pop[i]['F'][j] * (self.pop[i]['X'][R[0]] + self.pop[i]['X'][R[1]] - self.pop[i]['X'][R[2]] - self.pop[i]['X'][R[3]])
            # Crossover==>binomial
            self.pop[i]['OffspringPosition'] = self.pop[i]['X'].copy()
            rows = np.arange(self.ps)
            cols = np.random.randint(self.dim, size=self.ps)
            self.pop[i]['OffspringPosition'][rows, cols] = self.pop[i]['Donor'][rows, cols].copy()
            CrossoverBinomial = np.random.rand(self.ps, self.dim) < self.pop[i]['Cr'].reshape(self.ps, -1)
            self.pop[i]['OffspringPosition'][CrossoverBinomial] = self.pop[i]['Donor'][CrossoverBinomial].copy()
            lb_tmp1 = self.pop[i]['OffspringPosition'] < self.lb
            lb_tmp2 = ((self.lb + self.pop[i]['X']) * lb_tmp1) / 2
            self.pop[i]['OffspringPosition'][lb_tmp1] = lb_tmp2[lb_tmp1].copy()
            ub_tmp1 = self.pop[i]['OffspringPosition'] > self.ub
            ub_tmp2 = ((self.ub + self.pop[i]['X']) * ub_tmp1) / 2
            self.pop[i]['OffspringPosition'][ub_tmp1] = ub_tmp2[ub_tmp1].copy()
            tmp = problem.eval(self.pop[i]['OffspringPosition'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            self.pop[i]['OffspringFitness'] = tmp.copy()
            # Selection==>greedy
            better = self.pop[i]['OffspringFitness'] > self.pop[i]['FitnessValue']
            self.pop[i]['X'][better, :] = self.pop[i]['OffspringPosition'][better].copy()
            self.pop[i]['FitnessValue'][better] = self.pop[i]['OffspringFitness'][better].copy()
            bestid = np.argmax(self.pop[i]['FitnessValue'])
            self.pop[i]['BestPosition'] = self.pop[i]['X'][bestid].copy()
            self.pop[i]['BestFitness'] = self.pop[i]['FitnessValue'][bestid].copy()
            self.pop[i]['BestID'] = bestid
        # Exclusion
        if self.SwarmNumber > 1:
            Removelist = np.zeros(self.SwarmNumber)
            for i in range(self.SwarmNumber - 1):
                for j in range(i + 1, self.SwarmNumber):
                   if np.linalg.norm(self.pop[i]['BestPosition'] - self.pop[j]['BestPosition']) < self.ExclusionLimit:
                        if self.pop[i]['BestFitness'] < self.pop[j]['BestFitness']:
                            if self.FreeSwarmID != i:
                                Removelist[i] = 1
                            else:
                                self.pop[i] = self.sub_population_generator(problem)
                                if problem.RecentChange:
                                    return
                        else:
                            if self.FreeSwarmID != j:
                                Removelist[j] = 1
                            else:
                                self.pop[j] = self.sub_population_generator(problem)
                                if problem.RecentChange:
                                    return
            for k in reversed(range(self.SwarmNumber)):
                if Removelist[k]:
                    self.pop.pop(k)
                    self.SwarmNumber -= 1
                    if k < self.FreeSwarmID:
                        self.FreeSwarmID -= 1
        # FreeSwarm Convergence
        dist = cdist(self.pop[self.FreeSwarmID]['X'], self.pop[self.FreeSwarmID]['X']) > self.ConvergenceLimit
        if dist.sum() == 0:
            self.SwarmNumber += 1
            self.FreeSwarmID = self.SwarmNumber - 1
            self.pop.append(self.sub_population_generator(problem))
            if problem.RecentChange:
                return
        # Updating Thresholds
        if TmpSwarmNum != self.SwarmNumber:
            self.ExclusionLimit = 0.5 * ((self.ub - self.lb) / (self.SwarmNumber ** (1 / self.dim)))
            self.ConvergenceLimit = self.ExclusionLimit

    def change_reaction(self, problem):
        # Updating Shift Severity
        dummy = np.full((self.SwarmNumber, problem.EnvironmentNumber), np.nan)
        for j in range(self.SwarmNumber):
            if j != self.FreeSwarmID:
                if np.all(~np.isnan(self.pop[j]['Gbest_past_environment'])):
                    self.pop[j]['Shifts'].append(np.linalg.norm(self.pop[j]['Gbest_past_environment'] - self.pop[j]['BestPosition']))
                dummy[j, :len(self.pop[j]['Shifts'])] = self.pop[j]['Shifts']
        dummy = dummy[~np.isnan(dummy)].flatten()
        if dummy.size:
            self.ShiftSeverity = np.mean(dummy)
        # Introducing diversity (all except free swarm)
        for j in range(self.SwarmNumber):
            if j != self.FreeSwarmID:
                self.pop[j]['X'] = np.tile(self.pop[j]['BestPosition'], (self.ps, 1))
                self.pop[j]['X'] += self.ShiftSeverity * (2 * np.random.rand(self.ps, self.dim) - 1)
                self.pop[j]['X'][0, :] = self.pop[j]['BestPosition']
        # Updating memory
        for j in range(self.SwarmNumber):
            self.pop[j]['Gbest_past_environment'] = self.pop[j]['BestPosition']
            self.pop[j]['FitnessValue'] = problem.eval(self.pop[j]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist

    def run_episode(self, problem):
        self.initialize_swarm(problem)
        while problem.fes < problem.maxfes:
            self.iterative_components(problem)
            if problem.RecentChange == 1:
                problem.reset_RecentChange()
                self.change_reaction(problem)
                #print(f"Environment number: {problem.Environmentcounter}")

        result = {'cost': self.pop[-1]['FitnessValue'], 'fes': problem.fes, 'avg_dist': self.avg_dist}
        if hasattr(problem, 'CurrentError'):
            err = problem.CurrentError
            offlineerror = np.nanmean(err)
            result['current_error'] = offlineerror
        return result
