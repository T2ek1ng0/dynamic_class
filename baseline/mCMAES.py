"""
Danial Yazdani et al.,
        "Scaling up dynamic optimization problems: A divide-and-conquer approach"
        IEEE Transactions on Evolutionary Computation, vol. 24(1), pp. 1 - 15, 2019.
paper: https://ieeexplore.ieee.org/document/8657680
source: https://github.com/EvoMindLab/EDOLAB/blob/main/Algorithm/mCMAES
最大化优化，如果做最小化的话eval值取反
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy

class mCMAES(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.ps = 1
        self.SwarmNumber = 1
        self.FreeSwarmID = 0
        self.lamb = 5
        self.mu = 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        self.avg_dist = 0

    def __str__(self):
        return "mCMAES"

    def initialize_swarm(self, problem):
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        self.MinExclusionLimit = 0.5 * ((self.ub - self.lb) / (10 ** (1 / self.dim)))
        self.ExclusionLimit = self.MinExclusionLimit
        self.ConvergenceLimit = self.ExclusionLimit
        self.pop = [self.sub_population_generator(problem)]

    def sub_population_generator(self, problem):
        random_x = self.lb + (self.ub - self.lb) * np.random.rand(self.dim, 1)
        population = {
            'X': random_x.copy(),
            'FitnessValue': problem.eval(random_x.T),
            'Gbest_past_environment': np.full((self.dim, 1), np.nan),
            'FGbest_past_environment': np.nan,
            'sigma': np.round((self.ub - self.lb) / 3),
            'pc': np.zeros((self.dim, 1)),
            'ps': np.zeros((self.dim, 1)),
            'B': np.eye(self.dim),
            'D': np.ones(self.dim),
            'eigeneval': 0,
            'counteval': 0,
            'Shifts': [],
            'ShiftSeverity': 1,
        }
        population['C'] = population['B'] @ np.diag(population['D'] ** 2) @ population['B'].T
        population['invsqrtC'] = population['B'] @ np.diag(population['D'] ** -1) @ population['B'].T
        if problem.avg_dist:
            self.avg_dist += problem.avg_dist
        return population

    def iterative_components(self, problem):
        TmpSwarmNum = self.SwarmNumber
        # Sub-population movement
        for i in range(self.SwarmNumber):
            # Generate and evaluate Optimizer.lambda offspring
            arx = self.pop[i]['X'] + self.pop[i]['sigma'] * (self.pop[i]['B'] @ (self.pop[i]['D'].reshape(self.dim, -1) * np.random.randn(self.dim, self.lamb)))
            mask = arx > self.ub
            arx[mask] = 2 * self.ub - arx[mask]
            mask2 = mask & (arx < self.lb)
            arx[mask2] = self.lb
            mask = arx < self.lb
            arx[mask] = 2 * self.lb - arx[mask]
            mask2 = mask & (arx > self.ub)
            arx[mask2] = self.ub
            arfitness = problem.eval(arx.T)
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            self.pop[i]['counteval'] += self.lamb
            arindex = np.argsort(arfitness)[::-1]
            xold = self.pop[i]['X'].copy()
            self.pop[i]['X'] = (arx[:, arindex[:self.mu]] @ self.weights).reshape(self.dim, 1)
            self.pop[i]['FitnessValue'] = problem.eval(self.pop[i]['X'].T)
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            # Cumumulation: Update evolution paths
            self.pop[i]['ps'] = (1 - self.cs) * self.pop[i]['ps'] + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.pop[i]['invsqrtC'] @ (self.pop[i]['X'] - xold) / self.pop[i]['sigma']
            hsig = np.sum(self.pop[i]['ps'] ** 2) / (1 - (1 - self.cs) ** (2 * self.pop[i]['counteval'] / self.lamb)) / self.dim < 2 + 4 / (self.dim + 1)
            self.pop[i]['pc'] = (1 - self.cc) * self.pop[i]['pc'] + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (self.pop[i]['X'] - xold) / self.pop[i]['sigma']
            # Adapt covariance matrix C
            artmp = (arx[:, arindex[:self.mu]] - xold) / self.pop[i]['sigma']
            self.pop[i]['C'] = (1 - self.c1 - self.cmu) * self.pop[i]['C'] + self.c1 * (self.pop[i]['pc'] @ self.pop[i]['pc'].T + (1 - hsig) * self.cc * (2 - self.cc) * self.pop[i]['C']) + self.cmu * artmp @ np.diag(self.weights) @ artmp.T
            # Adapt step size Optimizer.sigma
            self.pop[i]['sigma'] *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.pop[i]['ps']) / self.chiN - 1))
            # Update B and D from C
            if self.pop[i]['counteval'] - self.pop[i]['eigeneval'] > self.lamb / (self.c1 + self.cmu) / self.dim / 10:
                self.pop[i]['eigeneval'] = self.pop[i]['counteval']
                self.pop[i]['C'] = np.triu(self.pop[i]['C']) + np.triu(self.pop[i]['C'], k=1).T
                # 使用 eigh 保证实数结果
                eigvals, eigvecs = np.linalg.eigh(self.pop[i]['C'])
                eigvals = np.maximum(eigvals, 1e-12)
                self.pop[i]['B'] = eigvecs
                self.pop[i]['D'] = np.sqrt(eigvals)
                self.pop[i]['invsqrtC'] = self.pop[i]['B'] @ np.diag(1 / self.pop[i]['D']) @ self.pop[i]['B'].T
        # Exclusion
        if self.SwarmNumber > 1:
            RemoveList = np.zeros(self.SwarmNumber)
            for i in range(self.SwarmNumber - 1):
                for j in range(i + 1, self.SwarmNumber):
                    if np.linalg.norm(self.pop[i]['X'].T - self.pop[j]['X'].T) < self.ExclusionLimit:
                        if self.pop[i]['FitnessValue'] < self.pop[j]['FitnessValue']:
                            if self.FreeSwarmID != i:
                                RemoveList[i] = 1
                            else:
                                self.pop[i] = self.sub_population_generator(problem)
                                if problem.RecentChange:
                                    return
                        else:
                            if self.FreeSwarmID != j:
                                RemoveList[j] = 1
                            else:
                                self.pop[j] = self.sub_population_generator(problem)
                                if problem.RecentChange:
                                    return
            for k in reversed(range(self.SwarmNumber)):
                if RemoveList[k]:
                    self.pop.pop(k)
                    self.SwarmNumber -= 1
                    if k < self.FreeSwarmID:
                        self.FreeSwarmID -= 1
            self.FreeSwarmID = max(self.FreeSwarmID, 0)
        # FreeSwarm Convergence
        arx = arx.T
        dist = cdist(arx, arx) > self.ConvergenceLimit
        if np.sum(dist) == 0:
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
                    self.pop[j]['Shifts'].append(np.linalg.norm(self.pop[j]['Gbest_past_environment'].T - self.pop[j]['X'].T))
                dummy[j, :len(self.pop[j]['Shifts'])] = self.pop[j]['Shifts']
        dummy = dummy[~np.isnan(dummy)].flatten()
        if dummy.size:
            self.ShiftSeverity = np.mean(dummy)
        # Introducing diversity (all except free swarm)
        for j in range(self.SwarmNumber):
            if j != self.FreeSwarmID:
                self.pop[j]['sigma'] = self.pop[j]['ShiftSeverity'] / 2
                self.pop[j]['pc'] = np.zeros((self.dim, 1))
                self.pop[j]['ps'] = np.zeros((self.dim, 1))
                self.pop[j]['B'] = np.eye(self.dim)
                self.pop[j]['D'] = np.ones(self.dim)
                self.pop[j]['C'] = self.pop[j]['B'] @ np.diag(self.pop[j]['D'] ** 2) @ self.pop[j]['B'].T
                self.pop[j]['invsqrtC'] = self.pop[j]['B'] @ np.diag(1 / self.pop[j]['D']) @ self.pop[j]['B'].T
                self.pop[j]['eigeneval'] = 0
                self.pop[j]['counteval'] = 0
        # Updating memory
        for j in range(self.SwarmNumber):
            self.pop[j]['Gbest_past_environment'] = self.pop[j]['X'].copy()
            self.pop[j]['FGbest_past_environment'] = self.pop[j]['FitnessValue']

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




