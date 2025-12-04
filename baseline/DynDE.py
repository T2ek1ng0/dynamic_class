"""
Mendes, R., Mohais, A.
  "DynDE: a differential evolution for dynamic optimization problems",
  Proceedings of the IEEE Congress on Evolutionary Computation (CEC 05), pp. 2808-2815. IEEE (2005)
paper: https://ieeexplore.ieee.org/document/1555047
source: https://github.com/EvoMindLab/EDOLAB/blob/main/Algorithm/DynDE
最大化优化，如果做最小化的话eval值取反
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy

class DynDE(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.avg_dist = 0
        self.ps = 100
        self.CR = 0.6
        self.F = 0.5
        self.SwarmNumber = 10
        self.IndivSize = 10
        self.BrownNumber = 5
        self.ConvergenceLimit = 0.01
        self.OverlapDegree = 0.1
        self.RandomFlag_map = {
            'Fixed CR/F': 0,
            'Random CR/F': 1
        }
        self.StrategyFlag_map = {
            'Random Strategy': 0,
            'DE/RAND/1': 1,
            'DE/RAND/2': 2,
            'DE/BEST/1': 3,
            'DE/BEST/2': 4,
            'DE/RAND-TO-BEST/1': 5,
            'DE/CURRENT-TO-RAND/1': 6,
            'DE/CURRENT-TO-BEST/1': 7
        }
        self.ExclusionLimit = 15.5
        self.HiberStep = 0.5
        self.FirstUpdate = 1
        self.DiversityFlag = 1
        self.RandomFlag = 0
        self.StrategyFlag = 4

    def __str__(self):
        return "DynDE"

    def initialize_swarm(self, problem):
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.pop = self.sub_population_generator(problem, self.SwarmNumber)
        self.SwarmNumber = len(self.pop)

    def sub_population_generator(self, problem, ps):
        pop = {
            'X': np.full((self.IndivSize, self.dim), np.nan),
            'Donor': None,
            'Trial': None,
            'CR': None,
            'F': None,
            'Strategy': None,
            'IndivType': np.full(self.IndivSize, np.nan),
            'Gbest_past_environment': np.full(self.dim, np.nan),
            'FitnessValue': np.full(self.IndivSize, np.nan),
            'BestValue': -np.inf,
            'BestPosition': None,
            'GbestID': None,
            'Center': None,
            'CurrentRadius': None,
            'ReInitState': 0
        }
        swarm = [copy.deepcopy(pop) for _ in range(ps)]
        Rcloud = 1
        for i in range(ps):
            if not self.StrategyFlag:
                swarm[i]['Strategy'] = np.random.randint(1, 8, size=self.IndivSize)
            else:
                swarm[i]['Strategy'] = np.full(self.IndivSize, self.StrategyFlag)
            if not self.RandomFlag:
                swarm[i]['CR'] = np.full(self.IndivSize, 0.6)
                swarm[i]['F'] = np.full(self.IndivSize, 0.5)
            else:
                swarm[i]['CR'] = np.random.rand(self.IndivSize)
                swarm[i]['F'] = np.random.rand(self.IndivSize)
            # IndivType:
            # 0: Normal Individual
            # 1: Brownian Individual
            # 2: Quantum Individual
            if self.DiversityFlag == 1:
                swarm[i]['IndivType'] = np.ones(self.IndivSize)
                swarm[i]['X'] = self.lb + (self.ub - self.lb) * np.random.rand(self.IndivSize, self.dim)
                swarm[i]['FitnessValue'] = problem.eval(swarm[i]['X'])
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
            elif self.DiversityFlag == 2:
                NormalIndiv = 5
                QuantumIndiv = self.IndivSize - 5
                swarm[i]['IndivType'][:NormalIndiv] = np.zeros(NormalIndiv)
                swarm[i]['IndivType'][NormalIndiv:] = np.full(QuantumIndiv, 2)
                swarm[i]['X'][:NormalIndiv] = self.lb + (self.ub - self.lb) * np.random.rand(NormalIndiv, self.dim)
                swarm[i]['FitnessValue'][:NormalIndiv] = problem.eval(swarm[i]['X'][:NormalIndiv])
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
                bestid = np.argmax(swarm[i]['FitnessValue'][:NormalIndiv])
                XQuantum = np.random.randn(QuantumIndiv, self.dim)
                swarm[i]['X'][NormalIndiv:] = swarm[i]['X'][bestid] + XQuantum * Rcloud * np.random.rand() / np.sqrt(np.sum(XQuantum ** 2, axis=1, keepdims=True))
                swarm[i]['FitnessValue'][NormalIndiv:] = problem.eval(swarm[i]['X'][NormalIndiv:])
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
            elif self.DiversityFlag == 3:
                swarm[i]['IndivType'] = np.zeros(self.IndivSize)
                swarm[i]['X'] = self.lb + (self.ub - self.lb) * np.random.rand(self.IndivSize, self.dim)
                swarm[i]['FitnessValue'] = problem.eval(swarm[i]['X'])
                if problem.avg_dist:
                    self.avg_dist += problem.avg_dist
            swarm[i]['Center'] = np.mean(swarm[i]['X'], axis=0)
            swarm[i]['CurrentRadius'] = np.mean(cdist(swarm[i]['X'], swarm[i]['Center'].reshape(1, -1))[:, 0])
            if not problem.RecentChange:
                bestid = np.argmax(swarm[i]['FitnessValue'])
                swarm[i]['GbestID'] = bestid
                swarm[i]['BestValue'] = swarm[i]['FitnessValue'][bestid]
                swarm[i]['BestPosition'] = swarm[i]['X'][bestid].copy()
            else:
                swarm[i]['FitnessValue'] = np.full(swarm[i]['X'].shape[0], -np.inf)
                bestid = np.argmax(swarm[i]['FitnessValue'])
                swarm[i]['GbestID'] = bestid
                swarm[i]['BestValue'] = swarm[i]['FitnessValue'][bestid]
                swarm[i]['BestPosition'] = swarm[i]['X'][bestid].copy()
        return swarm

    def iterative_components(self, problem):
        # Sub-pop movements
        for i in range(self.SwarmNumber):
            self.pop[i]['Donor'] = np.zeros_like(self.pop[i]['X'])
            self.pop[i]['Trial'] = np.zeros_like(self.pop[i]['X'])
            # Mutation
            self.pop[i]['Donor'] = self.Mutation(self.pop[i])
            # Crossover
            for p in range(self.pop[i]['X'].shape[0]):
                if self.pop[i]['IndivType'][p] != 2:
                    for d in range(self.dim):
                        rnd = np.random.rand()
                        randj = np.random.randint(self.dim)
                        if rnd < self.pop[i]['CR'][p] or d == randj:
                            self.pop[i]['Trial'][p, d] = self.pop[i]['Donor'][p, d]
                        else:
                            self.pop[i]['Trial'][p, d] = self.pop[i]['X'][p, d]
                        if self.pop[i]['Trial'][p, d] > self.ub:
                            self.pop[i]['Trial'][p, d] = self.ub
                        elif self.pop[i]['Trial'][p, d] < self.lb:
                            self.pop[i]['Trial'][p, d] = self.lb
                    # Select
                    tmpfit = problem.eval(self.pop[i]['Trial'][p, :]).item()
                    if problem.avg_dist:
                        self.avg_dist += problem.avg_dist
                    if tmpfit > self.pop[i]['FitnessValue'][p]:
                        self.pop[i]['X'][p, :] = self.pop[i]['Trial'][p, :].copy()
                        self.pop[i]['FitnessValue'][p] = tmpfit
                    if problem.RecentChange:
                        return
            bestid = np.argmax(self.pop[i]['FitnessValue'])
            if self.pop[i]['FitnessValue'][bestid] > self.pop[i]['BestValue']:
                self.pop[i]['BestPosition'] = self.pop[i]['X'][bestid].copy()
                self.pop[i]['BestValue'] = self.pop[i]['FitnessValue'][bestid]
                self.pop[i]['GbestID'] = bestid
            if self.DiversityFlag == 2:
                Rcloud = 1
                for p in range(self.pop[i]['X'].shape[0]):
                    if self.pop[i]['IndivType'][p] == 2:
                        if p != self.pop[i]['GbestID']:
                            XQuantum = np.random.randn(self.dim)
                            self.pop[i]['X'][p, :] = self.pop[i]['X'][self.pop[i]['GbestID']] + XQuantum * Rcloud * np.random.rand() / np.linalg.norm(XQuantum)
                            self.pop[i]['X'][p, :] = np.clip(self.pop[i]['X'][p, :], self.lb, self.ub)
                            self.pop[i]['FitnessValue'][p] = problem.eval(self.pop[i]['X'][p, :])
                            if problem.avg_dist:
                                self.avg_dist += problem.avg_dist
                    bestid = np.argmax(self.pop[i]['FitnessValue'])
                    if self.pop[i]['FitnessValue'][bestid] > self.pop[i]['BestValue']:
                        self.pop[i]['BestPosition'] = self.pop[i]['X'][bestid].copy()
                        self.pop[i]['BestValue'] = self.pop[i]['FitnessValue'][bestid]
                        self.pop[i]['GbestID'] = bestid
                    if problem.RecentChange:
                        return
        # Brownian Individuals
        if self.DiversityFlag == 1:
            for i in range(self.SwarmNumber):
                fitnesssort = self.pop[i]['FitnessValue']
                sortid = np.argsort(fitnesssort)
                sortid = sortid[:self.BrownNumber]
                for j in sortid:
                    sigma = 0.2
                    bestposition = self.pop[i]['X'][self.pop[i]['GbestID']].copy()
                    bestposition += sigma * np.random.randn(self.dim)
                    bestposition = np.clip(bestposition, self.lb, self.ub)
                    bestres = problem.eval(bestposition).item()
                    if problem.avg_dist:
                        self.avg_dist += problem.avg_dist
                    if bestres > self.pop[i]['FitnessValue'][j]:
                        self.pop[i]['FitnessValue'][j] = bestres
                        self.pop[i]['X'][j, :] = bestposition
                        bestid = np.argmax(self.pop[i]['FitnessValue'])
                        if self.pop[i]['FitnessValue'][bestid] > self.pop[i]['BestValue']:
                            self.pop[i]['BestPosition'] = self.pop[i]['X'][bestid].copy()
                            self.pop[i]['BestValue'] = self.pop[i]['FitnessValue'][bestid]
                            self.pop[i]['GbestID'] = bestid
                        if problem.RecentChange:
                            return
                self.pop[i]['Center'] = np.mean(self.pop[i]['X'], axis=0)
                self.pop[i]['CurrentRadius'] = np.mean(cdist(self.pop[i]['X'], self.pop[i]['Center'].reshape(1, -1))[:, 0])
        # Check overlapping
        for i in range(self.SwarmNumber - 1):
            for j in range(i + 1, self.SwarmNumber):
                dis = np.linalg.norm(self.pop[i]['X'][self.pop[i]['GbestID']] - self.pop[j]['X'][self.pop[j]['GbestID']])
                if self.pop[i]['ReInitState'] == 0 and self.pop[j]['ReInitState'] == 0 and dis < self.ExclusionLimit:
                    if self.pop[i]['FitnessValue'][self.pop[i]['GbestID']] > self.pop[j]['FitnessValue'][self.pop[j]['GbestID']]:
                        self.pop[j]['ReInitState'] = 1
                    else:
                        self.pop[i]['ReInitState'] = 1
        for i in range(self.SwarmNumber):
            if self.pop[i]['ReInitState'] == 1:
                self.pop[i] = self.sub_population_generator(problem, 1)[0]

    def Mutation(self, swarm):
        Donor = np.zeros_like(swarm['X'])
        for p in range(swarm['X'].shape[0]):
            id = np.arange(swarm['X'].shape[0])
            Xb = swarm['BestPosition'].copy()
            if swarm['IndivType'][p] != 2 and swarm['Strategy'][p] == 4:  # DE/BEST/2
                id = id[id != swarm['GbestID']]
                rand_id = np.random.choice(id, size=4, replace=False)
                X2 = swarm['X'][rand_id[0]].copy()
                X3 = swarm['X'][rand_id[1]].copy()
                X4 = swarm['X'][rand_id[2]].copy()
                X5 = swarm['X'][rand_id[3]].copy()
                Donor[p, :] = Xb + swarm['F'][p] * (X2 + X3 - X4 - X5)
            elif swarm['IndivType'][p] != 2 and swarm['Strategy'][p] == 3:  # DE/BEST/1
                id = id[id != swarm['GbestID']]
                rand_id = np.random.choice(id, size=2, replace=False)
                X2 = swarm['X'][rand_id[0]].copy()
                X3 = swarm['X'][rand_id[1]].copy()
                Donor[p, :] = Xb + swarm['F'][p] * (X2 - X3)
            elif swarm['IndivType'][p] != 2 and swarm['Strategy'][p] == 2:  # DE/RAND/2
                rand_id = np.random.choice(id, size=5, replace=False)
                X1 = swarm['X'][rand_id[0]].copy()
                X2 = swarm['X'][rand_id[1]].copy()
                X3 = swarm['X'][rand_id[2]].copy()
                X4 = swarm['X'][rand_id[3]].copy()
                X5 = swarm['X'][rand_id[4]].copy()
                Donor[p, :] = X1 + swarm['F'][p] * (X2 + X3 - X4 - X5)
            elif swarm['IndivType'][p] != 2 and swarm['Strategy'][p] == 1:  # DE/RAND/1
                rand_id = np.random.choice(id, size=3, replace=False)
                X1 = swarm['X'][rand_id[0]].copy()
                X2 = swarm['X'][rand_id[1]].copy()
                X3 = swarm['X'][rand_id[2]].copy()
                Donor[p, :] = X1 + swarm['F'][p] * (X2 - X3)
            elif swarm['IndivType'][p] != 2 and swarm['Strategy'][p] == 5:  # DE/RAND-TO-BEST/1
                id = id[id != swarm['GbestID']]
                rand_id = np.random.choice(id, size=3, replace=False)
                X1 = swarm['X'][rand_id[0]].copy()
                X2 = swarm['X'][rand_id[1]].copy()
                X3 = swarm['X'][rand_id[2]].copy()
                Donor[p, :] = X1 + swarm['F'][p] * (Xb - X1) + swarm['F'][p] * (X2 - X3)
            elif swarm['IndivType'][p] != 2 and swarm['Strategy'][p] == 6:  # DE/CURRENT-TO-RAND/1
                X_curr = swarm['X'][p, :].copy()
                id = id[id != p]
                rand_id = np.random.choice(id, size=3, replace=False)
                X1 = swarm['X'][rand_id[0]].copy()
                X2 = swarm['X'][rand_id[1]].copy()
                X3 = swarm['X'][rand_id[2]].copy()
                Donor[p, :] = X_curr + swarm['F'][p] * (X1 - X_curr) + swarm['F'][p] * (X2 - X3)
            elif swarm['IndivType'][p] != 2 and swarm['Strategy'][p] == 7:  # DE/CURRENT-TO-BEST/1
                X_curr = swarm['X'][p, :].copy()
                Xb = swarm['BestPosition'].copy()
                mask = (id != p) & (id != swarm['GbestID'])
                id = id[mask]
                rand_id = np.random.choice(id, size=2, replace=False)
                X2 = swarm['X'][rand_id[0]].copy()
                X3 = swarm['X'][rand_id[1]].copy()
                Donor[p, :] = X_curr + swarm['F'][p] * (Xb - X_curr) + swarm['F'][p] * (X2 - X3)
        return Donor

    def change_reaction(self, problem):
        # Updating memory
        for j in range(self.SwarmNumber):
            self.pop[j]['FitnessValue'] = problem.eval(self.pop[j]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            self.pop[j]['Gbest_past_environment'] = self.pop[j]['BestPosition'].copy()
            bestid = np.argmax(self.pop[j]['FitnessValue'])
            self.pop[j]['GbestID'] = bestid
            self.pop[j]['BestValue'] = self.pop[j]['FitnessValue'][bestid]
            self.pop[j]['BestPosition'] = self.pop[j]['X'][bestid].copy()

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





