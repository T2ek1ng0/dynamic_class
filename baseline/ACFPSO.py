"""
Danial Yazdani et al.,
        "Adaptive control of subpopulations in evolutionary dynamic optimization"
        IEEE Transactions on Cybernetics, vol. 52(7), pp. 6476 - 6489, 2020.
paper: https://ieeexplore.ieee.org/document/9284465
source: https://github.com/EvoMindLab/EDOLAB/tree/main/Algorithm/ACFPSO
最大化优化，如果做最小化的话eval值取反
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy

class ACFPSO(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.ps = 5
        self.w = 0.729843788
        self.c1 = 2.05
        self.c2 = 2.05
        self.ConvergenceSleepLayer = 0.05
        self.ConvergenceOuterLayer = 0.8
        self.ConvergenceInnerLayer = 0.3
        self.ExclusionOuterLayer = 1
        self.ExclusionInnerLayer = 0.3
        self.avg_dist = 0

    def __str__(self):
        return "ACFPSO"

    def initialize_swarm(self, problem):
        self.SwarmNumber = 1
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.FreeSwarmID = 0
        self.Radius = np.full(self.dim, (self.ub - self.lb) / self.SwarmNumber)
        self.DiversityPlus = 1
        self.ShiftSeverity = np.full(self.dim, 1)
        self.Relocations = np.full((problem.EnvironmentNumber, self.dim), np.nan)
        self.Relocations[0, :] = self.ShiftSeverity
        self.pop = []
        for _ in range(self.SwarmNumber):
            self.pop.append(self.sub_population_generator(self.ps, problem))

    def sub_population_generator(self, ps, problem):
        random_x = self.lb + (self.ub - self.lb) * np.random.rand(ps, self.dim)
        fitness = problem.eval(random_x)
        if problem.avg_dist:
            self.avg_dist += problem.avg_dist
        pop = {
            'X': random_x.copy(),
            'Velocity': np.zeros((ps, self.dim)),
            'Shifts': None,
            'Sleep': 0,
            'FitnessValue': fitness,
            'PbestPosition': random_x.copy(),
            'PbestValue': None,
            'BestValue': None,
            'GbestID': None,
            'BestPosition': None,
            'Center': np.mean(random_x, axis=0),
            'Diversity': None,
            'Gbest_past_environment': None,
            'PreviousGbestPosition': None,
            'IterationCounter': 0,
            'phase': 1,  # 1=explorere, 2= exploiter, 3=tracker
        }
        pop['Diversity'] = np.maximum(np.max(pop['PbestPosition'], axis=0) - pop['Center'], pop['Center'] - np.min(pop['PbestPosition'], axis=0))
        if not problem.RecentChange:
            pop['PbestValue'] = fitness.copy()
            pop['GbestID'] = np.argmax(pop['PbestValue'])
            pop['BestValue'] = pop['PbestValue'][pop['GbestID']]
            pop['BestPosition'] = pop['PbestPosition'][pop['GbestID'], :].copy()
        else:
            pop['FitnessValue'] = np.full(ps, -np.inf)
            pop['PbestValue'] = np.full(ps, -np.inf)
            pop['GbestID'] = np.argmax(pop['PbestValue'])
            pop['BestValue'] = pop['PbestValue'][pop['GbestID']].copy()
            pop['BestPosition'] = pop['PbestPosition'][pop['GbestID'], :].copy()
        return pop

    def iterative_components(self, problem):
        # Sub-swarm movement
        MaxValue = -np.inf
        BestIndex = None
        for j in range(self.SwarmNumber):
            if self.pop[j]['BestValue'] > MaxValue:
                MaxValue = self.pop[j]['BestValue']
                BestIndex = j
        # Optimizer.pop(BestIndex).Sleep=0
        GbestValue = np.array([[i, self.pop[i]['BestValue']] for i in range(self.SwarmNumber) if i != self.FreeSwarmID and not self.pop[i]['Sleep']])
        ActiveSubPopulationNumber = GbestValue.shape[0]
        if ActiveSubPopulationNumber:
            idx = np.argsort(GbestValue[:, 1])[::-1]
            Tickets = GbestValue[idx].T  # (2,n)
            row3 = np.arange(Tickets.shape[1], 0, -1, dtype=int)
            Tickets = np.vstack([Tickets, row3])
            Tickets = np.vstack([Tickets, np.zeros((2, Tickets.shape[1]), dtype=int)])
            Tickets[3, 0] = 1
            Tickets[4, 0] = Tickets[2, 0]
            for i in range(1, Tickets.shape[1]):
                Tickets[3, i] = Tickets[4, i - 1] + 1
                Tickets[4, i] = Tickets[3, i] + Tickets[2, i] - 1
            RandomTickets = np.random.randint(1, Tickets[2, :].sum() + 1, size=ActiveSubPopulationNumber)
            TicketWinners = np.full((len(RandomTickets), 2), np.nan)
            for i, rt in enumerate(RandomTickets):
                for j in range(Tickets.shape[1]):
                    if Tickets[3, j] <= rt <= Tickets[4, j]:
                        TicketWinners[i, 0] = Tickets[0, j]
                        TicketWinners[i, 1] = Tickets[1, j]
                        break
            idx2 = np.argsort(TicketWinners[:, 1])[::-1]
            TicketWinners = TicketWinners[idx2].T
            TicketWinners = np.delete(TicketWinners, 1, axis=0)
            TicketWinners = np.hstack([TicketWinners, np.array([[BestIndex, self.FreeSwarmID]])])
        else:
            TicketWinners = np.array([[self.FreeSwarmID]])
        while TicketWinners.shape[1] > 0:
            i = int(TicketWinners[0, 0])
            TicketWinners = np.delete(TicketWinners, 0, axis=1)
            r1 = np.random.rand(self.ps, self.dim)
            r2 = np.random.rand(self.ps, self.dim)
            self.pop[i]['Velocity'] = self.w * (self.pop[i]['Velocity'] + self.c1 * r1 * (self.pop[i]['PbestPosition'] - self.pop[i]['X']) + self.c2 * r2 * (self.pop[i]['BestPosition'] - self.pop[i]['X']))
            self.pop[i]['X'] += self.pop[i]['Velocity']
            mask = (self.pop[i]['X'] < self.lb) | (self.pop[i]['X'] > self.ub)
            self.pop[i]['X'] = np.clip(self.pop[i]['X'], self.lb, self.ub)
            self.pop[i]['Velocity'][mask] = 0
            tmp = problem.eval(self.pop[i]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            if problem.RecentChange:
                return
            self.pop[i]['FitnessValue'] = tmp
            improved = self.pop[i]['FitnessValue'] > self.pop[i]['PbestValue']
            self.pop[i]['PbestValue'][improved] = self.pop[i]['FitnessValue'][improved].copy()
            self.pop[i]['PbestPosition'][improved, :] = self.pop[i]['X'][improved, :].copy()
            BestPbestID = np.argmax(self.pop[i]['PbestValue'])
            if self.pop[i]['PbestValue'][BestPbestID] > self.pop[i]['BestValue']:
                self.pop[i]['BestValue'] = self.pop[i]['PbestValue'][BestPbestID].copy()
                self.pop[i]['PreviousGbestPosition'] = self.pop[i]['BestPosition'].copy()
                self.pop[i]['BestPosition'] = self.pop[i]['PbestPosition'][BestPbestID].copy()
                self.pop[i]['GbestID'] = BestPbestID
            self.pop[i]['Center'] = np.mean(self.pop[i]['PbestPosition'], axis=0)
            self.pop[i]['Diversity'] = np.maximum(np.max(self.pop[i]['PbestPosition'], axis=0) - self.pop[i]['Center'], self.pop[i]['Center'] - np.min(self.pop[i]['PbestPosition'], axis=0))
            if np.all(self.pop[i]['Diversity'] <= self.ConvergenceInnerLayer * self.Radius):
                self.pop[i]['phase'] = 3  # tracker
            elif np.all(self.pop[i]['Diversity'] <= self.ConvergenceOuterLayer * self.Radius):
                self.pop[i]['phase'] = 2  # exploiter
            else:
                self.pop[i]['phase'] = 1  # explorer
            if np.all(self.pop[i]['Diversity'] <= self.ConvergenceSleepLayer):
                self.pop[i]['Sleep'] = 1
                TicketWinners = TicketWinners[TicketWinners != i].reshape(1, -1)
            j = 0
            while j < self.SwarmNumber:
                if j == i:
                    j += 1
                    continue
                dist = np.abs(self.pop[i]['BestPosition'] - self.pop[j]['BestPosition'])
                # if two sub-population are inside the restricted exclusion area
                if np.all(dist <= self.ExclusionInnerLayer * self.Radius):
                    # when one of them is explorer, the explorer will be re-initialized
                    if self.FreeSwarmID in [i, j]:
                        self.pop[self.FreeSwarmID] = self.sub_population_generator(self.ps, problem)
                        if problem.RecentChange:
                            return
                    # when sub-populations are not explorer, so one of them must be removed
                    else:
                        if self.pop[i]['BestValue'] < self.pop[j]['BestValue']:
                            self.pop.pop(i)
                            TicketWinners = TicketWinners[TicketWinners != i].reshape(1, -1)
                            TicketWinners[TicketWinners > i] -= 1
                            self.FreeSwarmID -= 1
                            self.SwarmNumber -= 1
                            break
                        else:
                            self.pop.pop(j)
                            TicketWinners = TicketWinners[TicketWinners != j].reshape(1, -1)
                            TicketWinners[TicketWinners > j] -= 1
                            if j < i:
                                i -= 1
                            self.FreeSwarmID -= 1
                            self.SwarmNumber -= 1
                            j -= 1
                elif np.all(dist <= self.ExclusionOuterLayer * self.Radius):
                    if self.pop[i]['phase'] == 3 and self.pop[j]['phase'] == 3:
                        j += 1
                        continue  # Do nothing, trackers can continue until enter restericted radius
                    # when one of the involved sub-pops in the warning area is a tracker
                    elif self.pop[i]['phase'] == 3 or self.pop[j]['phase'] == 3:
                        tracker, other = (i, j) if self.pop[i]['phase'] == 3 else (j, i)
                        if self.pop[tracker]['BestValue'] < self.pop[other]['BestValue']:
                            j += 1
                            continue
                        else:
                            if other == self.FreeSwarmID:
                                self.pop[self.FreeSwarmID] = self.sub_population_generator(self.ps, problem)
                                if problem.RecentChange:
                                    return
                            else:
                                self.pop.pop(other)
                                TicketWinners = TicketWinners[TicketWinners != other].reshape(1, -1)
                                TicketWinners[TicketWinners > other] -= 1
                                self.FreeSwarmID -= 1
                                self.SwarmNumber -= 1
                                if other == i:
                                    break
                                else:
                                    if j < i:
                                        i -= 1
                                    j -= 1
                    else:  # if two exploiters, or the explorer and an exploiter are involved in the warning area
                        if self.FreeSwarmID in [i, j]:
                            self.pop[self.FreeSwarmID] = self.sub_population_generator(self.ps, problem)
                            if problem.RecentChange:
                                return
                        else:  # both are exploiters
                            if self.pop[i]['BestValue'] < self.pop[j]['BestValue']:
                                self.pop.pop(i)
                                TicketWinners = TicketWinners[TicketWinners != i].reshape(1, -1)
                                TicketWinners[TicketWinners > i] -= 1
                                self.FreeSwarmID -= 1
                                self.SwarmNumber -= 1
                                break
                            else:
                                self.pop.pop(j)
                                TicketWinners = TicketWinners[TicketWinners != j].reshape(1, -1)
                                TicketWinners[TicketWinners > j] -= 1
                                if j < i:
                                    i -= 1
                                self.FreeSwarmID -= 1
                                self.SwarmNumber -= 1
                                j -= 1
                j += 1
        # FreeSwarm Convergence
        if self.pop[self.FreeSwarmID]['phase'] > 1:
            if self.SwarmNumber > 30:
                worstID = None
                worstValue = np.inf
                for j in range(self.SwarmNumber):
                    if j != self.FreeSwarmID:
                        if self.pop[j]['BestValue'] < worstValue:
                            worstID = j
                            worstValue = self.pop[j]['BestValue']
                self.pop.pop(worstID)
                self.SwarmNumber -= 1
                self.FreeSwarmID -= 1
            self.SwarmNumber += 1
            self.pop.append(self.sub_population_generator(self.ps, problem))
            self.FreeSwarmID = len(self.pop) - 1
            if problem.RecentChange:
                return
        # Updating Thresholds
        TrackerNumber = 0
        for i in range(self.SwarmNumber):
            if self.pop[i]['phase'] == 3:
                TrackerNumber += 1
        self.Radius = np.full(self.dim, 1) * ((self.ub - self.lb) / max(1, TrackerNumber))

    def change_reaction(self, problem):
        # Updating Shift Severity
        dummy = np.full((self.SwarmNumber, self.dim), np.nan)
        for j in range(self.SwarmNumber):
            if self.pop[j]['phase'] == 3:
                if self.pop[j]['Gbest_past_environment'] is not None:
                    dummy[j, :] = np.abs(self.pop[j]['Gbest_past_environment'] - self.pop[j]['BestPosition'])
        dummy = dummy[~np.any(np.isnan(dummy), axis=1)]
        if dummy.shape[0]:
            self.Relocations[problem.Environmentcounter, :] = np.mean(dummy, axis=0)
        tmp = self.Relocations.copy()
        tmp = tmp[~np.any(np.isnan(tmp), axis=1)]
        self.ShiftSeverity = np.mean(tmp, axis=0)
        # Introducing diversity (all except free swarm)
        for j in range(self.SwarmNumber):
            if j != self.FreeSwarmID:
                if self.pop[j]['phase'] == 3:
                    tmp = np.random.randint(0, 2, size=(self.ps, self.dim))
                    tmp[tmp == 0] = -1  # generating 1 and -1 numbers randomly
                    self.pop[j]['X'] = self.pop[j]['BestPosition'] + self.ShiftSeverity * tmp
                    self.pop[j]['X'][0, :] = self.pop[j]['BestPosition'].copy()
        # Updating memory
        for j in range(self.SwarmNumber):
            self.pop[j]['Sleep'] = 0
            self.pop[j]['FitnessValue'] = problem.eval(self.pop[j]['X'])
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            self.pop[j]['PbestValue'] = self.pop[j]['FitnessValue'].copy()
            self.pop[j]['PbestPosition'] = self.pop[j]['X'].copy()
            if self.pop[j]['phase'] == 3:
                self.pop[j]['Gbest_past_environment'] = self.pop[j]['BestPosition']
            BestPbestID = np.argmax(self.pop[j]['PbestValue'])
            self.pop[j]['BestValue'] = self.pop[j]['PbestValue'][BestPbestID]
            self.pop[j]['BestPosition'] = self.pop[j]['PbestPosition'][BestPbestID]
            self.pop[j]['GbestID'] = BestPbestID

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







