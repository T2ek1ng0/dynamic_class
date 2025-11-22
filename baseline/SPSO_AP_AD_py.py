"""
Delaram Yazdani et al.,
        "A Species-based Particle Swarm Optimization with Adaptive Population Size and Deactivation of Species for Dynamic Optimization Problems"
        ACM Transactions on Evolutionary Learning and Optimization, 2023.
paper: https://dl.acm.org/doi/10.1145/3604812
source: https://github.com/EvoMindLab/EDOLAB/tree/main/Algorithm/SPSO_AP_AD
最大化优化
"""
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
import copy
from scipy.spatial.distance import cdist, pdist, squareform

class SPSO_AP_AD(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.__config = config
        self.InitialPopulationSize = 50
        self.SwarmMember = 5
        self.NewlyAddedPopulationSize = 5  # Number of new particles added when species count < Nmax
        self.w = 0.729843788  # Inertia weight used in velocity update
        self.c1 = 2.05
        self.c2 = 2.05
        self.rho = 0.7  # Ratio controlling the maximum deactivation threshold
        self.mu = 0.2  # Ratio controlling the minimum deactivation threshold
        self.beta = 1  # Initial factor for adjusting the deactivation threshold dynamically
        self.gama = 0.1  # Decay rate for updating beta
        self.Nmax = 30  # Maximum allowed number of species before anti‑convergence triggers
        self.avgdist = 0

    def initialize_swarm(self, problem):
        self.dim = problem.dim
        self.ub = problem.ub
        self.lb = problem.lb
        self.ShiftSeverity = 1
        self.ExclusionLimit = 0.5 * ((self.ub - self.lb) / (problem.peak_num ** (1 / self.dim)))
        self.GenerateRadious = 0.6 * self.ExclusionLimit
        self.teta = self.ShiftSeverity
        self.tracker = []
        self.MaxDeactivation = self.rho * self.ShiftSeverity
        self.MinDeactivation = self.mu * np.sqrt(self.dim)
        self.CurrentDeactivation = self.MaxDeactivation
        self.Particle = []  # list of dist
        for i in range(self.InitialPopulationSize):
            self.Particle.append(self.sub_population_generator(1, problem))
        self.Species = self.CreatingSpecies()  # list of dist

    def sub_population_generator(self, ps, problem):
        init_X = self.lb + (self.ub - self.lb) * np.random.rand(ps, self.dim)
        Velocity = np.zeros((ps, self.dim))
        if ps == 1:
            init_X = init_X[0]
            Velocity = Velocity[0]
        PbestPosition = init_X.copy()
        FitnessValue = problem.eval(init_X)
        if problem.avg_dist:
            self.avgdist += problem.avg_dist
        if not problem.RecentChange:
            PbestFitness = FitnessValue
        else:
            if ps == 1:
                FitnessValue = -np.inf
            else:
                FitnessValue = np.full((ps, 1), -np.inf)
            PbestFitness = FitnessValue
        population = {
            'X': init_X.copy(),
            'Velocity': Velocity.copy(),
            'Shifts': None,
            'FitnessValue': FitnessValue,
            'PbestPosition': PbestPosition.copy(),
            'PbestFitness': PbestFitness,
            'Processed': 0,
            'Pbest_past_environment': None
        }
        return population

    def CreatingSpecies(self):
        for p in self.Particle:
            p['Processed'] = 0
        pbest_list = [p['PbestFitness'].item() for p in self.Particle]
        SortIndex = np.argsort(pbest_list)[::-1]
        Species = {
            'seed': None,
            'member': [],
            'remove': 0,
            'Active': 1,
            'distance': None
        }
        Species_list = []
        for j in range(len(self.Particle)):
            PopList = np.full(len(self.Particle), np.nan)
            if not self.Particle[SortIndex[j]]['Processed']:
                species = copy.deepcopy(Species)
                species['seed'] = SortIndex[j]
                species['member'].append(SortIndex[j])
                self.Particle[SortIndex[j]]['Processed'] = 1
                for i in range(len(self.Particle)):
                    if not self.Particle[i]['Processed']:
                        PopList[i] = np.linalg.norm(self.Particle[i]['PbestPosition'] - self.Particle[SortIndex[j]]['PbestPosition'])
                SortDistance = np.argsort(np.nan_to_num(PopList, nan=np.inf))
                valid_count = np.sum(~np.isnan(PopList))
                species_size = min(self.SwarmMember - 1, valid_count)
                for i in range(species_size):
                    neighbor_idx = SortDistance[i]
                    species['member'].append(neighbor_idx)
                    self.Particle[neighbor_idx]['Processed'] = 1
                if species_size > 0:
                    species['distance'] = np.max(PopList[SortDistance[:species_size]])
                else:
                    species['distance'] = 0.0
                Species_list.append(species)
        return Species_list

    def iterative_components(self, problem):
        # create species, determine trackers and best tracker
        self.Species = self.CreatingSpecies()
        num_pre_iteration_tracker = len(self.tracker)
        self.tracker = []
        best_tracker_index = None
        tmp7 = -np.inf
        for i in range(len(self.Species)):
            if self.Species[i]['distance'] < self.teta:
                self.tracker.append(i)
                if self.Particle[self.Species[i]['seed']]['PbestFitness'].item() > tmp7:
                    tmp7 = self.Particle[self.Species[i]['seed']]['PbestFitness'].item()
                    best_tracker_index = i
        # Exclusion
        removed_particle_index = []
        for i in range(len(self.Species)):
            for j in range(i + 1, len(self.Species)):
                if np.linalg.norm(self.Particle[self.Species[i]['seed']]['PbestPosition'] - self.Particle[self.Species[j]['seed']]['PbestPosition']) < self.ExclusionLimit:
                    if self.Particle[self.Species[i]['seed']]['PbestFitness'].item() < self.Particle[self.Species[j]['seed']]['PbestFitness'].item():
                        self.Species[i]['remove'] = 1
                        removed_particle_index.extend(self.Species[i]['member'])
                    else:
                        self.Species[j]['remove'] = 1
                        removed_particle_index.extend(self.Species[j]['member'])
        # remove tracker index which were removed by exclusion
        if len(self.tracker):
            for i in reversed(range(len(self.tracker))):
                if self.Species[self.tracker[i]]['remove']:
                    self.tracker.pop(i)
        num_current_iteration_tracker = len(self.tracker)
        # compare the number of trackers with previous iteration
        # If it changed, reset the current deactivation value
        if num_pre_iteration_tracker < num_current_iteration_tracker:
            self.CurrentDeactivation = self.MaxDeactivation
        # deactive converged trackers, except for best tracker
        tmp9 = 1
        if len(self.tracker):
            for i in range(len(self.tracker)):
                if self.Species[self.tracker[i]]['distance'] > self.CurrentDeactivation:
                    tmp9 = 0
                else:
                    if self.tracker[i] != best_tracker_index:
                        self.Species[self.tracker[i]]['Active'] = 0
        # Update current deactivation value
        if tmp9 and len(self.tracker):
            if self.CurrentDeactivation > self.MinDeactivation:
                self.beta *= self.gama
                self.CurrentDeactivation = self.MinDeactivation + (self.MaxDeactivation - self.MinDeactivation) * self.beta
        # remove any species according to their "remove" field
        for i in reversed(range(len(self.Species))):
            if self.Species[i]['remove']:
                self.Species.pop(i)
        # update exclusion/generate radious
        self.ExclusionLimit = 0.5 * ((self.ub - self.lb) / (len(self.Species) ** (1 / self.dim)))
        self.GenerateRadious = 0.6 * self.ExclusionLimit
        # Check if all species are convereged
        allConverge = True
        for i in range(len(self.Species)):
            if self.Species[i]['distance'] > self.GenerateRadious:
                allConverge = False
                break
        # Check the number of sub-populations to trigger anti-convergence
        if len(self.Species) >= self.Nmax and allConverge:
            allConverge = False
            WorstSwarmValue = np.inf
            WorstSwarmIndex = None
            for i in range(len(self.Species)):
                if self.Particle[self.Species[i]['seed']]['PbestFitness'].item() < WorstSwarmValue:
                    WorstSwarmValue = self.Particle[self.Species[i]['seed']]['PbestFitness'].item()
                    WorstSwarmIndex = i
            for i in range(len(self.Species[WorstSwarmIndex]['member'])):
                self.Particle[self.Species[WorstSwarmIndex]['member'][i]] = self.sub_population_generator(1, problem)
                if problem.RecentChange:
                    self.Species = self.CreatingSpecies()
                    return
            self.Species[WorstSwarmIndex]['Active'] = 1
        # Run PSO
        for i in range(len(self.Species)):
            if self.Species[i]['Active']:
                for j in range(len(self.Species[i]['member'])):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    self.Particle[self.Species[i]['member'][j]]['Velocity'] = self.w * (self.Particle[self.Species[i]['member'][j]]['Velocity'] + self.c1 * r1 * (self.Particle[self.Species[i]['member'][j]]['PbestPosition'] - self.Particle[self.Species[i]['member'][j]]['X']) + self.c2 * r2 * (self.Particle[self.Species[i]['seed']]['PbestPosition'] - self.Particle[self.Species[i]['member'][j]]['X']))
                    self.Particle[self.Species[i]['member'][j]]['X'] += self.Particle[self.Species[i]['member'][j]]['Velocity']
                    tmp_mask = (self.Particle[self.Species[i]['member'][j]]['X'] < self.lb) | (self.Particle[self.Species[i]['member'][j]]['X'] > self.ub)
                    self.Particle[self.Species[i]['member'][j]]['X'] = np.clip(self.Particle[self.Species[i]['member'][j]]['X'], self.lb, self.ub)
                    self.Particle[self.Species[i]['member'][j]]['Velocity'][tmp_mask] = 0
                    self.Particle[self.Species[i]['member'][j]]['FitnessValue'] = problem.eval(self.Particle[self.Species[i]['member'][j]]['X'])
                    if problem.avg_dist:
                        self.avgdist += problem.avg_dist
                    if problem.RecentChange:
                        if len(removed_particle_index):
                            removed_particle_index = sorted(set(removed_particle_index), reverse=True)
                            for idx in removed_particle_index:
                                self.Particle.pop(idx)
                        self.Species = self.CreatingSpecies()
                        return
                    if self.Particle[self.Species[i]['member'][j]]['FitnessValue'].item() > self.Particle[self.Species[i]['member'][j]]['PbestFitness'].item():
                        self.Particle[self.Species[i]['member'][j]]['PbestFitness'] = self.Particle[self.Species[i]['member'][j]]['FitnessValue'].copy()
                        self.Particle[self.Species[i]['member'][j]]['PbestPosition'] = self.Particle[self.Species[i]['member'][j]]['X'].copy()
        # remove particles which were removed by exclusion
        if len(removed_particle_index):
            removed_particle_index = sorted(set(removed_particle_index), reverse=True)
            for idx in removed_particle_index:
                self.Particle.pop(idx)
            self.Species = self.CreatingSpecies()
        # Insert individuals if all species are convereged
        if len(self.Species) < self.Nmax and allConverge:
            for i in range(self.NewlyAddedPopulationSize):
                self.Particle.append(self.sub_population_generator(1, problem))
                if problem.RecentChange:
                    self.Species = self.CreatingSpecies()
                    return

    def change_reaction(self, problem):
        # determine trackers
        self.tracker = []
        for i in range(len(self.Species)):
            if self.Species[i]['distance'] < self.teta:
                self.tracker.append(i)
        # Updating shift severity
        dummy = np.full(len(self.tracker), np.nan)
        for j in range(len(self.tracker)):
            for i in range(len(self.Species[self.tracker[j]]['member'])):
                if self.Particle[self.Species[self.tracker[j]]['member'][i]]['Pbest_past_environment'] is not None:
                    self.Particle[self.Species[self.tracker[j]]['seed']]['Shifts'] = np.linalg.norm(self.Particle[self.Species[self.tracker[j]]['member'][i]]['Pbest_past_environment'] - self.Particle[self.Species[self.tracker[j]]['seed']]['PbestPosition'])
                    self.Particle[self.Species[self.tracker[j]]['member'][i]]['Pbest_past_environment'] = None
                    if i != 0:
                        self.Particle[self.Species[self.tracker[j]]['member'][i]]['Shifts'] = None
                    break
            if self.Particle[self.Species[self.tracker[j]]['seed']]['Shifts'] is not None:
                dummy[j] = self.Particle[self.Species[self.tracker[j]]['seed']]['Shifts']
        dummy = dummy[~np.isnan(dummy)]
        if dummy.size:
            self.ShiftSeverity = np.mean(dummy)
        # increase diversity for only trackers
        if len(self.tracker):
            for j in range(len(self.tracker)):
                for i in range(len(self.Species[self.tracker[j]]['member'])):
                    R = np.random.randn(self.dim)
                    norm_R = np.linalg.norm(R)
                    if norm_R == 0:
                        norm_R = 1e-8
                    shift = (R / norm_R) * self.ShiftSeverity
                    self.Particle[self.Species[self.tracker[j]]['member'][i]]['X'] = self.Particle[self.Species[self.tracker[j]]['seed']]['PbestPosition'] + shift
                self.Particle[self.Species[self.tracker[j]]['seed']]['X'] = self.Particle[self.Species[self.tracker[j]]['seed']]['PbestPosition'].copy()
                self.Particle[self.Species[self.tracker[j]]['seed']]['Pbest_past_environment'] = self.Particle[self.Species[self.tracker[j]]['seed']]['PbestPosition'].copy()
        # Check bound handling
        for i in range(len(self.Species)):
            for j in range(len(self.Species[i]['member'])):
                tmp_mask = (self.Particle[self.Species[i]['member'][j]]['X'] < self.lb) | (self.Particle[self.Species[i]['member'][j]]['X'] > self.ub)
                self.Particle[self.Species[i]['member'][j]]['X'] = np.clip(self.Particle[self.Species[i]['member'][j]]['X'], self.lb, self.ub)
                self.Particle[self.Species[i]['member'][j]]['Velocity'][tmp_mask] = 0
        # Updating memory for all
        for j in range(len(self.Particle)):
            self.Particle[j]['FitnessValue'] = problem.eval(self.Particle[j]['X'])
            if problem.avg_dist:
                self.avgdist += problem.avg_dist
            self.Particle[j]['PbestFitness'] = self.Particle[j]['FitnessValue'].copy()
            self.Particle[j]['PbestPosition'] = self.Particle[j]['X'].copy()
        # Updating thresholds parameters
        self.MaxDeactivation = self.rho * self.ShiftSeverity
        self.CurrentDeactivation = self.MaxDeactivation
        self.teta = self.ShiftSeverity
        self.beta = 1

    def run_episode(self, problem):
        self.initialize_swarm(problem)
        while problem.fes < problem.maxfes:
            self.iterative_components(problem)
            #print(f"fes: {problem.fes}/{problem.maxfes}")
            if problem.RecentChange == 1:
                problem.reset_RecentChange()
                self.change_reaction(problem)
                #print(f"Environment number: {problem.current_env}")

        gbest_list = [max(p['PbestFitness'].item() for p in self.Particle)]
        result = {'cost': gbest_list, 'fes': problem.fes, 'avg_dist': self.avgdist}
        return result







