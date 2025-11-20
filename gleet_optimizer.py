from scipy.spatial import distance
from metaevobox.environment.optimizer.learnable_optimizer import Learnable_Optimizer
import numpy as np


class GLEET_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    GLEET is a **G**eneralizable **L**earning-based **E**xploration-**E**xploitation **T**radeoff framework, which could explicitly control the exploration-exploitation tradeoff hyper-parameters of a given EC algorithm to solve a class of problems via reinforcement learning.
    # Original paper
    "[**Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning**](https://dl.acm.org/doi/abs/10.1145/3638529.3653996)." Proceedings of the Genetic and Evolutionary Computation Conference (2024).
    # Official Implementation
    [GLEET](https://github.com/GMC-DRL/GLEET)
    """

    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration and sets up internal parameters for optimization.
        # Args:
        - config (object): Config object containing optimizer settings.
            - The Attributes needed for the GLEET_Optimizer are the following:
                - log_interval (int): Interval at which logs are recorded.Default is config.maxFEs/config.n_logpoint.
                - n_logpoint (int): Number of log points to record.Default is 50.
                - full_meta_data (bool): Flag indicating whether to use full meta data.Default is False.
                - maxFEs (int): Maximum number of function evaluations.
                - __FEs (int): Counter for the number of function evaluations.Default is 0.
                - __config (object): Stores the config object from src/config.py.
                - PS (int): Population size.Default is 100.
        # Built-in Attribute:
        - self.__config (object): Stores the configuration object.
        - self.w_decay (bool): Flag to determine weight decay usage.Default is True.
        - self.w (float): Inertia weight, set based on `w_decay`.Default is 0.9 if `w_decay` is True, otherwise 0.729.
        - self.c (float): Acceleration coefficient.Default is 4.1.
        - self.reward_scale (int): Scaling factor for rewards.Default is 100.
        - self.ps (int): Population size or related parameter.Default is 100.
        - self.no_improve (int): Counter for iterations without improvement.Default is 0.
        - self.boarder_method (str): Method for handling boundaries.Default is 'clipping'.
        - self.reward_func (str): Reward function type.Default is 'direct'.
        - self.fes (Any): Tracks function evaluations (initialized as None).Default is None.
        - self.cost (Any): Tracks cost (initialized as None).Default is None.
        - self.log_index (Any): Logging index (initialized as None).Default is None.
        - self.log_interval (int): Interval for logging progress.
        # Returns:
        - None
        """

        super().__init__(config)
        self.__config = config

        self.w_decay = True
        if self.w_decay:
            self.w = 0.9
        else:
            self.w = 0.729
        self.c = 4.1

        self.reward_scale = 1

        self.ps = 100

        self.no_improve = 0

        #self.max_fes = config.maxFEs
        self.max_fes = None

        self.boarder_method = 'clipping'
        #self.reward_func = 'direct'

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

        self.k_neighbors = 4
        self.n_action = 3
        self.max_gen = None
        self.gen = 0
        self.archive_pos = []
        self.archive_val = []
        self.archive_prevval = []
        self.archive_newval = []
        self.dim = None
        self.avgdist = 0
        self.curr_error = None

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the GLEET optimizer instance.
        # Returns:
        - str: The name of the optimizer, "GLEET_Optimizer".
        """

        return "GLEET_Optimizer"

    def initialize_particles(self, problem):
        """
        # Introduction
        Initializes the particles for a particle swarm optimization (PSO) algorithm by generating random positions and velocities, evaluating initial costs, and setting up personal and global bests.
        # Args:
        - problem (object): The problem object, which has attributes `lb` (lower bounds), `ub` (upper bounds), and be compatible with the `get_costs` method.
        # Returns:
        - None: This method updates the internal state of the optimizer by initializing the `particles` attribute with positions, velocities, costs, and bests.
        # Notes:
        - The method uses the optimizer's random number generator (`self.rng`) and assumes the existence of attributes such as `ps` (particle size), `dim` (problem dimensionality), and `max_velocity`.
        - The `particles` dictionary stores all relevant information for each particle, including current and best positions, costs, velocities, and the global best.
        """

        self.dim = problem.dim
        # randomly generate the position
        self.gen = 1
        rand_pos = np.random.rand(self.ps, problem.dim) * (problem.ub - problem.lb) + problem.lb
        v = np.zeros_like(rand_pos)
        c_cost = self.get_costs(rand_pos, problem)  # (ps, 1)
        pop_dist = distance.cdist(rand_pos, rand_pos)
        neighbor_matrix = self.find_nei(pop_dist.copy())
        Species, slbest_idx, sgbest_idx = self.NBNC(c_cost, neighbor_matrix)
        # find out the gbest_val
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position = rand_pos[gbest_index]
        pbest = rand_pos.copy()
        pbest_val = c_cost.copy()
        guide_vec = np.zeros_like(rand_pos)
        for s_idx, s in enumerate(Species):
            best_idx = sgbest_idx[s_idx]
            for p_idx in s:
                guide_vec[p_idx] = rand_pos[best_idx]
        meandis = np.sum(pop_dist) / (self.ps * (self.ps - 1))
        # record
        self.max_cost = np.min(c_cost)
        # store all the information of the particles
        self.particles = {'current_position': rand_pos.copy(),  # ps, dim
                          'c_cost': c_cost.copy(),  # ps
                          'v': v,
                          'pop_dist': pop_dist.copy(),
                          'neighbor_matrix': neighbor_matrix.copy(),
                          'gbest_position': gbest_position.copy(),  # dim
                          'gbest_val': gbest_val,  # 1
                          'gbest_index': gbest_index,
                          'slbest_idx': slbest_idx.copy(),
                          'sgbest_idx': sgbest_idx.copy(),
                          'meandis': meandis,
                          'pbest': pbest,
                          'pbest_val': pbest_val,
                          'guide_vec': guide_vec,
                          'Species': Species,
                          }
        self.archive_pos.append(gbest_position.copy())
        self.archive_val.append(gbest_val)
        self.archive_newval = self.archive_val.copy()
        self.archive_prevval = self.archive_val.copy()

    def find_nei(self, pop_dist):
        pop_dist[range(self.ps), range(self.ps)] = np.inf
        pop_dist_arg = np.argsort(pop_dist.copy(), axis=-1)
        neighbor_matrix = np.zeros(shape=(self.ps, self.ps))
        for i in range(self.ps):
            neighbor_matrix[i][pop_dist_arg[i, :self.k_neighbors]] = 1
        return neighbor_matrix

    def NBNC(self, val=None, neighbor_matrix=None):
        if val is None:
            val = self.particles['c_cost'].copy()
        if neighbor_matrix is None:
            neighbor_matrix = self.particles['neighbor_matrix'].copy()
        # 生成种群
        Species = []
        visited = [False] * self.ps
        for i in range(self.ps):
            if visited[i]:
                continue
            stack = [i]
            cluster = []
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    cluster.append(node)
                    neighbors = np.where(neighbor_matrix[node] == 1)[0]
                    for nbr in neighbors:
                        if not visited[nbr]:
                            stack.append(nbr)
            Species.append(cluster)
        # 寻找每个粒子的最优邻居 slbest_idx 和每个种群的最优解 sgbest_idx
        sgbest_idx = []
        for s in Species:
            sgbest_idx.append(s[np.argmin(val[s])])
        slbest_idx = []
        for i in range(self.ps):
            neighbors = np.where(neighbor_matrix[i] == 1)[0]
            if len(neighbors) == 0:
                slbest_idx.append(i)
            else:
                slbest_idx.append(neighbors[np.argmin(val[neighbors])])
        return Species, slbest_idx, sgbest_idx

    def balancing_the_species(self, problem, alpha=1.0, max_species_size=20):
        pop = self.particles['current_position']
        val = self.particles['c_cost']
        self.particles['Species'].sort(key=lambda s: len(s), reverse=True)
        worstSet = []
        for i, s_i in enumerate(self.particles['Species']):
            N_worst = abs(len(s_i) - max_species_size)
            if len(s_i) > max_species_size:
                sorted_s_i = sorted(s_i, key=lambda idx: self.particles['c_cost'][idx], reverse=True)
                worst_particles = sorted_s_i[:N_worst]
                good_particles = sorted_s_i[N_worst:]
                worstSet.extend(worst_particles)
                self.particles['Species'][i] = good_particles
            else:
                if worstSet:
                    add_num = min(len(worstSet), N_worst) if i < len(self.particles['Species']) - 1 else len(worstSet)
                    if add_num == 0:
                        continue
                    add_Indis = worstSet[:add_num]
                    cut_dis = alpha * self.particles['meandis']
                    for j in add_Indis:
                        self.particles['pbest'][j] = pop[self.particles['sgbest_idx'][i]].copy()
                        self.particles['pbest_val'][j] = val[self.particles['sgbest_idx'][i]].copy()
                        self.particles['guide_vec'][j] = self.particles['pbest'][j].copy()
                        pop[j] = (self.particles['pbest'][j].copy() + np.random.uniform(-cut_dis, cut_dis, size=self.dim))
                        self.particles['v'][j] = 0.0
                    pop = np.clip(pop, problem.lb, problem.ub)
                    self.particles['Species'][i].extend(add_Indis)
                    worstSet = worstSet[add_num:]

    def get_cat_xy(self):
        """
        # Introduction
        Concatenates the current, personal best, and global best positions and their corresponding cost/fitness values for all particles in the optimizer.
        # Returns:
        - np.ndarray: A concatenated NumPy array containing the current positions and costs, personal best positions and values, and global best positions and values for all particles.
        # Notes:
        - The method assumes that the `self.particles` dictionary contains the keys: 'current_position', 'c_cost', 'pbest_position', 'pbest', 'gbest_position', and 'gbest_val'.
        - The concatenation is performed along the last axis for position-value pairs and along the first axis to combine all groups.
        """

        cur_x = self.particles['current_position']
        cur_y = self.particles['c_cost']
        cur_xy = np.concatenate((cur_x, cur_y), axis=-1)
        pbest_x = self.particles['pbest']
        pbest_y = self.particles['pbest_val']
        pbest_xy = np.concatenate((pbest_x, pbest_y), axis=-1)
        gbest_x = self.particles['gbest_position']
        gbest_y = self.particles['gbest_val']
        gbest_xy = np.concatenate((gbest_x, gbest_y), axis=-1)

        return np.concatenate((cur_xy, pbest_xy, gbest_xy), axis=0)

    # the interface for environment reseting
    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and related state variables for the optimizer, preparing it for a new optimization run.
        # Args:
        - problem (object): An object representing the optimization problem, expected to have attributes `ub` (upper bounds) and `lb` (lower bounds) for the search space.
        # Built-in Attribute:
        - self.fes (int): Function evaluation steps, initialized to 0.
        - self.per_no_improve (np.ndarray): Array to track the number of iterations without improvement for each particle, initialized to zeros.
        - self.max_velocity (np.ndarray): Maximum velocity for each particle, calculated based on the problem's bounds.
        - self.max_dist (float): Maximum distance in the search space, calculated based on the problem's bounds.
        - self.no_improve (int): Counter for the number of iterations without improvement, initialized to 0.
        - self.log_index (int): Index for logging progress, initialized to 1.
        - self.cost (list): List to store the best cost found at each logging interval, initialized with the global best value.
        - self.pbest_feature (np.ndarray): Array to store the personal best features of the particles.
        - self.gbest_feature (np.ndarray): Array to store the global best features of the particles.
        - self.meta_X (list): List to store the positions of the particles for meta-data logging, if configured.
        - self.meta_Cost (list): List to store the costs of the particles for meta-data logging, if configured.
        # Returns:
        - np.ndarray: The concatenated state of the population, including both the population state and additional features, with shape (ps, 27).
        # Notes:
        - Resets various counters and state variables to their initial values.
        - Initializes particle positions and velocities.
        - Optionally stores meta-data if configured.
        - Prepares features for exploration and exploitation tracking.
        """

        self.fes = 0
        self.max_fes = problem.maxfes
        self.max_gen = self.max_fes // self.ps
        self.per_no_improve = np.zeros((self.ps,))
        self.max_velocity = 0.1 * (problem.ub - problem.lb)
        # set the hyperparameters back to init value if needed
        if self.w_decay:
            self.w = 0.9

        self.max_dist = np.sqrt(np.sum((problem.ub - problem.lb) ** 2))

        self.no_improve -= self.no_improve
        self.fes -= self.fes
        self.per_no_improve -= self.per_no_improve

        # initialize the population
        self.initialize_particles(problem)

        self.log_index = 1
        self.cost = [self.particles['gbest_val']]

        # get state

        # get the population state
        state = self.observe()  # ps, 9

        # get the exploration state
        self.pbest_feature = state.copy()  # ps, 9

        # get the explotation state
        self.gbest_feature = state[self.particles['gbest_index']]  # 9

        # get and return the total state (population state, exploration state, exploitation state)
        gp_cat = self.gp_cat()  # ps, 18

        if self.__config.full_meta_data:
            self.meta_X = [self.particles['current_position'].copy()]
            self.meta_Cost = [self.particles['c_cost'].copy()]

        return np.concatenate((state, gp_cat), axis=-1)  # ps, 9+18

    # calculate costs of solutions
    def get_costs(self, position, problem):
        """
        # Introduction
        Calculates the cost(s) for a given position or set of positions in the search space, updating the function evaluation count.
        # Args:
        - position (np.ndarray): The position(s) in the search space for which the cost is to be evaluated. Shape is typically (n_samples, n_dimensions).
        - problem (object): The optimization problem instance, which must provide an `eval` method and an optional `optimum` attribute.
        # Returns:
        - np.ndarray or float: The evaluated cost(s) for the given position(s). If `problem.optimum` is defined, returns the difference between the evaluated value and the optimum.
        # Notes:
        - Increments the `fes` (function evaluation steps) counter by the number of positions evaluated.
        """

        ps = position.shape[0]
        self.fes += ps
        '''
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        '''
        cost = problem.eval(position)
        return cost

    # feature encoding
    def observe(self):
        max_step = self.max_fes // self.ps
        # cost cur
        #fea0 = self.particles['c_cost'] / self.max_cost
        # cost cur_gbest
        #fea1 = (self.particles['c_cost'] - self.particles['gbest_val']) / self.max_cost  # ps
        # cost archive
        if len(self.archive_val) >= 2:
            ratio_per_row = (np.array(self.archive_newval) + 1e-8) / (np.array(self.archive_prevval) + 1e-8)
            overall_ratio = np.mean(ratio_per_row)
            #print(self.archive_prevval, overall_ratio)
            fea1 = np.log10(overall_ratio)
            fea1 = np.clip(fea1, -8, 8) / 8
            fea1 = np.full(self.ps, fea1)
        else:
            fea1 = np.zeros(self.ps)  # ?
        # cost cur
        fea2 = (self.particles['c_cost'] - np.mean(self.particles['c_cost'])) / (np.std(self.particles['c_cost']) + 1e-8)
        # fes cur_fes
        fea3 = np.full(shape=(self.ps), fill_value=(self.max_fes - self.fes) / self.max_fes)
        # no_improve  per
        fea4 = self.per_no_improve / max_step
        # no_improve  whole
        fea5 = np.full(shape=(self.ps), fill_value=self.no_improve / max_step)
        # distance between cur and gbest
        fea6 = np.sqrt(
            np.sum((self.particles['current_position'] - np.expand_dims(self.particles['gbest_position'], axis=0)) ** 2,
                   axis=-1)) / self.max_dist
        # distance between cur and pbest
        fea7 = np.sqrt(np.sum((self.particles['current_position'] - self.particles['pbest']) ** 2,
                              axis=-1)) / self.max_dist

        # cos angle
        pbest_cur_vec = self.particles['pbest'] - self.particles['current_position']
        gbest_cur_vec = np.expand_dims(self.particles['gbest_position'], axis=0) - self.particles['current_position']
        fea8 = np.sum(pbest_cur_vec * gbest_cur_vec, axis=-1) / ((np.sqrt(
            np.sum(pbest_cur_vec ** 2, axis=-1)) * np.sqrt(np.sum(gbest_cur_vec ** 2, axis=-1))) + 1e-5)
        fea8 = np.where(np.isnan(fea8), np.zeros_like(fea8), fea8)

        return np.concatenate((fea1[:, None], fea2[:, None], fea3[:, None], fea4[:, None], fea5[:, None],
                               fea6[:, None], fea7[:, None], fea8[:, None]), axis=-1)

    def gp_cat(self):
        """
        # Introduction
        Concatenates the personal best features and the repeated global best feature for all particles.
        # Returns:
        - np.ndarray: A concatenated array of shape (ps, 18), where `ps` is the number of particles. The array consists of each particle's personal best features and the global best feature repeated for each particle.
        # Notes:
        - Assumes `self.pbest_feature` is an array of shape (ps, n_features).
        - Assumes `self.gbest_feature` is an array of shape (n_features,).
        - The concatenation is performed along the last axis.
        """

        return np.concatenate((self.pbest_feature, self.gbest_feature[None, :].repeat(self.ps, axis=0)),
                              axis=-1)  # ps, 18

    def cal_reward(self):
        ratio_per_row = np.array(self.archive_newval) / (np.array(self.archive_prevval) + 1e-8)
        overall_ratio = np.mean(ratio_per_row)
        if len(self.archive_val) < 2:
            return 0.0
        reward = np.log10(np.maximum(abs(overall_ratio * self.archive_val[-2]), 1e-8)) - np.log10(np.maximum(abs(self.archive_val[-1]), 1e-8))
        reward = np.maximum(reward, 0) / (np.log10(np.maximum(abs(overall_ratio * self.archive_val[-2]), 1e-8)) + 8 + 1e-8)
        return reward

    def act(self, action):
        if action.ndim != 2:
            raise ValueError(f"Expected 2D action (ps, n_action), but got {action.shape} here.")
        w_lb, w_ub = 0.4, 0.9
        c_lb, c_ub = 0.5, 2.5
        N = action.shape[0]  # ps
        w = w_lb + (w_ub - w_lb) * action[:, 0:1]
        c1 = c_lb + (c_ub - c_lb) * action[:, 1:2]
        c2 = c_lb + (c_ub - c_lb) * action[:, 2:3]
        return w, c1, c2  # (ps,)

    def update(self, action, problem, per=0.5, bt=0.25):
        """
        # Introduction
        Updates the state of the particle swarm optimizer (PSO) for one iteration based on the given action and problem definition. This includes updating particle velocities and positions, handling boundary conditions, evaluating costs, updating personal and global bests, managing stagnation counters, calculating rewards, and preparing the next state for further optimization or reinforcement learning.
        # Args:
        - action (np.ndarray): The action(s) to be applied to the particles, typically representing control parameters or decisions for the optimizer.
        - problem (object): The optimization problem instance, which must provide lower and upper bounds (`lb`, `ub`), and optionally an `optimum` attribute.
        # Returns:
        - next_state (np.ndarray): The updated state representation of the particle population after the current iteration.
        - reward (float): The reward signal calculated based on the improvement in global best value.
        - is_end (bool): Flag indicating whether the optimization process has reached its termination condition.
        - info (dict): Additional information (currently empty, but can be extended for logging or debugging).
        # Raises:
        - None explicitly, but may raise exceptions if input shapes are inconsistent or if required attributes are missing from `problem`.
        """

        is_end = False

        if self.gen == int(bt * self.max_gen):
            self.balancing_the_species(problem)
        for idx, species_idx in enumerate(
                self.particles['sgbest_idx'].copy() if self.gen > per * self.max_gen else self.particles['slbest_idx'].copy()):
            self.particles['guide_vec'][idx] = self.particles['current_position'][species_idx].copy()
        pop = self.particles['current_position']
        val = self.particles['c_cost']
        Species, slbest_idx, sgbest_idx = self.NBNC()
        v = self.particles['v']
        pbest = self.particles['pbest']
        pbest_val = self.particles['pbest_val']
        guide_vec = self.particles['guide_vec']

        r1 = np.random.rand(self.ps, self.dim)
        r2 = np.random.rand(self.ps, self.dim)
        w, c1, c2 = self.act(action)
        v = w * v + c1 * r1 * (pbest - pop) + c2 * r2 * (guide_vec - pop)
        new_velocity = np.clip(v, -self.max_velocity, self.max_velocity)

        # update position according the boarding method
        if self.boarder_method == "clipping":
            raw_position = self.particles['current_position'] + new_velocity
            new_position = np.clip(raw_position, problem.lb, problem.ub)
        elif self.boarder_method == "random":
            raw_position = self.particles['current_position'] + new_velocity
            filter = raw_position.abs() > problem.ub
            new_position = np.where(filter, self.rng.uniform(low=problem.lb, high=problem.ub, size=(self.ps, self.dim)),
                                    raw_position)
        elif self.boarder_method == "periodic":
            raw_position = self.particles['current_position'] + new_velocity
            new_position = problem.lb + ((raw_position - problem.ub) % (2. * problem.ub))
        elif self.boarder_method == "reflect":
            raw_position = self.particles['current_position'] + new_velocity
            filter_low = raw_position < problem.lb
            filter_high = raw_position > problem.ub
            new_position = np.where(filter_low, problem.lb + (problem.lb - raw_position), raw_position)
            new_position = np.where(filter_high, problem.ub - (new_position - problem.ub), new_position)

        # calculate the new costs
        new_cost = self.get_costs(new_position, problem)
        if problem.avg_dist:
            self.avgdist += problem.avg_dist

        # update particles
        filters = new_cost < self.particles['pbest_val']
        self.particles['pbest_val'][filters] = new_cost[filters]
        self.particles['pbest'][filters] = new_position[filters]

        per_filters = new_cost < val
        pop[per_filters] = new_position[per_filters].copy()
        val[per_filters] = new_cost[per_filters].copy()
        self.gen += 1
        new_pop_dist = distance.cdist(pop, pop)
        new_neighbor_matrix = self.find_nei(new_pop_dist.copy())
        gbest_position = self.particles['gbest_position'].copy()
        gbest_val = self.particles['gbest_val']
        no_improve = self.no_improve
        if np.min(val) < gbest_val:
            gbest_position = pop[np.argmin(val)].copy()
            gbest_val = np.min(val)
            no_improve = 0
        else:
            no_improve += 1
        self.no_improve = no_improve
        meandis = np.sum(new_pop_dist) / (self.ps * (self.ps - 1))
        new_particles = {'current_position': new_position.copy(),
                         'c_cost': new_cost,
                         'v': new_velocity,
                         'pop_dist': new_pop_dist.copy(),
                         'neighbor_matrix': new_neighbor_matrix.copy(),
                         'gbest_position': gbest_position.copy(),  # dim
                         'gbest_val': gbest_val,  # 1
                         'gbest_index': np.argmin(val),
                         'slbest_idx': slbest_idx.copy(),
                         'sgbest_idx': sgbest_idx.copy(),
                         'meandis': meandis,
                         'pbest': pbest,
                         'pbest_val': pbest_val,
                         'guide_vec': guide_vec,
                         'Species': Species
                         }

        # update the stagnation steps for singal particle in the population
        filter_per_patience = new_particles['c_cost'] < self.particles['c_cost']
        self.per_no_improve += 1
        tmp = np.where(filter_per_patience, self.per_no_improve, np.zeros_like(self.per_no_improve))
        self.per_no_improve -= tmp

        # cal the reward
        all_pos = np.stack(self.archive_pos, axis=0)
        self.archive_newval = problem.eval(all_pos)  # t-5,t-4,t-3,t-2,t-1
        if problem.avg_dist:
            self.avgdist += problem.avg_dist
        self.archive_val.append(gbest_val)
        reward = self.cal_reward()
        reward *= self.reward_scale

        # update the population
        self.particles = new_particles
        self.archive_pos.append(gbest_position.copy())
        self.archive_pos = self.archive_pos[-5:]
        self.archive_val = self.archive_val[-5:]
        # get the population next_state
        #next_state = self.observe()  # ps, 8
        next_state = np.full((self.ps, 8), 0.1)
        self.archive_prevval = np.concatenate((self.archive_newval.copy(), np.array([self.archive_val[-1]])), axis=0)[-5:]  # f_t-1(5,1)

        if self.__config.full_meta_data:
            self.meta_X.append(self.particles['current_position'].copy())
            self.meta_Cost.append(self.particles['c_cost'].copy())

        # see if the end condition is satisfied
        if problem.optimum is None:
            is_end = self.fes >= self.max_fes
        else:
            is_end = self.fes >= self.max_fes

        # update exploration state
        self.pbest_feature = np.where(self.per_no_improve[:, None] == 0, next_state, self.pbest_feature)
        # update exploitation state
        if self.no_improve == 0:
            self.gbest_feature = next_state[self.particles['gbest_index']]
        next_gpcat = self.gp_cat()
        next_state = np.concatenate((next_state, next_gpcat), axis=-1)

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.particles['gbest_val'])

        if is_end:
            if hasattr(problem, 'gmpb_problem'):
                self.curr_error = self.cal_Eo(problem)
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.particles['gbest_val']
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.particles['gbest_val'])

        info = {}
        #print(np.max(next_state), np.min(next_state))
        return next_state, reward, is_end, info

    def cal_Eo(self, problem):
        return sum(problem.gmpb_problem.CurrentError[0]) / len(problem.gmpb_problem.CurrentError[0])
