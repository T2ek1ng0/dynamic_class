"""
W. Luo, Y. Qiao, X. Lin, P. Xu and M. Preuss,
    "Hybridizing Niching, Particle Swarm Optimization, and Evolution Strategy for Multimodal Optimization,"
    in IEEE Transactions on Cybernetics, vol. 52, no. 7, pp. 6707-6720, July 2022
paper: https://ieeexplore.ieee.org/document/9295420
source: https://github.com/bonsur-ustc/NBNC-PSO-ES
最大化优化，如果做最小化的话eval值取反
"""
from metaevobox.environment import Basic_Problem
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import copy

class NBNC_PSO_ES(Basic_Optimizer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.avg_dist = 0

    def __str__(self):
        return "NBNC_PSO_ES"

    def initialize_swarm(self, problem):
        self.ps = 500
        self.gen = 1
        self.fes = 0
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.maxfes = problem.maxfes
        self.eval_cmaes = 0.2 * self.maxfes
        self.max_Gen_Pso = (self.maxfes - self.eval_cmaes - self.fes) // self.ps
        self.max_v = (self.ub - self.lb) / 2
        self.min_v = -self.max_v
        self.p_x = self.lb + (self.ub - self.lb) * np.random.rand(self.ps, self.dim)
        self.p_v = self.min_v + (self.max_v - self.min_v) * np.random.rand(self.ps, self.dim)
        self.p_cost = problem.eval(self.p_x)
        if problem.avg_dist:
            self.avg_dist += problem.avg_dist
        self.pbest = self.p_x.copy()
        self.pbest_cost = self.p_cost.copy()
        self.fes += self.ps

    def NBNC(self, fitness, x):
        alpha = 5  # scale factor for species
        slbest = np.zeros((self.ps, 3))
        slbest[:, 0] = np.arange(self.ps, dtype=int)
        matdist = cdist(x, x)
        # Look for the nearest particle
        np.fill_diagonal(matdist, np.inf)
        slbest[:, 2] = np.min(matdist, axis=1)
        slbest[:, 1] = np.argmin(matdist, axis=1)
        meandist = alpha * np.mean(slbest[:, 2])
        # get the minimum distance between the current one and its better neighbor one
        # find the better individual within meandis radius
        matdist[matdist >= meandist] = np.inf
        rank = np.argsort(fitness)[::-1]
        matdist = matdist[np.ix_(rank, rank)]
        matdist[np.triu_indices(self.ps)] = np.inf
        #matdist[rank, rank] = matdist
        slbest[:, 2] = np.min(matdist, axis=1)
        slbest[:, 1] = np.argmin(matdist, axis=1)
        mask = (slbest[:, 2] == np.inf)
        slbest[mask, 1] = slbest[mask, 0]
        sgbest = np.zeros((self.ps, 2), dtype=int)
        sgbest[:, 0] = np.arange(self.ps, dtype=int)
        for i in range(self.ps):
            j = int(slbest[i, 1])
            k = int(slbest[i, 0])
            while j != k:
                k = j
                j = int(slbest[k, 1])
            sgbest[i, 1] = k
        # Construct the raw species
        seed_index = np.unique(sgbest[:, 1]).astype(int)
        seed_len = seed_index.size
        species = []
        cost_seed = np.zeros(seed_len)
        costs_list = []
        seed_list = []
        for i in range(seed_len):
            spec = {
                'seed': seed_index[i],
                'idx': list(np.where(sgbest[:, 1] == seed_index[i])[0]),
                'num': None,
                'cost': fitness[seed_index[i]].copy(),
                'subcost': None
            }
            spec['num'] = len(spec['idx'])
            spec['subcost'] = fitness[np.array(spec['idx'])].copy()
            cost_seed[i] = fitness[seed_index[i]].copy()
            species.append(spec)
            costs_list.append(spec['cost'])
            seed_list.append(spec['seed'])
        # Sort clusters according to their fitness
        index = np.argsort(np.array(costs_list))
        seed_indices = np.array(seed_list)
        seed_x = x[seed_indices, :]
        seed_dist = cdist(seed_x, seed_x)
        np.fill_diagonal(seed_dist, np.inf)
        # the meachism of merging
        mark = np.zeros((seed_len, 2), dtype=int)
        mark[:, 0] = np.arange(seed_len, dtype=int)
        mark[:, 1] = mark[:, 0].copy()
        for i in range(seed_len):
            midx = np.argmin(seed_dist[i, :])
            if species[i]['cost'] < np.min(species[midx]['subcost']):
                mark[i, 1] = midx
        for i in range(seed_len):
            j = mark[i, 1]
            k = mark[i, 0]
            while j != k:
                k = j
                j = mark[k, 1]
            sgbest[i, 1] = k
        flag = np.zeros(seed_len)
        for i in range(seed_len):
            if mark[i, 0] != mark[i, 1]:
                flag[i] = 1
                sgbest[np.array(species[i]['idx']), 1] = species[mark[i, 1]]['seed']
                species[mark[i, 1]]['idx'] += species[i]['idx']
                species[mark[i, 1]]['num'] = len(species[mark[i, 1]]['idx'])
                species[mark[i, 1]]['subcost'] = fitness[np.array(species[mark[i, 1]]['idx'])].copy()
        for i in reversed(range(seed_len)):
            if flag[i] == 1:
                species.pop(i)
        sub_nums = len(species)
        per = 0.5
        if self.gen > per * self.max_Gen_Pso:
            guideIdx = sgbest.copy()
        else:
            guideIdx = slbest[:, :2].astype(int)
        guide = x[guideIdx[:, 1], :]
        return species, guide, sub_nums, meandist

    def reflect(self, x, ps, v=None):
        oneForNeedReflectLower = x < self.lb
        relectionAmount = np.mod(self.lb - x, self.ub - self.lb)
        relectionAmount = np.clip(relectionAmount, 0, None)
        x = np.clip(x, self.lb, None)
        x += relectionAmount
        oneForNeedReflectlarger = x > self.ub
        relectionAmount = np.mod(x - self.ub, self.ub - self.lb)
        relectionAmount = np.clip(relectionAmount, 0, None)
        x = np.clip(x, None, self.ub)
        x -= relectionAmount
        if v is None:
            v = np.zeros((ps, self.dim))
        else:
            v[oneForNeedReflectLower] = 0
            v[oneForNeedReflectlarger] = 0
        return x, v

    def balancePop(self, problem, species, sub_nums):
        m_subsize = 2
        seed_index = [spec['seed'] for spec in species]
        seed_index = np.array(seed_index)
        flag = np.zeros(sub_nums)
        if sub_nums > 1:
            t = 0
            while t < sub_nums:
                if species[t]['num'] >= m_subsize:
                    t += 1
                else:
                    dist_seed = cdist(self.pbest[seed_index, :], self.pbest[seed_index, :])
                    Idx = flag == 1
                    dist_seed[t, Idx] = np.inf
                    dist_seed[t, t] = np.inf
                    g2 = np.argmin(dist_seed[t, :])
                    species[g2]['idx'] += species[t]['idx']
                    species[g2]['num'] = species[g2]['num'] + species[t]['num']
                    species[g2]['subcost'] = self.pbest_cost[np.array(species[g2]['idx'])].copy()
                    max_index = np.argmax(self.pbest_cost[np.array(species[g2]['idx'])])
                    species[g2]['cost'] = np.max(self.pbest_cost[np.array(species[g2]['idx'])])
                    species[g2]['seed'] = species[g2]['idx'][max_index]
                    seed_index[g2] = species[g2]['seed']
                    flag[t] = 1
                    t += 1
        seed_index = list(seed_index)
        for i in reversed(range(len(species))):
            if flag[i] == 1:
                species.pop(i)
                seed_index.pop(i)
        sub_nums = len(seed_index)
        cost_seed = self.pbest_cost[np.array(seed_index)]
        index = np.argsort(cost_seed)[::-1]
        species = [species[i] for i in index]
        return species, sub_nums

    def cmaes(self, problem, pbest, pbestcost, stopval):
        pop_size = pbest.shape[0]
        mu = pop_size // 2
        pop = pbest[:mu, :]
        xmean = np.mean(pop, axis=0)
        if self.dim < 3:
            sigma = 0.1
        else:
            sigma = 0.2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        cs = (mueff + 2) / (self.dim + mueff + 5)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mueff)
        cmu = 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2) ** 2 + mueff)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (self.dim + 1)) - 1) + cs
        pc = np.zeros((self.dim, 1))
        ps = np.zeros((self.dim, 1))
        B = np.eye(self.dim)
        D = np.ones(self.dim)
        C = B @ np.diag(D ** 2) @ B.T
        chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))
        countval = 0
        while countval < stopval:
            arz = np.random.randn(self.dim, pop_size)
            arx = np.random.randn(self.dim, pop_size)
            for k in range(pop_size):
                arx[:, k] = xmean + sigma * B @ (D * arz[:, k])
            countval += pop_size
            arx = arx.T
            mask = arx > self.ub
            arx[mask] = self.ub - np.mod(arx[mask] - self.ub, self.ub - self.lb)
            mask = arx < self.lb
            arx[mask] = self.lb + np.mod(self.lb - arx[mask], self.ub - self.lb)
            x = arx
            artfitness = problem.eval(x)
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            arx = x.T
            arindex = np.argsort(artfitness)[::-1]
            pbest = pbest[arindex, :]
            pbestcost = pbestcost[arindex]
            xmean = arx[:, arindex[:mu]] @ weights
            zmean = arz[:, arindex[:mu]] @ weights
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * B @ zmean
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * countval / pop_size)) / chiN < 1.4 + 2 / (self.dim + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * B @ (D * zmean)
            artmp = B @ (D.reshape(self.dim, -1) * arz[:, arindex[:mu]])
            C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp @ np.diag(weights) @ artmp.T
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-12, 1e2)
            C = np.triu(C) + np.triu(C, k=1).T
            eigvals, eigvecs = np.linalg.eigh(C)
            B = eigvecs
            D = np.sqrt(np.abs(eigvals))
            D = np.clip(D, 1e-12, 1e5)
            improved = artfitness > pbestcost
            arx = arx.T
            pbest[improved, :] = arx[improved, :].copy()
            pbestcost[improved] = artfitness[improved]
            arx = arx.T
            if abs(artfitness[0] - artfitness[pop_size - 1]) < 1e-7:
                bestidx = np.argmax(pbestcost)
                break
        bestidx = np.argmax(pbestcost)
        bestcost = pbestcost[bestidx]
        return pbest, pbestcost, bestcost, bestidx, countval

    def balance_species(self, x, v, species, sub_nums, meandis, guide):
        MaxSpeciesSize = 20
        num = np.array([spec['num'] for spec in species])
        num_index = np.argsort(num)[::-1]
        species = [species[i] for i in num_index]
        # Balance species in descending order of seed fitness
        if sub_nums > 1:
            Overload_index = []
            for i in range(sub_nums):
                if species[i]['num'] > MaxSpeciesSize:
                    subindex = np.argsort(self.pbest_cost[np.array(species[i]['idx'])])[::-1]
                    worst = subindex[MaxSpeciesSize:]
                    over_index = [species[i]['idx'][j] for j in worst]
                    Overload_index.extend(over_index)
                    for j in reversed(sorted(set(worst))):
                        species[i]['idx'].pop(j)
                    species[i]['num'] = MaxSpeciesSize
                    species[i]['subcost'] = self.pbest_cost[np.array(species[i]['idx'])]
                else:
                    if len(Overload_index) == 0:
                        break
                    else:
                        if i < sub_nums - 1:
                            add_num = min(MaxSpeciesSize - species[i]['num'], len(Overload_index))
                        else:
                            add_num = len(Overload_index)
                        add_index = Overload_index[:add_num]
                        seed = species[i]['seed']
                        lb = -meandis + self.pbest[seed, :]
                        ub = meandis + self.pbest[seed, :]
                        max_v = (ub - lb) / 2
                        min_v = -max_v
                        x[add_index, :] = lb + (ub - lb) * np.random.rand(add_num, self.dim)
                        v[add_index, :] = min_v + (max_v - min_v) * np.random.rand(add_num, self.dim)
                        x[add_index, :], v[add_index, :] = self.reflect(x[add_index, :], add_num)
                        self.pbest[add_index, :] = np.tile(self.pbest[seed], (add_num, 1))
                        self.pbest_cost[add_index] = np.full(add_num, self.pbest_cost[seed])
                        guide[add_index, :] = np.tile(self.pbest[seed], (add_num, 1))
                        species[i]['idx'] += list(add_index)
                        species[i]['num'] = len(species[i]['idx'])
                        species[i]['subcost'] = np.concatenate([species[i]['subcost'], self.pbest_cost[add_index]])
                        max_index = np.argmax(species[i]['subcost'])
                        species[i]['cost'] = species[i]['subcost'][max_index]
                        species[i]['seed'] = species[i]['idx'][max_index]
        return species, x, v, guide

    def run_episode(self, problem):
        self.initialize_swarm(problem)
        # Phase1:pso evolution
        w = 0.729
        c1 = 2.05 * w
        c2 = 2.05 * w
        while self.gen < self.max_Gen_Pso:
            species, guide, num, meandist = self.NBNC(self.pbest_cost, self.pbest)
            if self.gen == self.max_Gen_Pso // 4:
                species, self.p_x, self.p_v, guide = self.balance_species(self.p_x, self.p_v, species, num, meandist, guide)
            r1 = np.random.rand(self.ps, self.dim)
            r2 = np.random.rand(self.ps, self.dim)
            v_tmp = w * self.p_v + c1 * r1 * (self.pbest - self.p_x) + c2 * r2 * (guide - self.p_x)
            v_tmp = np.clip(v_tmp, self.min_v, self.max_v)
            x_tmp = self.p_x + v_tmp
            self.p_x, self.p_v = self.reflect(x_tmp, self.ps, v_tmp)
            new_cost = problem.eval(self.p_x)
            if problem.avg_dist:
                self.avg_dist += problem.avg_dist
            self.fes += self.ps
            improved = new_cost > self.p_cost
            self.pbest[improved, :] = self.p_x[improved, :].copy()
            self.pbest_cost[improved] = new_cost[improved]
            self.gen += 1
            self.p_cost = new_cost.copy()
        # Phase2:CMAES
        stopval = self.maxfes - self.fes
        species, num = self.balancePop(problem, species, num)
        for i in range(num):
            index = species[i]['idx']
            sort_index = np.argsort(self.pbest_cost[index])[::-1]
            if self.dim < 3:
                cut_num = 20
            else:
                cut_num = 30
            cut = min(species[i]['num'], cut_num)
            index = np.array(index)
            index = index[sort_index[:cut]]
            # Evolution with CMA-ES
            self.pbest[index, :], self.pbest_cost[index], bestcost, bestidx, countval = self.cmaes(problem, self.pbest[index, :], self.pbest_cost[index], stopval)
            species[i]['seed'] = index[bestidx]
            species[i]['cost'] = bestcost
            self.fes += countval
            stopval = self.maxfes - self.fes
            if stopval <= 0:
                break
        result = {'cost': self.pbest_cost, 'fes': self.fes, 'avg_dist': self.avg_dist}
        if hasattr(problem, 'CurrentError'):
            err = problem.CurrentError
            offlineerror = np.nanmean(err)
            result['current_error'] = offlineerror
        return result





