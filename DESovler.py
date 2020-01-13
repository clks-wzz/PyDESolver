import os
import copy
import random
import numpy as np

from DELoss import *
from DESampler import *

class DESolver:
    def __init__(self, nDim=8, nPop=50, scale = 0.7, CR=.5, fusing_th=0.0, max_generation=50, samplers_list=None, energy_funtion=None, isArray=True):
        '''
        nDim: D
        nPop: M
        '''
        assert(not samplers_list is None)
        assert(nDim == len(samplers_list))
        assert(not energy_funtion is None)
        self.samplers_list = samplers_list
        self.energy_funtion = energy_funtion
        self.isArray = isArray

        self.nDim = nDim
        self.nPop = nPop
        self.scale = scale
        self.CR = CR
        self.fusing_th = fusing_th
        self.max_generation = max_generation
        self.bestSolution = [0.0 for x in range(self.nDim)]
        self.bestEnergy = 1e7

    def progress(self, generation):
        print('generation: %d, bestEnergy: %.6f, bestSolution:'%(generation, self.bestEnergy), self.bestSolution)

    def init_population(self):
        assert(len(self.samplers_list) == self.nDim)
        self.population = [
            [sampler.sample() for sampler in self.samplers_list] \
            for i in range(self.nPop)
        ]

        if self.isArray:
            self.population = [np.array(x) for x in self.population]

        self.trialpop = copy.deepcopy(self.population)
        self.pool = list(range(self.nPop))

    def force_vars_inrange(self, candidate):
        for i, sampler in enumerate(self.samplers_list):
            self.trialpop[candidate][i] = sampler.force_var_inrange(self.trialpop[candidate][i])

    def mutation(self, candidate):
        if candidate == 0:
            _pool = self.pool[1:]
        elif candidate == self.nPop-1:
            _pool = self.pool[:self.nPop-1]
        else:
            _pool = self.pool[0:candidate] + self.pool[candidate+1:] 
        r1, r2, r3 = random.sample(_pool, 3)
        self.trialpop[candidate] = self.population[r1] + self.scale * (self.population[r2] - self.population[r3])

    def crossover(self, candidate):
        probs = np.random.random((self.nDim))
        probs[random.randint(0, self.nDim-1)] = 1.0
        gather = np.array(probs>=self.CR, np.float) 
        #print(probs, gather)
        self.trialpop[candidate] = gather *  self.trialpop[candidate] + (1.0 - gather) * self.population[candidate]

    def greedy_selection(self, candidate):
        #print(self.population[candidate], self.trialpop[candidate])
        lst_loss = self.energy_funtion(self.population[candidate])
        cur_loss = self.energy_funtion(self.trialpop[candidate])
        if cur_loss < lst_loss:
            self.population[candidate] = self.trialpop[candidate]
            if cur_loss < self.bestEnergy:
                self.bestEnergy = cur_loss
                self.bestSolution = self.trialpop[candidate]

    def solve(self):
        self.init_population()
        t = 0
        while self.bestEnergy>=self.fusing_th and t < self.max_generation:
            #print(t)
            self.progress(t)
            for i in range(self.nPop):
                self.mutation(i)
                self.force_vars_inrange(i)
                self.crossover(i)
                self.greedy_selection(i)
                
            t += 1
        
        self.final_generation = t
        print('Optimizing Ending.')
        print('The final solution is:', self.bestSolution, 'at energy:', self.bestEnergy)


if __name__ == '__main__':
    samplers_list = [DESampler(-100.0, 100.0) for x in range(3)]
    loss_in = loss_test

    DE = DESolver(nDim=3, nPop=100, fusing_th=0.1, scale=0.7, CR=0.5, max_generation=50, samplers_list=samplers_list, energy_funtion=loss_in)
    DE.solve()

