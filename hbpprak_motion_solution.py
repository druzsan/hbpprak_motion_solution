#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Solution of the Motion Committee Challenge
# implemented by Raphael Fritz, William Saakyan and Alexander Druz

import logging
logging.disable(logging.INFO)
logging.getLogger('rospy').propagate = False
logging.getLogger('rosout').propagate = False

#import argparse
import os
import sys
import time
#import traceback
import numpy as np

try:
    from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
except ImportError as e:
    print(e)
    raise e

# Helper functions
def numpyArrayToInitStr(array):
    # array: 1- or 2-dimentional numpy array
    # return value: initialization string to create array
    if len(array.shape) == 1:
        string = '['
        for j in range(3):
            string += '['
            for i in range(3):
                string += str(array[j*3 + i]) + ','
            if string[-1] == ',':
                string = string[:-1]
            string += '],'
        if string[-1] == ',':
            string = string[:-1]
        string += ']'
        return string
    elif len(array.shape) == 2:
        string = '['
        for row in array:
            string += '['
            for element in row:
                string += str(element) + ','
            if string[-1] == ',':
                string = string[:-1]
            string += '],'
        if string[-1] == ',':
            string = string[:-1]
        string += ']'
        return string
    else:
        return ''

class MotionChallenge:
    # Constants
    
    # Experiment name
    EXC_NAME = 'ExDHBPPrak_Motion'
    
    # Csv file name to save distance
    CSV_NAME = 'distance.csv'
    
    # Placeholder for the smallest (unreal) distance value
    MIN_VAL = -1000.0
    
    # New transfer function to save distance in csv file
    RECORD_DISTANCE_TF = '''# Imported Python Transfer Function
@nrp.MapRobotSubscriber("ball_distance", Topic("/ball_distance", Float32))
@nrp.MapCSVRecorder("distance_recorder", filename="''' + CSV_NAME + '''", headers=["time", "distance"])
@nrp.Robot2Neuron()
def record_distance(t, ball_distance, distance_recorder):
    distance_recorder.record_entry(t, ball_distance.value.data if ball_distance.value else ''' + str(MIN_VAL) + ''')'''
    
    # Brain code from challenge with ability to set weigths
    BRAIN_TEMPLATE = '''# -*- coding: utf-8 -*-
# pragma: no cover

__author__ = 'Template'

from hbp_nrp_cle.brainsim import simulator as sim
import numpy as np
import logging

logger = logging.getLogger(__name__)

sim.setup(timestep=0.1, min_delay=0.1, max_delay=20.0, threads=1, rng_seeds=[1234])

# Following parameters were taken from the husky braitenberg brain experiment (braitenberg.py)

sensors = sim.Population(3, cellclass=sim.IF_curr_exp())
actors = sim.Population(3, cellclass=sim.IF_curr_exp())

sim.Projection(sensors, actors, sim.AllToAllConnector(), sim.StaticSynapse(weight={syn_weights}))

circuit = sensors + actors'''
    
    def __init__(self):
        # Handlers
        self._vc = None
        self._sim = None
        try:
            self._vc = VirtualCoach(environment='local')
        except:
            self._vc = None
        
        # Flags
        self._done = False
        
        # Data
        self._weights = None
        self._data = np.empty(0)

    def getData(self):
        return np.reshape(self._data, (-1, 10))
    
    def getDataCount(self):
        return self._data.shape[0]/10
    
    def clearData(self):
        self._data = np.empty(0)
    
    def popData(self):
        data = self.getData()
        self.clearData()
        return data
    
    def initialized(self):
        return self._vc != None
    
    def stopSim(self):
        try:
            self._done = True
            self._sim.stop()
            time.sleep(2.0)
        except:
            pass
    
    def receiveDistance(self):
        try:
            csv_data = self._sim.get_csv_data(self.CSV_NAME)
            if not csv_data:
                return self.MIN_VAL
            return float(csv_data[-1][-1])
        except:
            print('Unable to get csv data')
            return self.MIN_VAL
    
    def makeOnStatus(self):
        def onStatus(msg):
            if not self._done:
                print("Current simulation time: {}".format(msg['simulationTime']))
                distance = self.receiveDistance()
                if distance > self.MIN_VAL:
                    self._sim.pause()
                    self._done = True
                    self._data = np.append(self._data, np.append(self._weights, distance))
                    print('Weights:\n' + str(self._weights) + '\nDistance: ' + str(distance))
                    #print('Distance: ' + str(distance))
        return onStatus
    
    def run(self, weights):
        if not self.initialized():
            print('Virtual coach hasn\'t been initialized')
            return
        self._done = False
        
        # Launch experiment
        try:
            self._sim = self._vc.launch_experiment(self.EXC_NAME)
        except:
            print('Unable to launch experiment')
            if self._sim:
                self.stopSim()
            return
        
        # Set experiment
        try:
            self._sim.register_status_callback(self.makeOnStatus())
            self._sim.add_transfer_function(self.RECORD_DISTANCE_TF)
            self._weights = weights
            self._sim.edit_brain(self.BRAIN_TEMPLATE.format(**{'syn_weights': numpyArrayToInitStr(self._weights)}))
        except:
            print('Unable to set callback function, transfer function or brain')
            self._sim.stopSim()
            time.sleep(2.0)
            return
        
        # Start experiment
        try:
            self._sim.start()
        except:
            print('Unable to start simulation')
            self.stopSim()
            time.sleep(2.0)
            return
        
        # Wait until end of experiment and stop it
        while not self._done:
            time.sleep(0.2)
        self.stopSim()
        time.sleep(3.0)

def sort(array):
    # array (numpy array, shape=[n, 10]): population with its score in the last column
    # return value (numpy array, shape=[n, 10]): the same population sorted by the score
    return array[np.argsort(array[:, -1])]

def chooseTheBest(array):
    # array (numpy array, shape=[n, 10]): population with its score in the last column
    # return value ((numpy array, shape=[1, 9]), float): the best exemplar of population and its score
    temp = sort(np.copy(array))[-1]
    return (temp[:-1], temp[-1])

def initPopulation(count=1, scale=5):
    # count (int): size of population
    # scale (float): upper bound for population values
    # return value (numpy array, shape=[count, 9])
    return np.random.rand(count, 9) * scale

def testPopulation(sim, weights):
    # sim (initialized MotionChallenge() class instance):
    # weights (numpy array, shape=[n, 9]): current population
    # return value (numpy array, shape=[n, 10]): the same population with its score
    for i in range(weights.shape[0]):
        print('Experiment ' + str(i + 1) + ' of ' + str(weights.shape[0]))
        while True:
            try:
                sim.run(weights=weights[i])
                break
            except:
                sim.stopSim()
    return sim.popData()

def choosePopulation(array, countBest, countLucky):
    # array (numpy array, shape=[n, 10]): all exemplars from previous population with their score in the last column
    # countBest (int): count of best exemplars to be alived
    # countLucky (int): count of random (lucky) exemplars to be alived
    # return value (numpy array, shape=[countBest + countLucky, 9]): alived exemplars
    temp = sort(np.copy(array))[:, :-1]
    return np.concatenate((temp[temp.shape[0] - countBest:],
                           temp[np.random.choice(temp.shape[0] - countBest, countLucky, replace=False)]))
    
def getNewPopulation(array, countMutations, countRandom, f=0.5):
    # array (numpy array, shape=[n, 9]): alived exemplars from previous population
    # countMutations (int): count of mutations on each alived exemplar (unchanged exemplar will be saved too)
    # countRandom (int): count of full new exemplars in new population
    # f (float between 0.0 and 1.0): mutation factor
    # return value (numpy array, shape=[n*(countMutations + 1) + countRandom, 9]):
    population = np.empty(0)
    size = array.shape[0]
    for i in range(size):
        for _ in xrange(countMutations):
            j = np.random.randint(9)
            a = np.copy(array[i, :])
            a[j] += f*(array[np.random.randint(size), j] - array[np.random.randint(size), j])
            population = np.append(population, a)
    return np.concatenate((array, np.reshape(population, (-1, 9)), initPopulation(countRandom)))

if __name__ == '__main__':
    directory = '/home/nrpuser/HBPSolutions/hbpprak_motion_solution/'
    filename  = 'saved_state.npz'
    path = directory + filename
    if not os.path.isdir(directory):
        print('Target folder doesn\'t exist and will be created')
        os.makedirs(directory)
    if os.path.exists(path):
        print('file exist, reset the last training')
        params = np.load(path)
        (countIterations, curIteration) = (np.int(params['iters']), np.int(params['cur_iter']))
        (countRandom, countBest, countLucky, countMutations) =\
            (np.int(params['rand']), np.int(params['best']), np.int(params['lucky']), np.int(params['mut']))
        (bestWeights, bestDistances, weights) = (params['best_w'], params['best_d'], params['w'])
        
        populationSize = (countBest + countLucky)*(countMutations + 1) + countRandom
        
        sim = MotionChallenge()
    else:
        print('file doesn\'t exist, create new training')
        
        countIterations = np.int(20)
        curIteration    = np.int(0)
        
        countRandom             = np.int(5)
        (countBest, countLucky) = (np.int(8), np.int(2))
        countMutations          = np.int(3)
        
        sim = MotionChallenge()
        
        (bestWeights, bestDistances) = (np.zeros(9), np.array([sim.MIN_VAL]))
        
        populationSize = (countBest + countLucky)*(countMutations + 1) + countRandom
        
        weights = initPopulation(populationSize)
        
        # Save initial state
        np.savez(path, iters=countIterations, cur_iter=curIteration,
                       rand=countRandom, best=countBest, lucky=countLucky, mut=countMutations,
                       best_w=bestWeights, best_d=bestDistances, w=weights)
        
    print('Population size: ' + str(populationSize))
    print('Random exemplars in each population: ' + str(countRandom))
    print('Alived exemplars in each population: ' + str(countBest) + ' best and ' + str(countLucky) + ' lucky')
    print('Multyplying of alived exemplars by mutations: ' + str(countMutations + 1))
    print('Current iteration: ' + str(curIteration + 1) + ' of ' + str(countIterations))
    print('Best distances at the moment: ' + str(bestDistances))
    print('Best exemplar at the moment:' + str(bestWeights))
    #print('Current population:\n' + str(weights))
    
    while curIteration < countIterations:
        print('\nIteration ' + str(curIteration + 1) + ' of ' + str(countIterations) + '\n')
        
        # Test current weights    
        res = testPopulation(sim, weights)
                
        # Update weights and counter
        (curBestWeights, curBestDistance) = chooseTheBest(res)
        if curBestDistance > np.max(bestDistances):
            bestWeights = curBestWeights
        bestDistances = np.append(bestDistances, curBestDistance)
        print('Current best weights: ' + str(curBestWeights) + '\nBest distances: ' + str(bestDistances))
        chosen = choosePopulation(res, countBest, countLucky)
        weights = getNewPopulation(chosen, countMutations, countRandom, f=0.5)
        curIteration += 1
        
        # Save initial state
        np.savez(path, iters=countIterations, cur_iter=curIteration,
                       rand=countRandom, best=countBest, lucky=countLucky, mut=countMutations,
                       best_w=bestWeights, best_d=bestDistances, w=weights)
        
    print('Best weights overall: ' + str(bestWeights) + '\nBest distance overall: ' + str(bestDistances))
