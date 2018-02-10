from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import numpy as np
import csv

class Aktie:
    def __init__(self, csv_filename):
        o, c, l, h = init_aktie(csv_filename)       
        window = 25    # SMA window
        self.open_price_orig = o[window - 1:]
        self.close_price_orig = c[window - 1:]
        self.high_orig = h[window - 1:]
        self.low_orig = l[window - 1:]
        self.sma_orig = self.movingaverage(c, window)
        #self.time = np.arange(len(self.low))


    # simple moving avarage
    def movingaverage(self, values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma


    def oclh(self, time):
        return [self.open_price[time], self.close_price[time], \
                self.low[time], self.high[time]]


    def scale(self, factor):
        self.open_price = self.open_price_orig * factor
        self.close_price = self.close_price_orig * factor
        self.high = self.high_orig * factor
        self.low = self.low_orig * factor
        self.sma = self.sma_orig * factor 
        return None




class depot_env(object):
    """
    Base environment class.
    """

    def __init__(self, aktie):        
        self.reward_buy = 0.01   # Prozent
        self.reward_sell = 0.01   # Prozent
        
        self.aktie = aktie
        self.window_size = 10
        self.time_max = len(aktie.close_price_orig) - 1
        self.reset()
        
        self.state_size = len(self.state(self.time))     # Anzahl der Feature eines Zustandes
        self.action_size = 3    # Anzahl der möglichen Aktionen (do nothing, buy, sell)


    def __str__(self):
        return 'Depot_Environment'


    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """
        pass


    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments (e.g. ALE or some gym Envs) don't have to implement this method.
        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """
        return None


    def reset(self):
        self.time_start = np.random.randint(0, self.time_max - self.window_size)
        self.time_end = self.time_start + self.window_size - 1
        self.time = self.time_start  # set start time  #self.time_start
        
        self.reward = 0
        self.holdings = 0
        self.profit = 0

        # scale prices by 1/price(time_start)
        factor = 1 / self.aktie.close_price_orig[self.time_start]
        self.aktie.scale(factor)
        
        return self.state(self.time)


    def state(self, time):
        candles = []
        for t in range(4):
            candles += self.aktie.oclh(time - t)
        return np.array([self.holdings, self.aktie.sma[time]] + candles)


    def execute(self, actions):
        """
        Executes action, observes next state(s) and reward.
        Args:
            actions: Actions to execute.
        Returns:
            (Dict of) next state(s), boolean indicating terminal, and reward signal.
        """
        '''
        actions:    
            0 - do nothing  
            1 - buy one  
            2 - sell one
        '''        
        self.reward = 0
        self.profit += self.holdings * (self.aktie.close_price[self.time] - self.aktie.close_price[self.time - 1])

        if actions == 1: 
            if self.holdings < 1:
                self.holdings += 1
            self.reward -= self.reward_buy * self.aktie.close_price[self.time]            
        elif actions == 2: 
            if self.holdings > 0:
                self.holdings -= 1
            self.reward -= self.reward_sell * self.aktie.close_price[self.time]
        
        if self.time == self.time_end - 1:
            self.reward += self.profit
            done = True
        else:
            done = False
        
        self.time += 1
        self.reward += self.holdings * (self.aktie.close_price[self.time] - self.aktie.close_price[self.time - 1])          
        next_state = self.state(self.time)
            
        return next_state, done, self.reward


    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are available simultaneously.
        Returns: dict of state properties (shape and type).
        """
        return dict(type='float', shape=(self.state_size,))


    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are available simultaneously.
        Returns: dict of action properties (continuous, number of actions)
        """
        return dict(type='int', num_actions=self.action_size)
        
        
        
def init_aktie(csv_filename):           
    with open(csv_filename, 'r') as file:    
        reader = csv.DictReader(file)       
        data = {}    
        for row in reader:
            for header, value in row.items():
              try:
                data[header].append(value)
              except KeyError:
                data[header] = [value]
        
    open_price = np.flipud(np.array([float(i) for i in data['Open']]))
    close_price = np.flipud(np.array([float(i) for i in data['Close']]))
    high = np.flipud(np.array([float(i) for i in data['High']]))
    low = np.flipud(np.array([float(i) for i in data['Low']]))
       
    return open_price, close_price, low, high           