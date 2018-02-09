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

    

        
        


    
class Depot:
    
    def __init__(self, aktie):        
        self.reward_buy = 0.01   # Prozent
        self.reward_sell = 0.01   # Prozent
        
        self.aktie = aktie
        self.window_size = 60
        self.time_max = len(aktie.close_price_orig) - 1
        self.reset()
        
        self.state_size = len(self.state(self.time))     # Anzahl der Feature eines Zustandes
        self.action_size = 3    # Anzahl der möglichen Aktionen (do nothing, buy, sell)
                  
        
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
    
    
    def buy(self):
        if self.holdings < 1:
                self.holdings += 1
        
                

    def sell(self):
        if self.holdings > 0:
                self.holdings -= 1
        
   
        
    def step(self, action):
        '''
        actions:    
            0 - do nothing  
            1 - buy one  
            2 - sell one
        '''        
        self.reward = 0
        self.profit += self.holdings * (self.aktie.close_price[self.time] - self.aktie.close_price[self.time - 1])

        if action == 1: 
            self.buy()
            self.reward -= self.reward_buy * self.aktie.close_price[self.time]            
        elif action == 2: 
            self.sell()
            self.reward -= self.reward_sell * self.aktie.close_price[self.time]
        
        if self.time == self.time_end - 1:
            self.reward += self.profit
            done = True
        else:
            done = False
        
        self.time += 1
        self.reward += self.holdings * (self.aktie.close_price[self.time] - self.aktie.close_price[self.time - 1])          
        next_state = self.state(self.time)
            
        return next_state, self.reward, done, 'info'





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
    
    
    
    
    
        
        
    
    
    
    

        
    
       
