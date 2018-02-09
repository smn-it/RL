'''
import env_depot

aktie = env_depot.Aktie('google_stock_data.csv')        
env = env_depot.Depot(aktie)    
state_size = env.state_size
action_size = env.action_size
'''
import numpy as np

from tensorforce import Configuration