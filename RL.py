'''
import env_depot

aktie = env_depot.Aktie('google_stock_data.csv')        
env = env_depot.Depot(aktie)    
state_size = env.state_size
action_size = env.action_size
'''
import numpy as np
import env_depot_tensorforce
from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner


    



def main():
    max_episodes = None
    max_timesteps = None
    
    aktie = env_depot_tensorforce.Aktie('google_stock_data.csv')        
    env = env_depot_tensorforce.depot_env(aktie)
    states_spec = env.states
    actions_spec = env.actions
    


    network_spec=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ]

    agent = DQNAgent(
        states_spec=states_spec,
        actions_spec=actions_spec,
        network_spec=network_spec,
        batch_size=64
    )

    runner = Runner(agent, env)
    
    report_episodes = 10

    
    def episode_finished(r):
        if r.episode % report_episodes == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 10 rewards: {}".format(np.mean(r.episode_rewards[-10:])))
        return True
    


    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(max_episodes, max_timesteps, episode_finished=episode_finished)
    

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))



if __name__ == '__main__':
    main()