from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import time

import env_depot_tensorforce

from tensorforce.agents import Agent
from tensorforce.execution import Runner


    



def main():
    # Agent and Network
    agent_config_filename = './configs/dqn.json'
    network_spec_filename = './configs/dense_network.json'

    # learning parameters
    episodes = 100
    max_episode_timesteps = None
    timesteps = None
    report_episodes = 10

    # Create Environment
    aktie = env_depot_tensorforce.Aktie('google_stock_data.csv')        
    environment = env_depot_tensorforce.depot_env(aktie)


    # logging settings
    logging.basicConfig(filename="test_01.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    
    
    # agent_config
    with open(agent_config_filename, 'r') as fp:
        agent_config = json.load(fp=fp)
        
        
    # network_spec
    with open(network_spec_filename, 'r') as fp:
        network_spec = json.load(fp=fp)


    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec
        )
    )


    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )



    logger.info("Starting {agent} for Environment '{env}' \n".format(agent=agent, env=environment))

    def episode_finished(r):
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {} after {} timesteps. Steps Per Second {}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            logger.info("Average of last 100 rewards: {}\n".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))
        return True



    runner.run(
        timesteps=timesteps,
        episodes=episodes,
        max_episode_timesteps=max_episode_timesteps,
        deterministic=False,
        episode_finished=episode_finished
    )


    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))





if __name__ == '__main__':
    main()