# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from src.agent_learner import QLearningAgent
from src.trainer import Trainer
from src.utils import DisplayManager

import gym

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__=="__main__":
    
    env = gym.make('CliffWalking-v0', render_mode ='rgb_array')
    n_actions = env.action_space.n
    
    
    agent = QLearningAgent(
        alpha=0.5, epsilon=0.25, discount=0.99,
        get_legal_actions=lambda s: range(n_actions))
    trainer = Trainer(env, agent)
    
    trainer.train()
    
    plt.figure()
    DisplayManager.display_epoch(env, agent)
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------