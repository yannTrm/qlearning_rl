# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from .agent_learner import QLearningAgent, SarsaAgent
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class Trainer:
    def __init__(self, env, agent):
        """
        Trainer for Reinforcement Learning.

        Parameters:
        - environment: An instance of EnvironmentManager
        - agent: The reinforcement learning agent
        """
        self.env = env
        self.agent = agent
        self.rewards = []
       

    def train(self, num_epochs=1000):
        """
        Train the reinforcement learning agent.

        Parameters:
        - num_epochs: The number of training epochs
        """
        for epoch in range(num_epochs):
            total_reward = self.play_and_train()
            self.rewards.append(total_reward)
            self.agent.epsilon *= 0.99
            self.log_training_progress(epoch, total_reward)

            if epoch % 100 == 0:
                self.display_training_progress(epoch)
    def play_and_train(self, t_max=10**4):
        """
        Execute a complete game, with actions determined by the agent's policy.
        Train the agent using agent.update(...) whenever applicable.

        Returns:
        - total_reward: The total reward obtained during the game
        """
        total_reward = 0.0
        state, _ = self.env.reset()

        for t in range(t_max):

            action = self.agent.get_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
       

            # Check the type of agent and call the appropriate update method
            if type(self.agent) is QLearningAgent:
                self.agent.update(state, action, reward, next_state)
            elif type(self.agent) is SarsaAgent:
                next_action = self.agent.get_action(next_state)
                self.agent.update(state, action, reward, next_state, next_action)

            state = next_state
            total_reward += reward

            if done:
                break

        return total_reward

    def log_training_progress(self, epoch, total_reward):
        """
        Log the training progress.

        Parameters:
        - epoch: The current training epoch
        - total_reward: The total reward obtained during the epoch
        """
        print(f"Epoch: {epoch}, Total Reward: {total_reward}")
        
    def display_training_progress(self, epoch):
        clear_output(True)
        plt.title('eps = {:e}, mean reward = {:.1f}'.format(self.agent.epsilon, np.mean(self.rewards[-10:])))
        plt.plot(self.rewards)
        plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------