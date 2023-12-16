# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class DisplayManager:
    @staticmethod
    def show_state(environment):
        """
        Display the current state of the environment.

        Parameters:
        - environment: The current environment state
        """
        plt.imshow(environment.render())
        plt.axis('off')
        display(plt.gcf())
        clear_output(wait=True)
        time.sleep(0.1)  # Adjust the sleep time as needed

    @staticmethod
    def display_epoch(environment, agent):
        """
        Display the state of the environment for a specific epoch.

        Parameters:
        - environment: The environment instance
        - agent: The reinforcement learning agent
        """
        state, _ = environment.reset()
        done = False
        DisplayManager.show_state(environment)

        while not done:
            action = agent.get_action(state)
            next_state, _, done, _, _ = environment.step(action)
            state = next_state
            DisplayManager.show_state(environment)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------