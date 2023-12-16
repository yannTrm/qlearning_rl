# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import numpy as np
from collections import defaultdict
import random

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent.

        Parameters:
        - alpha: Learning rate
        - epsilon: Exploration probability
        - discount: Discount factor
        - get_legal_actions: Function to get legal actions for a given state
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))


    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value



    def get_value(self, state):
        """
       Calculate your agent's estimation of V(s) using the current q-values:

        \[ V(s) = \max_{\text{over\_action}} Q(\text{state, action}) \]

        Note: Consider that q-values may be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        value = max(self.get_qvalue(state, action) for action in possible_actions)

        return value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        current_qvalue = self.get_qvalue(state, action)
        next_value = self.get_value(next_state)
        updated_qvalue = (1 - learning_rate) * current_qvalue + learning_rate * (reward + gamma * next_value)

        self.set_qvalue(state, action, updated_qvalue)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        best_action =  max(possible_actions, key=lambda action: self.get_qvalue(state, action))
        return best_action

    def get_action(self, state):
        """
        Determine the action to take in the current state, incorporating exploration.
        With a probability of self.epsilon, choose a random action; otherwise, select the best policy action using self.get_best_action.

        Note: For randomly selecting from a list, use random.choice(list).
        To generate a True or False value based on a given probability,
        generate a uniform number in the range [0, 1] and compare it with the specified probability.
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        if random.random() < epsilon:
          chosen_action = random.choice(possible_actions)
        else :
          chosen_action = self.get_best_action(state)

        return chosen_action

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class SarsaAgent(QLearningAgent):
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        SARSA (State-Action-Reward-State-Action) Learning Agent.

        Parameters:
        - alpha: Learning rate
        - epsilon: Exploration probability
        - discount: Discount factor
        - get_legal_actions: Function to get legal actions for a given state
        """
        super(SarsaAgent, self).__init__(alpha, epsilon, discount, get_legal_actions)

    def update(self, state, action, reward, next_state, next_action):
        """
        SARSA (State-Action-Reward-State-Action) Q-Value update:

        Q(s, a) := Q(s, a) + alpha * (r + gamma * Q(s', a') - Q(s, a))
        """
        current_qvalue = self.get_qvalue(state, action)
        next_qvalue = self.get_qvalue(next_state, next_action)
        updated_qvalue = current_qvalue + self.alpha * (reward + self.discount * next_qvalue - current_qvalue)
        self.set_qvalue(state, action, updated_qvalue)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

