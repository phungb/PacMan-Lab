# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        #self.setNoise(0)

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates();

        # iterate over a certain amount of times
        for i in range(0, self.iterations):

            # old values used to compute new values
            old = self.values.copy()

            # review every state
            for state in states:
                # Assign new value for each state
                # new value is max of possible actions

                actions = self.mdp.getPossibleActions(state);
                vals = []

                # "value" of action is expectimax of its possible states
                for action in actions:
                    probs = self.mdp.getTransitionStatesAndProbs(state, action)
                    expectimax = [(prob * (self.mdp.getReward(state, action, nextState) + self.discount * old[nextState]))
                            for nextState, prob in probs]
                    vals.append(sum(expectimax))

                if not len(vals) == 0:
                    self.values[state] = max(vals)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        probs = self.mdp.getTransitionStatesAndProbs(state, action)
        expectimax = [(prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState)))
                for nextState, prob in probs]
        return sum(expectimax)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        vals = [self.computeQValueFromValues(state, action) for action in actions]
        if len(vals) == 0:
            return None
        bestVal = max(vals)
        bestIndices = [index for index in range(len(vals)) if vals[index] == bestVal]

        return actions[bestIndices[0]]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
