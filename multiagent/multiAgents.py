# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        total = successorGameState.getScore()

        for state in newGhostStates:
            ghostPos = state.getPosition();

            # if loss then that state sucks
            if newPos == ghostPos:
                return -100000

            # closer the ghost the worse the state is
            dist = util.manhattanDistance(newPos, ghostPos)
            total = total + dist

        # closer food is good
        newFood = newFood.asList()

        distances = []
        for food in newFood:
            dist = util.manhattanDistance(newPos, food)
            distances.append(dist)

        if not distances == []:
            total = total - min(distances)

        return total

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        deep = self.depth * numAgents

        legalMoves = gameState.getLegalActions(0)
        scores = [self.minimax(gameState.generateSuccessor(0, action),
                deep - 1, numAgents, 1 % numAgents) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def minimax(self, gameState, remaining, numAgents, turn):
        # if game is over or depth is reached
        # return the state's utility
        if gameState.isWin() or gameState.isLose() or remaining == 0:
            return self.evaluationFunction(gameState);

        legalMoves = gameState.getLegalActions(turn)

        scores = [self.minimax(gameState.generateSuccessor(turn , action), remaining - 1,
                numAgents, (turn + 1) % numAgents)
                for action in legalMoves]

        if turn == 0:
            return max(scores)
        else:
            return min(scores)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        numAgents = gameState.getNumAgents()
        deep = self.depth * numAgents

        scores = []
        legalMoves = gameState.getLegalActions(0)
        alpha = float("-inf")
        beta = float("inf")
        for action in legalMoves:
            scores.append(self.alphabeta(gameState.generateSuccessor(0 , action), deep - 1,
                    numAgents, 1 % numAgents, alpha,  beta))
            alpha = max(scores)

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def alphabeta(self, gameState, remaining, numAgents, turn, alpha, beta):
        # if game is over or depth is reached
        # return the state's utility
        if gameState.isWin() or gameState.isLose() or remaining == 0 or alpha > beta:
            return self.evaluationFunction(gameState);

        legalMoves = gameState.getLegalActions(turn)

        if turn == 0:
            v = float("-inf")
            for action in legalMoves:
                newV = self.alphabeta(gameState.generateSuccessor(turn , action), remaining - 1,
                        numAgents, (turn + 1) % numAgents, alpha, beta)
                #print newV
                v = max(v, newV)

                if v > beta:
                    return v

                alpha = max(v, alpha)
        else:
            v = float("inf")
            for action in legalMoves:
                newV = self.alphabeta(gameState.generateSuccessor(turn , action), remaining - 1,
                        numAgents, (turn + 1) % numAgents, alpha, beta)
                #print newV
                v = min(v, newV)

                if v < alpha:
                    return v

                beta = min(v, beta)

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        deep = self.depth * numAgents

        legalMoves = gameState.getLegalActions(0)
        scores = [self.expectimax(gameState.generateSuccessor(0, action),
                deep - 1, numAgents, 1 % numAgents) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def expectimax(self, gameState, remaining, numAgents, turn):
        # if game is over or depth is reached
        # return the state's utility
        if gameState.isWin() or gameState.isLose() or remaining == 0:
            return self.evaluationFunction(gameState);

        legalMoves = gameState.getLegalActions(turn)

        if turn == 0:
            scores = [self.expectimax(gameState.generateSuccessor(turn , action), remaining - 1,
                    numAgents, (turn + 1) % numAgents)
                    for action in legalMoves]

            return max(scores)
        else:
            scores = [self.expectimax(gameState.generateSuccessor(turn , action), remaining - 1,
                    numAgents, (turn + 1) % numAgents)
                    for action in legalMoves]

            return sum(scores) / len(legalMoves)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    total = currentGameState.getScore()

    # farther away from ghosts is better
    for state in newGhostStates:
        ghostPos = state.getPosition();
        dist = util.manhattanDistance(newPos, ghostPos)
        total = total + dist ** 0.5

    # get distances from food
    newFood = newFood.asList()
    distances = []
    for food in newFood:
        dist = util.manhattanDistance(newPos, food)
        distances.append(dist)

    # closer food is better
    if not distances == []:
        total = total - min(distances)

    return total

# Abbreviation
better = betterEvaluationFunction
