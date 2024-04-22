
from commons import AbstractAgent

class RandomAgent(AbstractAgent):
    def act(self, state, reward=-1000):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        print(1)
        # Randomly select an action from the action space
        return self.action_space.sample()

class FixedAgent(AbstractAgent):
    def act(self, state, reward=0):
        """
        This function represents the actual decision-making process of the agent. Given a 'state' and, possibly, a 'reward'
        the agent returns an action to take in that state.
        :param state: the state on which to act
        :param reward: the reward computed together with the state (i.e. the reward on the previous action). Useful for learning
        :params
        :return:
        """
        print(state)
        #if able to go down, move down
        if state == 1:
            return 1 #down
        #otherwise go right
        else:
            return 2 #right

class monteCarlo():
    def __init__(self, id, action_space):
        self.id = id
        self.action_space = action_space

        # Flag that you can change for distinguishing whether the agent is used for learning or for testing.
        # You may want to disable some behaviour when not learning (e.g. no update rule, no exploration eps = 0, etc.)
        self.learning = True

#SARSA
class TD_onPolicy():
    def __init__(self, id, action_space):
        self.id = id
        self.action_space = action_space

#Q-Learner
class TD_offPolicy():
    def __init__(self, id, action_space):
        self.id = id
        self.action_space = action_space

class dynaQ():
    def __init__(self, id, action_space):
        self.id = id
        self.action_space = action_space
