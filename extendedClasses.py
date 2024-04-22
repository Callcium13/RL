from time import sleep
from typing import Optional
import gym
from gym import spaces
import numpy as np
from commons import AbstractRLTask
import matplotlib.pyplot as plt

class GridWorld(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, n, m):
        super(GridWorld, self).__init__()
        self.n = n
        self.m = m
        self.action_space = spaces.Discrete(4)  # Up, Down, Right, Left
        self.observation_space = spaces.Tuple((spaces.Discrete(n), spaces.Discrete(m)))

        # Agent starts at (0, 0) and goal is at (n-1, m-1)
        self.position = (0, 0)
        self.goal = (n - 1, m - 1)

        # Initialize plot for rendering
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.img = None
    

    #Run one timestep of the environment's dynamics. 
    #When end of episode is reached, you are responsible for calling `reset()` to reset this environment's state.
    def step(self, action):
        # Dictionary mapping actions to movements
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, 1),   # Right
            3: (0, -1)   # Left
        }

        # Perform action
        movement = actions[action]
        new_position = (self.position[0] + movement[0], self.position[1] + movement[1])

        # Check if new position is within grid
        if 0 <= new_position[0] < self.n and 0 <= new_position[1] < self.m:
            self.position = new_position
           
        #observation (object): agent's observation of the current environment    
        observation = [self.position,self.goal]
        
        #reward (float) : amount of reward returned after previous action
        reward = -1
        
        #done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        done = self.position == self.goal
        
        #info (dict): contains auxiliary diagnostic information (helpful for debugging, logging, and sometimes learning)
        info = []

        return (observation, reward, done, info)

    # #Resets the environment to an initial state and returns an initial observation.
    # def reset(self,*, seed: Optional[int] = None, return_info: bool = False, 
    #           options: Optional[dict] = None):
    #     """
    #     This method should also reset the environment's random number
    #     generator(s) if `seed` is an integer or if the environment has not
    #     yet initialized a random number generator. If the environment already
    #     has a random number generator and `reset` is called with `seed=None`,
    #     the RNG should not be reset.
    #     Moreover, `reset` should (in the typical use case) be called with an
    #     integer seed right after initialization and then never again.
    #     """

    #     # Initialize the RNG if the seed is manually passed
    #     #if seed is not None:
    #     #    self._np_random, seed = seeding.np_random(seed)

    #     observation = []

    #     #info (optional dictionary): a dictionary containing extra information
    #     #this is only returned if return_info is set to true
    #     info = []

    #     if return_info:
    #         return observation, info
    #     return observation

    def render(self, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render_modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        if mode == 'ansi':
            ansi_grid = ""
            for i in range(self.n):
                for j in range(self.m):
                    if (i, j) == self.position:
                        ansi_grid += "A"  # Agent
                    elif (i, j) == self.goal:
                        ansi_grid += "G"  # Goal
                    else:
                        ansi_grid += "·"
                    ansi_grid += " "
                ansi_grid += "\n"

            return ansi_grid

        if mode == 'rgb_array':
            # Render the grid world as an RGB array
            rgb_array = np.zeros((self.n, self.m, 3), dtype=np.uint8)
            rgb_array[self.position] = [255, 0, 0]  # Red color for agent
            rgb_array[self.goal] = [0, 255, 0]   # Green color for goal

            return rgb_array
        
        elif mode == 'human':
            # Render the grid world
            grid = np.full((self.n, self.m), '·', dtype=str)  # Initialize with '0's
            grid[self.position] = 'A'  # Place agent
            grid[self.goal] = 'G'    # Place goal

            for row in grid:
                print(" ".join(row))
        
        elif mode == 'human2':
            # Render the grid world
            grid = np.zeros((self.n, self.m), dtype=float)  # Initialize with zeros
            grid[self.position] = 0.5  # Agent
            grid[self.goal] = 1.0    # Goal

            # Update plot
            if self.img is None:
                self.img = self.ax.imshow(grid, cmap='viridis', interpolation='nearest')
                self.ax.axis('off')
                self.fig.canvas.draw()
            else:
                self.img.set_data(grid)
                self.fig.canvas.flush_events()
        else:
            super(GridWorld, self).render(mode=mode) # just raise an exception

class RLTask(AbstractRLTask):
    def __init__(self, env, agent):
        """
        This class abstracts the concept of an agent interacting with an environment.


        :param env: the environment to interact with (e.g. a gym.Env)
        :param agent: the interacting agent
        """

        self.env = env
        self.agent = agent

    def interact(self, n_episodes):
        """
        This function executes n_episodes of interaction between the agent and the environment.

        :param n_episodes: the number of episodes of the interaction
        :return: a list of episode average returns  (see assignment for a definition
        """
        episode_returns = []  # Store episode returns

        for _ in range(n_episodes):
            episode_return = 0  # Initialize episode return
            obs = self.env.reset()  # Reset the environment
            done = False

            while not done:
                action = self.agent.act(obs)  # Get action from the agent
                next_obs, reward, done, _ = self.env.step(action)  # Take action in the environment
                episode_return += reward  # Accumulate reward for the episode
                obs = next_obs  # Update observation for the next step

            episode_returns.append(episode_return)  # Store episode return

        return episode_returns


    def visualize_episode(self, max_number_steps = None):
        """
        This function executes and plot an episode (or a fixed number 'max_number_steps' steps).
        You may want to disable some agent behaviours when visualizing(e.g. self.agent.learning = False)
        :param max_number_steps: Optional, maximum number of steps to plot.
        :return:
        """
        obs = self.env.reset()  # Reset the environment
        done = False
        steps = 0

        while not done and (max_number_steps is None or steps < max_number_steps):
            action = self.agent.act(obs,Grid5x5.position)  # Get action from the agent
            obs, _, done, _ = self.env.step(action)  # Take action in the environment
            print('\n\n\n\n\n\n\n\n')
            self.env.render()  # Render the environment
            sleep(0.01)
            steps += 1

        # Reset agent learning status after visualization
        self.agent.learning = True
    
