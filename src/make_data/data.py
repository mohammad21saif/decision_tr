import torch
import pickle
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm



class RandomSampler:
    '''
    Randomly samples trajectories from the environment and stores them in a dataset.

    Args:
    - rewardmap_path: path to the rewardmap file
    - grid_size: size of the map, grid_size x grid_size. Here it is 30 x 30
    - num_robot: number of robots in the environment
    - T: total number of timesteps in each trajectory
    - num_traj: number of trajectories to sample

    Returns:
    - dataset_dict: dataset containing the states, actions, rewards, returns_to_go, and timesteps of the sampled trajectories
    '''

    def __init__(self, rewardmap_path, grid_size, num_robot, T, num_traj, device) -> None:
        '''
        Initializes the RandomSampler object.

        Args:
        - rewardmap_path: path to the rewardmap file.
        - grid_size: size of the map, grid_size x grid_size. Here it is 30 x 30.
        - num_robot: number of robots in the environment.
        - T: total number of timesteps in each trajectory.
        - num_traj: number of trajectories to sample.
        
        Returns:
        - None
        '''

        self.device = device

        self.rewardmap = torch.tensor(pickle.load(open(rewardmap_path, "rb"), encoding="latin1")).to(self.device)
        self.grid_size = grid_size
        self.num_robot = num_robot
        self.T = T
        self.num_traj = num_traj
        
        #Initialize tensors to store
        self.states = torch.empty((self.num_traj, self.T, self.num_robot*2), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((self.num_traj, self.T, self.num_robot*2), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((self.num_traj, self.T, 1), dtype=torch.float32, device=self.device)
        self.rtg = torch.empty((self.num_traj, self.T, 1), dtype=torch.float32, device=self.device)
        self.timesteps = torch.empty((self.num_traj, self.T), dtype=torch.long, device=self.device)
        # self.visited = torch.empty((self.num_traj, self.T, self.num_robot*2), dtype=torch.float32).to(self.device)



    def give_direc(self, state, num_robot):
        '''
        Randomly selects a direction for each robot to move in.

        Args:
        - state: current state of the environment.
        - num_robot: number of robots in the environment.

        Returns:
        - action: action to be taken by each robot.
        - state_ret: new state of the environment after taking the action.
        '''

        while True:
            valid_direc = torch.tensor([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]], 
                                       dtype=torch.float32).to(self.device)
            indices = torch.randperm(valid_direc.size(0))[:num_robot] # randomly selecting a direction
            selected_direc = valid_direc[indices]
            action = selected_direc.reshape(-1)
            state_ret = torch.add(state, action)
            
            # Check if new state is within bounds
            if (state_ret.min() >= 0) and (state_ret[0::2].max() < self.grid_size) and (state_ret[1::2].max() < self.grid_size):
                
                return action, state_ret



    def sample(self):
        '''
        Samples trajectories from the environment.

        Args:
        - None

        Returns:
        - states: states of the sampled trajectories.
        - actions: actions taken in the sampled trajectories.
        - rewards: rewards obtained in the sampled trajectories.
        - rtg: returns to go in the sampled trajectories.
        - timesteps: timesteps in the sampled trajectories.
        '''

        for i in tqdm(range(0, self.num_traj), desc="Sampling Trajectories", ncols=100, unit="traj"):

            # 0th timestep of each trajectory initialized.
            self.states[i][0] = torch.randint(0, self.grid_size, (self.num_robot*2,)).to(self.device)
            # self.visited[i][0] = self.states[i][0]
            self.rewards[i][0] = self.rewardmap[self.states[0][0][0::2].to(int), self.states[0][0][1::2].to(int)].sum()
            self.rtg[i][0] = self.rewards[i][0]
            self.timesteps[i][0] = 0
            self.actions[i][0] = torch.zeros((self.num_robot*2,), dtype=torch.float32).to(self.device)

            for t in range(1, self.T):
                # action, state = torch.empty((self.num_robot*2,), dtype=torch.float32).to(self.device), torch.empty((self.num_robot*2,), dtype=torch.float32).to(self.device)
                action, state = self.give_direc(self.states[i][t-1], self.num_robot)
                self.states[i][t] = state
                self.actions[i][t] = action
                # self.visited[i][t] = self.states[i][t]
                self.rewards[i][t] = self.rewardmap[self.states[i][t][0::2].to(int), self.states[i][t][1::2].to(int)].sum()
                self.rtg[i][t] = torch.add(input=self.rtg[i][t-1], other=self.rewards[i][t])
                self.timesteps[i][t] = t
            self.rtg[i] = torch.flip(self.rtg[i], dims=[0])
        

        return self.states, self.actions, self.rewards, self.rtg, self.timesteps



    def make_data(self):
        '''
        Creates a dataset containing the states, actions, rewards, returns_to_go, and timesteps of the sampled trajectories.

        Args:
        - None

        Returns:
        - dataset_dict: dataset containing the states, actions, rewards, returns_to_go, and timesteps of the sampled trajectories.
        '''

        states, actions, rewards, returns_to_go, timesteps = self.sample()
        # states, actions, rewards, returns_to_go, timesteps = self.copy_every_tenth(states, actions, rewards, returns_to_go, timesteps)
        train_data = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps
        }
        train_dataset = Dataset.from_dict(train_data)
        dataset_dict = DatasetDict({"train": train_dataset})

        return dataset_dict




