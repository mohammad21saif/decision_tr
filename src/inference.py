from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import pickle

from model.decision_transformer import DecisionTransformer



class MakeAnime:
    def __init__(self, max_episode_length, num_robot, target_return, grid_size, K) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = '/home/moonlab/decision_transformer/decision_tr/saved_models/trained_model.pt'
        self.rewardmap_path = '/home/moonlab/decision_transformer/decision_tr/maps/gaussian_mixture_training_data.pkl'

        self.target_return = target_return
        self.num_robot = num_robot
        self.max_episode_length = max_episode_length
        self.grid_size = grid_size

        self.state_dim = 2 * num_robot
        self.act_dim = 2 * num_robot
        self.max_length = K

        self.states = torch.empty((1, self.max_episode_length, self.num_robot*2), dtype=torch.float32).to(self.device)
        self.actions = torch.empty((1, self.max_episode_length, self.num_robot*2), dtype=torch.float32).to(self.device)
        self.rewards = torch.empty((1, self.max_episode_length, 1), dtype=torch.float32).to(self.device)
        self.rtg = torch.empty((1, self.max_episode_length, 1), dtype=torch.float32).to(self.device)
        self.timesteps = torch.empty((1, self.max_episode_length), dtype=torch.long).to(self.device)

        self.fig, self.ax = plt.subplots()
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']



    def get_actions(self, model, states, actions, rewards, returns_to_go, timesteps):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = model.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask)

        return action_preds[0,-1]
        


    def get_frames(self):
        model = torch.load(self.model_path).to(self.device)
        model.eval()
        rewardmap = torch.tensor(pickle.load(open(self.rewardmap_path, "rb"), encoding="latin1")).to(self.device)

        
        state = torch.randint(low=0, high=self.grid_size, size=(self.num_robot*2,)).to(self.device)
        action = torch.zeros((self.num_robot*2,)).to(self.device)
        reward = rewardmap[state[0::2].to(int), state[1::2].to(int)].sum()
        target_return = torch.tensor(self.target_return).to(self.device)
        timestep = torch.tensor(0).reshape(1, 1).to(self.device)
        
        self.states[0] = state
        self.actions[0] = action
        self.rewards[0] = reward
        self.timesteps[0][0] = timestep
        # print(state, action, reward, target_return, timestep)

        episode_return = reward
        for i in range(1, self.max_episode_length):
            action = self.get_actions(model, state, action, reward, target_return, timestep)
            self.states[0][i] = torch.add(input=self.states[0][i-1], other=action)
            self.actions[0][i] = action
            self.timesteps[0][i] = i
            self.rewards[0][i] = rewardmap[self.states[0][i][0::2].to(int), self.states[0][i][1::2].to(int)].sum()
            # episode_return += self.rewards[i]
        # print("States: ", self.states, "Actions: ", self.actions, "Rewards: ", self.rewards, "Timesteps: ", self.timesteps)

        return


    def update_frame(self, i):
        for j in range(self.num_robot):
            self.ax.plot
            self.ax.plot(self.states[0, :i, j*2].cpu().detach().detach().numpy(), self.states[0, :i, j*2+1].cpu().detach().numpy(), color=self.colors[j], marker='o', markersize=5)
        return


                

    def anime(self):
        self.get_frames()
        map = torch.tensor(pickle.load(open(self.rewardmap_path, "rb"), encoding="latin1")).to(self.device)
        self.ax.imshow(map.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
        anim = animation.FuncAnimation(fig=self.fig, func=self.update_frame, interval=30, frames=self.max_episode_length)
        # plt.show()
        anim.save(filename='/home/moonlab/decision_transformer/decision_tr/res/videos/trajectory_video.mp4', writer='ffmpeg', fps=30)

    
