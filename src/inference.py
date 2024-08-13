from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from model.decision_transformer import DecisionTransformer



class MakeAnime:
    def __init__(self, max_episode_length, num_robot, target_return, grid_size, K) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = '/home/moonlab/decision_transformer/decision_tr/saved_models/trained_model.pt'

        self.target_return = target_return
        self.num_robot = num_robot
        self.max_episode_length = max_episode_length
        self.grid_size = grid_size

        self.state_dim = 2 * num_robot
        self.act_dim = 2 * num_robot
        self.max_length = K



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

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask)

        return action_preds[0,-1]
        

    def get_frames(self):
        model = torch.load(self.model_path).to(self.device)

        states = torch.zeros((self.max_episode_length, self.num_robot*2), dtype=torch.float32).to(self.device)
        actions = torch.zeros((self.max_episode_length, self.num_robot*2), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((self.max_episode_length, 1), dtype=torch.float32).to(self.device)
        timesteps = torch.zeros((self.max_episode_length), dtype=torch.long).to(self.device)
        target_return = torch.tensor(self.target_return, dtype=torch.float32).reshape(1,1).to(self.device)


        self.get_actions(model, states, actions, rewards, target_return, timesteps)

    def update_frame():
        pass

    def anime():
        pass

    
