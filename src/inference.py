from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from model.decision_transformer import DecisionTransformer



class MakeAnime:
    def __init__(self, max_episode_length, num_robot, target_return, grid_size) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = '/home/moonlab/decision_transformer/decision_tr/saved_models/trained_model.pt'
        self.model = load_from_disk(self.model_path).to(self.device)

        self.target_return = target_return
        self.num_robot = num_robot
        self.max_episode_length = max_episode_length
        self.grid_size = grid_size



    def get_actions():
        pass

    def get_frame():
        pass

    def update_frame():
        pass

    def anime():
        pass

    
