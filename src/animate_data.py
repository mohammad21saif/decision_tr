import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import argparse
from datasets import load_from_disk
import torch
import pickle
from datetime import datetime
import tqdm




class AnimateData:
    def __init__(self, vars):
        self.data_path = vars['data_path']
        self.rewardmap_path = vars['rewardmap_path']
        self.num_traj = vars['num_traj']
        self.T = vars['T']
        self.num_robot = vars['num_robot']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = load_from_disk(self.data_path)

        self.state = torch.empty((1, self.T, self.num_robot * 2), dtype=torch.float32).to(self.device)
        
        self.fig, self.ax = plt.subplots()
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    


    def get_states(self, traj):
        states = self.dataset['train'][traj]['states']
        states = torch.tensor(states).to(self.device)
        return states



    def update_frame(self, frame_number):   
        for j in range(self.num_robot):
            self.ax.plot(self.state[frame_number, 2 * j].cpu().detach().numpy(),
                         self.state[frame_number, 2 * j + 1].cpu().detach().numpy(),
                         'o', color=self.colors[j], markersize=5)
            map = torch.tensor(pickle.load(open(self.rewardmap_path, "rb"), encoding="latin1")).to(self.device)
            self.ax.imshow(map.cpu().detach().numpy(), cmap='viridis', interpolation='nearest')
        self.ax.set_xlim(0, 30)
        self.ax.set_ylim(0, 30) 

        return 
    


    def animate(self):
        
        # for traj in range(self.num_traj//50):
        for traj in tqdm(range(self.num_traj//50), desc='Animating Data', unit=traj):
            self.state = self.get_states(traj)
            
            anim = animation.FuncAnimation(fig=self.fig, func=self.update_frame, frames=self.T, interval=30)

            current_time = datetime.now().strftime("%Y%m%d__%H%M%S")
            trajectory_num = traj + 1

            filename = f'/home/moonlab/decision_transformer/decision_tr/res/videos/trajectory_video_{current_time}_{trajectory_num}.mp4'
            anim.save(filename=filename, writer='ffmpeg', fps=30)
        return
    
    
    def plot_states(self):
        for traj in tqdm(range(self.num_traj//50), desc='Plotting Data', unit=traj):
            self.state = self.get_states(traj)
            for j in range(self.num_robot):
                self.ax.plot(self.state[:, j*2].cpu().detach().numpy(), self.state[:, j*2+1].cpu().detach().numpy(), color=self.colors[j], marker='o', markersize=5)
                self.ax.set_xlim(0, 30)
                self.ax.set_ylim(0, 30)

                #save the plot
                current_time = datetime.now().strftime("%Y%m%d__%H%M%S")
                trajectory_num = traj + 1
                filename = f'/home/moonlab/decision_transformer/decision_tr/res/images/trajectory_plot_{current_time}_{trajectory_num}.png'
            plt.savefig(filename)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/moonlab/decision_transformer/data/')
    parser.add_argument('--rewardmap_path', type=str, default="/home/moonlab/decision_transformer/decision_tr/maps/gaussian_mixture_training_data.pkl")
    parser.add_argument('--num_traj', type=int)
    parser.add_argument('--T', type=int)
    parser.add_argument('--num_robot', type=int)

    args = parser.parse_args()
    animate_data = AnimateData(vars(args))
    animate_data.plot_states()


if __name__ == "__main__":
    main()
