from data import RandomSampler
import argparse
import torch


def generate(variant):
    rewardmap_path = variant['rewardmap_path']
    grid_size = variant['grid_size']
    num_robot = variant['num_robot']
    T = variant['T']
    num_traj = variant['num_traj']

    device = torch.device("cpu")
    
    data = RandomSampler(rewardmap_path, grid_size, num_robot, T, num_traj, device=device).make_data()
    data.save_to_disk("/home/moonlab/decision_transformer/data/")



def main():

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--rewardmap_path', type=str, default="/home/moonlab/decision_transformer/decision_tr/maps/gaussian_mixture_training_data.pkl")
    parser.add_argument('--grid_size', type=int, default=30)
    parser.add_argument('--num_robot', type=int, default=3)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--num_traj', type=int, default=50)

    args = parser.parse_args()

    generate(variant=vars(args))

    torch.cuda.empty_cache()



if __name__ == '__main__':
    main()