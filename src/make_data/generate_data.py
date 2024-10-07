from data import RandomSampler
import argparse
import torch
import os


def generate(variant):
    rewardmap_path = variant['rewardmap_path']
    grid_size = variant['grid_size']
    num_robot = variant['num_robot']
    T = variant['T']
    num_traj = variant['num_traj']
    # save_limit_factor = variant['save_limit_factor']
    num_shards = variant['num_shards']

    device = torch.device("cpu")

    for shard in range(0, num_shards):
        print(f"Generating data for shard {shard}")
        num_traj_per_shard = num_traj // num_shards
        data = RandomSampler(rewardmap_path, grid_size, num_robot, T, num_traj_per_shard, device=device).make_data()
        data.save_to_disk(f"/home/moonlab/decision_transformer/data/data_{shard}")
    


def main():

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--rewardmap_path', type=str, default="/home/moonlab/decision_transformer/decision_tr/maps/gaussian_mixture_training_data.pkl")
    parser.add_argument('--grid_size', type=int, default=30)
    parser.add_argument('--num_robot', type=int, default=3)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--num_traj', type=int, default=50)
    parser.add_argument('--num_shards', type=int, default=10)

    args = parser.parse_args()

    generate(variant=vars(args))
    # combine_data(variant=vars(args))

    torch.cuda.empty_cache()



if __name__ == '__main__':
    main()