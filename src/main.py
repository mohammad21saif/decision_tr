from make_data.data import RandomSampler
from make_data.data_collator import DataCollate
from training.seq_trainer import SequenceTrainer
from model.decision_transformer import DecisionTransformer
from evaluation.evaluate_episodes import evaluate_episode_rtg
from inference import MakeAnime

import torch
import numpy as np
import os
import argparse




def make_dataset(device, rewardmap_path, grid_size, num_robot, T, num_traj):
    '''
    Makes the dataset for training the model.

    Args:
    - device: device to run the model on, cpu or cuda.
    - rewardmap_path: path to the rewardmap.
    - grid_size: size of the grid.
    - num_robot: number of robots.
    - T: number of time steps.
    - num_traj: number of trajectories

    Returns:
    - saved_data: saved dataset.
    '''


    #Making the dataset
    data = RandomSampler(rewardmap_path, grid_size, num_robot, T, num_traj, device='cuda').make_data()
    data.save_to_disk("/home/moonlab/decision_transformer/data/")

    feature = data['train']
    state_dim = len(feature['states'][0][0])
    act_dim = len(feature['actions'][0][0])

    states = []
    for traj in feature['states']:
        states.append(traj)
    
    states_concatenated = np.concatenate(states, axis=0)
    state_mean = np.mean(states_concatenated, axis=0)
    state_std = np.std(states_concatenated, axis=0) + 1e-6



    
    return state_mean, state_std, state_dim, act_dim


def save_checkpoint(model, optimizer, scheduler, iteration, path):
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)



def experiment(variant):


    K = variant['K']
    batch_size = variant['batch_size']
    embed_dim = variant['embed_dim']
    n_layer = variant['n_layer']
    n_head = variant['n_head']
    activation_function = variant['activation_function']
    dropout = variant['dropout']
    learning_rate = variant['learning_rate']
    weight_decay = variant['weight_decay']
    warmup_steps = variant['warmup_steps']
    num_eval_episodes = variant['num_eval_episodes']
    max_iters = variant['max_iters']
    num_steps_per_iter = variant['num_steps_per_iter']
    max_ep_len = variant['max_ep_len']


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    rewardmap_path = "/home/moonlab/decision_transformer/decision_tr/maps/gaussian_mixture_training_data.pkl"

    grid_size = 30
    num_robot = 3
    T = 10
    num_traj = 50
    env_targets = [500, 400]

    state_mean, state_std, state_dim, act_dim = make_dataset(device, rewardmap_path, grid_size, num_robot, T, num_traj)
    collate_data = DataCollate(batch_size=batch_size, max_len=10, max_episode_len=max_ep_len, num_traj=num_traj, device=device).make_batch()

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=embed_dim,
        activation_function=activation_function,
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
    )
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    warmup_steps = 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        target_return=target_rew,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=collate_data,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    checkpoint_dir = '/home/moonlab/decision_transformer/decision_tr/saved_models'
    for iter in range(max_iters):
        outputs = trainer.train_iteration(num_steps=num_steps_per_iter, iter_num=iter+1, print_logs=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iter+1}.pt")
        save_checkpoint(model, optimizer, scheduler, iter+1, checkpoint_path)
    
    final_model_path = os.path.join(checkpoint_dir, "trained_model.pt")
    torch.save(model.state_dict(), final_model_path)

    MakeAnime()


    


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=str, default=20)
    parser.add_argument('--batch_size', type=str, default=32)
    parser.add_argument('--embed_dim', type=str, default=128)
    parser.add_argument('--n_layer', type=str, default=3)
    parser.add_argument('--n_head', type=str, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=str, default=0.1)
    parser.add_argument('--learning_rate', type=str, default=1e-4)
    parser.add_argument('--weight_decay', type=str, default=1e-4)
    parser.add_argument('--warmup_steps', type=str, default=10)
    parser.add_argument('--num_eval_episodes', type=str, default=10)
    parser.add_argument('--max_iters', type=str, default=10)
    parser.add_argument('--num_steps_per_iter', type=str, default=10)
    parser.add_argument('--max_ep_len', type=str, default=10)


    args = parser.parse_args()
    experiment(variant=vars(args))
    
    torch.cuda.empty_cache()