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
from datetime import datetime
import wandb
from datasets import load_from_disk, concatenate_datasets
from progiter.manager import ProgressManager
from scipy import stats




def make_dataset(device, num_shards, T):
    '''
    Loads dataset.

    Args:
    - device: device to load the dataset.
    - num_shards: number of shards to load.
    - T: number of time steps.

    Returns:
    - state_mean: mean of the states.
    - state_std: standard deviation of the states.
    - state_dim: dimension of the state.
    - act_dim: dimension of the action.
    - data: dataset.
    '''

    device = device
    pman = ProgressManager()
    dataset = concatenate_datasets([
                load_from_disk(f"/home/moonlab/decision_transformer/data/test_data_{shard_idx}")['train']
                for shard_idx in range(num_shards)
                ])
    states = []
    means = []
    stds = []
    sample_size = []
    with pman:
        for shard_idx in pman(range(num_shards)):
            data = load_from_disk(f"/home/moonlab/decision_transformer/data/test_data_{shard_idx}")['train']
            feature = data
            state_dim = len(feature['states'][0][0])
            act_dim = len(feature['actions'][0][0])
            for traj in feature['states']:
                states.append(traj)

            states_concatenated = np.concatenate(states, axis=0)
            state_mean = np.mean(states_concatenated, axis=0)
            means.append(state_mean)

            state_std = np.std(states_concatenated, axis=0) + 1e-6
            stds.append(state_std)

            sample_size.append(T)

        combined_means = np.mean(np.array(means), axis=0)
        
        total_sample_size = sum(sample_size)
        pooled_variance = np.sum([((sample_size[i] - 1) * stds[i]**2) for i in range(num_shards)], axis=0) / (total_sample_size - num_shards)
        combined_stds = np.sqrt(pooled_variance)
        
    return combined_means, combined_stds, state_dim, act_dim, dataset



def save_checkpoint(model, optimizer, scheduler, iteration, path):
    '''
    Saves the model, optimizer and scheduler to a checkpoint.

    Args:
    - model: model to save.
    - optimizer: optimizer to save.
    - scheduler: scheduler to save.
    - iteration: iteration number.

    Returns:
    - None
    '''

    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)



def experiment(variant):
    '''
    Main function to run the experiment.

    Args:
    - variant: dictionary containing the hyperparameters.

    Returns:
    - None
    '''

    K = variant['K'] #max_context_length
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
    # max_ep_len = variant['max_ep_len']
    num_robot = variant['num_robot']
    T = variant['T']
    num_traj = variant['num_traj']
    grid_size = variant['grid_size']
    rewardmap_path = variant['rewardmap_path']
    num_shards = variant['num_shards']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    log_to_wandb = variant.get('log_to_wandb', False)

    rewardmap_path = "/home/moonlab/decision_transformer/decision_tr/maps/gaussian_mixture_training_data.pkl"

    env_targets = [500]

    state_mean, state_std, state_dim, act_dim, data = make_dataset(device, num_shards, T)
    print("Starting experiment")
    collate_data = DataCollate(dataset=data, 
                               batch_size=batch_size, 
                               max_len=10, 
                               max_episode_len=T, 
                               num_traj=num_traj,
                               state_mean=state_mean,
                               state_std=state_std,
                               device=device).make_batch()
    print("Data collated")

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_context_length=K,
        max_ep_len=T,
        hidden_size=embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=embed_dim,
        activation_function=activation_function,
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
    )
    print("Model created")

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
                        max_ep_len=T,
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

    #TODO: try cross-entropy loss
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
    save_interval = max_iters // 3  # Calculate save interval

    if log_to_wandb:
        wandb.init(
            name="dt",
            project='decision-transformer',
            config=variant
        )

    print("Starting Iterations")
    progress = ProgressManager()
    with progress:
        for iter in range(max_iters):
            outputs = trainer.train_iteration(num_steps=num_steps_per_iter, iter_num=iter+1, print_logs=True)
            if log_to_wandb:
                wandb.log(outputs)
    
            # Save checkpoint only every max_iters/3 iterations
            if (iter + 1) % save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iter+1}_{timestamp}.pt")
                # save_checkpoint(model, optimizer, scheduler, iter+1, checkpoint_path)
    
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(checkpoint_dir, f"trained_model_{final_timestamp}.pt")
        # torch.save(model, final_model_path)


    


if __name__ == "__main__":

    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=2)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=2)
    # parser.add_argument('--max_ep_len', type=int, default=15)
    parser.add_argument('--num_robot', type=int, default=3)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--num_traj', type=int, default=50)
    parser.add_argument('--grid_size', type=int, default=30)
    parser.add_argument('--rewardmap_path', type=str, default="/home/moonlab/decision_transformer/decision_tr/maps/gaussian_mixture_training_data.pkl")
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--num_shards', type=int, default=3)

    args = parser.parse_args()

    experiment(variant=vars(args))
    
    torch.cuda.empty_cache()