import numpy as np
import torch
import pickle


rewardmap_path = "/home/moonlab/decision_transformer/decision_tr/maps/gaussian_mixture_training_data.pkl"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
rewardmap = torch.tensor(pickle.load(open(rewardmap_path, "rb"), encoding="latin1")).to(device)
grid_size = 30

def take_step(action, state):
    '''
    Takes a step in the environment.

    Args:
    - action: action to take.
    - state: current state.

    Returns:
    - state_ret: next state.
    - reward: reward for taking the action.
    '''

    # state_ret = np.add(action, state)
    # reward = rewardmap[state_ret[0::2], state_ret[1::2]].sum()

    # return state_ret, reward
    state_ret = np.add(action, state)
    
    # Clamp the state to ensure it stays within the grid boundaries
    state_ret[0::2] = np.clip(state_ret[0::2], 0, grid_size - 1)
    state_ret[1::2] = np.clip(state_ret[1::2], 0, grid_size - 1)
    
    reward = rewardmap[state_ret[0::2], state_ret[1::2]].sum()

    return state_ret, reward



def evaluate_episode(
        state_dim,
        act_dim,
        model,
        max_ep_len,
        device,
        target_return,
        state_mean,
        state_std,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = np.zeros((state_dim,), dtype=np.float32)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward = take_step(action, state)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

    return episode_return, episode_length



def evaluate_episode_rtg(
        state_dim,
        act_dim,
        model,
        max_ep_len,
        state_mean,
        state_std,
        device,
        target_return,
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = np.zeros((state_dim,), dtype=np.float32)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward = take_step(action, state)
   
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        # if mode != 'delayed':
        #     pred_return = target_return[0,-1] - (reward)
        # else:
        #     pred_return = target_return[0,-1]

        pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

    return episode_return, episode_length