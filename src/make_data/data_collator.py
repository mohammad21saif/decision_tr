import torch
from datasets import load_from_disk
import numpy as np


class DataCollate:
    def __init__(self, batch_size, max_len, max_episode_len, num_traj, device):
        '''
        Initializes the DataCollate object.

        Args:
        - batch_size: batch size for training the model.
        - max_len: subset of episode to consider for training.
        - max_episode_len: maximum length of an episode.
        - num_traj: number of trajectories to sample.   
        - device: device to run the model on, cpu or cuda.

        Returns:
        - None
        '''

        self.dataset = load_from_disk('/home/moonlab/decision_transformer/data/')
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_episode_len = max_episode_len
        self.num_traj = num_traj
        self.device = device

        self.state_dim = len(self.dataset['train'][0]['states'][0])
        self.act_dim = len(self.dataset['train'][0]['actions'][0])



    def _discount_cumsum(self, x, gamma) -> np.ndarray:
        '''
        Calculates the discounted cumulative sum of the rewards.

        Args:
        - x: rewards.
        - gamma: discount factor.

        Returns:
        - discount_cumsum: discounted cumulative sum of the rewards.
        '''

        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    


    def make_batch(self):
        '''
        Makes a batch of data for training the model.

        Args:
        - None

        Returns:
        - {
            'states': s,
            'actions': a,
            'rewards': r,
            'returns_to_go': rtg,
            'timesteps': timesteps,
            'attention_mask': mask
        }
        '''

        batch_indices = np.random.choice(a=np.arange(self.num_traj), size=self.batch_size, replace=True) # used to randomly select batch indices
        s, a, r, rtg, timesteps, mask = [], [], [], [], [], []

        for i in batch_indices:
            feature = self.dataset['train'][int(i)]
            # feature = self.dataset['train']
            s_i = np.random.randint(0, len(feature['states'])-1) # randomly selecting a starting index

            s.append(np.array(feature['states'][s_i:s_i+self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature['actions'][s_i:s_i+self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature['rewards'][s_i:s_i+self.max_len]).reshape(1, -1, 1))
            timesteps.append(np.array(feature['timesteps'][s_i:s_i+self.max_len]).reshape(1, -1)) #check. Dataset already has timesteps.
            # timesteps.append(np.arange(s_i, s_i + s[-1].shape[1]).reshape(1, -1)) 
            # timesteps[-1][timesteps[-1] >= self.max_episode_len] = self.max_episode_len - 1
            # rtg.append(np.array(feature['returns_to_go'][s_i:s_i+self.max_len]).reshape(1, -1, 1)) #check. Dataset already has rtg.
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][s_i:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            # mask.append(np.concatenate([np.zeros((1, self.max_len - s[-1].shape[1])), np.ones((1, s[-1].shape[1]))], axis=1)) #0 for padding and 1 for actual data
            mask.append(np.ones((1, self.max_len)))

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            # mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))


        s = torch.from_numpy(np.concatenate(s, axis=0)).float().to(self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).float().to(self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).float().to(self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long().to(self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float().to(self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float().to(self.device)

        # print("From data collator: ", "state: ", s.shape, "action: ", a.shape, r.shape, "returns: ", rtg.shape, timesteps.shape, mask.shape)

        return {
            'states': s,
            'actions': a,
            'rewards': r,
            'returns_to_go': rtg,
            'timesteps': timesteps,
            'attention_mask': mask
        }

            

            



        
