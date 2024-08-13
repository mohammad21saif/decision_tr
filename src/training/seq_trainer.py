import numpy as np
import torch

from training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        batch = self.get_batch
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        rtg = batch['returns_to_go']
        timesteps = batch['timesteps']
        attention_mask = batch['attention_mask']

        action_target = torch.clone(actions)

        # state_preds, action_preds, reward_preds = self.model.forward(
        #     states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        # )

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg, timesteps, attention_mask=attention_mask,
        ) #using all rtgs (incluiding the last entry)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds - action_target) ** 2).detach().cpu().item()

        return loss.detach().cpu().item()