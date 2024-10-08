import numpy as np
import torch
import matplotlib.pyplot as plt

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
        print("rtg shape in seq: ", rtg.shape)

        action_target = torch.clone(actions)
        # print("Action target before reshaping: ", action_target)
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        # state_preds, action_preds, reward_preds = self.model.forward(
        #     states, actions, rewards, rtg, timesteps, attention_mask=attention_mask,
        # ) #using all rtgs (incluiding the last entry)
        # print("Action pred before reshaping: ", action_preds)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # print("Action preds after reshaping: ", action_preds)
        # print("Action target after reshaping: ", action_target)

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        # print("Loss: ", loss)



        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds - action_target) ** 2).detach().cpu().item()

        return loss.detach().cpu().item()