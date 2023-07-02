import torch
import torch.nn as nn

class actor_critic(nn.Module):
    def __init__(self, actor, critic):
        super(actor_critic, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, state):
        pred_act = self.actor(state)
        pred_rwd = self.critic(state)
        return pred_act, pred_rwd