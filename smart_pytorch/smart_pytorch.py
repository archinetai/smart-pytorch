from typing import List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)

def to_list(x):
    return x if isinstance(x, list) else [x]

class SMARTLoss(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Union[Callable, List[Callable]],
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = to_list(loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed: Tensor, state: Union[Tensor, List[Tensor]]) -> Tensor:
        states = to_list(state)
        noise = torch.randn_like(embed, requires_grad=True) * self.noise_var

        for i in range(self.num_steps + 2):
            # Compute perturbed states 
            embed_perturbed = embed + noise 
            states_perturbed = to_list(self.eval_fn(embed_perturbed))
            loss = 0
            # Compute perturbation loss over all states 
            for j in range(len(states)):
                loss += self.loss_fn[j](states_perturbed[j], states[j].detach())
            if i == self.num_steps + 1: 
                return loss 
            # Compute noise gradient     
            noise_gradient = torch.autograd.grad(loss, noise)[0]
            # Move noise towards gradient to change state as much as possible 
            step = noise + self.step_size * noise_gradient 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()