from pathlib import Path
from tarfile import DIRTYPE
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearQNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LinearQNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def save(self, filename: Path = Path("model.pth")) -> None:
        # model_dir
        model_dir = Path("./chkpts")
        if not model_dir.exists():
            model_dir.mkdir()

        # filepath
        file_path  = model_dir / filename
        torch.save(self.state_dict(), file_path)

class QNetTrainer(object):
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.model = model
        self.lr = lr # learning rate
        self.gamma = gamma # discount factor

        # Create optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)

        # Set up loss function
        self.loss_func = nn.MSELoss()

    def train_step(self, state: Union[np.ndarray, list[np.ndarray]],
        action: Union[list, list[list]],
        reward: Union[int, list[int]],
        next_state: Union[np.ndarray, list[np.ndarray]],
        done: Union[bool, list[bool]]) -> None:

        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        # Handle batch of inputs
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            next_state = torch.unsqueeze(next_state, dim=0)
            done = (done, )

        # Compute q_targets
        q_preds = self.model(state)
        if len(action.shape) == 2: # only updating taken actions
            action_idxs = torch.argmax(action, dim=1)
        q_preds = torch.gather(q_preds, dim=1, index=action_idxs.unsqueeze(1)).squeeze(1)

        q_targets = reward
        q_1 = self.model(next_state).clone()
        q_1, _ = torch.max(q_1, dim=1)

        # Add on look-ahead if next_state is not terminal
        for idx in range(len(done)):
            if not done[idx]:
                q_targets[idx] = q_targets[idx] + self.gamma*q_1[idx]

        # Compute loss
        loss = self.loss_func(q_preds, q_targets)

        # Zero grad, backprop, optim step,
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
