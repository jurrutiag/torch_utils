import copy
import numpy as np
import torch


class Checkpoint:
    def __init__(self, model, keep_best=True, minimize_value=True, verbose=1):
        self.minimize_value = minimize_value
        self.keep_best = keep_best
        self.verbose = verbose

        self.model = model
        self.checkpoint_value = np.inf if minimize_value else -np.inf
        self.checkpoint_model_state_dict = copy.deepcopy(model.state_dict())

    def step(self, value, savename=None):
        if (self.minimize_value and value < self.checkpoint_value) or (not self.minimize_value and value > self.checkpoint_value) or not self.keep_best:
            self.checkpoint_value = value
            self.checkpoint_model_state_dict = copy.deepcopy(self.model.state_dict())

            if savename is not None:
                self.save_checkpoint(savename)

            if self.verbose >= 1:
                print("Checkpoint updated.")

    def save_checkpoint(self, path):
        torch.save(self.checkpoint_model_state_dict, path)

    def load_checkpoint(self):
        self.model.load_state_dict(self.checkpoint_model_state_dict)

