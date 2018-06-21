"""NTM Read and Write Heads."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class NTMHeadBase(nn.Module):
    """An NTM Read/Write Head."""

    def __init__(self, memory, controller_size):
        """Initilize the read/write head.

        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(NTMHeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, β, g, s, γ, w_prev):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)

        w = self.memory.address(k, β, g, s, γ, w_prev)

        return w


class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ sizes from the paper
        self.read_lengths = [self.M, 1, 1, 3, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings, w_prev):
        """NTMReadHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_read(embeddings)
        k, β, g, s, γ = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        r = self.memory.read(w)

        return r, w


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev):
        """NTMWriteHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_write(embeddings)
        k, β, g, s, γ, e, a = _split_cols(o, self.write_lengths)

        # e should be in [0, 1]
        e = F.sigmoid(e)

        # Write to memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        self.memory.write(w, e, a)

        return w
