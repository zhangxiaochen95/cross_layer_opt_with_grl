import random
from collections import deque

from components.misc import *


class ReplayBuffer:
    """Replay buffer storing sequences of transitions."""

    def __init__(self, args):

        self.pre_decision_fields = set(args.pre_decision_fields)  # Field names before agent decision
        self.post_decision_fields = set(args.post_decision_fields)  # Field names after agent decision
        self.fields = self.pre_decision_fields.union(self.post_decision_fields)  # Overall fields
        self.fields.add('filled')

        self.capacity = args.buffer_size  # Total number of data sequences that can be held by memory
        self.memory = deque(maxlen=self.capacity)  # Memory holding samples
        self.data_chunk_len = args.data_chunk_len  # Maximum length of data sequences

        self.sequence = None  # Data sequence holding up-to-date transitions
        self.ptr = None  # Recorder of data sequence length
        self._reset_sequence()

    def _reset_sequence(self) -> None:
        """cleans up the data sequence."""
        self.sequence = {k: [] for k in self.fields}
        self.ptr = 0

    def insert(self, transition):
        """Stores a transition into memory. A transition is first held by data sequence.
        When maximum length is reached, contents of data sequence is stored to memory.
        """

        # When maximum sequence length is reached,
        if self.ptr == self.data_chunk_len:
            # Append the pre-decision data beyond the last timestep to data sequence.
            for k in self.pre_decision_fields:
                self.sequence[k].append(transition.get(k, ''))
            # Move data sequence to memory.
            self.memory.append(self.sequence)
            # Clear the sequence and reset pointer.
            self._reset_sequence()
            # Pseudo transition is no longer added to the beginning of next sequence.
            if not transition.get('filled'):
                return

        # Store data specified by fields.
        # Note that pseudo transition is stored if not appended to the end of sequence.
        for k, v in transition.items():
            if k in self.fields:
                self.sequence[k].append(v)
        self.ptr += 1  # A complete transition is stored.

    def recall(self, batch_size: int):
        """Selects a random batch of samples."""
        assert len(self) >= batch_size, "Samples are insufficient."
        samples = random.sample(self.memory, batch_size)  # List of samples
        batched_samples = {k: [] for k in self.fields}  # Dict holding batch of samples.

        # Construct input sequences.
        for t in range(self.data_chunk_len):
            for k in self.fields:
                batched_samples[k].append(cat([samples[idx][k][t] for idx in range(batch_size)]))

        # Add pre-decision data beyond the last timestep for bootstrapping.
        for k in self.pre_decision_fields:
            batched_samples[k].append(cat([samples[idx][k][self.data_chunk_len] for idx in range(batch_size)]))

        return batched_samples

    def can_sample(self, batch_size: int) -> bool:
        """Whether sufficient samples are available."""
        return batch_size <= len(self)

    def __len__(self):
        return len(self.memory)

    def __repr__(self):
        return f"ReplayBuffer, holding {len(self)}/{self.capacity} sequences."


if __name__ == '__main__':
    a = {'apple'}
    b = {'pear', 'banana', 'apple'}
    buffer = ReplayBuffer(a, b, 10, 10)

    a = [1, 2]
    from typing import Iterable
    print(isinstance(a, Iterable))