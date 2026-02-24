import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.rewards = []
        self.states = []
        self.actions = []
        self.values = []
        self.probs = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.values), \
               np.array(self.rewards), \
               batches
    
