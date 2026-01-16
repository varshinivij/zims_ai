import random
from collections import deque
from config import REPLAY_BUFFER_SIZE


class ReplayBuffer:
    #init
    def __init__(self):
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    #add 
    def add(self, transition):
       return self.replay_buffer.append(transition)
    #sample
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
