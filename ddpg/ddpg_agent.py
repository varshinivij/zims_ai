from network import Actor, Critic
from replay_buffer import ReplayBuffer
import numpy as np

class DDPGAgent:
    def __init__(self):
        self.actor_network = Actor()
        self.critic_network = Critic()

        self.target_actor_network = Actor()
        self.target_critic_network = Critic()

        self.target_actor_network.set_weights(self.actor_network.get_weights())
        self.target_critic_network.set_weights(self.critic_network.get_weights())

        self.replay_buffer = ReplayBuffer()

        #actor and critic optimizers

    def update_replay_memory(self, transition):
        self.replay_memory.add(transition)

    def choose_action(self, state):
        actor_model = self.actor_network.actor_model()
        action = actor_model.predict(state)[0]
        action = np.clip(action, -1, 1)
        return action
    
    #train


    pass