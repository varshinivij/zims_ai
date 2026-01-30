import time, math
import random
import tensorflow as tf
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import TensorBoard

#values will need to be tuned/changed based on spider bot design                                              #
REPLAY_MEMORY_SIZE = 50000 
MIN_REPLAY_MEMORY_SIZE = 5000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
LIDAR_BINS = 8


class DQNAgent:
    def create_model(self):
        imu_input = Input(shape=(6,), name="IMU_input")
        x_imu = Dense(64, activation='relu')(imu_input)
        x_imu = Dense(32, activation='relu')(x_imu)

        servo_input = Input(shape=(12,), name="Servo_input")
        x_servo = Dense(64, activation='relu')(servo_input)
        x_servo = Dense(32, activation='relu')(x_servo)

        lidar_input = Input(shape=(LIDAR_BINS,), name="Lidar_input")
        x_lidar = Dense(64, activation='relu')(lidar_input)
        x_lidar = Dense(32, activation='relu')(x_lidar)

        combined_inputs = Concatenate()([x_imu, x_servo, x_lidar])
        x = Dense(64, activation='relu')(combined_inputs)

        output = Dense(12, activation='linear', name="servo_output")(x)

        model = Model(
            inputs=[imu_input, servo_input, lidar_input],
            outputs=output
        )
        return model

    def __init__(self):
        # Main model
        self.model = self.create_model()

        #Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        #replay memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    #transition = (current_state, action, reward, next_state, done)
    #current_state = (imu, servo, lidar)
    #next_state    = (imu, servo, lidar)
    def update_replay_memory(self, transition):
       self.replay_memory.append(transition)                

    def train(self, terminal_state, step): 
        # Only start training if we have enough samples in replay memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Sample a random minibatch of past experiences for training
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)


        # Prepare current states for prediction
        # Extract IMU, servo, and LiDAR data from each transition
        current_imu_state = np.array([transition[0][0] for transition in minibatch])
        current_servo_state = np.array([transition[0][1] for transition in minibatch])
        current_lidar_state = np.array([t[0][2] for t in minibatch])
        current_states = [current_imu_state, current_servo_state, current_lidar_state]
       
        # Predict Q-values for current states using the main network
        current_qs_list = self.model.predict(current_states)

        # Prepare next states for prediction
        new_imu_state = np.array([transition[3][0] for transition in minibatch])
        new_servo_state = np.array([transition[3][1] for transition in minibatch])
        new_lidar_state = np.array([t[3][2] for t in minibatch])
        new_current_states = [new_imu_state, new_servo_state, new_lidar_state]
        
        # Predict Q-values for the next states using the target network
        future_qs_list = self.target_model.predict(new_current_states)

        X_imu = []
        X_servo = []
        X_lidar = []
        Y = []

        # Compute target Q-values and prepare training data
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                #Bellmanford equation 
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                # If episode ended, target is just the reward
                new_q = reward
            
            # Update Q-value for the action taken
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Add input states and updated Q-values to training batches
            X_imu.append(current_state[0])
            X_servo.append(current_state[1])
            X_lidar.append(current_state[2])
            Y.append(current_qs)
        
        # Train the main model on the minibatch
        self.model.fit([X_imu, X_servo, X_lidar], np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network weights periodically
        if terminal_state:
            self.target_update_counter += 1
            
        # Copy main network weights to target network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        imu, servo, lidar = state
        imu = np.array(imu).reshape(1, -1)    # shape (1, 6)
        servo = np.array(servo).reshape(1, -1) # shape (1, 12)
        lidar = np.array(lidar).reshape(1, -1) # shape (1, LIDAR_BINS)
        
        return self.model.predict([imu, servo, lidar])[0]
