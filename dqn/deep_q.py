
import time, math
import random
import tensorflow as tf
import numpy as np
from collections import deque
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import TensorBoard

#values will need to be tuned/changed based on SLAP design
ACTION_SPACE_SIZE = 4
REPLAY_MEMORY_SIZE = 50000 
MIN_REPLAY_MEMORY_SIZE = 5000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
class DQNAgent:
    def create_model(self):
        #IMU data may need normalization depending on how big numbers are
        imu_input = Input(shape =(6,), name="IMU_input") #servo input layer
        x_imu = Dense(64, activation='relu')(imu_input)
        x_imu = Dense(32, activation='relu')(x_imu)
       
        #Servo layer may also need normalization
        servo_input = Input(shape=(12,), name="Servo_input")
        x_servo = Dense(64, activation='relu')(servo_input)
        x_servo = Dense(32, activation='relu')(x_servo)

        #combine the feature vectors
        combined_inputs = Concatenate()([x_imu, x_servo])
        x = Dense(64, activation='relu')(combined_inputs)

        output = Dense(ACTION_SPACE_SIZE, activation='linear')(x)
        
        model = Model(inputs=[imu_input, servo_input], outputs=output)

        return model

    def __init__(self):
        # Main model
        self.model = self.create_model()

        #Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        #replay memory
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        #tensorboard for logs
        #self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
       self.replay_memory.append(transition)

    def train(self, terminal_state, step): 
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_imu_state = np.array([transition[0][0] for transition in minibatch])
        current_servo_state = np.array([transition[0][1] for transition in minibatch])
        current_states = [current_imu_state, current_servo_state]
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        new_imu_state = np.array([transition[3][0] for transition in minibatch])
        new_servo_state = np.array([transition[3][1] for transition in minibatch])
        new_current_states = [new_imu_state, new_servo_state]
        future_qs_list = self.target_model.predict(new_current_states)

        X_imu = []
        X_servo = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X_imu.append(current_state[0])
            X_servo.append(current_state[1])
            Y.append(current_qs)
        
        self.model.fit([X_imu, X_servo], np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        if terminal_state:
            self.target_update_counter += 1
            
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        imu, servo = state
        imu = np.array(imu).reshape(1, -1)    # shape (1, 6)
        servo = np.array(servo).reshape(1, -1) # shape (1, 12)
        return self.model.predict([imu, servo])[0]
