
"""
This code defines the robot state variables and actions the robot may take.
"""
import numpy as np

#these need to be defined
#DEFAULT_SERVO_POSITIONS = 
#DEFAULT_ORIENTATION = 
#SERVO_MIN = 
#SERVO_MAX = 
#LIDAR_SIZE = 

class Robot:
    def __init__(self):
        #Need normalization for easier learning 
        self.servo_positions = DEFAULT_SERVO_POSITIONS
        self.orientation = DEFAULT_ORIENTATION
        self.angular_velocity = np.zeros(3)
        self.lidar = np.zeros(LIDAR_SIZE)
        
    def choose_action(self, servo_positions):
        pass

    def update_orientation(self):
        pass

    def get_state(self):
        return np.concatenate([
            self.orientation,        # 3
            self.angular_velocity,   # 3
            self.lidar,              # LIDAR_BINS
            self.servo_positions     # 12
        ])