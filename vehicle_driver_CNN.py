import os
import cv2
import numpy as np
from vehicle import Driver  # Webots
from controller import Supervisor
import random  # To simulate random steering adjustments
import gym
from gym import Env
from gym.spaces import Box
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch

# Settings
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 500
MAX_STEERING_ANGLE = 0.8  # Maximum allowable steering angle
MAX_SPEED = 20.0
t1 = 0

# Just for handling an error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def make_dots(image,line):
    # print (line)
    # line[np.isnan(line)==True]=random.randint(0,2)
    slope,intercept=line
    y1=image.shape[0]
    y2=int(y1*(2/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
        # return np.array([0,0,1,1])


def line_av(image,lines):
    black=np.zeros_like(image)
    left_side=[]
    right_side=[]

    try:
        for line in lines:
            croods=line[0]
            parameters=np.polyfit((croods[0],croods[2]),(croods[1],croods[3]),1)
            slope=parameters[0]
            intercept=parameters[1]
            if slope<0:
                left_side.append((slope,intercept))
            else:
                right_side.append((slope,intercept))
            
        if len(left_side)!=0 and len(right_side)!=0:
            # left
            left_fit_av=np.average(left_side,axis=0)
            left_dots=make_dots(image,left_fit_av)
            left_coords = left_dots
            cv2.line(black, (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3]), [0,255,0], 2)
            # print(1/((left_coords[3]-left_coords[1])/(left_coords[2]-left_coords[0]))-1.6 , (199-left_coords[1])/((left_coords[3]-left_coords[1])/(left_coords[2]-left_coords[0]))+left_coords[0]-152)
            distance_left = (199-left_coords[1])/((left_coords[3]-left_coords[1])/(left_coords[2]-left_coords[0]))+left_coords[0]
            # right
            right_fit_av=np.average(right_side,axis=0)
            right_dots=make_dots(image,right_fit_av)
            right_coords = right_dots
            cv2.line(black, (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3]), [0,255,0], 2)
            # print(1/((right_coords[3]-right_coords[1])/(right_coords[2]-right_coords[0]))+1.6 , (199-right_coords[1])/((right_coords[3]-right_coords[1])/(right_coords[2]-right_coords[0]))+right_coords[0]+152)
            distance_right = (199-right_coords[1])/((right_coords[3]-right_coords[1])/(right_coords[2]-right_coords[0]))+right_coords[0]
        elif len(left_side)!=0:
            distance_right = 0
            left_fit_av=np.average(left_side,axis=0)
            left_dots=make_dots(image,left_fit_av)
            left_coords = left_dots
            cv2.line(black, (left_coords[0], left_coords[1]), (left_coords[2], left_coords[3]), [0,255,0], 2)
            # print(1/((left_coords[3]-left_coords[1])/(left_coords[2]-left_coords[0]))-1.6 , (199-left_coords[1])/((left_coords[3]-left_coords[1])/(left_coords[2]-left_coords[0]))+left_coords[0]-152)
            distance_left = (199-left_coords[1])/((left_coords[3]-left_coords[1])/(left_coords[2]-left_coords[0]))+left_coords[0]
        elif len(right_side)!=0:
            distance_left = 0
            right_fit_av=np.average(right_side,axis=0)
            right_dots=make_dots(image,right_fit_av)
            right_coords = right_dots
            cv2.line(black, (right_coords[0], right_coords[1]), (right_coords[2], right_coords[3]), [0,255,0], 2)
            # print(1/((right_coords[3]-right_coords[1])/(right_coords[2]-right_coords[0]))+1.6 , (199-right_coords[1])/((right_coords[3]-right_coords[1])/(right_coords[2]-right_coords[0]))+right_coords[0]+152)
            distance_right = (199-right_coords[1])/((right_coords[3]-right_coords[1])/(right_coords[2]-right_coords[0]))+right_coords[0]
        else:
            distance_right = 0
            distance_left = 0
    except:
        distance_right = 0
        distance_left = 0

    return black,distance_left,distance_right


def line_analysis(image,left_line,right_line):
    if left_line != 0 and right_line != 0 :
        distance = image.shape[1]//2 - (left_line + right_line)//2
        # print(image.shape[1]//2 - (left_line + 160),image.shape[1]//2 - (right_line - 160))
    elif left_line != 0:
        distance = image.shape[1]//2 - (left_line + 160)
    elif right_line != 0:
        distance = image.shape[1]//2 - (right_line - 160)
    else:
        distance = 0

    return distance


def make_canny(image_copy_func):
    image_gray=cv2.cvtColor(image_copy_func,cv2.COLOR_BGR2GRAY)
    image_blur=cv2.GaussianBlur(image_gray,(5,5),0)
    canny=cv2.Canny(image_blur,50,100)
    return canny


def region_interest(image):
    heigh=image.shape[0]
    triangel=np.array([[(29,199),(470,199),(339,140),(169,140)]])
    mask=np.zeros_like(image)
    mask=cv2.fillPoly(mask,triangel,255)
    image[mask==0] = 0
    return image


class LaneFollowingEnv(Env):
    def __init__(self, car_def_name="MY_ROBOT"):
        super().__init__()
        self.observation_space = Box(low=0, high=1, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())
        self.driver = Driver()

        self.car_def_name = car_def_name
        self.car_node = self.supervisor.getFromDef(self.car_def_name)
        if self.car_node is None:
            raise ValueError(f"Error: Could not find Automobile with DEF '{self.car_def_name}'")

        self.camera = self.supervisor.getDevice("camera")
        self.camera.enable(self.time_step)

    def step(self, action):
        steer = action[0] * MAX_STEERING_ANGLE
        
        self.driver.setSteeringAngle(steer)
        self.driver.setCruisingSpeed(20)
        self.supervisor.step(self.time_step)

        image_data = self.camera.getImage()
        if image_data is not None:
            image = np.frombuffer(image_data, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
        else:
            image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)

        # Calculate distance and reward based on the original algorithm
        image_copy = np.copy(image)
        image_canny = make_canny(image_copy)
        mask = region_interest(image_canny)
        HoughLines = cv2.HoughLinesP(mask, 2, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=20)
        black_line, left_line_pix, right_line_pix = line_av(image, HoughLines)
        distance = line_analysis(image, left_line_pix, right_line_pix)

        reward = 20 - abs(distance)
        reward /= 100

        done = False
        translation_field = self.car_node.getField("translation")
        translation = translation_field.getSFVec3f()
        x, y, _ = translation

        if x < 69.5 and x > 66.0 and y < -44.7 and y > -50.3:
            reward = 5
            done = True
        elif abs(distance) > 85:
            reward = -7
            done = True
        
        image = image[:, :, :3] / 255.0  # Normalize RGB values

        return image, reward, done, {}

    def reset(self):
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(0)

        translation_field = self.car_node.getField("translation")
        rotation_field = self.car_node.getField("rotation")
        translation_field.setSFVec3f([-47.73, 53.03, 0.4])
        rotation_field.setSFRotation([0, 0, -1, -1.5708])

        self.supervisor.simulationResetPhysics()
        self.supervisor.step(self.time_step)

        image_data = self.camera.getImage()
        if image_data is not None:
            image = np.frombuffer(image_data, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
            return image[:, :, :3] / 255.0  # Normalize RGB values
        else:
            return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

# Custom CNN for feature extraction
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations.permute(0, 3, 1, 2)))

# Register custom policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# Create environment
env = LaneFollowingEnv()

# Train the PPO model
model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=60000)

# Save the model
model.save("cnn_lane_following_agent")
