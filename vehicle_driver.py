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


def make_line(image, HoughLines):
    black=np.zeros_like(image)
    coords = [0,0,0,0] 
    if HoughLines is not None:
        for line in HoughLines:
            coords = line
            print((coords[0], coords[1]), (coords[2], coords[3]))
            if(((coords[3]-coords[1])/(coords[2]-coords[0])) > 0): # right
                print(1/((coords[3]-coords[1])/(coords[2]-coords[0]))-1.6 , (199-coords[1])/((coords[3]-coords[1])/(coords[2]-coords[0]))+coords[0]-152)
            else:
                print(1/((coords[3]-coords[1])/(coords[2]-coords[0]))+1.6 , (199-coords[1])/((coords[3]-coords[1])/(coords[2]-coords[0]))+coords[0]+152)
            cv2.line(black, (coords[0], coords[1]), (coords[2], coords[3]), [0,255,0], 2)
            # print((coords[0], coords[1]), (coords[2], coords[3]) , (199-coords[1])/((coords[3]-coords[1])/(coords[2]-coords[0]))+coords[0])
    coords = np.array([0,0,0,0])
    if HoughLines is not None:
        for line in HoughLines:
            coords += np.array(line)
        coords = coords//2
        cv2.line(black, (coords[0], coords[1]), (coords[2], coords[3]), [0,255,0], 2)
        print(1/((coords[3]-coords[1])/(coords[2]-coords[0])) , (199-coords[1])/((coords[3]-coords[1])/(coords[2]-coords[0]))+coords[0])
    # print((coords[0]//2, coords[1]//2), (coords[2]//2, coords[3]//2) , (199-coords[1]//2)/((coords[3]//2-coords[1]//2)/(coords[2]//2-coords[0]//2))+coords[0]//2)
    return black


def region_interest(image):
    heigh=image.shape[0]
    triangel=np.array([[(29,199),(470,199),(339,140),(169,140)]])
    mask=np.zeros_like(image)
    mask=cv2.fillPoly(mask,triangel,255)
    image[mask==0] = 0
    return image


class LaneFollowingEnv(Env):
    # car_def_name: The DEF name of your Automobile/Car node in the .wbt file.
    def __init__(self, car_def_name="MY_ROBOT"):

        super().__init__()

        # Define action/observation spaces
        self.observation_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # [distance, speed]
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)       # [steer, speed]

        # Create the Supervisor
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())

        # Create the Driver
        self.driver = Driver()

        # Keep track of the vehicle node
        self.car_def_name = car_def_name
        self.car_node = self.supervisor.getFromDef(self.car_def_name)
        if self.car_node is None:
            print(f"Error: Could not find Automobile with DEF '{self.car_def_name}'")

        # Get camera
        self.camera = None
        try:
            self.camera = self.supervisor.getDevice("camera")
            self.camera.enable(self.time_step)
        except Exception:
            print("Warning: No camera device found. Camera code will be skipped.")


    def seed(self, seed=None):
        """Sets the random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        global t1

        # 1) Parse the action
        steer = action[0] * MAX_STEERING_ANGLE  # scale factor, adapt as needed
        speed = (action[1] + 1.0) * MAX_SPEED

        # 2) Set steering angle & speed by updating the relevant fields
        self.driver.setSteeringAngle(steer)
        self.driver.setCruisingSpeed(speed)

        # 3) Step the simulation
        self.supervisor.step()

        # 4) Get observation from camera and do your lane detection
        if self.camera:
            image_data = self.camera.getImage()
            if image_data is not None:
                image_size = len(image_data) // 4  # RGBA
                actual_height = int(image_size // IMAGE_WIDTH)
                image = np.frombuffer(image_data, dtype=np.uint8).reshape((actual_height, IMAGE_WIDTH, 4))

                # do your lane detection
                image_copy=np.copy(image)
                image_canny=make_canny(image_copy)
                mask = region_interest(image_canny)
                HoughLines = cv2.HoughLinesP(mask,2, np.pi/180, threshold = 40, minLineLength = 30, maxLineGap = 20)
                black_line,left_line_pix,right_line_pix=line_av(image,HoughLines)
                distance = line_analysis(image,left_line_pix,right_line_pix)
                # final=cv2.addWeighted(image_copy,0.8,black_line,1,1)
                # print(distance)
                # cv2.imshow("debug", final)
                # cv2.waitKey(1)

                # Construct normalized state
                state = np.array([distance/IMAGE_WIDTH, self.driver.getTargetCruisingSpeed()/MAX_SPEED], dtype=np.float32)
            else:
                # Fallback if no camera image
                state = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # Fallback if no camera device
            state = np.array([0.0, 0.0], dtype=np.float32)

        # print(self.driver.getTargetCruisingSpeed())
        # 5) Compute reward
        reward = 20-abs(distance) + self.driver.getTargetCruisingSpeed() 
        reward /= 100

        # 6) Determine if done
        #    E.g. if lane departure is too high
        done = False
        translation_field = self.car_node.getField("translation")
        rotation_field    = self.car_node.getField("rotation")
        translation = translation_field.getSFVec3f()
        #rotation = rotation_field.getSFRotation()
        x,y,_ = translation

        if(x < 69.5 and x > 66.0 and y < -44.7 and y > -50.3):
            reward = 5 - (self.supervisor.getTime()-t1)/60
            done = True
        elif abs(distance) > 85:
            reward = -7 + (self.supervisor.getTime()-t1)/60
            done = True

        info = {}
        return state, reward, done, info

    def reset(self):
        global t1
        # Reset the vehicle to a known position & orientation.
        if self.car_node is None:
            print(f"Error: Could not find Automobile with DEF '{self.car_def_name}'")
            return np.array([0.0, 0.0], dtype=np.float32)

        # Access translation & rotation fields
        translation_field = self.car_node.getField("translation")
        rotation_field    = self.car_node.getField("rotation")
        # Example: set new translation
        # Adjust these coordinates to your desired reset pose
        translation_field.setSFVec3f([-47.73, 53.03, 0.4])
        rotation_field.setSFRotation([0, 0, -1, -1.5708])

        # Reset physics so the car doesn't keep old velocity
        self.supervisor.simulationResetPhysics()
        t1 = self.supervisor.getTime()

        # Reset steer and speed
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(0)

        # You might step once to let Webots apply the changes
        self.supervisor.step(self.time_step)

        print("Vehicle has been reset to its initial position/orientation.")
        return np.array([0.0, 0.0], dtype=np.float32)

    def render(self, mode='human'):
        # Optional: anything you want to display in real-time
        pass

    def close(self):
        cv2.destroyAllWindows()


env = LaneFollowingEnv()

# # Vectorized environment for faster training
# vec_env = make_vec_env(lambda: env, n_envs=4)

# load pre-trained model
# try:
#     model = PPO.load("lane_following_agent", env=env)  # Load the pretrained model
#     print("Pretrained model loaded successfully.")
# except FileNotFoundError:
#     print("Pretrained model not found. Training a new model.")

# Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=60000)

# Save the model
model.save("lane_following_agent")

