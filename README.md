# Autonomous-Vehicle-RL
Reinforcement Learning for Autonomous Lane Following in Webots

This repository demonstrates the implementation of a reinforcement learning (RL) environment for autonomous lane-following vehicles using Webots, OpenAI Gym, and Stable-Baselines3.

## Demo Video

Below is a demo video showcasing the RL-trained agent's performance:

https://github.com/user-attachments/assets/de004a60-82bb-4ff9-a836-b9d285cd3d88

## Features

- **Webots Integration**: Seamlessly integrates with the Webots robot simulator to provide a realistic environment for testing autonomous vehicles.
- **Reinforcement Learning**: Utilizes Proximal Policy Optimization (PPO) from Stable-Baselines3 for training.
- **Lane Detection (Sample)**: Includes a basic image processing and lane-detection logic using OpenCV. **This algorithm is provided as a sample and is highly recommended to replace with a more robust and accurate lane-detection solution for better performance**.
- **Custom Environment**: Implements a custom OpenAI Gym-compatible environment to simulate lane-following behavior.
- **Training and Testing**: Provides scripts to train the agent or load pretrained models for evaluation.
- **Custom CNN**: Includes a Convolutional Neural Network (CNN) for feature extraction, enabling enhanced perception and better handling of complex inputs.

## Prerequisites

- [Webots](https://cyberbotics.com/) installed
- Python 3.8 or higher
- Required Python libraries:
  - `numpy`
  - `opencv-python`
  - `gym`
  - `stable-baselines3`

## File Structure

- `vehicle_driver.py`: Main script containing the RL environment, sample lane-detection logic, and training loop.
- `vehicle_driver_CNN.py`: Advanced script implementing a custom CNN for feature extraction, enabling improved processing of visual inputs for lane-following tasks. This file integrates a more sophisticated RL setup using the CNN architecture for enhanced decision-making.
- `lane_following_agent.zip`: Pretrained PPO model for lane following (optional).

### Running the Simulation

1. Open your Webots world file and ensure the vehicle is configured with the correct DEF name (default: `MY_ROBOT`).
2. Verify that the Supervisor node is enabled in the Webots simulation.
3. Set the `vehicle_driver` or `vehicle_driver_CNN` as the controller for your robot in the Webots world file.
4. Run the simulation.

### Training the Agent

The script will automatically train the PPO agent if no pretrained model is found. Modify training parameters directly in the script.

### Loading a Pretrained Model

To use a pretrained model, place the model file (`lane_following_agent.zip`) in the same directory and run the script.

## Key Components

- **Lane Detection**: Utilizes OpenCV for edge detection, Hough transform for lane line extraction, and geometry-based lane following logic. 
  - **Note**: The current lane-detection algorithm is provided as a sample implementation. It is recommended to replace it with a more robust and accurate method tailored to your specific needs for better results.
- **RL Environment**: A custom `LaneFollowingEnv` class derived from Gym's `Env` base class.
- **Reward Function**: Encourages staying centered within the lane and maintaining optimal speed.
- **Custom CNN for Feature Extraction**: Utilizes a convolutional neural network to process image data and extract meaningful features for reinforcement learning. This improves the model's ability to understand and adapt to complex environments.

## Results

The agent learns to:
- Steer the vehicle to stay within the lane.
- Maintain optimal speed based on distance and time.
- Adapt to dynamic lane detection using CNN-based feature extraction (in `vehicle_driver_CNN.py`).
