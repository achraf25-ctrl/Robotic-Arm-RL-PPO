import os
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 1. Enregistrement dyal les environnements d robotics
gym.register_envs(gymnasium_robotics)

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("FetchPickAndPlace-v4", max_episode_steps=100)

env = Monitor(env, log_dir)

model = PPO(
    "MultiInputPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./ppo_robot_tensorboard/"
)

TOTAL_TIMESTEPS = 100_000
print("training is starting some seconds...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)


model.save("ppo_robot_arm_model")
print("model has saved")