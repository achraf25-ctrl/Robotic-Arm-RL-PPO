import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPickAndPlace-v4", render_mode="human")

model = PPO.load("ppo_robot_arm_model")

obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()