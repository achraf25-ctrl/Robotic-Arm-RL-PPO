import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(CustomRobotEnv, self).__init__()
        self.render_mode = render_mode
        
        self.max_steps = 200  
        self.current_step = 0
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        low_obs = np.array([-np.pi, -np.pi, -np.pi, -10.0, -10.0, -10.0], dtype=np.float32)
        high_obs = np.array([np.pi, np.pi, np.pi, 10.0, 10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        self.state = None
        self.target_position = np.array([2.0, 2.0, 1.0], dtype=np.float32) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0 
        
        initial_joints = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.state = np.concatenate([initial_joints, self.target_position])
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        
        joints = self.state[0:3] + action * 0.1
        joints = np.clip(joints, -np.pi, np.pi)
        
        x = np.cos(joints[0]) * 2.0
        y = np.sin(joints[1]) * 2.0
        z = np.sin(joints[2]) * 1.0
        current_pos = np.array([x, y, z], dtype=np.float32)
        
        distance = np.linalg.norm(current_pos - self.target_position)
        reward = -distance
        
        terminated = False
        if distance < 0.2:
            reward += 100.0
            terminated = True
            
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            
        self.state = np.concatenate([joints, self.target_position])
        return self.state, reward, terminated, truncated, {}