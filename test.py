# Gym stuff
import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from finta import TA

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

# Processing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('spy_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', ascending=True, inplace=True)
df.set_index('Date', inplace=True)
env = gym.make('stocks-v0', df=df, frame_bound=(5,250), window_size=5)

state = env.reset()
while True: 
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    if done:
        print("info:", info)
        break
        
plt.figure(figsize=(15,6))
plt.cla()
env.unwrapped.render_all()
df['Volume'] = df['Volume'].apply(lambda x: float(x.replace(",", "")))

df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)
df.fillna(0, inplace=True)



#for later
env = MyCustomEnv(df=df, window_size=12, frame_bound=(80,250))
obs = env.reset()
while True: 
    observation = obs[0]
    observation = observation[np.newaxis, ...]
    profits = obs[1]
    new_obs = (observation, profits)
    action, _states = model.predict(new_obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        print("info", info)
        break