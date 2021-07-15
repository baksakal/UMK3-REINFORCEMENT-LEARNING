
import argparse
import retro
from baselines.common.vec_env import SubprocVecEnv
from baselines.common.retro_wrappers import make_retro, wrap_deepmind_retro
from baselines.ppo2 import ppo2
import stable_baselines
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from baselines.common.retro_wrappers import *
import gym
from stable_baselines import SAC
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

#import os
#os.chdir(C:\Users\draks\miniconda3\envs\py36\Library\bin)

def main():

    env = stable_baselines.common.vec_env.subproc_vec_env.SubprocVecEnv([lambda: retro.make(game='aaa',state="noob",scenario='scenario')] * 9)
    eval_env = stable_baselines.common.vec_env.subproc_vec_env.SubprocVecEnv([lambda: retro.make(game='aaa',state="noob",scenario='scenario')])

    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1000,
                             deterministic=True, render=False, n_eval_episodes=1)
  
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10000, callback=eval_callback)
    model.save("model")

    env = stable_baselines.common.vec_env.subproc_vec_env.SubprocVecEnv([lambda: retro.make(game='aaa',state="noob",scenario='scenario')])   
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()