import argparse
import random
import numpy as np
from dreamerV2 import *
import gymnasium as gym
from Trainer import Trainer
from collections import deque, namedtuple, defaultdict
import matplotlib.pyplot as plt
from gymnasium.wrappers import NormalizeObservation


def evaluate(env=None, n_episodes=1, render=False):
    agent = Policy()
    agent.load()

    env = gym.make('CarRacing-v2', continuous=agent.continuous)
    if render:
        env = gym.make('CarRacing-v2', continuous=agent.continuous, render_mode='human')
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        while not done:
            action = agent.act(s)
            
            s, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        
    print('Mean Reward:', np.mean(rewards))



def train():
    train_loss = defaultdict(list)
    n_epochs = 10
    n_epochs_wm = 100
    #env = NormalizeObservation(gym.make('CarRacing-v2', continuous=False), epsilon = 1e-6)
    env = gym.make('CarRacing-v2', continuous=False)    
    wm_training_loss = []
    trainer = Trainer(env)
    for i in tqdm(range(n_epochs_wm)):
        loss = trainer.train_wm_epoch(16, 50)
        wm_training_loss.append(loss)

    for i in tqdm(range(n_epochs)):
        wm_loss, act_loss, crt_loss = trainer.train_epoch(16, 50, 15)
        train_loss['wm'].append(wm_loss)
        train_loss['actor'].append(act_loss)
        train_loss['critic'].append(crt_loss)
        #if i % 10 == 0:
        #trainer.fill_episodes(1000)
    x_wm = range(n_epochs_wm)
    plt.plot(x_wm, wm_training_loss)
    plt.show()

    x = range(n_epochs)
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(x, train_loss['wm'])
    axs[0].set_title('World Model loss')
    axs[1].plot(x, train_loss['actor'], 'tab:orange')
    axs[1].set_title('Actor loss')
    axs[2].plot(x, train_loss['critic'], 'tab:green')
    axs[2].set_title('Critic loss')
    fig.show()
        

def learn_from_experience():
    env = gym.make('CarRacing-v2', continuous=False)
    trainer = Trainer()
    trainer.train_from_experience(env, 32, 50 ,15)

def learn_from_imagination():

    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-ex', '--experience', action='store_true')
    parser.add_argument('-im', '--imagination', action = 'store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()
        #f args.experience:
        #    learn_from_experience()
        #elif args.imagination:
            #learn_from_imagination()
        #else:
        #    raise Exception('Please choose to learn from experience o from the imagination!')        

    if args.evaluate:
        evaluate(render=args.render)

    
if __name__ == '__main__':
    main()
