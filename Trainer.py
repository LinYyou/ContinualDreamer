import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.distributions import Categorical, Normal, Bernoulli, OneHotCategorical, Independent
import matplotlib.pyplot as plt
from collections import deque, namedtuple, defaultdict
from copy import deepcopy
from tqdm import tqdm
import random

from ActorCritic import *
from Networks import *
#from replay_buffer import ReplayBuffer_M
from memory import ReplayBuffer
from dreamerV2 import WorldModel




class Trainer():
	def __init__(self, env, prefill_steps = 1000, device= torch.device('cpu')):
		self.device = device
		self.wm = WorldModel(16, 32)
		self.buffer = ReplayBuffer()
		self.prefill_steps = prefill_steps
		self.env = env

		self.initialize()

	def initialize(self):
		while self.buffer.stats["total_steps"] < self.prefill_steps:
			ob, _= self.env.reset()
			episode = defaultdict(list)
			done = False
			ep_length = 0
			while (not done):
				action = self.env.action_space.sample()
				next_ob, reward, terminated, truncated, _ = self.env.step(action)
				done = terminated or truncated
				episode["action"].append(action)
				episode["reward"].append(reward)
				episode["observation"].append(ob)
				episode["next_observation"].append(next_ob)
				episode["done"].append(done)
				ob = next_ob
			self.buffer.add_episode(episode)


	def train_wm_epoch(self, batch_size, seq_len):
				
		batch = self.buffer.sample(batch_size, seq_len)
		wm_loss = []
		for episode in batch:
			###World model learning
			prev_posterior = self.wm.init_stoch_state(1)

			obs = episode['observation']
			actions = episode['action']  
			
			rewards = episode['reward']
			done = episode['done']
			priors, posteriors, losses = self.wm.observe_rollout(actions, obs, rewards, prev_posterior)

			kl_loss = torch.stack(losses['kl_loss']).mean()
			rw_loss = torch.stack(losses['reward_loss']).mean()
			ds_loss = torch.stack(losses['discount_loss']).mean()
			im_loss = torch.stack(losses['image_loss']).mean()
			
			ep_loss = self.wm.c_kl*kl_loss + self.wm.c_im*im_loss + self.wm.c_ds*ds_loss +self.wm.c_rw*rw_loss
			
			self.wm.WM_optimizer.zero_grad()
			ep_loss.backward()
			self.wm.WM_optimizer.step()
			print(ep_loss)
			wm_loss.append(ep_loss.item())

		wm_loss_mean = sum(wm_loss)/len(wm_loss)
		return wm_loss_mean

	def train_epoch(self, batch_size, seq_len, horizon):
		batch = self.buffer.sample(batch_size, seq_len)
		wm_losses = []
		act_losses = []
		crt_losses = []

		for episode in batch:
			###World model learning
			prev_posterior = self.wm.init_stoch_state(1)

			obs = episode['observation']
			actions = episode['action']  
			
			rewards = episode['reward']
			done = episode['done']
			priors, posteriors, losses = self.wm.observe_rollout(actions, obs, rewards, prev_posterior)

			kl_loss = torch.stack(losses['kl_loss']).mean()
			rw_loss = torch.stack(losses['reward_loss']).mean()
			ds_loss = torch.stack(losses['discount_loss']).mean()
			im_loss = torch.stack(losses['image_loss']).mean()
			
			wm_loss = self.wm.c_kl*kl_loss + self.wm.c_im*im_loss + self.wm.c_ds*ds_loss +self.wm.c_rw*rw_loss
			
			self.wm.WM_optimizer.zero_grad()
			wm_loss.backward()
			self.wm.WM_optimizer.step()
			#######
			#behaviour learning
			posterior_initial = random.choice(posteriors)
			stoch_states, actions, rewards, discounts = self.wm.imagine_rollout(posterior_initial, horizon = horizon)
			
			stoch_states_samples = torch.stack([st.sample.squeeze(0) for st in stoch_states], dim=0)
			
			act_loss, crt_loss = self.wm.AC.update(stoch_states_samples, actions, rewards, discounts)
			
			wm_losses.append(wm_loss.item())
			act_losses.append(act_loss.item())
			crt_losses.append(crt_loss.item())
		wm_loss_mean = sum(wm_losses)/len(wm_losses)
		act_loss_mean = sum(act_losses)/len(act_losses)
		crt_loss_mean = sum(crt_losses)/len(crt_losses)
		return wm_loss_mean, act_loss_mean, crt_loss_mean

	def fill_episodes(self, n_episodes):
		for i in range(n_episodes):
			ob, _= self.env.reset()
			episode = defaultdict(list)
			done = False
			
			action = 0
			posterior = self.wm.init_stoch_state(1)
			while (not done):
				posterior, action = self.wm.act_from_ob(posterior, action, ob)
				next_ob, reward, terminated, truncated, _ = self.env.step(action.item())
				done = terminated or truncated
				episode["action"].append(action.item())
				episode["reward"].append(reward)
				episode["observation"].append(ob)
				episode["next_observation"].append(next_ob)
				episode["done"].append(done)
				ob = next_ob
				
			self.buffer.add_episode(episode)
		print('Total steps in buffer: ', self.buffer.stats['total_steps'])


		