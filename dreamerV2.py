import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.distributions import Categorical, Normal, Bernoulli, OneHotCategorical, independent
import matplotlib.pyplot as plt
from collections import deque, namedtuple, defaultdict
from copy import deepcopy
from tqdm import tqdm
import random
from torchvision import transforms

from ActorCritic import *
from Networks import *
from replay_buffer import ReplayBuffer_M


##This file aims to build a universal Dreamer V2 model, the model will be firstly trained so tested on car racing enviroment,
##then it will be used to train our Square Escape environment.

##https://github.com/danijar/dreamerv2/blob/main/dreamerv2/agent.py

'''
Variables:
h_t = deterministic state in recurrent state space model propagation
z_prior = stochastic prior, representation of the state computed 
z_posterior = stochastic posterior, representation of the state with access to the current image
'''

prior_state = namedtuple('prior_state', ['sample', 'dist', 'deter'])
posterior_state = namedtuple('posterior_state', ['sample', 'dist', 'deter'])
wm_loss = namedtuple('wm_loss', ['img_loss', 'dis_loss', 'r_loss','kl_loss'])
################		
class WorldModel(nn.Module):
	def __init__(self, h_size=16, z_size=32):
		super().__init__()
		

		self.AC_lr = 1e-3
		self.WD_lr = 1e-3
		
		self.h_size = h_size
		self.z_size = z_size
		action_size = 1 
		##RSSM block
		self.rnn = nn.GRUCell(z_size + action_size, h_size)##h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) The use of nn.GRUCell aims us to obtain only one time step output 
		self.Repre_Model = Representation_Model(h_size, z_size) #z_posterior_t(z_posterior) = q(z_t| h_t, x_t) 
		self.Prior_Predictor = DNN(h_size, z_size, n_layers = 2, hidden_sizes = [32], act_fn = nn.ELU()) #z_prior_t = p(z_prior_t|h_t)
		#self.Posterior_Predictor = DNN(z_size*2, z_size, n_layers = 2, hidden_size = [32], act_fn=nn.ELU())

		self.Image_Predictor = IMG_Decoder((3, 96, 96), z_size, h_size)
		self.R_Predictor = DNN(h_size + z_size, 1, n_layers = 2, hidden_sizes = [32], act_fn = nn.ELU())
		self.Discount_Predictor = DNN(h_size + z_size, 1, n_layers = 2, hidden_sizes = [32], act_fn = nn.ELU())
		
		self.AC = A2C()
		self.KL_loss = nn.KLDivLoss()

		self.obs_transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
			])

		self.c_kl = 0.1
		self.c_im = 1
		self.c_rw = 1 
		self.c_ds = 1

		#self.WM_parameters = [param for param in self.parameters() if param not in self.AC.parameters()]
		self.AC_optimizer = torch.optim.Adam(self.AC.parameters(), lr=self.AC_lr)
		self.WM_optimizer = torch.optim.Adam(self.parameters(), lr=self.WD_lr)

	def init_stoch_state(self, batch_size):

		sample = torch.zeros(batch_size, self.z_size)
		
		dist = OneHotCategorical(logits = sample)
		deter = torch.zeros(batch_size, self.h_size)
		return posterior_state(sample, dist, deter)

	def imagine_rollout(self, initial_state, horizon):
		states = []
		actions = []
		rewards = []
		discounts = []

		state = initial_state
		for i in range(horizon):
			
			action = self.AC.act(state.sample)

			next_state, reward, discount = self.imagine_step(state, action)
			actions.append(action.squeeze(0))
			states.append(state)
			rewards.append(reward)
			discounts.append(discount)

			state = next_state
		
		actions = torch.stack(actions)
		rewards = torch.stack(rewards)
		discounts = torch.stack(discounts)
		
		return states, actions, rewards, discounts 

		
	def observe_rollout(self, actions, images, rewards, previous_posterior):
		priors = []
		posteriors = []
		
		losses = defaultdict(list)
		for i in range(len(actions)):
			prior, posterior, image_loss, reward_loss, discount_loss, kl_loss = self.observe_step(previous_posterior, actions[i], images[i], rewards[i])
			priors.append(prior)
			posteriors.append(posterior)
			previous_posterior = posterior
			losses['kl_loss'].append(kl_loss)
			losses['reward_loss'].append(reward_loss)
			losses['image_loss'].append(image_loss)
			losses['discount_loss'].append(discount_loss)
		
		return priors, posteriors, losses
		
	def imagine_step(self, posterior, action):##Take as input the current state and action, it should return the prior and posterior
		action = torch.LongTensor([action])
		state_action = torch.cat((posterior.sample, action.unsqueeze(0)), dim = -1)
		
		hidden = self.rnn(state_action)
		
		#prior , posterior_raw = torch.chunk(state[-1,:], chunks = 2, dim = -1)
		prior_logits = self.Prior_Predictor(hidden)
		
		prior_dist = OneHotCategorical(logits = prior_logits)
		prior_sample = prior_dist.sample()
		posterior_probs = nn.Softmax()(prior_logits)
	
		prior_sample += prior_dist.probs - prior_dist.probs.detach()
		prior = prior_state(prior_sample, prior_dist, hidden) 

		reward, reward_dist = self.reward_predict(prior)
		discount, discount_dist = self.discount_predict(prior)
		
		return prior, reward, discount

	def observe_step(self, posterior, action, image, reward):
		'''
		Input: 
		Output: prior, posterior
		'''
		prior, _, _ = self.imagine_step(posterior, action)
		image = self.obs_transform(image)
		posterior_logits = self.Repre_Model(image.unsqueeze(0), prior.deter)
		posterior_probs = nn.Softmax()(posterior_logits)
		posterior_dist = OneHotCategorical(logits = posterior_logits)
		posterior = posterior_dist.sample()
		
		posterior += posterior_dist.probs - posterior_dist.probs.detach()
		posterior = posterior_state(posterior, posterior_dist, prior.deter)

		image_loss, reward_loss, discount_loss, kl_loss = self.observe_loss(prior, posterior, image, reward)

		return prior, posterior, image_loss, reward_loss, discount_loss, kl_loss


	def act_from_ob(self, posterior, action, image):
		reward = 0 
		with torch.no_grad():
			_, next_posterior, _, _, _, _ = self.observe_step(posterior, action, image, reward)
			next_action = self.AC.get_action(posterior.sample)

		return next_posterior, next_action

	def reward_predict(self, posterior):
		
		post = torch.cat((posterior.sample, posterior.deter), dim = -1)
		reward_mean = self.R_Predictor(post)
		reward_std = torch.ones(reward_mean.shape)
		reward_dist = Independent(Normal(reward_mean, reward_std), 1)
		reward = reward_dist.sample()
		return reward, reward_dist

	def discount_predict(self, posterior):
		post = torch.cat((posterior.sample, posterior.deter), dim = -1)
		discount_logits = self.Discount_Predictor(post)
		discount_dist = Independent(Bernoulli(logits = discount_logits), 1)
		discount = discount_dist.sample()
		return discount, discount_dist

	def observe_loss(self, prior, posterior, image, reward):

		reward = torch.tensor(reward)
		#image = transforms.Normalize(mean=mean, std = std)(image/255.0)
		
		image_sample, image_dist = self.Image_Predictor(posterior.sample, posterior.deter)
		reward_sample, reward_dist = self.reward_predict(posterior)
		discount, discount_dist = self.discount_predict(posterior)
		
		img_log_loss = self.log_loss(image_dist, image)
		reward_log_loss = self.log_loss(reward_dist, reward)
		discount_log_loss = self.log_loss(discount_dist, discount)
		kl_loss = self.KL_loss(posterior.dist.probs, prior.dist.probs)
		
		return img_log_loss, reward_log_loss, discount_log_loss, kl_loss

	def log_loss(self, dist, item):	
		return -torch.mean(dist.log_prob(item))


