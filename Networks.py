
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.distributions import Normal, Independent
from replay_buffer import ReplayBuffer_M
from torchvision import transforms


#This file contain all elementary networks 
class Representation_Model(nn.Module):##Input: image_t + h_t Output: discrete latent representation
	##To DO: Value estimation not correct, it should use Reinforce value estimation.
	def __init__(self, hidden_size, z_size):
		super(Representation_Model, self).__init__()

		self.tranform = transforms.Compose([transforms.ToTensor(),
			transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
			])


		num_channels=3
		self.cnn_block = nn.ModuleList([ ##Input(batch, 96, 96, 3)
            nn.Conv2d(num_channels, 16, (5, 5), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), (1, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(16, 32, (3, 3), (1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (4, 4), (1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Flatten(),
            nn.Linear(32*20*20, 32)
        ])##Output (batch, 20, 20, 32)

		self.dense = nn.Sequential(
            nn.Linear(32 + hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, z_size)
        )

	def forward(self, image, deter):

		#x = self.preprocess(image)
		#x = image.transpose(-1, -3)
		x = image
		for layer in self.cnn_block:
			x = layer(x)
		
		integrate = torch.cat((x, deter), 1)

		out = self.dense(integrate)
		return out

	def preprocess(self, state_raw):

		#x = state_raw/255.0##Normalize every pixels value from [0, 255] to [0, 1]
		#state = x.transpose(-1, -3)##Input shape (batch, 96, 96, 3), CNN2D required shape (batch, 3, 96, 96)
		state = self.transform(state_raw)
		return state
	
class IMG_Decoder(nn.Module):  # Decoder for image reconstruction from latent state
	def __init__(self, output_size, z_size, h_size):
		super().__init__()
		num_channels, heigth, witdh = output_size
		self.input_size = z_size + h_size 

		self.dense = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32 * 20 * 20),
            nn.ReLU()
        )

		self.cnn_block = nn.ModuleList([
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.ConvTranspose2d(32, 32, (4, 4), (1, 1)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, (3, 3), (1, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(16, 16, (3, 3), (1, 1)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, num_channels, (5, 5), (1, 1)),
            nn.Tanh()
        ])
		self.output_shape = (3, 96, 96)
	def forward(self, z_posterior, h):
		batch_shape = z_posterior.shape[:-1]
		x = self.dense(torch.cat((z_posterior, h), 1))
		x = x.view(-1, 32, 20, 20)  # Reshape to match the shape before flattening in the encoder
		
		for layer in self.cnn_block:
			x = layer(x)
		
		mean = torch.reshape(x, (*batch_shape, *self.output_shape))
		std = torch.ones(mean.shape)
		obs_dist = Independent(Normal(mean, std), len(self.output_shape))
		return obs_dist.sample(), obs_dist
class DNN(nn.Module):
	def __init__(self, input_size, output_size, n_layers: int, hidden_sizes, act_fn):
		super().__init__()
		assert n_layers > 0
		self.n_layers = n_layers
		self.act_fn = act_fn
		if hidden_sizes:
			assert len(hidden_sizes) == n_layers - 1

		if n_layers == 1:
			self.dense = nn.ModuleList([nn.Linear(input_size, output_size)])
		else:
			self.dense = nn.ModuleList(
				[nn.Linear(input_size, hidden_sizes[0])] + 
				[nn.Linear(hidden_sizes[i-1], hidden_sizes[i]) for i in range(len(hidden_sizes))]+ 
				[nn.Linear(hidden_sizes[-1], output_size)])

	def forward(self, input):
		x = input
		for i, layer in enumerate(self.dense):
			x = layer(x)
			if i <self.n_layers:
				x = self.act_fn(x)
			
		return x #it returns logits