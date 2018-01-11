import gym

import torch
import torch.nn as nn
from torch.autograd import Variable

import random

class Untargeted(gym.Wrapper):
	def __init__(self, env, scale = 1, norm = None):
		super(Untargeted, self).__init__(env)
		self.reward_range = (-scale, scale)
		self.norm = norm

	def _get_reward(self, obs, action):
		# Establish ground truth for image
		ground_truth = obs[1]

		# Run through target model
		action = Variable(action, volatile = True)
        outs = self.target_model(action)
        outs = nn.functional.sigmoid(outs)
		confidence, prediction = torch.max(outs, 1)

		# Determine norm penalty
		if self.norm is not None:
            norm_penalty = self.norm_on_batch(action.data - obs[0], self.norm)
        else:
			norm_penalty = 0

		# Calculate reward and track info
		reward = (ground_truth != prediction.data).float() * confidence.data
			- (ground_truth == prediction.data).float() * confidence.data
			- norm_penalty
		info = {'label':ground_truth,
			'prediction':prediction.data,
			'confidence':confidence.data,
			'norm':norm_penalty if self.norm is not None else None}

		return reward, info

	def norm_on_batch(self, input, p):
		# Assume dimension 0 is batch dimension
        norm_penalty = input
        while len(norm_penalty.size())>1:
            norm_penalty = torch.norm(norm_penalty, p, -1)
		return norm_penalty

class Targeted(gym.Wrapper):
	def __init__(self, env, target_label = 0, skip_target_class = True, scale = 1, norm = None):
		super(Targeted, self).__init__(env)
		self.reward_range = (-scale, scale)
		self.norm = norm

	def _step(self, action, **kwargs):
		try:
            current_obs = self.successor
            if self.skip_target_class:
				self.successor = self.iterator.__next__()
                while (self.successor[1] == self.target_class).any():
                	newly_sampled = self._sample_for_skip(self.batch_size)
                	target_mask = (self.successor[1] == self.target_class)
                    self.successor[0][target_mask] = newly_sampled[0][target_mask]
                    self.successor[1][target_mask] = newly_sampled[1][target_mask]
            else:
                self.successor = self.iterator.__next__()
            self.ix += 1
            if self.ix >= self.episode_length:
                raise StopIteration
        except StopIteration:
            self.done = True
        if self.use_cuda:
            action = action.cuda()
            self.successor[0] = self.successor[0].cuda()
            self.successor[1] = self.successor[1].cuda()
        else:
            action = action.cpu()
        reward, info = self._get_reward(current_obs, action)
        return self.successor, reward, self.done, info

    def _sample_for_skip(self, batch_size):
    	# Set up tensor types
    	Tensor = torch.cuda.Tensor if self.use_cuda else torch.Tensor
    	LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
    	# Sample from dataset -- necessary so we don't screw up the DataLoader
    	collection = [self.dataset[int(Tensor(1).random_(0,len(self.dataset), generator = self.torch_rng)[0])] for x in range(batch_size)]
    	# Reformat back to standard format for get_item
    	collection = tuple(zip(*collection))
    	collection[0] = torch.cat(map(lambda x: torch.unsqueeze(x, 0), collection[0]))
    	collection[1] = torch.cuda.LongTensor(collected[1]) if self.use_cuda else torch.LongTensor(collected[1])
    	return collection

	def _get_reward(self, obs, action):
		# Establish ground truth for image
		#ground_truth
		pass