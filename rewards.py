import gym

import torch
import torch.nn as nn
from torch.autograd import Variable


class RewardWrapper(gym.Wrapper):
	def __init__(self, env, scale, out_function)
		super(RewardWrapper, self).__init__(env)
		self.reward_range = (-scale, scale)
		self.out_function = out_function

	def _attack(self, action):
		action = Variable(action, volatile = True)
		outs = self.target_model(action)
		return self.out_function(outs)


class Untargeted(RewardWrapper):
	def __init__(self, env, scale = 1., out_function = nn.functional.sigmoid):
		super(Untargeted, self).__init__(env, scale, out_function)

	def _get_reward(self, obs, action, **kwargs):
		# Establish ground truth for image
		ground_truth = obs[1]

		# Attack target model
		# TODO: Consider abstracting this and allowing it to be overridden,
		# or at least allow for different output functions to be used
		outs = self._attack(action)
		confidence, prediction = torch.max(outs, 1)

		# Determine norm penalty
		if self.norm is not None:
            norm_penalty = self.norm_on_batch(action.data - obs[0], self.norm)
        else:
			norm_penalty = 0

		# Compute reward, gather info
		reward = (ground_truth != prediction.data).float() * confidence.data
			- (ground_truth == prediction.data).float() * confidence.data
			- norm_penalty
		info = {'label':ground_truth,
			'prediction':prediction.data,
			'confidence':confidence.data,
			'norm':norm_penalty if self.norm is not None else None}

		return reward, info


class SingleTargeted(RewardWrapper):
	def __init__(self, env, target_class = 0, skip_target_class = True, scale = 1., out_function = nn.functional.sigmoid):
		super(SingleTargeted, self).__init__(env, scale, out_function)
		self.target_class = target_class
		self.skip_target_class = skip_target_class

	def _step(self, action, **kwargs):
		try:
            current_obs = self.successor
            if self.skip_target_class:
				self.successor = self.iterator.__next__()
                while (self.successor[1] == self.target_class).any():
                	newly_sampled = self._sample_for_skip((self.successor[1] == self.target_class).sum())
                	target_mask = (self.successor[1] == self.target_class)
                    self.successor[0][target_mask] = newly_sampled[0]
                    self.successor[1][target_mask] = newly_sampled[1]
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
        reward, info = self._get_reward(current_obs, action, **kwargs)
        return self.successor, reward, self.done, info

	def _get_reward(self, obs, action, **kwargs):
		# Establish ground truth for image
		ground_truth = obs[1]

		# Just here for testing
		if self.skip_target_class:
			assert ground_truth != self.target_class

		# Attack target model
		outs = self._attack(action)
		confidence, prediction = torch.max(outs, 1)

		# Determine norm penalty
		if self.norm is not None:
            norm_penalty = self.norm_on_batch(action.data - obs[0], self.norm)
        else:
			norm_penalty = 0

		# Compute reward, gather info
		reward = reward = (self.target_class == prediction.data).float() * confidence.data
			- (self.target_class != prediction.data).float() * confidence.data
			- norm_penalty
		info = {'label':ground_truth,
			'prediction':prediction.data,
			'confidence':confidence.data,
			'norm':norm_penalty if self.norm is not None else None}

		return reward, info

    def _sample_for_skip(self, batch_size):
    	# Set up tensor types
    	Tensor = torch.cuda.Tensor if self.use_cuda else torch.Tensor
    	LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

    	# Sample from dataset -- necessary so we don't mess up the index of the DataLoader
    	collection = [self.dataset[int(Tensor(1).random_(0,len(self.dataset), generator = self.torch_rng)[0])] for x in range(batch_size)]

    	# Collect batch into tensor
    	collection = tuple(zip(*collection))
    	collection[0] = torch.cat(map(lambda x: torch.unsqueeze(x, 0), collection[0]))
    	collection[1] = LongTensor(collected[1]))

    	return collection


class DynamicTargeted(RewardWrapper):
	def __init__(self, env, scale = 1., out_function = nn.functional.sigmoid):
		super(DynamicTargeted, self).__init__(env, scale, out_function)

	def _get_reward(self, obs, action, **kwargs):
		# Establish ground truth for image
		ground_truth = obs[1]

		# Attack target model
		outs = self._attack(action)
		confidence, prediction = torch.max(outs, 1)

		# Unpack target_class and make sure it's in the right range
		try:
			target_class = kwargs['target_class']
			assert (type(target_class) is int) and (target_class >= 0) and (target_class <= outs.size(-1) - 1)
		except KeyError:
			raise gym.error.Error('DynamicTargeted wrapper requires keyword argument \'target_class\' for each call of \'step\'.')
		except AssertionError:
			raise gym.error.Error('Keyword argument \'target_class\' must be within range of model.')

		# Determine norm penalty
		if self.norm is not None:
            norm_penalty = self.norm_on_batch(action.data - obs[0], self.norm)
        else:
			norm_penalty = 0

		# Compute reward, gather info
		reward = reward = (target_class == prediction.data).float() * confidence.data
			- (target_class != prediction.data).float() * confidence.data
			- norm_penalty
		info = {'label':ground_truth,
			'prediction':prediction.data,
			'confidence':confidence.data,
			'norm':norm_penalty if self.norm is not None else None}

		return reward, info
