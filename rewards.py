import gym

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class RewardWrapper(gym.Wrapper):
	"""
	Base class for reward wrappers.
	This should not actually be used as a Wrapper in your code.

	Args:
		env (subclass of AdvEnv): the environment instance to wrap.
		scale (float): controls reward scaling.
		out_function (function): the final activation function to use for computing confidence
			for each attack. If None, uses the identity function.
	"""
	def __init__(self, env, scale, out_function, **kwargs):
		super(RewardWrapper, self).__init__(env)
		self.reward_range = (-scale, scale)
		self.scale = scale
		self.out_function = lambda x: x if out_function is None else out_function

		self.use_cuda = self.unwrapped.use_cuda
		self.Tensor = self.unwrapped.Tensor
		self.LongTensor = self.unwrapped.LongTensor

		self.target_model = self.unwrapped.target_model
		self.dataset = self.unwrapped.dataset
		self.norm = self.unwrapped.norm
		self.strict_epsilon = self.unwrapped.strict_epsilon
		self.action_space = self.unwrapped.action_space
		self.observation_space = self.unwrapped.observation_space
		self.episode_length = self.unwrapped.episode_length
		self.sampler = self.unwrapped.sampler
		self.torch_rng = self.unwrapped.torch_rng
		self.batch_size = self.unwrapped.batch_size
		self.num_workers = self.unwrapped.num_workers
		self.data_loader = self.unwrapped.data_loader
		self.iterator = iter(self.data_loader)

	def _step(self, action, **kwargs):
		# Iterate until StopIteration 
		# (either episode_length has been reached or DataLoader iterator is exhausted)
		try:
			current_obs = self.successor
			self.successor = self.iterator.__next__()
			self.unwrapped.ix += 1
			if self.unwrapped.ix >= self.episode_length:
				raise StopIteration
		except StopIteration:
			self.unwrapped.done = True

		# CUDA conversion
		if self.use_cuda:
			action = action.cuda()
			self.successor[0] = self.successor[0].cuda()
			self.successor[1] = self.successor[1].cuda()
		else:
			action = action.cpu()

		# Get reward and return results of environment transition
		reward, info = self._get_reward(current_obs, action, **kwargs)
		return self.successor, reward, self.unwrapped.done, info

	def _get_reward(self, obs, action, **kwargs):
		# Must be overridden by a subclass
		raise NotImplementedError

	def _seed(self, seed):
		return self.unwrapped._seed(seed)

	def _reset(self):
		return self.unwrapped._reset()

	def norm_on_batch(self, input, p):
		return self.unwrapped.norm_on_batch(input, p)

	def _attack(self, action):
		action = Variable(action, volatile = True)
		outs = self.target_model(action)
		return self.out_function(outs)

	def _strict(self, norm_penalty):
		if self.strict_epsilon:
			return norm_penalty < self.strict_epsilon
		else:
			return True

	def  _failed_strict(self,**kwargs):
		return min(reward_range)


class Untargeted(RewardWrapper):
	"""
	Reward computed for untargeted attack.

	Args:
		env (subclass of AdvEnv, required): the environment instance to wrap.
		scale (float, default: 1): controls reward scaling.
		out_function (function, default: nn.functional.sigmoid): the final activation function to use for computing confidence
			for each attack. If None, uses the identity function.
	"""
	def __init__(self, env, scale = 1., out_function = nn.functional.sigmoid):
		super(Untargeted, self).__init__(env, scale, out_function)
		self.out_function = self.env.out_function if out_function is None else out_function

	def _get_reward(self, obs, action, **kwargs):
		# Establish ground truth for image
		ground_truth = obs[1]

		# Attack target model
		outs = self._attack(action)
		confidence, prediction = torch.max(outs, 1)

		# Determine norm penalty
		if self.norm is not None:
			norm_penalty = self.norm_on_batch(action.data - obs[0], self.norm)
		else:
			norm_penalty = 0

		# Compute reward, gather info
		if self.unwrapped._check_norm_validity() and self._strict(norm_penalty):
			reward = ((ground_truth != prediction.data).float() * confidence.data
				- (ground_truth == prediction.data).float() * confidence.data
				- norm_penalty)
		else:
			reward = self._failed_strict()
		info = {'label':ground_truth,
			'prediction':prediction.data,
			'confidence':confidence.data,
			'norm':norm_penalty if self.norm is not None else None}

		return reward, info


class StaticTargeted(RewardWrapper):
	"""
	Reward computed for targeted attack.

	Args:
		env (subclass of AdvEnv, required): the environment instance to wrap.
		target_class (integer, required): the class we want the target_model to misclassify as.
			Must be an integer corresponding to an output neuron in the target network.
		skip_target_class (boolean, default: True): If True, skips images belonging to the target
			class.  Otherwise, the agent is rewarded for correct classification of images belonging
			to the target_class.
		scale (float, default: 1): controls reward scaling.
		out_function (function, default: nn.functional.sigmoid): the final activation function to use for computing confidence
			for each attack. If None, uses the identity function.

	"""
	def __init__(self, env, target_class, skip_target_class = True, scale = 1., out_function = nn.functional.sigmoid):
		super(StaticTargeted, self).__init__(env, scale, out_function)
		self.target_class = target_class
		self.skip_target_class = skip_target_class
		self.out_function = self.env.out_function if out_function is None else out_function

	def _step(self, action, **kwargs):
		# Same as base _step method, but allows skipping images belonging to the target class
		try:
			current_obs = self.successor
			if self.skip_target_class:
				self.successor = self.iterator.__next__()
				while (self.successor[1] == self.target_class).any():
					newly_sampled = self._sample_for_skip((self.successor[1] == self.target_class).sum())
					target_mask = (self.successor[1] == self.target_class)
					expanded_mask = target_mask.view(-1, *[1]*(len(self.successor[0].size())-1))
					expanded_mask = target_mask.expand_as(self.successor[0])
					self.successor[0][expanded_mask] = newly_sampled[0]
					self.successor[1][target_mask] = newly_sampled[1]
			else:
				self.successor = self.iterator.__next__()
			self.ix += 1
			if self.ix >= self.episode_length:
				raise StopIteration
		except StopIteration:
			self.unwrapped.done = True
		if self.use_cuda:
			action = action.cuda()
			self.successor[0] = self.successor[0].cuda()
			self.successor[1] = self.successor[1].cuda()
		else:
			action = action.cpu()
		reward, info = self._get_reward(current_obs, action, **kwargs)
		return self.successor, reward, self.unwrapped.done, info

	def _get_reward(self, obs, action, **kwargs):
		# Establish ground truth for image
		ground_truth = obs[1]

		# Just here for testing -- TODO: Remove
		if self.skip_target_class:
			assert (ground_truth != self.target_class).all()

		# Attack target model
		outs = self._attack(action)
		confidence, prediction = torch.max(outs, 1)

		# Determine norm penalty
		if self.norm is not None:
			norm_penalty = self.norm_on_batch(action.data - obs[0], self.norm)
		else:
			norm_penalty = 0

		# Compute reward, gather info
		if self.unwrapped._check_norm_validity() and self._strict(norm_penalty):
			reward = ((self.target_class == prediction.data).float() * confidence.data
				- (self.target_class != prediction.data).float() * confidence.data
				- norm_penalty)
			reward += self.scale
		else:
			reward = self._failed_strict()

		info = {'label':ground_truth,
			'prediction':prediction.data,
			'confidence':confidence.data,
			'norm':norm_penalty if self.norm is not None else None}

		return reward, info

	def _reset(self):
		if self.skip_target_class:
			self.data_loader = DataLoader(self.dataset, batch_size = self.batch_size, sampler = self.sampler, num_workers = self.num_workers)
			self.iterator = iter(self.data_loader)
			self.successor = self.iterator.__next__()
			while (self.successor[1] == self.target_class).any():
				newly_sampled = self._sample_for_skip((self.successor[1] == self.target_class).sum())
				target_mask = (self.successor[1] == self.target_class)
				expanded_mask = target_mask.view(-1, *[1]*(len(self.successor[0].size())-1))
				expanded_mask = target_mask.expand_as(self.successor[0])
				self.successor[0][expanded_mask] = newly_sampled[0]
				self.successor[1][target_mask] = newly_sampled[1]
			if self.use_cuda:
				self.successor[0] = self.successor[0].cuda()
				self.successor[1] = self.successor[1].cuda()
			self.unwrapped.done = False
			self.ix = 0
			return self.successor
		else:
			return self.unwrapped._reset()

	def _sample_for_skip(self, batch_size):
		# Sample from dataset -- necessary so we don't mess up the index of the DataLoader
		collection = [self.dataset[int(self.Tensor(1).cpu().random_(0, len(self.dataset), generator = self.torch_rng)[0])] for x in range(batch_size)]

		# Collect batch into tensor
		collection = list(zip(*collection))
		collection[0] = torch.cat(map(lambda x: torch.unsqueeze(x, 0), collection[0]))
		collection[1] = torch.LongTensor(collection[1])

		return collection


class DynamicTargeted(RewardWrapper):
	"""
	Reward computed for targeted attack, but this allows the user to specify which class they want
	to target at each step in the MDP.

	Args:
		env (subclass of AdvEnv, required): the environment instance to wrap.
		target_class (LongTensor, required): the class we want the target_model to misclassify as.
			Must be a 1-D LongTensor of length batch_size containing integers corresponding to the
			desired output neuron in the target network.
		skip_target_class (boolean, default: True): If True, skips images belonging to the target
			class.  Otherwise, the agent is rewarded for correct classification of images belonging
			to the target_class.
		scale (float, default: 1): controls reward scaling.
		out_function (function, default: nn.functional.sigmoid): the final activation function to use for computing confidence
			for each attack. If None, uses the identity function.

	"""
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
			assert isinstance(target_class, self.LongTensor) and (target_class >= 0).all() and (target_class <= outs.size(-1) - 1).all()
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
		reward = ((target_class == prediction.data).float() * confidence.data
			- (target_class != prediction.data).float() * confidence.data
			- norm_penalty)
		info = {'label':ground_truth,
			'prediction':prediction.data,
			'confidence':confidence.data,
			'norm':norm_penalty if self.norm is not None else None}

		return reward, info



class DefendMode(RewardWrapper):
	def __init__(self, env, scale = 1., out_function = nn.functional.sigmoid):
		super(DefendMode, self).__init__(env, scale, out_function)

	def _get_reward(self, obs, action, **kwargs):
		# Establish ground truth for image
		ground_truth = obs[1]

		# Attack target model
		outs = self._attack(action)
		confidence, prediction = torch.max(outs, 1)

		# Determine norm penalty
		if self.norm is not None:
			norm_penalty = self.norm_on_batch(action.data - obs[0], self.norm)
		else:
			norm_penalty = 0

		# Compute reward, gather info
		if self.unwrapped._check_norm_validity() and self._strict(norm_penalty):
			reward = ((ground_truth == prediction.data).float() * confidence.data
				- (ground_truth != prediction.data).float() * confidence.data
				- norm_penalty)
			reward += self.scale
		else:
			reward = self._failed_strict()

		info = {'label':ground_truth,
			'prediction':prediction.data,
			'confidence':confidence.data,
			'norm':norm_penalty if self.norm is not None else None}

		return reward, info

# TODO
class BadNets(RewardWrapper):
	def __init__(self, env, scale =  1., out_function = nn.functional.sigmoid):
		super(BadNets, self).__init__(env, scale, out_function)