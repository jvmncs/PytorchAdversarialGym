import gym

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

import warnings

from .space import TensorBox
from .util import UniformSampler


class AdvEnv(gym.Env):
	"""
	An environment for generating and defending against adversarial examples with PyTorch models
		and Datasets.

	Args:
		target_model (subclass of torch.nn.Module): The model we're attacking.
			Currently, only torch.nn.Module is supported.
		dataset (subclass of torch.utils.data.Dataset): The dataset we're using to attack.
			Currently, only Pytorch Dataset objects are supported.
			Note: ToTensor transform must be included.
		batch_size (int, optional): Number of instances for the target_model to classify per step.
		episode_length (positive integer, optional): Specifies the number of steps to include in
			each episode.  Default is len(dataset)//batch_size.
		sampler (subclass of torch.utils.data.Sampler, optional): Specifies the sampling strategy.
			Default is uniform random sampling with replacement over the entire dataset.
		num_workers (integer, optional): Argument to be passed to DataLoader specifying number of
			subprocess threads to use for data loading.
		use_cuda: (bool, optional): Whether to place tensors and model on GPU.
			Defaults to True if GPU is available.
		seed: (int, optional): integer to use for random seed.  If None, use default Pytorch RNG.
			Note: setting a seed does not guarantee determinism when using CUDNN backend.
			For this reason, we disable CUDNN if a seed is specified.

	"""
	def __init__(self, target_model, dataset, batch_size = 1,
			episode_length = None, sampler = None, num_workers = 0,
			use_cuda = torch.cuda.is_available(), seed = None):
		super(AdvEnv).__init__()
		# Set up tensor types
		self.use_cuda = use_cuda
		self.Tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
		self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

		# Fail early and often
		if not self._check_model(target_model):
			raise gym.error.Error('Model type {} not supported.'.format(type(target_model)) +
							  ' Currently, target_model must be a subclass of torch.nn.Module.')
		if not self._check_dataset(dataset):
			raise gym.error.Error('Dataset type {} not supported.'.format(type(dataset)) +
							  ' Currently, dataset must be a subclass of torch.utils.data.Dataset containing (FloatTensor, LongTensor) instances.')

		# Unpack args
		if seed is not None:
			torch.backends.cudnn.enabled = False
		self.seedey = self.seed(seed)
		self.target_model = target_model.cuda() if use_cuda else target_model.cpu()
		self.dataset = dataset
		self.sampler = UniformSampler(self.dataset, self.torch_rng, len(self.dataset)) if not sampler else sampler

		if not self._check_sampler(self.sampler):
			raise gym.error.Error('Sampler type {} not supported.'.format(type(self.sampler)) +
							   ' Currently, sampler must be a subclass of torch.utils.data.sampler.Sampler.')

		# Construct necessaries
		space_shape = self.dataset[0][0].size()
		space_shape = (batch_size, *space_shape[1:])
		self.action_space = TensorBox(0, 1, space_shape, self.use_cuda)
		self.observation_space = TensorBox(0, 1, space_shape, self.use_cuda)
		self.episode_length = len(self.dataset)//batch_size if not episode_length else episode_length
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.data_loader = DataLoader(self.dataset, batch_size = self.batch_size, sampler = self.sampler, num_workers = self.num_workers)
		self.iterator = iter(self.data_loader)
		self.reset()

	def step(self, action, **kwargs):
		# Overridden in RewardWrapper
		raise NotImplementedError

	def _get_reward(self, obs, action, **kwargs):
		# Must be overridden by a subclass of RewardWrapper
		raise NotImplementedError

	def seed(self, seed):
		integer_types = (int,)
		if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
			raise gym.error.Error('Seed must be a non-negative integer or omitted, not {}.'.format(type(seed)))
		self.torch_rng = torch.manual_seed(seed) if seed is not None else torch.default_generator
		self.seedey = seed
		return [seed]

	def reset(self):
		self.data_loader = DataLoader(self.dataset, batch_size = self.batch_size, sampler = self.sampler, num_workers = self.num_workers)
		self.iterator = iter(self.data_loader)
		self.successor = next(self.iterator)
		if self.use_cuda:
			self.successor[0] = self.successor[0].cuda()
			self.successor[1] = self.successor[1].cuda()
		self.done = False
		self.ix = 0
		return self.successor

	def norm_on_batch(self, input, p):
		# Assume dimension 0 is batch dimension
		norm_penalty = input
		while len(norm_penalty.size())>1:
			norm_penalty = torch.norm(norm_penalty, p, -1)
		return norm_penalty

	def _check_model(self, model):
		return isinstance(model, nn.Module)

	def _check_dataset(self, dataset):
		return isinstance(dataset, Dataset) and (isinstance(dataset[0][0], self.Tensor)) and (isinstance(dataset[0][1], self.LongTensor))

	def _check_sampler(self, sampler):
		return isinstance(sampler, Sampler)
