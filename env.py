import gym

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from .space import TensorBox
from .util import UniformSampler
class AdversarialEnv(gym.Env):
    """
    An environment for generating and defending against adversarial examples with PyTorch models
        and Datasets.

    Args:
        target_model (subclass of torch.nn.Module): The model we're attacking.
            Currently, only torch.nn.Module is supported.
        dataset (subclass of torch.utils.data.Dataset): The dataset we're using to attack.
            Currently, only Pytorch Dataset objects are supported.
            Note: ToTensor transform must be included.
        target_class (int, optional): If an integer, positive reward requires target_model to
            classify all instances as target_class. Otherwise, positive reward requires
            misclassification.
        skip_target_class (boolean, optional): Whether to allow instances of label target_class to
            be observations.  Can works if target_class is a valid integer.
        defend_mode (bool, optional): If True, positive reward requires correctly classifying
            specific class.  Default is False.
        batch_size (int, optional): Number of instances for the target_model to classify per step.
        episode_length (positive integer, optional): Specifies the number of steps to include in
            each episode.  Default is len(dataset)/.
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
    def __init__(self, target_model, dataset, target_class = None, skip_target_class = True,
                norm = None, defend_mode = False, batch_size = 1, episode_length = None,
                sampler = None, num_workers = 0, use_cuda = torch.cuda.is_available(),
                seed = None):
        super().__init__()
        self.use_cuda = use_cuda
        if seed is not None:
            torch.backend.cudnn.enabled = False
        self.seedey = self._seed(seed)
        self.target_model = target_model.cuda() if use_cuda else target_model.cpu()
        self.dataset = dataset
        space_shape = self.dataset[0][0].size()
        space_shape = (batch_size, *space_shape[1:])
        self.action_space = TensorBox(0, 1, space_shape)
        self.observation_space = TensorBox(0, 1, space_shape)
        self.episode_length = len(self.dataset)//batch_size if not episode_length else episode_length
        self.sampler = UniformSampler(self.dataset, self.torch_rng, len(self.dataset)) if not sampler else sampler

        if not self._check_dataset():
            raise gym.error.Error('Dataset type {} not supported.'.format(type(self.dataset)) +
                              'Currently, dataset must be a subclass of torch.utils.data.Dataset containing FloatTensors')

        if not self._check_model():
            raise gym.error.Error('Model type {} not supported.'.format(type(self.target_model)) +
                              ' Currently, target_model must be a subclass of torch.nn.Module.')

        if not self._check_sampler():
            raise gym.error.Error('Sampler type {} not supported.'.format(type(self.sampler)) +
                               'Currently, sampler must be a subclass of torch.utils.data.sampler.Sampler.')

        self.target_class = target_class
        self.skip_target_class = self.target_class is not None and skip_target_class
        self.defend_mode = defend_mode
        self.batch_size = batch_size
        self.norm = norm
        self.num_workers = num_workers
        self.data_loader = DataLoader(self.dataset, batch_size = self.batch_size, sampler = self.sampler, num_workers = self.num_workers)
        self.iterator = iter(self.data_loader)
        self._reset()

    def _step(self, action):
        try:
            current_obs = self.successor
            if self.skip_target_class:
                self.successor = self.iterator.__next__()
                while (self.successor[1] == self.target_class).any():
                    self.successor = self.iterator.__next__()
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

    def _seed(self, seed):
        integer_types = (int,)
        if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
            raise gym.error.Error('Seed must be a non-negative integer or omitted, not {}.'.format(type(seed)))
        self.torch_rng = torch.manual_seed(seed) if seed is not None else torch.default_generator
        self.seedey = seed
        return [seed]

    def _reset(self):
        self.data_loader = DataLoader(self.dataset, batch_size = self.batch_size, sampler = self.sampler, num_workers = self.num_workers)
        self.iterator = iter(self.data_loader)
        self.successor = self.iterator.__next__()
        if self.use_cuda:
            self.successor[0] = self.successor[0].cuda()
            self.successor[1] = self.successor[1].cuda()
        self.done = False
        self.ix = 0
        return self.successor

    def _get_reward(self, obs, action):
        ground_truth = obs[1]
        target_label = self.target_class
        if self.target_class is not None and self.defend_mode:
            raise gym.error.Error("Argument target_class must be None if using defend_mode.")
        if self.defend_mode:
            target_label = ground_truth
        action = Variable(action, volatile = True)
        outs = self.target_model(action)
        outs = nn.functional.sigmoid(outs)
        confidence, prediction = torch.max(outs, 1)
        if self.norm is not None:
            norm_penalty = self.norm_on_batch(action.data - obs[0], self.norm)
        else:
            norm_penalty = 0
        if target_label is not None:
            reward = (target_label == prediction.data).float()*confidence.data - (target_label != prediction.data).float()*confidence.data - norm_penalty
        else:
            reward = (ground_truth != prediction.data).float()*confidence.data - (ground_truth == prediction.data).float()*confidence.data - norm_penalty
        return reward, {'label':ground_truth, 'prediction':prediction.data, 'confidence':confidence.data, 'norm':norm_penalty if self.norm is not None else None}

    def norm_on_batch(self, input, p):
        # Assume dimension 0 is batch dimension
        norm_penalty = input
        while len(norm_penalty.size())>1:
            norm_penalty = torch.norm(norm_penalty, p, -1)
        return norm_penalty

    def _check_model(self):
        return isinstance(self.target_model, nn.Module)

    def _check_dataset(self):
        return isinstance(self.dataset, Dataset) and (isinstance(self.dataset[0][0], torch.FloatTensor) or isinstance(self.dataset[0][0], torch.cuda.FloatTensor))

    def _check_sampler(self):
        return isinstance(self.sampler, Sampler)