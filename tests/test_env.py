import unittest
from PytorchAdversarialGym import AdvEnv, UniformSampler
import gym
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class FakeModel(nn.Module):
    def __init__(self):
        super(FakeModel, self).__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)

class FakeDataset(Dataset):
    def __init__(self):
        super(FakeDataset, self).__init__()
    
    def __getitem__(self, index):
        return torch.FloatTensor([index, index + 1]), torch.LongTensor([index])
    
    def __len__(self):
        return 10

class TestNotImplemented(unittest.TestCase):
	def setUp(self):
		self.model = FakeModel()
		self.dataset = FakeDataset()
		self.env = AdvEnv(self.model, self.dataset)
		perturb = torch.FloatTensor([.1, .2])
		self.action_0 = perturb + self.env.successor[0]

	def tearDown(self):
		del self.model, self.dataset, self.env

	def test_step(self):
		self.assertRaises(NotImplementedError, self.env.step, self.action_0)

	def test_get_reward(self):
		self.assertRaises(NotImplementedError, self.env._get_reward, self.env.successor, self.action_0)

class TestSanityChecks(unittest.TestCase):
	def setUp(self):
		self.wrong_model = 'Model'
		self.wrong_dataset = 'Dataset'
		self.wrong_sampler = 'Sampler'
		self.right_model = FakeModel()
		self.right_dataset = FakeDataset()
		self.right_sampler = UniformSampler(self.right_dataset)

	def tearDown(self):
		del self.wrong_model, self.wrong_dataset, self.wrong_sampler, self.right_model, self.right_dataset, self.right_sampler

	def test_check_model(self):
		self.assertRaises(gym.error.Error, AdvEnv, self.wrong_model, self.right_dataset, sampler = self.right_sampler)
		self.assertRaises(gym.error.Error, AdvEnv, self.right_model, self.wrong_dataset, sampler = self.right_sampler)
		self.assertRaises(gym.error.Error, AdvEnv, self.right_model, self.right_dataset, sampler = self.wrong_sampler)
		env = AdvEnv(self.right_model, self.right_dataset, sampler = self.right_sampler)
		del env

