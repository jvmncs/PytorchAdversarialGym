import torch
from torch.utils.data.sampler import Sampler


class UniformSampler(Sampler):
    def __init__(self, data_source, seed_generator, n_samples):
        super().__init__(data_source)
        m = len(data_source)
        p = torch.DoubleTensor(1, m).fill_(1 / m)
        # TODO: Find a better way to do this; memory-intensive for large `n_samples`
        # Went this route because I wanted to feed in the seed_generator (I think)
        self.choices = p.multinomial(n_samples, replacement = True, generator = seed_generator)

    def __iter__(self):
        return iter(self.choices[0])

    def __len__(self):
        return len(self.choices)