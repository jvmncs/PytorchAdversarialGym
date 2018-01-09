import gym
import torch

class TensorBox(gym.Space):
    """
    Implementing the Box space with torch.Tensor instead of ndarray.

    Args:
        low (float or int):
        high (float or int):
        shape (torch.Size or subclass of tuple):
    """
    def __init__(self, low, high, shape):
        super().__init__()
        assert isinstance(low, (int, float)) and isinstance(high, (int, float)) and isinstance(shape, tuple)
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self):
        return torch.Tensor(*self.shape).uniform_(self.low, self.high).unsqueeze(0)

    def contains(self, x):
        return x.size() == (1, *self.shape) and (x >= self.low).all() and (x <= self.high).all()

    def to_jsonable(self, sample_n):
        tensor_bool = isinstance(sample_n, torch.Tensor)
        seq_bool = issubclass(sample_n, tuple) or issubclass(sample_n, list)
        assert tensor_bool or seq_bool
        if tensor_bool:
            if len(sample_n.size()) == 3:
                return [sample_n.tolist()]
            else:
                return sample_n.tolist()
        else:
            return torch.cat([x.unsqueeze(0) for x in sample_n]).tolist()

    def from_jsonable(self, sample_n):
        return torch.Tensor(sample_n)
