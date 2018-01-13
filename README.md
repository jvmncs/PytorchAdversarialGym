PytorchAdversarialGym
=====================
**A PyTorch-integrated Gym environment for enabling and evaluating new research into attacking and defending neural networks with adversarial perturbations.**  While most research into adversarial perturbations has focused on gradient, decision boundary, or transferability methods, this environment generalizes the adversarial attack problem as a Markov decision process.  This allows for methods from reinforcement learning to be easily applied, a possibility that has only been rarely suggested or investigated in the ML security literature.  In the spirit of Gym, no assumptions are made about the structure of your agent; in particular, adversaries can be either MDP-aware or MDP-agnostic.

Note: Still in development, but master branch should be self-contained.

Dependencies:
- [OpenAI Gym](https://github.com/openai/gym)
- [PyTorch](https://pytorch.org/)

Optional:
- [torchvision](https://github.com/pytorch/vision)
- [pytorch-classification](https://github.com/bearpaw/pytorch-classification)

Roadmap:
- [x] Build out core functionality
- [x] Minimally test environment with pytorch-classification
- [x] Generalize to non-CIFAR Datasets
- [x] Extend to custom samplers to expose environment dynamics
- [x] Extend for different reward functions
- [x] Refactor to incorporate reward Wrappers
- [x] Generalize confidence calculation to other activations
- [x] Implement strict argument for strict epsilon ball enforcement
- [x] Test basic functionality of Untargeted and StaticTargeted wrappers
- [ ] Test basic functionality of DynamicTargeted and DefendMode wrappers
- [ ] Comment/document reward Wrappers (might need to do some more)
- [ ] Test new features
- [ ] Add [BadNets](https://arxiv.org/abs/1708.06733) functionality
- [ ] Make UniformSampler memory-efficient
- [ ] Minimally test environment with torchvision models (esp. non-CIFAR datasets)
- [ ] Spec out rewriting for Foolbox integration/decide if it's worth it