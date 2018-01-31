from setuptools import setup

def read(filepath, lines = 3):
	return_string = ''
	with open(filepath) as file:
		for line in range(lines):
			return_string += '{}\n'.format(line)
	return return_string

setup(
    name = "PytorchAdversarialGym",
    version = "0.0.1",
    author = "Jason Mancuso",
    author_email = "jvmancuso87@gmail.com",
    description = ("""A PyTorch-integrated Gym environment for enabling and evaluating new 
    	research into attacking and defending neural networks with adversarial perturbations."""),
    license = "MIT",
    keywords = "reinforcement learning machine learning adversarial artificial intelligence gym pytorch",
    url = "http://packages.python.org/PytorchAdversarialGym",
    packages=['PytorchAdversarialGym', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
