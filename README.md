  # DATA-DRIVEN LATTICES FOR VECTOR QUANTIZATION
![LATTICE_DIAGRAM](https://github.com/BokoAssaf/DeepLatticeUVEQ/assets/143960995/0111f515-3797-4c60-ae0e-8666dd969e45)
## Introduction

In this work we propose an optimization of lattices matrices when using vector quantization methods by using neural networks while performing backpropogation. This repository contains a basic PyTorch implementation of Data-Driven Lattice Learning.   Please refer to our [paper](https://drive.google.com/file/d/1HFgmjkefbeS7VPKzJMlHkIPQede9Id5S/view?usp=sharing) for more details.


## Usage
This code has been tested on Python 3.7.3 and PyTorch 2.1.0.

### Prerequisite
1. PyTorch=2.1.0: https://pytorch.org
2. scipy
3. tqdm
4. matplotlib
5. torchinfo
6. TensorboardX: https://github.com/lanpa/tensorboardX

### Training
```
python main.py

```

### Testing

Our evaluation divided into two parts - Synthetic data and Real data.
For running the code with the synthetic data, change the variable as follow - 
**evaluation = 'synth'**.
For running the the code with the real data, we used a [Federated Learning paradigm](https://github.com/langnatalie/JoPEQ) that developed in Shlezinger lab by Natalie Lang and DR. Nir Shlezinger, so you have to change the variable - 
**evaluation = 'real'** and download the code in the link above.


