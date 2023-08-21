import gc
import sys
from statistics import mean
import time
import torch
from configurations import args_parser
from tqdm import tqdm
import utils
import models
import federated_utils
from torchinfo import summary
import numpy as np
# from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython import display
import torch.nn as nn
import os
import math
# from torchmetrics import MeanSquaredError as MSE


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_hidden1=400, num_hidden2 = 300, num_hidden3 = 500, output_dim = 2):
        super(PyTorchMLP, self).__init__()
        self.output_dim = output_dim
        self.layer1 = torch.nn.Linear(100, num_hidden1)
        self.layer2 = torch.nn.Linear(num_hidden1, num_hidden2)
        self.layer3 = torch.nn.Linear(num_hidden2, num_hidden3)
        self.layer4 = torch.nn.Linear(num_hidden3, output_dim**2)
        self.relu = nn.LeakyReLU() ## The Activation FunctionSSSS
        self.sigmoid = torch.tanh # nn.Sigmoid()#
    def forward(self, inp):
        inp = inp.reshape([-1, 100])
        first_layer = self.relu(self.layer1(inp))
        second_layer = self.relu(self.layer2(first_layer))
        third_layer = self.relu(self.layer3(second_layer))
        forth_layer = self.sigmoid(self.layer4(third_layer))
        return torch.reshape(forth_layer, [self.output_dim, self.output_dim])

def train_model(model, learning_rate=1e-5 , dim = 2):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # built-in L2
    # Adam for our parameter updates
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # built-in L2
    train_acc = []
    epochs = []
    sum_hex = torch.zeros([dim, dim])
    # avg_hex = torch.zeros(self)
    vec_alpha = []
    padded_size = 100 + (dim - 100 % dim) % dim
    loss_vec = []
    # Training
    for t in range(1 , 100):
        # Divide data into mini batches

        for i in range(0, 100):
            # Feed forward to get the logits
            F = torch.normal(mean=1, std=2, size=(dim, dim))
            F = torch.kron(torch.eye(math.ceil(100 / dim)), F)
            local_weights_orig = torch.matmul(F, torch.rand(padded_size, 1))
            hex_mat = model(torch.ones(100))        #torch.ones(100))local_weights_orig

            alpha = 1 ### change it #####
            # hex_mat = torch.reshape(hex_mat, [2, 2])
            # hex_mat = hex_mat.detach().numpy()
            mechanism = federated_utils.JoPEQ(args, alpha, hex_mat)
            local_weights = mechanism(local_weights_orig)
            # print(local_weights.detach())
            local_weights.requires_grad_(requires_grad=True)
            local_weights_orig.requires_grad_(requires_grad=True)
            # Compute the training loss and accuracy
            if type(mechanism) == federated_utils.JoPEQ:
                overloading = mechanism.quantizer.print_overloading_vec()
            loss = criterion(local_weights.reshape(-1, padded_size), local_weights_orig.reshape(-1, padded_size))
            # loss = criterion(local_weights.reshape(-1, 100) - local_weights_orig.reshape(-1, 100), torch.zeros(100).reshape(-1, 100)) #+ overloading
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute training accuracy
        overloading = 0
        sum_hex += hex_mat
        avg_hex = sum_hex/t
        avg_alpha = alpha
        if (t % 20) == 0:
            print("[EPOCH]: %i, [LOSS]: %.6f, [Alpha]: %.3f, [AVG_ALPHA]: %.3f, [Overloading]: %.3f, [Avg_hex_mat]:" % (
            t, loss.item(), alpha, avg_alpha, overloading))
            print(avg_hex) #/torch.linalg.det(hex_mat).to(torch.float32).to(args.device) + 0.00001)#?
        display.clear_output(wait=True)

        # Save error on each epoch
        epochs.append(t)
        #train_acc.append(acc)
        vec_alpha.append(avg_alpha)
        loss_vec.append((loss.item()))

    np_hex_mat = avg_hex.detach().numpy()
    np.save(f'hex_mat_rate{args.R}_dim{args.lattice_dim}.npy', np_hex_mat)

    # plotting

    plt.figure()
    plt.title("Loss vs Epochs")
    plt.plot(epochs, loss_vec, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    if evaluation == "synth": # that part allow us to evaluate our model with *** Synthetic Data ***
        print('Starting the synthetic Evaluation')
        eval_mechanism = federated_utils.JoPEQ(args, alpha, hex_mat, our_round= False)
        val = torch.rand(padded_size)
        # val = convert_uncorrelated2correlated(val, 100)
        new_val = eval_mechanism(val)
        noise = new_val - val
        final_SNR = torch.var(val) / torch.var(noise)
        print(f'the final SNR in dB is: {10*torch.log10(final_SNR)} for R ={args.R} and dim = {args.lattice_dim}')
        return 10*torch.log10(final_SNR)         #, args.R, args.lattice_dim
    elif evaluation == "real":
        print('Starting the real Evaluation')
        fail = os.system(f"python main.py --exp_name=jopeq --quantization --lattice_dim {args.lattice_dim} --R {args.R} --hex_mat")
        # fail = os.system(f"python main.py --hex_mat --lattice_dim 4")
        if fail:
            exit()
        return 1


if __name__ == '__main__':
    args = args_parser()
    args.privacy = False
    # args.R = 6rand
    # args.lattice_dim = 2
    evaluation = "real"
    # pytorchmlp11 = PyTorchMLP(output_dim=args.lattice_dim)
    # train_model(pytorchmlp11, dim=args.lattice_dim)
    SNR_list =[]
    for dim in (3, 9): #range(1,9):
        for rate in range(1,9):
            args.R = rate*dim
            args.lattice_dim = dim
            pytorchmlp11 = PyTorchMLP(output_dim=args.lattice_dim)
            SNR_list.append(train_model(pytorchmlp11, dim=args.lattice_dim))
        # print(f'the list of SNR of dim = {args.lattice_dim} is: {SNR_list}')
        if evaluation == "synth":
            SNR_tensor = torch.stack(SNR_list)
            SNR_numpy = SNR_tensor.detach().numpy()
            np.save(f'output{dim}.npy', SNR_numpy)
        SNR_list = []

    # randomize_zeta()
    print("end of zeta simulation")

##     !!!!!!!!!!!!!!!!   if we run this file, we have to remove the hex_mat line in init (JOPEQ) !!!!!!!!!!!!              ##


