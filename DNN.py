import quantizer
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torch
import numpy as np

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

def train_model(model, args, F, learning_rate=1e-5):

    evaluation = "real"
    
    dim = args.lattice_dim
    # setting the training model
    criterion = torch.nn.MSELoss()                                          # setting the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)      # setting the optimizer

    # initializing the data
    epochs = []                                                             # setting the epochs vec for convergance plot
    loss_vec = []                                                           # setting the loass vec for convergance plot
    sum_hex = torch.zeros([dim, dim])                                       # setting the sum of all matrices for avraging at the end
    padded_size = 100 + (dim - 100 % dim) % dim                             # setting the padded vector size to get a whole number of sub vectors (depends on the dimantion)

    # Training
    # epochs
    for t in range(1 , 100):
        
        # iterations
        for i in range(0, 100):
            # Feed forward to get the logits
            # setting the data to be a set of correlated sub vectors
            iid_vec = torch.rand(padded_size, 1)                            # randomizing an iid vector
            local_weights_orig = torch.matmul(F, iid_vec)                   # creating the correlation for each subvector
            hex_mat = model(torch.ones(100))                                # creating the current generator matrix using the NLP

            # creating the quantizer with the curent generator matrix and apllying it on the generated data
            mechanism = quantizer.LatticeQuantization(args, hex_mat, True)
            local_weights = mechanism(local_weights_orig)
            local_weights.requires_grad_(requires_grad=True)
            local_weights_orig.requires_grad_(requires_grad=True)
    
            # Computing the loss and backpropegating
            loss = criterion(local_weights.reshape(-1, padded_size), local_weights_orig.reshape(-1, padded_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Displaying the results
        overloading = 0
        sum_hex += hex_mat
        avg_hex = sum_hex/t
        avg_alpha = 1
        if (t % 20) == 0:
            print("[EPOCH]: %i, [LOSS]: %.6f, [Avg_hex_mat]:" % (
            t, loss.item()))
            print(avg_hex)
        # display.clear_output(wait=True)                                 #!!!!!!!! what is that???!!!!!!

        # Save error for each epoch
        epochs.append(t)
        loss_vec.append((loss.item()))

    np_hex_mat = avg_hex.detach().numpy()
    np.save(f'hex_mat_rate{args.R}_dim{args.lattice_dim}.npy', np_hex_mat)

    # Plotting the results
    plt.figure()
    plt.title("Loss vs Epochs")
    plt.plot(epochs, loss_vec, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    if evaluation == "synth": # that part allow us to evaluate our model with *** Synthetic Data ***
        print('Starting the synthetic Evaluation')
        eval_mechanism = quantizer.LatticeQuantization(args, hex_mat, our_round= False)
        val = torch.rand(padded_size)
        new_val = eval_mechanism(val)
        noise = new_val - val
        final_SNR = torch.var(val) / torch.var(noise)
        print(f'the final SNR in dB is: {10*torch.log10(final_SNR)} for R ={args.R} and dim = {args.lattice_dim}')
        return 10*torch.log10(final_SNR)         #, args.R, args.lattice_dim
    elif evaluation == "real":
        print('Starting the real Evaluation')
        fail = os.system(f"python main.py --exp_name=jopeq --quantization --lattice_dim {args.lattice_dim} --R {args.R} --hex_mat")
        if fail:
            exit()
        return 1