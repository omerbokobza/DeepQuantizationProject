from configurations import args_parser
from DNN import *
import math

if __name__ == '__main__':
    args = args_parser()
    evaluation = "real"
    SNR_list =[]
    for dim in (2, 9): #range(1,9):
        F = torch.normal(mean=1, std=2, size=(dim, dim))                # randomizing a gaussian matrix to create correlation in each sub vector
        F = torch.kron(torch.eye(math.ceil(100 / dim)), F)              # adjusting a larger matrix that suits the size of all vectors concatinate
        for rate in range(2,9):
            args.R = rate*dim
            args.lattice_dim = dim
            pytorchmlp11 = PyTorchMLP(output_dim=args.lattice_dim)
            SNR_list.append(train_model(pytorchmlp11, args, F))
        if evaluation == "synth":
            SNR_tensor = torch.stack(SNR_list)
            SNR_numpy = SNR_tensor.detach().numpy()
            np.save(f'output{dim}.npy', SNR_numpy)
        SNR_list = []

    # randomize_zeta()
    print("end of zeta simulation")

##     !!!!!!!!!!!!!!!!   if we run this file, we have to remove the hex_mat line in init (JOPEQ) !!!!!!!!!!!!              ##