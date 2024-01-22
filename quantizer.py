import torch
import numpy as np

class LatticeQuantization:
    def __init__(self, args, hex_mat, our_round = True):
        self.gamma = args.gamma
        self.args = args
        self.our_round = our_round
        if self.our_round:   # our differentiable round !!!!
            self.round = self.sigm
        else:           # the original step function - *** not differentiable ***
            self.round = torch.round
        self.dim = args.lattice_dim
        self.gen_mat = hex_mat

        # estimate P0_cov
        self.delta = (2 * args.gamma) / (2 ** args.R + 1)
        self.egde = args.gamma - (self.delta / 2)

        orthog_domain_dither = torch.from_numpy(np.random.uniform(low=-self.delta / 2, high=self.delta / 2, size=[args.lattice_dim, 1000])).float()
        lattice_domain_dither = torch.matmul(self.gen_mat, orthog_domain_dither)
        self.P0_cov = torch.cov(lattice_domain_dither)

    def sigm(self, x):
        return x+0.2*(torch.cos(2*torch.pi*(x+0.25)))

    def divide_into_blocks(self, input, dim=2):
        # Zero pad if needed
        input = input.view(-1)
        modulo = len(input) % dim
        if modulo:
            pad_with = dim - modulo
            input = torch.cat((input, torch.zeros(pad_with).to(input.dtype).to(input.device)))
        else:
            pad_with = 0
        input_vec = input.view(dim, -1)  # divide input into blocks
        return input_vec, pad_with

    def __call__(self, input):
        original_shape = input.shape
        input, pad_with = self.divide_into_blocks(input, self.args.lattice_dim)
        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.norm(input - mean) / (input.shape[-1] ** 0.5)
        
        input_vec = (input - mean) / std
        
        dither = torch.zeros_like(input_vec, dtype=input_vec.dtype)
        dither = torch.matmul(self.gen_mat, dither.uniform_(-self.delta / 2, self.delta / 2))  # generate dither

        input_vec = input_vec + dither

        # quantize
        orthogonal_space = torch.matmul(torch.inverse(self.gen_mat), input_vec)
        q_orthogonal_space = self.delta * self.round(orthogonal_space / self.delta)
        # self.calc_overloading_vec(q_orthogonal_space)
        q_orthogonal_space[q_orthogonal_space >= self.egde] = self.egde
        q_orthogonal_space[q_orthogonal_space <= -self.egde] = -self.egde
        input_vec = torch.matmul(self.gen_mat, q_orthogonal_space)
        
        input = input_vec - dither
        
        input = (input * std) + mean

        input = input.view(-1)[:-pad_with] if pad_with else input  # remove zero padding

        input = input.reshape(original_shape)
        # print(f"at the end:{input.shape}")
        return input

