import torch
import numpy as np

# class our_round(torch.nn.Module):
#     def __init__(self):
#         super(our_round, self).__init__()
#         self.round = torch.round
#
#     def forward(self, input):
#         # self.save_for_backward(x)
#         return self.round(input)#x+0.2+0.2*(torch.cos(2*torch.pi*(x+0.25))-1)
#
#     def backward(self, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = sigm(input).grad
#         return grad_input


def sigm(x):
    return x+0.2*(torch.cos(2*torch.pi*(x+0.25))) #1 / (1 + torch.exp(-50*x))

def grad_round(x):
    # ctx.save_for_backward
    output = sigm(x + 100)
    for i in range(-99,100):
        output = output + sigm(x - i)
    # print(output - 100)
    return output - 100

    # def backward(ctx, grad_output):
    #     return grad_output

class LatticeQuantization:
    def __init__(self, args, hex_mat, our_round = True):
        self.gamma = args.gamma
        self.args = args
        self.overloading_vec = []
        self.our_round = our_round
        if self.our_round:   # our differentiable round !!!!
            self.round = sigm
        else:           # the original step function - *** not differentiable ***
            self.round = torch.round
        self.dim = args.lattice_dim
        self.gen_mat = hex_mat
        # hex_mat = hex_mat
        # # lattice generating matrix
        # dete = (torch.linalg.det(hex_mat).to(torch.float32).to(args.device))
        # if dete == 0:
        #     dete = 0.00000000000001
        # self.gen_mat = hex_mat/dete#?
        # self.gen_mat = torch.from_numpy(gen_mat).to(torch.float32).to(args.device)
        # print(self.gen_mat)

        # estimate P0_cov
        self.delta = (2 * args.gamma) / (2 ** args.R + 1)
        self.egde = args.gamma - (self.delta / 2)

        orthog_domain_dither = torch.from_numpy(np.random.uniform(low=-self.delta / 2, high=self.delta / 2, size=[args.lattice_dim, 1000])).float()
        lattice_domain_dither = torch.matmul(self.gen_mat, orthog_domain_dither)
        self.P0_cov = torch.cov(lattice_domain_dither)

    def print_overloading_vec(self):
        sum = 0
        len = 0
        for item in self.overloading_vec:
            sum += item[0]
            len += item[1][0] * item[1][1]
        self.overloading_vec = []
        return (sum * 100 / len)

    def calc_overloading_vec(self, q_orthogonal_space):
        overloading = torch.zeros_like(q_orthogonal_space)
        overloading[q_orthogonal_space >= self.egde] = 1
        overloading[q_orthogonal_space <= -self.egde] = 1
        overloading_sum = float(torch.sum(torch.sum(overloading, dim=-1), dim=0))
        overloading_shape = q_orthogonal_space.shape
        # overloading = (float(torch.sum(torch.sum(overloading, dim=-1), dim=0)) / int(q_orthogonal_space.shape[0] * q_orthogonal_space.shape[1]))* 100
        self.overloading_vec.append([overloading_sum, overloading_shape])

    def __call__(self, input_vec):
        # input_vec = input_vec.view(self.dim,-1)#!!!!!!!!!!!!!!!!!!
        dither = torch.zeros_like(input_vec, dtype=input_vec.dtype)
        # print(f'the genmat shape is:{self.gen_mat.shape}')
        # print(f'the dither vec is{dither.shape}')
        # if self.args.hex_mat:
            # print(f'the input vec:{input_vec}, the dither is:{dither}, the gen mat:{self.gen_mat}')
        dither = torch.matmul(self.gen_mat, dither.uniform_(-self.delta / 2, self.delta / 2))  # generate dither

        input_vec = input_vec + dither

        # quantize
        orthogonal_space = torch.matmul(torch.inverse(self.gen_mat), input_vec)
        # print((self.round(orthogonal_space / self.delta)).detach()) #!!!!!!
        # if self.our_round == False:
        #     print((self.round(orthogonal_space / self.delta)).detach())
        q_orthogonal_space = self.delta * self.round(orthogonal_space / self.delta)
        self.calc_overloading_vec(q_orthogonal_space)
        q_orthogonal_space[q_orthogonal_space >= self.egde] = self.egde
        q_orthogonal_space[q_orthogonal_space <= -self.egde] = -self.egde
        input_vec = torch.matmul(self.gen_mat, q_orthogonal_space)

        return input_vec - dither


class ScalarQuantization:
    def __init__(self, args):
        # quantization levels spacing
        self.delta = (2 * args.gamma) / (2 ** args.R + 1)
        self.egde = args.gamma - (self.delta/2)

    def __call__(self, input):
        # decode
        dither = torch.zeros_like(input, dtype=torch.float32)
        dither = dither.uniform_(-self.delta / 2, self.delta / 2)  # generate dither
        input = input + dither

        # mid-tread
        q_input = self.delta * torch.round(input / self.delta)
        q_input[q_input >= self.egde] = self.egde
        q_input[q_input <= -self.egde] = -self.egde

        return q_input - dither

if __name__ == '__main__':
    x = torch.rand(5)
    print(x)
    print(sigm(x))
