import torch
import torch.optim as optim
import copy
import math
from quantization import LatticeQuantization, ScalarQuantization
from privacy import Privacy
import numpy as np

def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users)
    local_models = {}
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models


def distribute_model(local_models, global_model):
    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))


def aggregate_models(local_models, global_model, mechanism):  # FeaAvg
    mean = lambda x: sum(x) / len(x)
    state_dict = copy.deepcopy(global_model.state_dict())
    SNR_layers = []
    for key in state_dict.keys():
        local_weights_average = torch.zeros_like(state_dict[key])
        SNR_users = []
        for user_idx in range(0, len(local_models)):
            local_weights_orig = local_models[user_idx]['model'].state_dict()[key] - state_dict[key]
            local_weights_orig = local_weights_orig
            local_weights = mechanism(local_weights_orig)
            SNR_users.append(torch.var(local_weights_orig) / torch.var(local_weights_orig - local_weights))
            local_weights_average += local_weights
        SNR_layers.append(mean(SNR_users))
        state_dict[key] = state_dict[key]
        state_dict[key] += (local_weights_average / len(local_models)).to(state_dict[key].dtype)
    global_model.load_state_dict(copy.deepcopy(state_dict))


    return mean(SNR_layers)


class JoPEQ:  # Privacy Quantization class
    def __init__(self, args, alpha = 0.6, hex_mat = torch.tensor([[np.sqrt(3) / 2, 0], [1 / 2, 1]]).to(torch.float32), our_round = True):
        # hex_mat = torch.eye(args.lattice_dim).to(torch.float32)
        # hex_mat = (torch.tensor([[0.0618, 0.98], [0.98, 0.043]])).to(torch.float32)
        # hex_mat = torch.tensor([[2,0,0],[1,1,0],[1,0,1]]).to(torch.float32)
        self.vec_normalization = args.vec_normalization
        self.alpha = alpha
        self.args = args
        self.our_round = our_round
        self.hex_mat = hex_mat
        dither_var = None
        if args.quantization:
            if args.lattice_dim > 1:
                # print(f"the dimention is  {self.args.lattice_dim}")
                self.quantizer = LatticeQuantization(args, self.hex_mat, self.our_round)
                dither_var = self.quantizer.P0_cov
            else:
                # print(f"the dimention is  {self.args.lattice_dim}")
                self.quantizer = ScalarQuantization(args)
                dither_var = (self.quantizer.delta ** 2) / 12
        else:
            self.quantizer = None
        if args.privacy:
            self.privacy = Privacy(args, dither_var)
        else:
            self.privacy = None

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
        return input_vec, pad_with,

    def __call__(self, input):
        original_shape = input.shape
        if self.vec_normalization:  # normalize
            input, pad_with = self.divide_into_blocks(input, self.args.lattice_dim)
        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.norm(input - mean) / (input.shape[-1] ** 0.5)

        std = self.alpha * std    ## was 3 * std

        input = (input - mean) / std
        # print(f"input:{input.shape}, mean:{mean.shape}, std:{std.shape}")
        if self.privacy is not None:
            input = self.privacy(input)

        if self.quantizer is not None:
            input = self.quantizer(input)
        # print(f"input:{input.shape}, mean:{mean.shape}, std:{std.shape}")

        # denormalize
        input = (input * std) + mean

        if self.vec_normalization:
            input = input.view(-1)[:-pad_with] if pad_with else input  # remove zero padding

        input = input.reshape(original_shape)
        # print(f"at the end:{input.shape}")
        return input
