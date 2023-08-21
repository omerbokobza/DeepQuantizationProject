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


if __name__ == '__main__':
    for mdl in ['cnn2', 'mlp', 'linear']:
        start_time = time.time()
        args = args_parser()
        args.model = mdl
        args.alpha = 1
        boardio, textio, best_val_acc, path_best_model = utils.initializations(args)
        textio.cprint(str(args))

        # data
        train_data, test_loader = utils.data(args)
        input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)

        # model
        if args.model == 'mlp':
            global_model = models.FC2Layer(input, output)
        elif args.model == 'cnn2':
            global_model = models.CNN2Layer(input, output, args.data)
        elif args.model == 'cnn3':
            global_model = models.CNN3Layer()
        else:
            global_model = models.Linear(input, output)
        print(str(summary(global_model)))

        textio.cprint(str(summary(global_model)))
        global_model.to(args.device)

        train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
        test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

        # learning curve
        train_loss_list = []
        val_acc_list = []
        overloading_list = []
        avg_SNR_list = []

        #  inference
        if args.eval:
            global_model.load_state_dict(torch.load(path_best_model))
            test_acc = utils.test(test_loader, global_model,test_creterion, args.device)
            textio.cprint(f'eval test_acc: {test_acc:.0f}%')
            gc.collect()
            sys.exit()

        # training loops
        local_models = federated_utils.federated_setup(global_model, train_data, args)
        if args.hex_mat: # if we ran the code with load hex_mat from the gen_mat file
            loaded_hex_mat = torch.from_numpy(np.load(f'hex_mat_rate{args.R}_dim{args.lattice_dim}.npy')).to(torch.float32)
            # print(f"working on loaded mat: {loaded_hex_mat} with shape {loaded_hex_mat.shape} from type {type(loaded_hex_mat)}")
            mechanism = federated_utils.JoPEQ(args, args.alpha, loaded_hex_mat, our_round=False)
        else: # running the regular federated process
            mechanism = federated_utils.JoPEQ(args, args.alpha, our_round=False)

        SNR_list = []

        for global_epoch in tqdm(range(0, args.global_epochs)):
            federated_utils.distribute_model(local_models, global_model)
            users_loss = []

            for user_idx in range(args.num_users):
                user_loss = []
                for local_epoch in range(0, args.local_epochs):
                    user = local_models[user_idx]
                    train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                       train_creterion, args.device, args.local_iterations)
                    if args.lr_scheduler:
                        user['scheduler'].step(train_loss)
                    user_loss.append(train_loss)
                users_loss.append(mean(user_loss))

            train_loss = mean(users_loss)
            SNR = federated_utils.aggregate_models(local_models, global_model, mechanism)  # FeaAvg
            SNR_list.append(SNR)

            val_acc = utils.test(val_loader, global_model, test_creterion, args.device)

            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)

            boardio.add_scalar('train', train_loss, global_epoch)
            boardio.add_scalar('validation', val_acc, global_epoch)
            gc.collect()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(global_model.state_dict(), path_best_model)

            if type(mechanism) == federated_utils.JoPEQ:
                overloading = mechanism.quantizer.print_overloading_vec()
                overloading_list.append(overloading)
            avg_SNR_list.append(10 * torch.log10(sum(SNR_list) / len(SNR_list)))
            textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | '
                          f'val_acc: {val_acc:.0f}% | '
                          f'SNR: {10 * torch.log10(SNR):.3f} | '
                          f'avg SNR: {10 * torch.log10(sum(SNR_list) / len(SNR_list)):.3f} | '
                          f'OL: {overloading:.4f}%')

        np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
        np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)
        np.save(f'checkpoints/{args.exp_name}/SNR_dim{args.lattice_dim}_rate{args.R}_{mdl}_list.npy', 10 * torch.log10(sum(SNR_list) / len(SNR_list)))
        elapsed_min = (time.time() - start_time) / 60
        textio.cprint(f'total execution time: {elapsed_min:.0f} min')
        # textio.cprint(f'The overloading vector is: {overloading_list}')
        # textio.cprint(f'The AVG SNR vector is: {avg_SNR_list}')