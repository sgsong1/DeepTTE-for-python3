import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import numpy as np

# Argument parser setup
parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)

# evaluation args
parser.add_argument('--weight_file', type=str)
parser.add_argument('--result_file', type=str)

# cnn args
parser.add_argument('--kernel_size', type=int)

# rnn args
parser.add_argument('--pooling_method', type=str)

# multi-task args
parser.add_argument('--alpha', type=float)

# log file name
parser.add_argument('--log_file', type=str)

args = parser.parse_args()

# Configuration loading
config = json.load(open('./config.json', 'r'))

def train(model, elogger, train_set, eval_set):
    # Record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'train with {device}')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        print(f'Training on epoch {epoch}')
        model.train()
        for input_file in train_set:
            print(f'Train on file {input_file}')

            # Data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            running_loss = 0.0


            for idx, (attr, traj) in enumerate(data_iter):
                # Transform the input to pytorch variable
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                attr = {k: v.to(device) for k, v in attr.items()}
                
                # Handle map objects in traj by converting them to lists or tensors
                traj = {k: torch.tensor(list(v)).to(device) if isinstance(v, map) else v.to(device) for k, v in traj.items()}
                # Evaluate the model on the batch and calculate the loss
                _, loss = model.eval_on_batch(attr, traj, config)

                # Update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track running loss
                running_loss += loss.item()
                print(f'\rProgress: {(idx + 1) * 100.0 / len(data_iter):.2f}%, average loss: {running_loss / (idx + 1):.6f}', end='')


                # 만약 루프가 실행되지 않았으면 idx는 여전히 -1이므로 이를 체크
                if idx == -1:
                    print(f'Training on file {input_file}, no data found.')
                    elogger.log(f'Training Epoch {epoch}, File {input_file}, no data found.')
                else:
                    print(idx)
                    elogger.log(f'Training Epoch {epoch}, File {input_file}, Loss {running_loss / (idx + 1.0)}')


        # Evaluate the model after each epoch
        evaluate(model, elogger, eval_set, save_result=False)

        # Save the weight file after each epoch
        weight_name = f'{args.log_file}_{datetime.datetime.now()}'
        # Ensure the directory exists before saving
        save_dir = './saved_weights'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        elogger.log(f'Save weight file {weight_name}')
        torch.save(model.state_dict(), f'{save_dir}/{weight_name}')

def write_result(fs, pred_dict, attr):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    for i in range(pred_dict['pred'].size()[0]):
        fs.write(f'{label[i][0]:.6f} {pred[i][0]:.6f}\n')

        dateID = attr['dateID'].data[i]
        timeID = attr['timeID'].data[i]
        driverID = attr['driverID'].data[i]

def evaluate(model, elogger, files, save_result=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    result_dir = os.path.dirname(args.result_file)

    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if save_result:
        fs = open(f'{args.result_file}', 'w')

    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)
        
        idx = -1

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)
            
            attr = {k: v.to(device) for k, v in attr.items()}
            
            # Handle map objects in traj by converting them to lists or tensors
            traj = {k: torch.tensor(list(v)).to(device) if isinstance(v, map) else v.to(device) for k, v in traj.items()}

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            if save_result:
                write_result(fs, pred_dict, attr)

            running_loss += loss.item()
        


        if idx == -1:
            print(f'Evaluate on file {input_file}, no data found.')
            elogger.log(f'Evaluate File {input_file}, no data found.')
        else:
            print(f'Evaluate on file {input_file}, loss {running_loss / (idx + 1.0):.6f}')
            elogger.log(f'Evaluate File {input_file}, Loss {running_loss / (idx + 1.0)}')

    if save_result:
        fs.close()

def get_kwargs(model_class):
    model_args = inspect.getfullargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if arg not in model_args:
            kwargs.pop(arg)

    return kwargs

def run():
    # Get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)

    # Model instance
    model = models.DeepTTE.Net(**kwargs)

    # Experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set=config['train_set'], eval_set=config['eval_set'])

    elif args.task == 'test':
        # Load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result=True)

if __name__ == '__main__':
    run()
