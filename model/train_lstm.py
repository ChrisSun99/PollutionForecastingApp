# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

使用多维数据构建lstm
"""
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import numpy as np
import json
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.build_samples_and_targets import build_train_samples_dict, build_train_targets_array
from mods.loss_criterion import criterion
from mods.models import LSTM, AlphaLayer, WeightsLayer, initialize_lstm_params, initialize_alpha_layer_params, \
    initialize_weights_layer_params


def save_models_and_records(train_loss_record, verify_loss_record, lstm, alpha_layer, weights_layer):
    """保存模型和相应记录"""
    # 参数
    target_column = config.conf['model_params']['target_column']

    # 损失函数记录
    train_loss_list = [float(p.detach().cpu().numpy()) for p in train_loss_record]
    verify_loss_list = [float(p.cpu().numpy()) for p in verify_loss_record]

    with open('../tmp/lstm_train_loss.pkl', 'w') as f:
        json.dump(train_loss_list, f)
    with open('../tmp/lstm_verify_loss.pkl', 'w') as f:
        json.dump(verify_loss_list, f)

    # 保存模型文件
    torch.save(lstm.state_dict(), '../tmp/lstm_state_dict_{}.pth'.format(target_column))
    torch.save(alpha_layer.state_dict(), '../tmp/alpha_layer_state_dict_{}.pth'.format(target_column))
    torch.save(weights_layer.state_dict(), '../tmp/weights_layer_state_dict_{}.pth'.format(target_column))

    # 保存模型结构参数
    model_struc_params = {
        'lstm': {
            'input_size': lstm.input_size
        },
        'alpha_layer': {
            'features_num': alpha_layer.features_num
        },
        'weights_layer': {
            'weights_num': weights_layer.weights_num,
            'output_size': weights_layer.output_size
        }
    }

    with open('../tmp/model_struc_params.pkl', 'w') as f:
        json.dump(model_struc_params, f)


def build_train_and_verify_datasets():
    """构建训练和验证数据集"""
    pred_dim = config.conf['model_params']['pred_dim']
    batch_size = config.conf['model_params']['batch_size']
    use_cuda = config.conf['model_params']['train_use_cuda']

    # 载入训练样本和目标数据集
    train_samples_dict = build_train_samples_dict()
    train_targets_arr = build_train_targets_array()

    # 构造训练集
    X = np.concatenate([train_samples_dict[col] for col in train_samples_dict.keys()], axis=2).astype(np.float32)
    X = np.hstack((X, np.zeros([X.shape[0], pred_dim, X.shape[2]]).astype(np.float32)))
    y = train_targets_arr.astype(np.float32)

    # shuffle操作
    id_list = np.random.permutation(range(X.shape[0]))
    X, y = X[list(id_list), :, :], y[list(id_list), :, :]

    # 划分训练集和验证集
    split_num = int(0.9 * X.shape[0])
    X_train, y_train = X[:split_num, :, :], y[:split_num, :, :]
    X_verify, y_verify = X[split_num:, :, :], y[split_num:, :, :]

    train_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    verify_dataset = Data.TensorDataset(torch.from_numpy(X_verify), torch.from_numpy(y_verify))
    verifyloader = DataLoader(verify_dataset, batch_size=X_verify.shape[0], pin_memory=False)
    # if use_cuda:
    #     torch.cuda.empty_cache()
    #     trainloader = [(train_x.cuda(), train_y.cuda()) for (train_x, train_y) in trainloader]
    #     verifyloader = [(verify_x.cuda(), verify_y.cuda()) for (verify_x, verify_y) in verifyloader]

    return trainloader, verifyloader, X_train, y_train, X_verify, y_verify


if __name__ == '__main__':
    # 设定参数
    target_column = config.conf['model_params']['target_column']
    selected_columns = config.conf['model_params']['selected_columns']
    pred_dim = config.conf['model_params']['pred_dim']
    use_cuda = config.conf['model_params']['train_use_cuda']
    lr = config.conf['model_params']['lr']
    epochs = config.conf['model_params']['epochs']
    batch_size = config.conf['model_params']['batch_size']

    # 构建训练和验证数据集 ——————————————————————————————————————————————————————————————————————————————————————————
    trainloader, verifyloader, X_train, y_train, X_verify, y_verify = build_train_and_verify_datasets()

    # 构建模型 ———————————————————————————————————————————————————————————————————————————————————————————————————
    lstm_input_size = X_train.shape[2]
    lstm = LSTM(lstm_input_size)  # 样本的特征数
    alpha_layer = AlphaLayer(lstm_input_size)
    weights_layer = WeightsLayer(X_train.shape[1], y_train.shape[1])
    lstm = initialize_lstm_params(lstm)
    alpha_layer = initialize_alpha_layer_params(alpha_layer)
    weights_layer = initialize_weights_layer_params(weights_layer)

    # if use_cuda:
    #     lstm = lstm.cuda()
    #     alpha_layer = alpha_layer.cuda()
    #     weights_layer = weights_layer.cuda()

    # 设定优化器 —————————————————————————————————————————————————————————————————————————————————————————————————
    optimizer = torch.optim.Adam(
        [
            {'params': lstm.parameters()},
            {'params': alpha_layer.parameters()},
            {'params': weights_layer.parameters()}
        ],
        lr=lr
    )

    # 模型训练和保存 ——————————————————————————————————————————————————————————————————————————————————————————————
    train_loss_record, verify_loss_record = [], []
    early_stop_steps = 200
    sum = torch.tensor(early_stop_steps - 50).int()
    stop_criterion = torch.tensor(1).byte()
    if use_cuda:
        sum = sum.cuda()
        stop_criterion = stop_criterion.cuda()

    for epoch in range(epochs):
        # 训练集
        for train_x, train_y in trainloader:
            input_weights = alpha_layer(train_x)[:, :, 0]
            output_weights = weights_layer(input_weights)
            lstm_train_out = lstm(train_x)
            lstm_train_out = lstm_train_out[:, -pred_dim:, 0]
            train_loss = criterion(torch.mul(lstm_train_out, output_weights), train_y[:, :, 0])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        train_loss_record.append(train_loss)

        # 验证集
        with torch.no_grad():
            for verify_x, verify_y in verifyloader:
                input_weights = alpha_layer(verify_x)[:, :, 0]
                output_weights = weights_layer(input_weights)
                lstm_verify_out = lstm(verify_x)
                lstm_verify_out = lstm_verify_out[:, -pred_dim:, 0]
                verify_loss = criterion(torch.mul(lstm_verify_out, output_weights), verify_y[:, :, 0])
            verify_loss_record.append(verify_loss)

        if epoch % 5 == 0:
            print(epoch, train_loss, verify_loss)

            # 早停
            if epoch > 10 + early_stop_steps:
                for i in range(-early_stop_steps, 0):
                    if i == -early_stop_steps:
                        s = (verify_loss_record[i] > verify_loss_record[i - 1]).int()
                    else:
                        s += (verify_loss_record[i] > verify_loss_record[i - 1]).int()

                if (s > sum) == stop_criterion:
                    break

        if epoch % 500 == 0:
            save_models_and_records(train_loss_record, verify_loss_record, lstm, alpha_layer, weights_layer)

    save_models_and_records(train_loss_record, verify_loss_record, lstm, alpha_layer, weights_layer)

