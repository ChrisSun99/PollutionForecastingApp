"""

@author: mengtisun

使用多维数据构建rnn
"""
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from trash import models
import sys

sys.path.append('../')

from mods.config_loader import config
from mods.build_samples_and_targets import build_train_samples_dict, build_train_targets_array


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


if __name__ == "__main__":
    # 设定参数
    target_column = config.conf['model_params']['target_column']
    selected_columns = config.conf['model_params']['selected_columns']
    pred_dim = config.conf['model_params']['pred_dim']
    # use_cuda = config.conf['model_params']['pred_use_cuda']
    use_cuda = False
    # lr = config.conf['model_params']['lr']
    lr = 0.001
    # n_epochs = config.conf['model_params']['epochs']
    n_epochs = 6
    # batch_size = config.conf['model_params']['batch_size']

    # 载入训练样本和目标数据集
    train_samples_dict = build_train_samples_dict()
    train_targets_arr = build_train_targets_array()

    # 构建训练和验证数据集
    trainloader, verifyloader, X_train, y_train, X_verify, y_verify = build_train_and_verify_datasets()

    # 构建模型
    model = models.RNN(input_size=1, hidden_size=64, output_size=1, cell="RNN", num_layers=1, use_cuda=False)

    # 设定优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    train_loss_record = []
    verify_loss_record = []
    for i in range(n_epochs):
        for i, (train_x, train_y) in enumerate(trainloader):

            # x, y = Variable(train_x), Variable(train_y)

            # Run the forward pass
            optimizer.zero_grad()
            train_x = train_x[:, :, :1]
            pred = model.forward(train_x)
            train_y = train_y[:, :10, :]
            train_loss = criterion(pred, train_y)
            train_loss_record.append(train_loss.item())
            lossSum += train_loss.item()
            if i % 10 == 0 and i != 0:
                print("batch: %d , loss is:%f" % (i, lossSum / 10))
                train_loss_record.append(lossSum / 10)
                lossSum = 0

            train_loss.backward()
            optimizer.step()

        print("%d epoch is finished!" % (i + 1))
