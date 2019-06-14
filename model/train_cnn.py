import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from mods import model_evaluations
from mods import models
import json

import sys

sys.path.append('../')

from mods.config_loader import config
from mods.build_samples_and_targets import build_train_samples_dict, build_train_targets_array

"""
A univariate CNN model: support multiple features or types of observation at each time step. 
Params: 
n_input: The number of lag observations to use as input to the model.
n_filters: The number of parallel filters.
n_kernel: The number of time steps considered in each read of the input sequence.
n_epochs: The number of times to expose the model to the whole training dataset.
n_batch: The number of samples within an epoch after which the weights are updated.
"""

if __name__ == "__main__":
    # 设定参数
    target_column = config.conf['model_params']['target_column']
    selected_columns = config.conf['model_params']['selected_columns']
    pred_dim = config.conf['model_params']['pred_dim']
    use_cuda = config.conf['model_params']['pred_use_cuda']
    #lr = config.conf['model_params']['lr']
    lr = 0.001
    #n_epochs = config.conf['model_params']['epochs']
    n_epochs = 6
    #batch_size = config.conf['model_params']['batch_size']
    batch_size = 100
    n_test = pred_dim
    # for data split

    # 载入训练样本和目标数据集
    train_samples_dict = build_train_samples_dict()
    train_targets_arr = build_train_targets_array()

    # 构造训练集
    X_train = np.concatenate([train_samples_dict[col] for col in train_samples_dict.keys()], axis=2).astype(np.float32)
    X_train = np.hstack((X_train, np.zeros([X_train.shape[0], pred_dim, X_train.shape[2]]).astype(np.float32)))
    y_train = train_targets_arr.astype(np.float32)

    torch_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    # load = (weight, 32, 1, 5)
    if use_cuda:
        loader = [(tx.cuda(), ty.cuda()) for (tx, ty) in loader]

    # # 构造测试集
    # train_samples_dict = build_train_samples_dict()
    # train_targets_arr = build_train_targets_array()
    #
    # X_test = X_train[-1]
    # y_test = y_train
    # test_dataset = Data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    model = models.ConvNet()

    # 设定优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    total_step = len(loader)
    loss_record = []
    acc_list = []
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(loader):
            # Run the forward pass
            print(images.shape, labels.shape)
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss_record.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

    # 测试模型
    # model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in test_loader:
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    #     print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # #保存模型
    # with open('../tmp/model_struc_params_cnn.pkl', 'w') as f:
    #     json.dump(model, f)
    #
    # # 保存损失函数记录
    # with open('../tmp/train_loss.pkl', 'w') as f:
    #     json.dump(loss_record, f)
    #
    #
