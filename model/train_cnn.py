import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from mods import models

from mods.build_samples_and_targets import build_train_samples_dict, build_train_targets_array
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
    #use_cuda = config.conf['model_params']['pred_use_cuda']
    use_cuda= False
    # lr = config.conf['model_params']['lr']
    lr = 0.001
    # n_epochs = config.conf['model_params']['epochs']
    n_epochs = 6
    # batch_size = config.conf['model_params']['batch_size']
    batch_size = 100
    n_test = pred_dim
    # for data split


    # 载入训练样本和目标数据集
    train_samples_dict = build_train_samples_dict()
    train_targets_arr = build_train_targets_array()

    # 构建训练和验证数据集
    trainloader, verifyloader, X_train, y_train, X_verify, y_verify = build_train_and_verify_datasets()


    # 定义模型
    model = models.ConvNet()
    input_size = X_train.shape[2]
    alpha_layer = models.AlphaLayer(input_size)
    weights_layer = models.WeightsLayer(X_train.shape[1], y_train.shape[1])


    # 设定优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    total_step = len(trainloader)
    loss_list = []
    acc_list = []
    for epoch in range(n_epochs):
        for i, (train_x, train_y) in enumerate(trainloader):
            # Run the forward pass
            train_x = train_x.unsqueeze(1)
            train_out = model(train_x)
            #train_out = outputs[:, -pred_dim:, 0]
            train_y = train_y[:, :10, :]
            print(train_out.shape, train_y.shape)

            loss = criterion(train_out, train_y[:, :, 0])
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = train_out.size()
            _, predicted = torch.max(train_out.data, 1)
            correct = (predicted == np.long(train_y)).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

    # # Train the model
    # total_step = len(trainloader)
    # verify_loss_record = []
    # train_loss_record = []
    # acc_list = []
    # for epoch in range(n_epochs):
    #     # 训练集
    #     for train_x, train_y in trainloader:
    #         input_weights = alpha_layer(train_x)[:, :, 0]
    #         output_weights = weights_layer(input_weights)
    #         train_x = train_x.unsqueeze(1)
    #         print("train_x: ", train_x.shape)
    #         train_out = model(train_x)
    #         #train_out = train_out[:, -pred_dim:, 0]
    #         print(train_out.shape, output_weights.shape)
    #         train_loss = criterion(torch.mul(train_out, output_weights), train_y[:, :, 0])
    #         optimizer.zero_grad()
    #         train_loss.backward()
    #         optimizer.step()
    #     train_loss_record.append(train_loss)
    #
    #     # 验证集
    #     with torch.no_grad():
    #         for verify_x, verify_y in verifyloader:
    #             input_weights = models.alpha_layer(verify_x)[:, :, 0]
    #             output_weights = models.weights_layer(input_weights)
    #             cnn_verify_out = model(verify_x)
    #             cnn_verify_out = cnn_verify_out[:, -pred_dim:, 0]
    #             verify_loss = criterion(torch.mul(cnn_verify_out, output_weights), verify_y[:, :, 0])
    #         verify_loss_record.append(verify_loss)
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
