import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
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


def save_models_and_records(train_loss_record, verify_loss_record, model):
    # 参数
    target_column = config.conf['model_params']['target_column']

    # 损失函数记录
    train_loss_list = [float(p.detach().cpu().numpy()) for p in train_loss_record]
    verify_loss_list = [float(p.cpu().numpy()) for p in verify_loss_record]

    with open('../tmp/cnn_train_loss.pkl', 'w') as f:
        json.dump(train_loss_list, f)
    with open('../tmp/cnn_verify_loss.pkl', 'w') as f:
        json.dump(verify_loss_list, f)

    # 保存模型文件
    torch.save(model.state_dict(), '../tmp/cnn_state_dict_{}.pth'.format(target_column))

    # 保存模型结构参数
    cnn_model_struc_params = {
        'cnn': {
            'batch_size': model.batch_size,
            'learning_rate': model.lr,
            'epoch': model.n_epochs
        }
    }

    with open('../tmp/cnn_model_struc_params.pkl', 'w') as f:
        json.dump(cnn_model_struc_params, f)


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

    # 设定优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    total_step = len(trainloader)
    train_loss_record = []
    verify_loss_record = []
    for epoch in range(n_epochs):
        for i, (train_x, train_y) in enumerate(trainloader):
            # Run the forward pass
            train_x = train_x.unsqueeze(1)
            train_out = model(train_x)
            # train_out = outputs[:, -pred_dim:, 0]
            train_y = train_y[:, :10, :]
            train_loss = criterion(train_out, train_y[:, :, 0])
            train_loss_record.append(train_loss.item())

            # Back-propagation and perform Adam optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # 验证集
        with torch.no_grad():
            for i, (verify_x, verify_y) in enumerate(verifyloader):
                verify_x = verify_x.unsqueeze(1)
                cnn_verify_out = model(verify_x)
                verify_y = verify_y[:, :10, :]
                verify_loss = criterion(cnn_verify_out, verify_y[:, :, 0])
            verify_loss_record.append(verify_loss)

        if epoch % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Verify Loss: {:.4f}'
                  .format(epoch + 1, n_epochs, i + 1, total_step, train_loss.item(), verify_loss))

    # 保存模型
    #save_models_and_records(train_loss_record, verify_loss_record, model)
