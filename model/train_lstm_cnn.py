
from mods.models import ConvLSTM
import torch
import numpy as np
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mods.config_loader import config
from mods.build_samples_and_targets import build_train_samples_dict, build_train_targets_array
from mods.loss_criterion import criterion


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
    use_cuda = config.conf['model_params']['train_use_cuda']
    lr = config.conf['model_params']['lr']
    epochs = config.conf['model_params']['epochs']
    batch_size = config.conf['model_params']['batch_size']

    # 建立模型
    convlstm = ConvLSTM(input_channels=1, hidden_channels=[1, 192, 192, 11, 11], kernel_size=3, step=5,
                        effective_step=[4])
    loss_fn = torch.nn.MSELoss()

    # 构建训练和验证数据集
    trainloader, verifyloader, X_train, y_train, X_verify, y_verify = build_train_and_verify_datasets()

    output_list = []
    target_list = []

    # input = Variable(torch.randn(1, 512, 64, 32))
    #expected = (1, 2048, 192, 11)
    # target = Variable(torch.randn(1, 32, 64, 32)).double()
    # print(type(input), type(target))
    # output = convlstm(input)
    # output = output[0][0].double()
    # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)
    for i, ii in trainloader:
        i = i.unsqueeze(1)


        input = Variable(i)
        print(input.shape)

        output = convlstm(input)
        output_list.append(output)


    # for i, data, label in enumerate(input):
    #     print(type(data))
    #     output = convlstm(data)
    #     output = output[0][0].double()
    #     output_list.append(output)

    # for i, data in enumerate(verifyloader):
    #     target_list.append(data)
    #
    # for target in target_list:
    #     for output in output_list:
    #         res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    #         print(res)
