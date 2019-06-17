## time_series_forecast_app
时间序列预测app

### TODO
1. 接入鄂尔多斯数据（同时包好了天气和污染物浓度的小时级数据）
2. 补充湿度字段数据
3. 出具模型在不同样本长度，不同地区的计算效果报告
4. 模型可以考虑首先使用天气数据预测与目标相关的污染物浓度数据, 然后同时结合天气数据和相关污染物数据来预测目标污染物浓度

### 项目目的
使用某固定站点的历史污染物浓度、温度、压力、湿度等数据对未来污染物浓度变化进行预测

### 项目依赖
* 依赖python>=3.6
* 安装lake: git clone https://github.com/CosmosShadow/lake，切换至develop分支，加入环境变量
* 安装numpy>=1.16.3
* 安装scipy>=1.2.1
* 安装pandas>=0.24.0
* 安装torch>=1.0.1
* 安装statsmodel>=0.9.0

### 代码结构
```markdown
|--config
    |--config.yml                       # 项目配置文件
|--bin
    |--app.py                           # web服务代码
|--lib
    |--manifold_prediction.py           # 使用流形样本进行预测
|--logs
    |--web_error.log                    # error日志
    |--web_info.log                     # info日志
|--mods
    |--config_loader.py                 # 配置加载器
    |--models.py                        # 长短期记忆模型以及基于pytorch的神经网络训练和预测模型
    |--data_filtering.py                # 清理数据
    |--granger_causality.py             # 格兰杰检验
    |--build_samples_and_targets.py     # 运用流行学习对数据进行非线性降维，用更少的数据表示更多的信息 
    |--extract_data_and_normalize.py    # 清理异常值，缺省值并进行归一化
    |--loss_criterion.py                # 损失函数：平均百分比误差
    |--model_evaluations.py             # 误差值分析
           
|--model 
    |--train_cnn.py                     # 训练卷积神经网络模型
    |--train_lstm.py                    # 训练长短期记忆模型   
    |--train_lstm_cnn.py                # 卷积和长短期记忆的结合模型 (还有bug) 
|--other
    |--get_raw_data_and_normalize.py 
    |--data_correlation_and_analysis.py # 拉取天气数据并进行分析
|--analysis 
    |--acf_pacf_test.py                 # acf和pacf检验稳定性
    |--correlation_analysis.py          # 数据相关性分析
    |--feature_importance.py            # 特征重要性分析
    |--granger_causality_test.py        # 格兰杰检验
|--tmp                                  # 资源文档

```

### Documentation
* config 
    * `record_start_time`: '2016050510'   # 数据起始时间
    * `record_end_time`: '2019050513'     # 数据结束时间
    * `exist_record_time`: '2018050513'   
    * `model_params`                      # 模型参数
    * `logging`                           
* mods
    * models.py                         
        * LSTM Model 
            * 参数
                * `input_size`              # 输入数据维度
                * `batch_first`             # 设定为True则使second dimension为seq_len
                * `hidden_size`             # 隐藏层层数
                * `num_layers`              # 神经网络层数
        * CNN Model 
            * 我们建立一个nn.Module的subclass, 建立一个两层的神经网络, 每一层有convolution + RELU + pooling的系列操作，
            由Conv2d方法调用。两层建立方式相同，除了layer2的input_size = 32, output_size = 64.
            * 参数
                * `number_of_input_channels`    # 我们这里输入一段时间序列，因此是1
                * `number_of_output_channels`   # 
                * `kernel_size`                 # convolutional filter的大小，这里设置成5 * 5
                * `padding_argument`            
                    ![](http://latex.codecogs.com/gif.latex?\\\Wout=\frac{Win-F+2P}{S}+1)
                    `Win`表示输入数据的宽度，`F`表示filter的大小，`P`表示Padding, `S`是步幅(Stride).我们使input和output大小相同，
                    S= 1, F = 5, 所以P = 2, 即padding argument. 
                    
                
                        
    * data_filtering.py                 # 平滑数据
        * savitzky_golay_filtering()    # 用线性最小二乘法把相邻数据点fit到低阶多项式 
        * band_pass_filtering()         # 带通滤波，去掉高频低频数据      
    * build_test_samples_and_targets.py # 构建LSTM模型样本
    * granger_causality.py              # 检验时间序列x是否是y的原因。x,y须具平稳性
* model
    * train_lstm.py
        * 参数
            * `target_column`           # 预测数据列
            * `selected columns`        # 输入数据列
            * `lr`                      # 学习速率
            * `pred_dim`                # 输出数据维度
            * `use_cuda`                # 是否使用CUDA GPU 
            * `batch_size`              # 每个forward/backward pass中训练样本数量
            * `epoch`                   # 训练次数
        * 优化器
            * 调用方法
              ``````
              for input, target in dataset:
                    optimizer.zero_grad()
                    output = model(input)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
             `````` 
            * Adam算法    # 它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，
            每一次迭代学习率都有个确定范围，使得参数比较平稳。

    * train_cnn.py 
        * 参数 
            * `target_column`           # 预测数据列
            * `selected columns`        # 输入数据列
            * `lr`                      # 学习速率
            * `pred_dim`                # 输出数据维度
            * `use_cuda`                # 是否使用CUDA GPU 
            * `batch_size`              # 每个forward/backward pass中训练样本数量
            * `epoch`                   # 训练次数
        * 算法概述
            * 在训练过程中,我们重复两个循环。外循环循环`n_epoch`次，内循环循环`trainloader`.通过神经网络后，我们得到输出数据，并计算loss.
            然后我们进行back-propagation和一个optimized training step. 我们先通过调用`zero_grad()`让gradients归零，然后调用`backward()`
            进行back-propagation.此时我们已经计算了gradients.我们需要调用`optimizer.step()`进行Adam optimizer training step.
            
            
                                
