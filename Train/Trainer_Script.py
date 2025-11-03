# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/31 10:53
import numpy as np
import torch
from torch import nn
from Model import EEGNet, CCNN, SSVEPNet, FBtCNN, ConvCA, SSVEPformer, DDGCNN,SSVEPformerX,SSVEPformerX_CVFusion
from Utils import Constraint, LossFunction, Script
from etc.global_config import config

# [DEPRECATED on 2025-11-02] def data_preprocess(EEGData_Train, EEGData_Test,generator=None, worker_init_fn=None):
# [ADD] 仅用于测试数据管线；不在正式训练中启用
def data_preprocess(EEGData_Train, EEGData_Test, generator=None, worker_init_fn=None, whether_dual=False):
    '''
    Parameters
    ----------
    EEGData_Train: EEG Training Dataset (Including Data and Labels)
    EEGData_Test: EEG Testing Dataset (Including Data and Labels)

    Returns: Preprocessed EEG DataLoader
    -------
    '''
    algorithm = config['algorithm']
    ws = config["data_param"]["ws"]
    Fs = config["data_param"]["Fs"]
    Nf = config["data_param"]["Nf"]
    bz = config[algorithm]["bz"]

    train_time_tensor = None
    train_freq_tensor = None
    test_time_tensor = None
    test_freq_tensor = None

    '''Loading Training Data'''
    EEGData_Train, EEGLabel_Train = EEGData_Train[:]
    EEGData_Train = EEGData_Train[:, :, :, :int(Fs * ws)]

    if algorithm == "ConvCA":
        EEGData_Train = torch.swapaxes(EEGData_Train, axis0=2, axis1=3) # (Nh, 1, Nt, Nc)
        EEGTemp_Train = Script.get_Template_Signal(EEGData_Train, Nf)  # (Nf × 1 × Nt × Nc)
        EEGTemp_Train = torch.swapaxes(EEGTemp_Train, axis0=0, axis1=1)  # (1 × Nf × Nt × Nc)
        EEGTemp_Train = EEGTemp_Train.repeat((EEGData_Train.shape[0], 1, 1, 1))  # (Nh × Nf × Nt × Nc)
        EEGTemp_Train = torch.swapaxes(EEGTemp_Train, axis0=1, axis1=3)  # (Nh × Nc × Nt × Nf)

        print("EEGData_Train.shape", EEGData_Train.shape)
        print("EEGTemp_Train.shape", EEGTemp_Train.shape)
        print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGTemp_Train, EEGLabel_Train)

    else:
        if algorithm == "CCNN":
            EEGData_Train = CCNN.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)

        elif algorithm == "SSVEPformer":
            EEGData_Train = SSVEPformer.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)
            EEGData_Train = EEGData_Train.squeeze(1)

        #elif algorithm == "SSVEPformerX":

        #     EEGData_Train = SSVEPformer.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
        #    EEGData_Train = torch.from_numpy(EEGData_Train)
        #     EEGData_Train = EEGData_Train.squeeze(1)

            # [ADD] 双支路“预览”模式（仅当 preview_dual=True 时启用）
        elif algorithm in ["SSVEPformerX", "SSVEPformerX_CVFusion"]:
            # ------- Train -------
            if whether_dual:
                # 时域
                x_time_train = EEGData_Train.squeeze(1).contiguous()  # (Nh, C, Nt)
                # 频域（沿用你原来的特征函数）
                x_freq_train = SSVEPformer.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
                x_freq_train = torch.from_numpy(x_freq_train).squeeze(1).contiguous()  # (Nh, C or 2C, F)

                print("[dual] x_time_train:", tuple(x_time_train.shape))
                print("[dual] x_freq_train:", tuple(x_freq_train.shape))

                # 赋值到统一占位
                train_time_tensor = x_time_train
                train_freq_tensor = x_freq_train
            else:
                # 仅频域
                x_freq_train = SSVEPformer.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
                x_freq_train = torch.from_numpy(x_freq_train).squeeze(1).contiguous()
                print("[single] x_freq_train:", tuple(x_freq_train.shape))

                train_freq_tensor = x_freq_train


    '''Loading Testing Data'''
    EEGData_Test, EEGLabel_Test = EEGData_Test[:]
    EEGData_Test = EEGData_Test[:, :, :, :int(Fs * ws)]

    if algorithm == "ConvCA":
        EEGData_Test = torch.swapaxes(EEGData_Test, axis0=2, axis1=3)  # (Nh, 1, Nt, Nc)
        EEGTemp_Test = Script.get_Template_Signal(EEGData_Test, Nf)  # (Nf × 1 × Nt × Nc)
        EEGTemp_Test = torch.swapaxes(EEGTemp_Test, axis0=0, axis1=1)  # (1 × Nf × Nt × Nc)
        EEGTemp_Test = EEGTemp_Test.repeat((EEGData_Test.shape[0], 1, 1, 1))  # (Nh × Nf × Nt × Nc)
        EEGTemp_Test = torch.swapaxes(EEGTemp_Test, axis0=1, axis1=3)  # (Nh × Nc × Nt × Nf)

        print("EEGData_Test.shape", EEGData_Test.shape)
        print("EEGTemp_Test.shape", EEGTemp_Test.shape)
        print("EEGLabel_Test.shape", EEGLabel_Test.shape)
        EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGTemp_Test, EEGLabel_Test)

    else:
        if algorithm == "CCNN":
            EEGData_Test = CCNN.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)

        elif algorithm == "SSVEPformer":
            EEGData_Test = SSVEPformer.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)
            EEGData_Test = EEGData_Test.squeeze(1)
        #elif algorithm == "SSVEPformerX":
           # EEGData_Test = SSVEPformer.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
           # EEGData_Test = torch.from_numpy(EEGData_Test)
           # EEGData_Test = EEGData_Test.squeeze(1)
        elif algorithm in ["SSVEPformerX", "SSVEPformerX_CVFusion"]:
            if whether_dual:
                x_time_test = EEGData_Test.squeeze(1).contiguous()
                x_freq_test = SSVEPformer.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
                x_freq_test = torch.from_numpy(x_freq_test).squeeze(1).contiguous()

                print("[dual] x_time_test:", tuple(x_time_test.shape))
                print("[dual] x_freq_test:", tuple(x_freq_test.shape))

                test_time_tensor = x_time_test
                test_freq_tensor = x_freq_test
            else:
                x_freq_test = SSVEPformer.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
                x_freq_test = torch.from_numpy(x_freq_test).squeeze(1).contiguous()
                print("[single] x_freq_test:", tuple(x_freq_test.shape))

                test_freq_tensor = x_freq_test
        elif algorithm == "DDGCNN":
            EEGData_Test = torch.swapaxes(EEGData_Test, axis0=1, axis1=3)  # (Nh, 1, Nc, Nt) => (Nh, Nt, Nc, 1)

        print("EEGData_Test.shape", EEGData_Test.shape)
        print("EEGLabel_Test.shape", EEGLabel_Test.shape)
    if whether_dual:
        # 双支路三元组
        train_dataset = torch.utils.data.TensorDataset(train_time_tensor, train_freq_tensor, EEGLabel_Train)
        test_dataset = torch.utils.data.TensorDataset(test_time_tensor, test_freq_tensor, EEGLabel_Test)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bz, shuffle=True,
                                                   drop_last=True, generator=generator, worker_init_fn=worker_init_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bz, shuffle=False,
                                                  drop_last=True, )
        return train_loader, test_loader

    else:
        # 单支路二元组（兼容原训练器）

        eeg_train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=bz, shuffle=True,
                                                           drop_last=True, generator=generator,
                                                           worker_init_fn=worker_init_fn)
        eeg_test_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=bz, shuffle=False,
                                                          drop_last=True, )

        return eeg_train_dataloader, eeg_test_dataloader
    # Create DataLoader for the Dataset



def build_model(devices):
    '''
    Parameters
    ----------
    device: the device to save DL models
    Returns: the building model
    -------
    '''
    algorithm = config['algorithm']
    Nc = config["data_param"]['Nc']
    Nf = config["data_param"]['Nf']
    Fs = config["data_param"]['Fs']
    ws = config["data_param"]['ws']
    lr = config[algorithm]['lr']
    wd = config[algorithm]['wd']
    Nt = int(Fs * ws)

    if algorithm == "EEGNet":
        net = EEGNet.EEGNet(Nc, Nt, Nf)

    elif algorithm == "CCNN":
        net = CCNN.CNN(Nc, 220, Nf)

    elif algorithm == "FBtCNN":
        net = FBtCNN.tCNN(Nc, Nt, Nf, Fs)

    elif algorithm == "ConvCA":
        net = ConvCA.convca(Nc, Nt, Nf)

    elif algorithm == "SSVEPformer":
        net = SSVEPformer.SSVEPformer(depth=2, attention_kernal_length=31, chs_num=Nc, class_num=Nf,
                                      dropout=0.5)
        net.apply(Constraint.initialize_weights)
    elif algorithm == "SSVEPformerX":

        a = config[algorithm]  # == config["SSVEPformerX"]

        net = SSVEPformerX.SSVEPformerX(
            depth=a["depth"],
            # 兼容命名差异：kernel/kernal 都能读到
            attention_kernal_length=a.get("attention_kernal_length",
                                          a.get("attention_kernel_length", 31)),
            chs_num=Nc,
            class_num=Nf,
            token_dim=a.get("token_dim", 560),  # YAML 没写就用默认
            dropout=a["dropout"],
            use_subband=a["use_subband"],
            subband_proj=a.get("subband_proj", "linear"),
            use_crossview=a["use_crossview"],
            crossview_alpha=a.get("crossview_alpha", 1.0),
            use_harmonic_pe=a["use_harmonic_pe"],
            harmonic_beta=a.get("harmonic_beta", 1.0),
            pe_mode=a.get("pe_mode", "harmonic"),
        )
        net.apply(Constraint.initialize_weights)
    elif algorithm == "SSVEPformerX_CVFusion":
        # 新模型：频域+时域双支路，CrossView 前通道维拼接融合
        a = config[algorithm]  # == config["SSVEPformerX_CVFusion"]

        net = SSVEPformerX_CVFusion.SSVEPformerX_CVFusion(
            depth_time=a.get("depth_time", 2),
            depth_freq=a.get("depth_freq", 2),
            attention_kernal_length=a.get("attention_kernal_length",
                                          a.get("attention_kernel_length", 31)),  # 兼容拼写
            chs_num=Nc,
            class_num=Nf,
            token_dim=a.get("token_dim", int(Fs * 2.1875) if "token_dim" not in a else a["token_dim"]),  # 默认给个560附近的安全值
            dropout=a.get("dropout", 0.5),

            # 模块开关与超参（都给默认，保持兼容）
            use_subband=a.get("use_subband", True),
            subband_proj=a.get("subband_proj", "linear"),
            use_crossview=a.get("use_crossview", True),
            crossview_alpha=a.get("crossview_alpha", 1.0),
            use_harmonic_pe=a.get("use_harmonic_pe", True),
            harmonic_beta=a.get("harmonic_beta", 1.0),
            pe_mode=a.get("pe_mode", "harmonic"),

            # 时域分支（可选）：与数据管线 whether_dual 搭配
            enable_time_branch=a.get("enable_time_branch", True),
            time_token_dim=a.get("time_token_dim", int(Fs * ws)),  # 通常等于 Nt
            resize_time_to_freq=a.get("resize_time_to_freq", True),

            fusion_mode=a.get("fusion_mode", "chan_cat"),
            use_len_fusion_block=a.get("use_len_fusion_block", True)
        )
        net.apply(Constraint.initialize_weights)  # 与旧模型保持一致的初始化策略
    elif algorithm == "SSVEPNet":
        net = SSVEPNet.ESNet(Nc, Nt, Nf)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "DDGCNN":
        bz = config[algorithm]["bz"]
        norm = config[algorithm]["norm"]
        act = config[algorithm]["act"]
        trans_class = config[algorithm]["trans_class"]
        n_filters = config[algorithm]["n_filters"]
        net = DDGCNN.DenseDDGCNN([bz, Nt, Nc], k_adj=3, num_out=n_filters, dropout=0.5, n_blocks=3, nclass=Nf,
                                 bias=False, norm=norm, act=act, trans_class=trans_class, device=devices)


    if algorithm == 'SSVEPNet':
        stimulus_type = str(config[algorithm]["stimulus_type"])
        criterion = LossFunction.CELoss_Marginal_Smooth(Nf, stimulus_type=stimulus_type)

    else:
        criterion = nn.CrossEntropyLoss(reduction="none")

    if algorithm == "SSVEPformer":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    elif algorithm == "SSVEPformerX":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    elif algorithm == "SSVEPformerX_CVFusion":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)

    return net, criterion, optimizer
