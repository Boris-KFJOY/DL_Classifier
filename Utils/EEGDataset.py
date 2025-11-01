# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/10/6 22:47
from torch.utils.data import Dataset
import torch
import scipy.io


class getSSVEP12Inter(Dataset):
    def __init__(self, subject=1, mode="train"):
        self.Nh = 180
        self.Nc = 8
        self.Nt = 1024
        self.Nf = 12
        self.Fs = 256
        self.eeg_raw_data = self.read_EEGData()
        self.label_raw_data = self.read_EEGLabel()
        if mode == 'train':
            self.eeg_data = torch.cat(
                (self.eeg_raw_data[0:(subject - 1) * self.Nh], self.eeg_raw_data[subject * self.Nh:]), dim=0)
            self.label_data = torch.cat(
                (self.label_raw_data[0:(subject - 1) * self.Nh:, :], self.label_raw_data[subject * self.Nh:, :]), dim=0)

        if mode == 'test':
            self.eeg_data = self.eeg_raw_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_raw_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def get_DataSub(self, index):
        # load file into dict
        subjectfile = scipy.io.loadmat(f'../data/Dial/DataSub_{index}.mat')
        # extract numpy from dict
        samples = subjectfile['Data']
        # (num_trial, sample_point, num_trial) => (num_trial, num_channels, sample_point)
        eeg_data = samples.swapaxes(1, 2)
        eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)
        print(eeg_data.shape)
        return eeg_data

    def read_EEGData(self):
        eeg_data = self.get_DataSub(1)
        for i in range(1, 10):
            single_subject_eeg_data = self.get_DataSub(i + 1)
            eeg_data = torch.cat((eeg_data, single_subject_eeg_data), dim=0)
        return eeg_data

    # get the single label data
    def get_DataLabel(self, index):
        # load file into dict
        labelfile = scipy.io.loadmat(f'../data/Dial/LabSub_{index}.mat')
        # extract numpy from dict
        labels = labelfile['Label']
        label_data = torch.from_numpy(labels)
        print(label_data.shape)
        return label_data - 1

    def read_EEGLabel(self):
        label_data = self.get_DataLabel(1)
        for i in range(1, 10):
            single_subject_label_data = self.get_DataLabel(i + 1)
            label_data = torch.cat((label_data, single_subject_label_data), dim=0)
        return label_data


class getSSVEP12Intra(Dataset):
    def __init__(self, subject=1, train_ratio=0.8, KFold=None, n_splits=5, mode="train"):
        super(getSSVEP12Intra, self).__init__()
        self.Nh = 180  # number of trials
        self.Nc = 8  # number of channels
        self.Nt = 1024  # number of time points
        self.Nf = 12  # number of target frequency
        self.Fs = 256  # Sample Frequency
        self.subject = subject  # current subject
        self.eeg_data = self.get_DataSub()
        self.label_data = self.get_DataLabel()
        self.num_trial = self.Nh // self.Nf  # number of trials of each frequency
        self.train_idx = []
        self.test_idx = []
        if KFold is not None:
            fold_trial = self.num_trial // n_splits  # number of trials in each fold
            self.valid_trial_idx = [i for i in range(KFold * fold_trial, (KFold + 1) * fold_trial)]

        for i in range(0, self.Nh, self.Nh // self.Nf):
            for j in range(self.Nh // self.Nf):
                if n_splits == 2 and j == self.num_trial - 1:
                    continue  # if K = 2, discard the last trial of each category
                if KFold is not None:  # K-Fold Cross Validation
                    if j not in self.valid_trial_idx:
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)
                else:  # Split Ratio Validation
                    if j < int(self.num_trial * train_ratio):
                        self.train_idx.append(i + j)
                    else:
                        self.test_idx.append(i + j)

        self.eeg_data_train = self.eeg_data[self.train_idx]
        self.label_data_train = self.label_data[self.train_idx]
        self.eeg_data_test = self.eeg_data[self.test_idx]
        self.label_data_test = self.label_data[self.test_idx]

        if mode == 'train':
            self.eeg_data = self.eeg_data_train
            self.label_data = self.label_data_train
        elif mode == 'test':
            self.eeg_data = self.eeg_data_test
            self.label_data = self.label_data_test

        print(f'eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # get the single subject data
    def get_DataSub(self):
        subjectfile = scipy.io.loadmat(f'../data/Dial/DataSub_{self.subject}.mat')
        samples = subjectfile['Data']  # (8, 1024, 180)
        eeg_data = samples.swapaxes(1, 2)  # (8, 1024, 180) -> (8, 180, 1024)
        eeg_data = torch.from_numpy(eeg_data.swapaxes(0, 1))  # (8, 180, 1024) -> (180, 8, 1024)
        eeg_data = eeg_data.reshape(-1, 1, self.Nc, self.Nt)  # (180, 1, 8, 1024)
        print(eeg_data.shape)
        return eeg_data

    # get the single label data
    def get_DataLabel(self):
        labelfile = scipy.io.loadmat(f'../data/Dial/LabSub_{self.subject}.mat')
        labels = labelfile['Label']
        label_data = torch.from_numpy(labels)
        print(label_data.shape)
        return label_data - 1


class getBETADataset(Dataset):
    """
    BETA数据集加载器 (预处理后版本)
    - 使用S1-S15被试
    - 8个SSVEP目标
    - 8个枕区通道: O1, Oz, O2, PO7, PO3, POz, PO4, PO8
    - 时间窗口: 0.5-2.5秒 (500个采样点 @ 250Hz)
    - 每个被试32个trial (8目标 × 4重复)
    """

    def __init__(self, subject=1, mode="train"):
        self.Nh = 32  # 每个被试的试验次数 (8目标 × 4重复)
        self.Nc = 8  # 通道数 (8个枕区通道)
        self.Nt = 500  # 时间点数 (2s × 250Hz)
        self.Nf = 8  # SSVEP目标数
        self.Fs = 250  # 采样率
        self.Ns = 15  # 被试总数 (S1-S15)

        self.eeg_raw_data = self.read_EEGData()
        self.label_raw_data = self.read_EEGLabel()

        # Inter-Subject留一法: 训练集用其他被试，测试集用当前被试
        if mode == 'train':
            self.eeg_data = torch.cat((self.eeg_raw_data[0:(subject - 1) * self.Nh],
                                       self.eeg_raw_data[subject * self.Nh:]), dim=0)
            self.label_data = torch.cat((self.label_raw_data[0:(subject - 1) * self.Nh:, :],
                                         self.label_raw_data[subject * self.Nh:, :]), dim=0)
        elif mode == 'test':
            self.eeg_data = self.eeg_raw_data[(subject - 1) * self.Nh:subject * self.Nh]
            self.label_data = self.label_raw_data[(subject - 1) * self.Nh:subject * self.Nh]

        print(f'BETA eeg_data for subject {subject}:', self.eeg_data.shape)
        print(f'BETA label_data for subject {subject}:', self.label_data.shape)

    def __getitem__(self, index):
        return self.eeg_data[index], self.label_data[index]

    def __len__(self):
        return len(self.label_data)

    # 获取单个被试的预处理后数据
    def get_DataSub(self, index):
        """
        加载预处理后的单个被试数据
        预处理后的数据格式: (n_trials, n_channels, n_samples)
        需要转换为: (n_trials, 1, n_channels, n_samples)
        """
        subjectfile = scipy.io.loadmat(f'data/BETA/BETASub_{index}.mat')
        samples = subjectfile['Data']  # (48, 8, 500)

        # 转换为torch tensor并添加维度
        eeg_data = torch.from_numpy(samples).float()
        eeg_data = eeg_data.unsqueeze(1)  # (48, 1, 8, 500)

        print(f'BETA Subject {index} eeg_data shape:', eeg_data.shape)
        return eeg_data

    def read_EEGData(self):
        """读取所有15个被试的EEG数据"""
        eeg_data = self.get_DataSub(1)
        for i in range(1, self.Ns):  # S1-S15
            single_subject_eeg_data = self.get_DataSub(i + 1)
            eeg_data = torch.cat((eeg_data, single_subject_eeg_data), dim=0)
        print(f'Total BETA eeg_data shape: {eeg_data.shape}')
        return eeg_data

    # 获取单个被试的标签数据
    def get_DataLabel(self, index):
        """
        加载预处理后的单个被试标签
        标签范围: 1-8 (需要转换为0-7)
        """
        labelfile = scipy.io.loadmat(f'data/BETA/BETALabSub_{index}.mat')
        labels = labelfile['Label']  # (48, 1)
        label_data = torch.from_numpy(labels).long()
        print(f'BETA Subject {index} label_data shape:', label_data.shape)
        return label_data - 1  # 转换为0-7

    def read_EEGLabel(self):
        """读取所有15个被试的标签数据"""
        label_data = self.get_DataLabel(1)
        for i in range(1, self.Ns):  # S1-S15
            single_subject_label_data = self.get_DataLabel(i + 1)
            label_data = torch.cat((label_data, single_subject_label_data), dim=0)
        print(f'Total BETA label_data shape: {label_data.shape}')
        return label_data