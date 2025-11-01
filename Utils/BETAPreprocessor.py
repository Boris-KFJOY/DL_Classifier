# Designer: AI Assistant
# Time: 2025/10/28
# Purpose: BETA数据集预处理脚本 - 提取前15个被试、8个通道、8个目标、0.5-2.5秒时间窗

import scipy.io
import numpy as np
import os
from pathlib import Path


class BETAPreprocessor:
    """
    BETA数据集预处理器
    处理S1-S15被试数据，提取特定通道和时间窗口
    """

    def __init__(self, raw_data_dir='15Subject', output_dir='data/BETA'):
        """
        参数:
        - raw_data_dir: 原始BETA数据集目录（包含S1-S15的.mat文件）
        - output_dir: 预处理后数据保存目录
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir

        # SSVEP常用的8个枕区视觉通道
        self.target_channels = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']

        # SSVEP黄金频段选择 (12-15 Hz范围，间隔0.4 Hz)
        # BETA频率范围: 8.0-15.8 Hz, 步进0.2 Hz
        # 推荐频率: 12.0, 12.4, 12.8, 13.2, 13.6, 14.0, 14.4, 14.8 Hz
        self.target_freqs = [12.0, 12.4, 12.8, 13.2, 13.6, 14.0, 14.4, 14.8]

        # 计算频率对应的索引 (0-based)
        # BETA索引公式: idx = round((f - 8.0) / 0.2)
        self.freq_indices = [int(round((f - 8.0) / 0.2)) for f in self.target_freqs]
        # 对应索引: [20, 22, 24, 26, 28, 30, 32, 34]

        print(f"选择的SSVEP频率: {self.target_freqs} Hz")
        print(f"对应的0-based索引: {self.freq_indices}")
        print(f"对应的1-based索引(MATLAB): {[idx + 1 for idx in self.freq_indices]}")

        # 数据参数
        self.fs = 250  # 采样率 250Hz
        self.n_subjects = 15  # 使用前15个被试
        self.n_targets = 8  # 使用8个目标频率
        self.n_repeats = 4  # 每个目标4次重复 (实际BETA S1-S15是4个block)
        self.n_trials = self.n_targets * self.n_repeats  # 32 trials/被试

        # 时间窗口: 0.5-2.5秒
        self.time_start = 0.5  # 开始时间(秒)
        self.time_end = 2.5  # 结束时间(秒)

        # 对于S1-S15: 试次长度3秒，0.5s前 + 2s刺激 + 0.5s后
        # 原始数据: 3s × 250Hz = 750个采样点
        # 0-0.5s: 0-125, 0.5-2.5s: 125-625
        self.sample_start = int(self.time_start * self.fs)  # 125
        self.sample_end = int(self.time_end * self.fs)  # 625
        self.n_samples = self.sample_end - self.sample_start  # 500个采样点

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def find_channel_indices(self, channel_info):
        """
        根据通道名称找到对应的索引

        参数:
        - channel_info: 通道信息矩阵 (64 x 4)
          第0列: 通道索引, 第1列: 度数, 第2列: 半径, 第3列: 通道名

        返回:
        - channel_indices: 目标通道的索引列表
        """
        channel_indices = []

        # 提取通道名（第4列，索引3）
        channel_names = []
        for i in range(channel_info.shape[0]):
            ch_name = channel_info[i, 3]  # 第4列是通道名
            if isinstance(ch_name, np.ndarray):
                ch_name = ch_name[0] if len(ch_name) > 0 else ''
            channel_names.append(str(ch_name).strip().upper())

        # 查找目标通道
        for target_ch in self.target_channels:
            found = False
            for idx, ch_name in enumerate(channel_names):
                if target_ch.upper() == ch_name:
                    channel_indices.append(idx)
                    found = True
                    break
            if not found:
                print(f"警告: 未找到通道 {target_ch}")

        if len(channel_indices) != len(self.target_channels):
            print(f"警告: 只找到 {len(channel_indices)}/{len(self.target_channels)} 个通道")
            print(f"找到的索引: {channel_indices}")
            print(f"前10个通道名: {channel_names[:10]}")

        return channel_indices

    def preprocess_single_subject(self, subject_idx):
        """
        预处理单个被试的数据

        参数:
        - subject_idx: 被试编号 (1-15)

        返回:
        - eeg_data: 预处理后的EEG数据 (n_trials, n_channels, n_samples)
        - labels: 标签 (n_trials, 1)
        """
        # 加载原始数据
        mat_file = os.path.join(self.raw_data_dir, f'S{subject_idx}.mat')

        print(f"\n处理被试 S{subject_idx}...")
        print(f"加载文件: {mat_file}")

        try:
            data = scipy.io.loadmat(mat_file)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {mat_file}")
            return None, None

        # BETA数据集结构: data.EEG (channel x time x block x condition)
        # 获取EEG数据
        if 'data' in data:
            eeg_raw = data['data']['EEG'][0, 0]  # (64, 750, 4, 40) for S1-S15
            suppl_info = data['data']['suppl_info'][0, 0]

            # 获取通道信息 (字段名是 'chan' 不是 'chan_info')
            channel_info = suppl_info['chan'][0, 0]

            # 获取频率信息
            freqs = suppl_info['freqs'][0, 0].flatten()
            print(f"数据集全部频率范围: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")

            # 验证选择的频率
            selected_freqs = freqs[self.freq_indices]
            print(f"选择的8个频率: {selected_freqs}")
            print(f"期望频率: {self.target_freqs}")

        else:
            print("警告: 数据格式可能不同，尝试直接读取...")
            # 尝试其他可能的数据结构
            for key in data.keys():
                if not key.startswith('__'):
                    print(f"发现数据键: {key}")
            return None, None

        # 找到目标通道的索引
        channel_indices = self.find_channel_indices(channel_info)
        print(f"选择的通道索引: {channel_indices}")
        print(f"对应通道名: {self.target_channels}")

        # 数据维度: (64, 750, 6, 40)
        # 使用指定的频率索引提取数据，而不是前8个
        eeg_raw = eeg_raw[:, :, :, self.freq_indices]  # (64, 750, 6, 8)

        # 提取目标通道
        eeg_raw = eeg_raw[channel_indices, :, :, :]  # (8, 750, 6, 8)

        # 提取时间窗口 0.5-2.5秒
        eeg_raw = eeg_raw[:, self.sample_start:self.sample_end, :, :]  # (8, 500, 6, 8)

        print(f"切片后数据形状: {eeg_raw.shape}")

        # 重塑数据: (channel, time, block, condition) -> (trial, channel, time)
        # 每个condition有6个block (重复)
        n_channels, n_times, n_blocks, n_conditions = eeg_raw.shape

        # 初始化输出数组
        eeg_data = np.zeros((self.n_trials, n_channels, n_times))
        labels = np.zeros((self.n_trials, 1), dtype=int)

        trial_idx = 0
        for cond in range(n_conditions):
            for block in range(n_blocks):
                eeg_data[trial_idx, :, :] = eeg_raw[:, :, block, cond]
                labels[trial_idx, 0] = cond  # 标签 0-7
                trial_idx += 1

        print(f"最终数据形状: eeg_data={eeg_data.shape}, labels={labels.shape}")
        print(f"标签范围: {labels.min()} - {labels.max()}")

        return eeg_data, labels

    def save_preprocessed_data(self, subject_idx, eeg_data, labels):
        """
        保存预处理后的数据为.mat格式

        参数:
        - subject_idx: 被试编号
        - eeg_data: EEG数据
        - labels: 标签
        """
        # 保存数据文件
        data_file = os.path.join(self.output_dir, f'BETASub_{subject_idx}.mat')
        scipy.io.savemat(data_file, {'Data': eeg_data})
        print(f"保存数据: {data_file}")

        # 保存标签文件
        label_file = os.path.join(self.output_dir, f'BETALabSub_{subject_idx}.mat')
        scipy.io.savemat(label_file, {'Label': labels + 1})  # 标签转为1-8
        print(f"保存标签: {label_file}")

    def preprocess_all_subjects(self):
        """
        预处理所有15个被试的数据
        """
        print("=" * 60)
        print("开始预处理BETA数据集")
        print(f"被试数: {self.n_subjects}")
        print(f"目标数: {self.n_targets}")
        print(f"SSVEP频率: {self.target_freqs} Hz")
        print(f"频率索引(0-based): {self.freq_indices}")
        print(f"试次数: {self.n_trials} (每个被试)")
        print(f"通道数: {len(self.target_channels)}")
        print(f"通道名: {self.target_channels}")
        print(f"采样点: {self.n_samples} ({self.time_start}s - {self.time_end}s @ {self.fs}Hz)")
        print("=" * 60)

        success_count = 0

        for subject_idx in range(1, self.n_subjects + 1):
            eeg_data, labels = self.preprocess_single_subject(subject_idx)

            if eeg_data is not None and labels is not None:
                self.save_preprocessed_data(subject_idx, eeg_data, labels)
                success_count += 1
            else:
                print(f"跳过被试 S{subject_idx}")

        print("\n" + "=" * 60)
        print(f"预处理完成! 成功处理 {success_count}/{self.n_subjects} 个被试")
        print(f"预处理后的数据保存在: {self.output_dir}")

        # 保存频率信息到文本文件
        freq_info_file = os.path.join(self.output_dir, 'frequency_info.txt')
        with open(freq_info_file, 'w', encoding='utf-8') as f:
            f.write("BETA数据集频率配置信息\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"选择的8个SSVEP频率 (Hz):\n")
            for i, freq in enumerate(self.target_freqs):
                f.write(
                    f"  目标 {i + 1}: {freq} Hz (索引: {self.freq_indices[i]}, MATLAB索引: {self.freq_indices[i] + 1})\n")
            f.write(f"\n频率范围: {self.target_freqs[0]} - {self.target_freqs[-1]} Hz\n")
            f.write(f"频率间隔: 0.4 Hz\n")
            f.write(f"黄金频段: 12-15 Hz (SSVEP最佳响应范围)\n")
            f.write(f"\n标签映射:\n")
            for i in range(self.n_targets):
                f.write(f"  标签 {i} → {self.target_freqs[i]} Hz\n")

        print(f"频率配置信息已保存到: {freq_info_file}")
        print("=" * 60)


if __name__ == '__main__':
    # 使用示例
    # 假设原始数据在 '15Subject' 目录下
    # 预处理后保存到 'data/BETA' 目录

    preprocessor = BETAPreprocessor(
        raw_data_dir='15Subject',
        output_dir='data/BETA'
    )

    preprocessor.preprocess_all_subjects()