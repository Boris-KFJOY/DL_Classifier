# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/30 17:16
import sys
import time
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.append('../')
import torch
import Utils.EEGDataset as EEGDataset
from Utils import Ploter
from Train import Classifier_Trainer, Trainer_Script
from etc.global_config import config
from Utils.exp import build_model_name, build_run_tag, annotate_meta
from Utils.Seedlock import set_seed, seed_worker
import os


def run():
    # 1、Define parameters of eeg
    algorithm = config['algorithm']
    print(f"{'*' * 20} Current Algorithm usage: {algorithm} {'*' * 20}")
    seeds = config["train_param"].get("seeds", [config["train_param"].get("seed", 0)])

    '''Parameters for training procedure'''
    UD = config["train_param"]['UD']
    ratio = config["train_param"]['ratio']
    if ratio == 1 or ratio == 3:
        Kf = 5
    elif ratio == 2:
        Kf = 2

    Kf = 1


    '''Parameters for ssvep data'''
    ws = config["data_param"]["ws"]
    Ns = config["data_param"]['Ns']
    dual= config["data_param"]["dual"]
    a = config[algorithm]
    '''Parameters for DL-based methods'''
    epochs = config[algorithm]['epochs']
    lr_jitter = config[algorithm]['lr_jitter']

    devices = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in seeds:
        # --- 固定本次 run 的随机性 ---
        set_seed(seed)
        g = torch.Generator(device='cpu')  # 或者直接 g = torch.Generator()
        g.manual_seed(seed)

        # --- 每个 seed 单独的模型名&保存目录 ---
        model_name = build_model_name(algorithm, a, seed)  # 你的 build_model_name 已经支持 seed
        save_dir = os.path.join('Result', 'BETA', model_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"[seed={seed}] save_dir => {save_dir}")

        # --- 开始训练（按被试留一）---
        final_acc_list = []
        for fold_num in range(Kf):
            final_test_acc_list = []
            for testSubject in range(1, Ns + 1):
                EEGData_Train = EEGDataset.getBETADataset(subject=testSubject, mode='train')
                EEGData_Test = EEGDataset.getBETADataset(subject=testSubject, mode='test')

                eeg_train_dataloader, eeg_test_dataloader = Trainer_Script.data_preprocess(
                    EEGData_Train, EEGData_Test, generator=g, worker_init_fn=seed_worker,whether_dual=dual
                )

                net, criterion, optimizer = Trainer_Script.build_model(devices)
                test_acc = Classifier_Trainer.train_on_batch(
                    epochs, eeg_train_dataloader, eeg_test_dataloader,
                    optimizer, criterion, net, devices, lr_jitter=lr_jitter
                )
                final_test_acc_list.append(test_acc)
        # 收集一轮 seed 的全部被试结果
        final_acc_list.append(final_test_acc_list)

        # --- 出图 & 存 CSV（和你现有流程一致；每个 seed 一个文件夹）---
        Ploter.plot_save_Result(
            final_acc_list, model_name=model_name, dataset='BETA',
            UD=UD, ratio=ratio, win_size=str(ws), text=True, save_dir=save_dir
        )

if __name__ == '__main__':
    run()