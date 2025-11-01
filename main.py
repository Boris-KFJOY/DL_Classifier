import os
import torch
import random
import numpy as np

from etc.global_config import config  # 自动加载 etc/config.yaml
from Utils.EEGDataset import getSSVEP12Inter
from Train.Trainer_Script import data_preprocess, build_model
from Train.Classifier_Trainer import train_on_batch

def set_seed(seed=42):
    """固定随机种子，保证复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # ========== 初始化部分 ==========
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Using device: {device}\n")

    # ========== 读取配置 ==========
    algorithm = config["algorithm"]
    data_param = config["data_param"]
    model_param = config[algorithm]

    print(f"Algorithm: {algorithm}")
    print(f"Data parameters: {data_param}")
    print(f"Model parameters: {model_param}\n")

    # ========== 构建跨被试数据集 ==========
    subject_id = 1  # 可以改成 1~10 任意被试编号
    print(f"Loading cross-subject data... (Subject {subject_id} as test)")

    train_dataset = getSSVEP12Inter(subject=subject_id, mode='train')
    test_dataset  = getSSVEP12Inter(subject=subject_id, mode='test')

    # ========== 数据预处理 ==========
    train_loader, test_loader = data_preprocess(train_dataset, test_dataset)

    # ========== 构建模型 ==========
    net, criterion, optimizer = build_model(device)
    epochs = model_param.get("epochs", 60)

    print(f"Training for {epochs} epochs...\n")

    # ========== 训练与验证 ==========
    best_acc = train_on_batch(
        num_epochs=epochs,
        train_iter=train_loader,
        test_iter=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        net=net,
        device=device,
    )

    print(f"\n✅ [Subject {subject_id}] Best Validation Accuracy = {best_acc:.2f}%\n")


if __name__ == "__main__":
    main()
