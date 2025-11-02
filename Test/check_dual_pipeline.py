# Test/check_dual_pipeline.py
import torch
import Utils.EEGDataset as EEGDataset
from Train import Trainer_Script
from etc.global_config import config

# ========= 简单可见的总开关 =========
# 设为 True：预览双支路 -> 期望 batch 有 3 项 (x_time, x_freq, y)
# 设为 False：单支路 -> 期望 batch 有 2 项 (x_freq, y)
PREVIEW_DUAL = True   # <- 你只需要改这里 True/False

def inspect_loader(loader, preview_dual, split_name):
    print(f"=== {split_name.upper()} LOADER ===")
    batch = next(iter(loader))

    if preview_dual:
        assert len(batch) == 3, f"Expect 3 items (x_time, x_freq, y), got {len(batch)}."
        x_time, x_freq, y = batch
        print("x_time:", tuple(x_time.shape))  # (B, Nc, Nt)
        print("x_freq:", tuple(x_freq.shape))  # (B, Nc or 2*Nc, Df)
        print("y:", tuple(y.shape))
        b_time, c_time, _ = x_time.shape
        b_freq, c_freq, _ = x_freq.shape
        assert b_time == b_freq and c_time == c_freq, "Batch/Channel mismatch between time and freq views."
        print("[OK] dual-branch tensors look consistent.")
    else:
        assert len(batch) == 2, f"Expect 2 items (x_freq, y), got {len(batch)}."
        x_freq, y = batch
        print("x_freq:", tuple(x_freq.shape))
        print("y:", tuple(y.shape))
        print("[OK] single-branch tensors look consistent.")

def main(preview_dual):
    algorithm = "SSVEPformerX"
    config["algorithm"] = algorithm

    seed = config["train_param"].get("seed", 0)
    generator = torch.Generator().manual_seed(seed)

    # 取一个被试快速检查
    subject = 1
    eeg_train = EEGDataset.getBETADataset(subject=subject, mode="train")
    eeg_test  = EEGDataset.getBETADataset(subject=subject, mode="test")

    train_loader, test_loader = Trainer_Script.data_preprocess(
        eeg_train, eeg_test, generator=generator, worker_init_fn=None, preview_dual=preview_dual
    )

    inspect_loader(train_loader, preview_dual, "train")
    inspect_loader(test_loader, preview_dual, "test")

if __name__ == "__main__":
    # [DEPRECATED on 2025-11-02] 之前通过命令行参数控制：
    # import argparse
    # parser = argparse.ArgumentParser(description="Preview dual-branch pipeline outputs.")
    # parser.add_argument("--preview-dual", action="store_true",
    #                     help="Enable preview mode to inspect dual-branch (time/freq) outputs.")
    # args = parser.parse_args()
    # main(preview_dual=args.preview_dual)

    # 现在通过文件内的 PREVIEW_DUAL 开关：
    main(preview_dual=PREVIEW_DUAL)
