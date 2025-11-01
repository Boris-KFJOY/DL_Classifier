# Utils/Ploter.py
import os, numpy as np, pandas as pd, matplotlib, matplotlib.pyplot as plt

def plot_save_Result(final_acc_list, model_name, dataset='Benchmark', UD=0, ratio=1,
                     win_size='1', text=True, save_dir=None):

    # ------- 文案 -------
    if ratio == -1: proportion = 'Training-Free'
    elif ratio == 1: proportion = '8vs2'
    elif ratio == 2: proportion = '5vs5'
    elif ratio == 3: proportion = '2vs8'
    else:           proportion = 'N-1vs1'
    val_way = 'Unsupervised' if UD == -1 else ('PerSubject' if UD == 0 else 'CrossSubject')

    # ------- 统计（final_acc_list: shape = (folds, subjects)）-------
    final_acc_list = np.asarray(final_acc_list, dtype=float)
    mean_per_subj = np.mean(final_acc_list, axis=0)          # (S,)
    std_per_subj  = np.std(final_acc_list, axis=0, ddof=0)   # (S,)
    overall_mean  = float(mean_per_subj.mean())
    overall_std   = float(mean_per_subj.std(ddof=0))
    mean_list = np.append(mean_per_subj, overall_mean)       # (S+1,)
    std_list  = np.append(std_per_subj,  overall_std)

    # ------- 单位自适配：小数(<=1.5)→乘100；百分数(>1.5)→直接用 -------

    src_max = float(np.nanmax(mean_list))
    if src_max <= 1.5:
        data_for_plot = mean_list * 100.0
        txt_scale = 100.0
        ylim = (0, 105)  # 原来是 (0, 100)
    else:
        data_for_plot = mean_list
        txt_scale = 1.0
        ylim_top = 100 if src_max <= 100 else int(np.ceil(src_max / 10) * 10)
        ylim = (0, ylim_top + 5)  # 在原上限基础上 +5

    # ------- 保存 CSV（以百分比写盘，便于阅读） -------
    if save_dir is None:
        save_dir = os.path.join('Result', dataset, model_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'{proportion}_{val_way}_Classification_Result({win_size}S).csv')
    df = pd.DataFrame()
    for i in range(final_acc_list.shape[0]):
        df[f'Fold{i+1}'] = [f'{x*100:.2f}' for x in np.append(final_acc_list[i], np.mean(final_acc_list[i]))]
    df['Mean±Std'] = [f'{m*100:.2f}±{s*100:.2f}' for m, s in zip(mean_list, std_list)]
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ------- 画图（统一百分比坐标） -------
    plt.figure(figsize=(20, 8), dpi=80)
    matplotlib.rcParams['ytick.direction'] = 'in'
    plt.ylim(*ylim)
    plt.yticks(list(range(0, int(ylim[1])+1, 10)))
    xs = list(range(len(data_for_plot)))
    for i in xs:
        plt.bar(xs[i], data_for_plot[i], width=0.35)

    xticklabels = [str(i+1) for i in range(len(data_for_plot)-1)] + ['M']
    plt.xticks(xs, xticklabels, fontsize=10)
    plt.xlabel('Subject', fontsize=15)
    plt.ylabel('Accuracy(%)', fontsize=15)
    plt.title(f'{model_name} {proportion} {val_way} Classification Result({win_size}S)', fontsize=15)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    if text:
        for i in xs:
            delta = 2.0
            if i != len(data_for_plot) - 1:
                plt.text(xs[i] - 0.175, data_for_plot[i] + delta, f'{mean_list[i]*txt_scale:.2f}')
            else:
                plt.text(xs[i] - 0.30, data_for_plot[i] + delta,
                         f'{mean_list[-1]*txt_scale:.2f}±{std_list[-1]*txt_scale:.2f}', color='r')

    png_path = os.path.join(save_dir, f'{proportion}_{val_way}_Classification_Result({win_size}S).png')

    # ------- 自检打印（看一眼是否单位匹配） -------
    print("[PLOTER] src_max=", src_max, "| ylim=", ylim, "| plot_min/max=",
          float(np.min(data_for_plot)), float(np.max(data_for_plot)))

    plt.savefig(png_path)
    plt.show()
