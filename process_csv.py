import numpy as np
import pandas as pd


# def moving_average(a, n=5) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


def moving_average(x, smooth=5):
    y = np.ones(smooth)
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    return smoothed_x


def smooth_curves(filename):
    df = pd.read_csv(f'./{filename}.csv')
    # max_index = 51
    # df = df[0:max_index]

    for col in df.columns:
        seq = df[col].values
        if col == 'Step':
            df.loc[:, col] = seq / 1e6
        else:
            # print(f"df.index = {df.index}")
            # print(f"seq.shape = {seq.shape}")
            smooth_seq = moving_average(seq, 5)
            # print(f"smooth_seq.shape = {smooth_seq.shape}")
            # print(f"df[col] = {df[col]}")
            df.loc[:, col] = smooth_seq

    df.to_csv(f'./smoothed_{filename}.csv')


if __name__ == '__main__':
    for exp_idx in ['3_1fb']:
        files = [f'exp{exp_idx}_agent_bottleneck_rate',
                 f'exp{exp_idx}_c2dst_full_bottleneck_rate', f'exp{exp_idx}_c2dst_rand_bottleneck_rate',
                 f'exp{exp_idx}_msinr_full_bottleneck_rate', f'exp{exp_idx}_msinr_rand_bottleneck_rate']
        for file in files:
            smooth_curves(file)
