"""
数据预处理模块：把长振动信号切分成固定长度的样本
"""
import numpy as np


def split_signal(signal, window_size=2048, stride=2048):
    """
    用滑动窗口把一条长信号切分成多个固定长度的样本。

    参数：
        signal (np.ndarray): 一维的长振动信号
        window_size (int): 每个样本的长度（点数），默认 2048
        stride (int): 窗口每次滑动的步长，默认 2048（不重叠）
                      设小于 window_size 可让样本重叠，增加样本数

    返回：
        np.ndarray: 形状为 (样本数, window_size) 的二维数组，
                    每一行是一个样本
    """
    samples = []  # 先用列表收集每个切出来的样本

    # 从 0 开始，每次前进 stride，直到剩余长度不够一个窗口为止
    start = 0
    while start + window_size <= len(signal):
        # 切出一段 [start, start+window_size)
        segment = signal[start:start + window_size]
        samples.append(segment)
        start += stride

    # 把样本列表堆叠成一个二维数组 (样本数, window_size)
    samples = np.array(samples)
    return samples

def build_dataset(file_label_map, data_dir="data/raw",
                  window_size=2048, stride=2048, sensor="DE"):
    """
    读取多个文件，切分并打标签，组装成完整数据集。

    参数：
        file_label_map (dict): 文件名 -> 标签 的映射，
                               例如 {"Normal_0.mat": 0, "IR007_0.mat": 1}
        data_dir (str): 数据文件所在目录
        window_size (int): 每个样本长度
        stride (int): 滑窗步长
        sensor (str): 传感器位置，"DE" 或 "FE"

    返回：
        X (np.ndarray): 形状 (样本数, window_size)
        y (np.ndarray): 形状 (样本数,)，每个元素是该样本的标签
    """
    import os
    from src.data.data_loader import load_signal

    X_list = []
    y_list = []

    # 逐个文件处理
    for file_name, label in file_label_map.items():
        file_path = os.path.join(data_dir, file_name)

        # 1) 读取这个文件的长信号
        signal = load_signal(file_path, sensor=sensor)

        # 2) 切分成样本
        samples = split_signal(signal, window_size=window_size, stride=stride)

        # 3) 给这批样本打上同一个标签
        labels = np.full(len(samples), label)

        # 4) 收集起来
        X_list.append(samples)
        y_list.append(labels)

        print(f"{file_name}: 切出 {len(samples)} 个样本, 标签={label}")

    # 5) 把所有文件的样本纵向拼接成一个大数组
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y

def normalize(X, method="zscore"):
    """
    对每个样本单独进行归一化。

    参数：
        X (np.ndarray): 形状 (样本数, 样本长度)
        method (str): "zscore" = 标准化（均值0、标准差1）
                      "minmax" = 缩放到 [-1, 1]

    返回：
        np.ndarray: 归一化后的数组，形状不变

    说明：
        按行（每个样本）独立归一化，使不同幅值水平的样本
        统一到相同尺度，避免模型仅依据幅值大小做判别。
    """
    X = X.astype(np.float32)  # 统一成 float32，省内存、适配深度学习框架

    if method == "zscore":
        # keepdims=True 保持维度，方便广播运算
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        # 加一个极小值 1e-8，防止除以 0
        X_norm = (X - mean) / (std + 1e-8)

    elif method == "minmax":
        x_min = X.min(axis=1, keepdims=True)
        x_max = X.max(axis=1, keepdims=True)
        X_norm = 2 * (X - x_min) / (x_max - x_min + 1e-8) - 1

    else:
        raise ValueError(f"不支持的归一化方法：{method}，请用 'zscore' 或 'minmax'")

    return X_norm

def compute_balanced_stride(signal_length, window_size, target_samples):
    """
    计算为达到目标样本数所需的滑窗步长。

    参数：
        signal_length (int): 原始信号长度
        window_size (int): 窗口长度
        target_samples (int): 希望切出的样本数

    返回：
        int: 应使用的步长（至少为 1）

    说明：
        样本数 n 与步长 s 的关系近似为：
            n = (signal_length - window_size) / s + 1
        反解得：
            s = (signal_length - window_size) / (n - 1)
        对少数类使用更小的步长，可切出更多重叠样本，
        从而缓解类别不平衡。
    """
    if target_samples <= 1:
        return window_size

    usable = signal_length - window_size
    stride = int(usable // (target_samples - 1))

    # 步长至少为 1，且不超过窗口长度（超过就浪费数据了）
    stride = max(1, min(stride, window_size))
    return stride


def build_balanced_dataset(file_label_map, data_dir="data/raw",
                           window_size=2048, target_per_class=None,
                           sensor="DE"):
    """
    构建类别平衡的数据集：对每个文件自动计算步长，
    使各类样本数尽量接近 target_per_class。

    参数：
        file_label_map (dict): 文件名 -> 标签
        data_dir (str): 数据目录
        window_size (int): 样本长度
        target_per_class (int): 每类目标样本数；None 表示自动取最大可能值
        sensor (str): "DE" 或 "FE"

    返回：
        X (np.ndarray), y (np.ndarray)
    """
    import os
    from src.data.data_loader import load_signal

    # 先读取所有信号，记录长度
    signals = {}
    for file_name, label in file_label_map.items():
        file_path = os.path.join(data_dir, file_name)
        signals[file_name] = load_signal(file_path, sensor=sensor)

    # 若未指定目标数，取"不重叠切分时最多的那一类"的样本数
    if target_per_class is None:
        counts = [(len(s) - window_size) // window_size + 1
                  for s in signals.values()]
        target_per_class = max(counts)

    X_list, y_list = [], []
    for file_name, label in file_label_map.items():
        sig = signals[file_name]
        stride = compute_balanced_stride(len(sig), window_size, target_per_class)
        samples = split_signal(sig, window_size=window_size, stride=stride)

        # 若切多了，截断到目标数量
        samples = samples[:target_per_class]

        X_list.append(samples)
        y_list.append(np.full(len(samples), label))
        print(f"{file_name}: stride={stride}, 切出 {len(samples)} 个样本, 标签={label}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

def compute_class_weights(y):
    """
    计算类别权重，用于缓解类别不平衡（配合加权损失函数使用）。

    参数：
        y (np.ndarray): 标签数组

    返回：
        np.ndarray: 各类别的权重，样本越少权重越大

    说明：
        采用 sklearn 的 'balanced' 策略：
            weight_c = n_samples / (n_classes * count_c)
        少数类得到更大权重，使其在损失中占更大比重。
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = n_samples / (n_classes * counts)
    return weights.astype(np.float32)

def add_noise(X, snr_db):
    """
    按指定信噪比给信号添加高斯白噪声。

    参数：
        X (np.ndarray): 形状 (样本数, 信号长度) 的信号
        snr_db (float): 目标信噪比（分贝）。越小噪声越强。
                        例如 10=轻微干扰, 0=噪声与信号等强, -4=强干扰

    返回：
        np.ndarray: 加噪后的信号，形状不变

    说明：
        对每个样本单独按其自身功率计算所需噪声强度，
        保证每条样本都达到指定的 SNR。
        推导：SNR_dB = 10*log10(P_signal / P_noise)
             => P_noise = P_signal / 10^(SNR_dB/10)
    """
    X = X.astype(np.float32)

    # 1) 计算每个样本的信号功率（均方值），保持维度便于广播
    signal_power = np.mean(X ** 2, axis=1, keepdims=True)

    # 2) 由 SNR 反推所需的噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    # 3) 生成对应强度的高斯白噪声（标准差 = 根号功率）
    noise = np.random.randn(*X.shape).astype(np.float32) * np.sqrt(noise_power)

    return X + noise