"""
数据加载模块：读取 CWRU 轴承振动数据（.mat 格式）
"""
import scipy.io as sio


def load_signal(file_path, sensor="DE"):
    """
    从一个 CWRU .mat 文件中读取振动信号。

    参数：
        file_path (str): .mat 文件的路径，例如 "data/raw/Normal_0.mat"
        sensor (str): 传感器位置，"DE"=驱动端，"FE"=风扇端，默认 "DE"

    返回：
        numpy.ndarray: 一维的振动信号数组

    说明：
        CWRU 每个文件里振动信号的变量名前缀数字都不同
        （如 X097_DE_time、X105_DE_time），但都以 "_DE_time"
        或 "_FE_time" 结尾。本函数自动匹配对应结尾的键，
        因此读任何文件都无需手动指定变量名。
    """
    # 1) 读取 .mat 文件（得到一个字典）
    mat_data = sio.loadmat(file_path)

    # 2) 我们要找的键，是以 "_DE_time" 或 "_FE_time" 结尾的那个
    target_suffix = f"_{sensor}_time"

    # 3) 遍历所有键，找到匹配的那个
    signal_key = None
    for key in mat_data.keys():
        if key.endswith(target_suffix):
            signal_key = key
            break

    # 4) 如果没找到，主动报错提示（异常处理，专业代码必备）
    if signal_key is None:
        raise ValueError(
            f"在文件 {file_path} 中没有找到以 '{target_suffix}' 结尾的信号变量。"
            f"该文件包含的键有：{list(mat_data.keys())}"
        )

    # 5) 取出信号，并压平成一维数组返回
    signal = mat_data[signal_key].flatten()
    return signal