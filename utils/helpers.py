\"\"\"
工具函数库：日志记录、数据验证、可视化辅助等
\"\"\"
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logger(name: str, level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    return logging.getLogger(name)

def plot_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette(\"Set2\")
