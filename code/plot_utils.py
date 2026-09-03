"""
绘图工具模块 - 统一的中文字体配置（旧脚本兼容层）

新的重绘脚本请改用 code/textbook_style.py（统一色板、面板编号、PDF+PNG 双输出，
并优先使用 Noto Sans CJK 以保证对数坐标的负号可显示）。

在所有绘图脚本开头添加:
from plot_utils import setup_chinese_font
setup_chinese_font()
"""

import matplotlib.pyplot as plt
import matplotlib

def setup_chinese_font():
    """配置matplotlib使用中文字体"""
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

# 自动执行
setup_chinese_font()
