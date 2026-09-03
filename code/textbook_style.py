"""
教科书统一绘图风格（全书所有重绘图共用）
==========================================

用法（在任何绘图脚本开头）::

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
    from textbook_style import setup_style, save_figure, panel_label, COLORS

    setup_style()
    fig, ax = plt.subplots(figsize=fig_size(1))   # 单栏
    ...
    save_figure(fig, 'figs_chapXX/name')           # 同时输出 name.pdf 与 name.png

设计原则（与正文"每张图都有思路"的要求对应）：

1. **图里不放大标题**：图的"结论"写在正文 caption 里，图内只保留坐标轴、
   图例和 (a)(b)(c) 面板编号；避免图与 caption 各说一套。
2. **中文标签**：坐标轴、图例、注记一律中文（公式用 mathtext），
   字体优先 Noto Sans CJK SC（Ubuntu: apt install fonts-noto-cjk），
   缺失时退化到 DejaVu Sans + 文泉驿正黑。
3. **克制的配色**：固定 8 色色板（色盲友好、印刷友好），同一概念在
   全书中颜色一致——精确解/参考解用黑色，数值解/学习结果用蓝色，
   误差/警示用红色，约束/可行域用绿色。
4. **统一尺寸**：单栏图宽 6.3 in（约等于正文 \\textwidth），
   高度按面板数自动给出；线宽 1.6，字号 9–10 pt，印刷后可读。
5. **去噪**：去掉上、右两条轴线；浅灰网格；图例无边框。
6. **双格式输出**：PDF（矢量，供 LaTeX 用）+ PNG（300 dpi，供预览）。
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 色板：语义化命名，保证全书同一概念颜色一致
# ---------------------------------------------------------------------------
COLORS = {
    'blue':   '#2A6F97',   # 数值解 / 学习结果 / 主曲线
    'red':    '#C1443C',   # 误差 / 警示 / 最优路径
    'green':  '#3A7D44',   # 约束 / 可行域 / 安全集
    'orange': '#E07B39',   # 第二条对比曲线
    'purple': '#7B5EA7',   # 第三条对比曲线
    'gray':   '#6C757D',   # 参考线 / 辅助元素
    'gold':   '#D4A017',   # 高亮
    'teal':   '#2A9D8F',   # 第四条对比曲线
    'black':  '#222222',   # 精确解 / 参考解
}
CYCLE = [COLORS[k] for k in ('blue', 'orange', 'green', 'red', 'purple', 'teal', 'gold', 'gray')]

# 正文宽度（A4，book 类默认边距下约 6.3 in）
TEXT_WIDTH_IN = 6.3


def fig_size(ncols=1, nrows=1, width=TEXT_WIDTH_IN, aspect=0.62):
    """按面板数给出统一的 figsize：每个面板宽 width/ncols，高宽比 aspect。"""
    w = width
    h = (width / ncols) * aspect * nrows
    return (w, h)


def _available_fonts(candidates):
    """只保留本机安装了的字体，避免 matplotlib 逐字回退时反复报"找不到字体"。"""
    from matplotlib import font_manager as fm
    installed = {f.name for f in fm.fontManager.ttflist}
    found = [name for name in candidates if name in installed]
    return found or ['DejaVu Sans']


def setup_style():
    """设置 matplotlib 的全局风格。"""
    mpl.rcParams.update({
        # 字体：显式给出回退列表（matplotlib>=3.6 按字符逐个回退）。
        # 首选 Noto Sans CJK SC：同时包含中文、西文与数学负号 U+2212，
        # 使对数坐标刻度（mathtext 的 \mathdefault）也能正确显示；
        # 缺失时退到 DejaVu Sans（西文）+ 文泉驿正黑（中文）。
        'font.family': _available_fonts(['Noto Sans CJK SC', 'DejaVu Sans',
                                         'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
                                         'SimHei']) + ['sans-serif'],
        'axes.unicode_minus': False,
        'mathtext.fontset': 'dejavusans',
        # 字号
        'font.size': 9.5,
        'axes.titlesize': 10,
        'axes.labelsize': 9.5,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        # 线条与标记
        'lines.linewidth': 1.6,
        'lines.markersize': 4.5,
        'axes.prop_cycle': mpl.cycler(color=CYCLE),
        # 轴与网格
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'legend.frameon': False,
        # 输出
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'pdf.fonttype': 42,   # 嵌入 TrueType，保证中文在 PDF 中可复制
        'ps.fonttype': 42,
    })


def panel_label(ax, text, x=-0.08, y=1.04, **kw):
    """在面板左上角加 (a)(b)(c) 编号。"""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5,
            fontweight='bold', ha='right', va='bottom', **kw)


def save_figure(fig, path_no_ext, formats=('pdf', 'png'), **kw):
    """同时保存 PDF 与 PNG。path_no_ext 不带扩展名。"""
    os.makedirs(os.path.dirname(path_no_ext) or '.', exist_ok=True)
    for ext in formats:
        fig.savefig(f'{path_no_ext}.{ext}', **kw)
    plt.close(fig)
    print(f'已保存 {path_no_ext}.{{{",".join(formats)}}}')


# 导入即生效，兼容旧脚本 `from plot_utils import setup_chinese_font` 的用法
setup_style()
