# PINN实验代码

本文件夹包含第14章《物理信息神经网络》的所有实验代码。

## 文件说明

| 文件 | 说明 | 对应章节 |
|------|------|----------|
| `pinn_poisson_1d.py` | PINN求解一维泊松方程的完整代码 | 2.3节 |
| `pinn_poisson_2d.py` | PINN求解二维泊松方程（含3D可视化） | 综合实验 |
| `spectral_bias.py` | 频谱偏差现象演示 | 4.1节 |
| `fourier_features.py` | 傅里叶特征映射实验 | 4.1节 |
| `hard_constraint.py` | 硬约束vs软约束对比实验 | 3节 |
| `adaptive_weight.py` | 自适应权重PINN实现 | 5节 |
| `gradient_diagnosis.py` | 梯度竞争诊断 | 4.2节 |
| `lambda_effect.py` | 罚系数ρ（损失权重）的影响实验 | 2.4节 |

## 运行环境

需要安装以下Python包：
```bash
pip install torch numpy matplotlib
```

## 运行方法

每个Python文件都可以独立运行：
```bash
python pinn_poisson_1d.py
python spectral_bias.py
# ... 等等
```

运行后会生成对应的结果图片（.png格式）。

## 实验结果摘要

### 1. 一维泊松方程 (`pinn_poisson_1d.py`)
- MSE: ~4.60×10⁻⁸
- 训练10000个epoch

### 2. 二维泊松方程 (`pinn_poisson_2d.py`)
- MSE: ~1.0×10⁻⁸
- 训练15000个epoch
- 包含3D表面图、等高线图、切片对比图

### 3. 频谱偏差 (`spectral_bias.py`)
- 标准MLP难以学习高频成分
- 10000 epoch后MSE仍为~2.6×10⁻³

### 4. 傅里叶特征 (`fourier_features.py`)
- 傅里叶特征显著加速高频学习
- scale=10时效果最佳，MSE可达10⁻⁷量级

### 5. 硬约束vs软约束 (`hard_constraint.py`)
- 硬约束：MSE = 6.64×10⁻⁹，边界精确满足
- 软约束：MSE = 4.60×10⁻⁸，边界近似满足

### 6. 梯度竞争 (`gradient_diagnosis.py`)
- 约42%的训练步骤存在梯度冲突
- 梯度范数不平衡

### 7. 罚系数（损失权重）ρ 的影响 (`lambda_effect.py`)
- λ=10~100为最优范围
- λ过大或过小都会损害精度
