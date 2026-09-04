# 物理智能（Physical AI）的理论基础——以约束优化为主线

中文 LaTeX 教科书源码。全书 17 章 + 4 个附录 + 后记，主线是把每一章的问题写成
`L = f + λᵀg`，追问"目标是什么、约束是什么、λ 代表什么（影子价格）"。

## 目录结构

```
book_outline.tex      主文件（导言区、前言/目录/导论、四个部分与附录的组织）
chapters/             各章 .tex：chap01–chap17、preface、epilogue、appendix_{index,env,llm,eng}
figures/chapNN/       各章的图（PDF + PNG，由 code/chapNN 下的脚本生成；TikZ 图直接写在 .tex 里）
code/chapNN/          各章的实验与绘图脚本；code/appC/ 为附录 C 的 Transformer 实现
code/textbook_style.py  全书统一绘图风格（字体、色板、尺寸、双格式输出）
docs/figures/chapNN.md  每章每张图的思路说明（画什么、为什么这样画、由哪个脚本生成）
REVIEW_NOTES.md       历次修订记录与待作者复核的事项
build/                编译输出（不入库）
```

## 编译

需要 TeX Live（XeLaTeX + ctex）和中文字体（Fandol 随 TeX Live 自带；生僻字回退到 Noto CJK）。

```bash
mkdir -p build
for i in 1 2 3; do xelatex -interaction=nonstopmode -output-directory=build book_outline.tex; done
```

三遍之后 `build/book_outline.pdf` 即为成书；`.log` 里应当没有 `!` 错误、未定义引用和 5pt 以上的 `Overfull \hbox`。

## 重新生成图

脚本都假定在仓库根目录运行，并通过 `sys.path.insert(0, 'code')` 载入统一风格：

```bash
pip install numpy scipy matplotlib torch
python code/chap14/lambda_effect.py     # 例：第 14 章罚系数实验，输出到 figures/chap14/
```

每个脚本对应的图、参数与结论见 `docs/figures/`。
