# 实验一：新闻文本分类

这次实验围绕三个问题展开：四种分类器在相同文本表示下有什么差别；BoW、TF-IDF 和平均 Word2Vec 哪一种更适合当前数据；标题与正文分开处理能否带来稳定收益。完整过程和输出见 `experiment.ipynb`，两页报告的 LaTeX 源文件放在 `report/`。

## 文件说明

| 文件 | 内容 |
| --- | --- |
| `experiment.ipynb` | 按实验顺序展示数据、搜索结果、图表和分析 |
| `data.py` | 读取数据、分层划分、检查重复文本 |
| `text_features.py` | 解析标题与正文，构造 BoW、TF-IDF 和 Word2Vec 特征 |
| `models.py` | 定义 NB、SVM、LR 和 MLP |
| `search.py` | 运行参数搜索、结构对照与划分复核 |
| `run_experiments.py` | 命令行入口 |
| `summarize.py` | 把逐次运行记录整理成结果表 |
| `plot_results.py` | 从结果文件生成报告图 |
| `report/report.tex` | 两页实验报告源文件 |

## 环境

实验使用 Python 3.11，具体依赖版本见 `requirements.txt`。建议新建环境后安装：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境；Linux 或 macOS 使用 `source .venv/bin/activate`。

## 运行实验

`--data-dir` 指向直接包含 `train_data.csv` 和 `test_data_unlabeled.csv` 的目录：

```bash
python run_experiments.py --data-dir ../data-Project1/data-Project1 --predict
python plot_results.py
```

主划分采用随机种子 66，另用种子 88 复核最终入选的几组配置。训练集与验证集按类别分层划分为 8:2。词表、IDF 和 Word2Vec 都只在训练部分拟合；无标签测试集在方案选定后才用于预测。

数据预处理保留标题和正文，去掉作者、组织、主机等邮件头字段，并清除标题开头重复的 `Re:`。正文中的引用与签名没有删除。分词时转为小写，保留长度至少为 2 的词元，不使用停用词表、词干化或数据增强。

BoW 和 TF-IDF 共用一元词表设置：`max_features=15000`、`min_df=2`、`max_df=0.98`，最后做 L2 归一化。Word2Vec 只用训练文本学习 200 维 Skip-gram 词向量，窗口为 5，最低词频为 2，训练 20 轮；文档向量取词向量均值后归一化，没有使用外部语料或预训练权重。

参数搜索范围写在 `config.json`。首轮尝试：

- NB：`alpha = 0.1, 1, 10`
- 线性 SVM：`C = 0.1, 1, 10`
- LR：`C = 1, 10, 100`
- MLP：64 个隐层单元，`alpha = 0.0001, 0.01, 1`

若 NB、SVM 或 LR 的最好结果落在搜索边界，再向外补查一个数量级。标题结构实验比较仅正文、标题正文直接拼接，以及标题正文分别表示后取 `0.5、1、2` 三种标题权重。MLP 容量实验比较 `(64,)`、`(128,)` 和 `(64, 32)`，并检查弱、强两种 L2 正则化。

主指标为 Macro-F1，同时保存 Accuracy、各类 precision/recall/F1、训练时间、迭代次数和混淆矩阵。清洗后若验证文本与训练文本完全相同，该条预测仍会保存，但不计入模型选择分数。`results/splits/` 记录划分行号和被排除的条目。

## 输出

`results/results.json` 保存各组参数、指标、耗时、迭代次数和混淆矩阵，便于 Notebook 展示完整搜索过程；整理后的 CSV 也放在 `results/`。逐条验证预测和划分行号可由随机种子重新生成，因此不上传仓库。最终预测为 `output/prediction.csv`，只有一列、没有表头，顺序与测试文件一致。

## 编译报告

进入报告目录运行两次 XeLaTeX：

```bash
cd report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

## 参考资料

- [scikit-learn：文本特征提取](https://scikit-learn.org/1.2/modules/feature_extraction.html#text-feature-extraction)
- [scikit-learn：多层感知机](https://scikit-learn.org/1.2/modules/neural_networks_supervised.html)
- [Gensim：Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
