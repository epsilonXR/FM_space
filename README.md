# FM_space

DeepDRiD 上基础模型视觉 embedding 的特征空间分析：模型 → 提取特征 → 分析 → 画图。

| 步骤 | 代码 | 输出 |
|---|---|---|
| 模型 | `src/fmspace/models.py` | — |
| 提取特征 | `src/fmspace/extract.py`、`notebooks/extract.ipynb` | `features/<model>.npz`（不进 git） |
| 分析 | `src/fmspace/analysis.py`、`notebooks/analysis.ipynb` | `results/` |
| 画图 | `src/fmspace/plots.py`、`notebooks/figures.ipynb` | `figures/` |

metadata 和患者划分在 `src/fmspace/data.py`，路径在 `src/fmspace/paths.py`。每个 notebook 的第一个代码格是初始化（import、读数据），其余各格只依赖它，顺序随意。

## 环境

```powershell
py -3.12 -m venv D:\envs\fmspace
D:\envs\fmspace\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # 命令以 pytorch.org 为准；无 GPU 去掉 --index-url
pip install -e <仓库目录>
```

VS Code 的 notebook 内核选 `D:\envs\fmspace\Scripts\python.exe`。

## 数据

`paths.py` 的 `DATA_DIR` 指向 DeepDRiD 目录：

```text
DATA_DIR/
├── regular-fundus-training.csv
└── regular-fundus-training/Images/<patient_id>/<image_id>.jpg
```

## 模型

| 名称 | checkpoint | 输出 | 维度 |
|---|---|---|---|
| siglip | google/siglip-so400m-patch14-384 | `pooler_output` | 1152 |
| medsiglip | google/medsiglip-448 | `pooler_output` | 1152 |
| conceptclip | JerrryNie/ConceptCLIP | `encode_image` 的 CLS 向量 | 1152 |
| flair | jusiro2/FLAIR | `vision_model` 的输出 | 512 |

新增模型：在 `models.py` 写一个加载函数（或复用已有的），在 `MODELS` 里加一行。加载函数返回 `encode(images) -> Tensor[B, D]`。
