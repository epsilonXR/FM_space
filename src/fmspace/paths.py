"""路径。换机器或换数据位置时改 DATA_DIR；FEATURE_DIR、RESULT_DIR、FIGURE_DIR 在仓库根目录下。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(r"C:\M1\master paper\DataSet\DeepDRiD")
OLD_FEATURE_DIR = Path(r"C:\M1\SAMIT\project\SAMIT\features\DeepDRiD")  # 旧仓库 SAMIT 的特征，只在 extract.ipynb 里比较用
FEATURE_DIR = ROOT / "features"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
