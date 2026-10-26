"""路径。换机器或换数据位置时改 DATA_DIR；其余目录都在仓库根目录下。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(r"C:\M1\master paper\DataSet\DeepDRiD")
FEATURE_DIR = ROOT / "features"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
