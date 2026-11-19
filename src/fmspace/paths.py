"""路径。换机器或换数据位置时改 DATA_DIR；FEATURE_DIR、RESULT_DIR、FIGURE_DIR 在仓库根目录下。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path.home() / "workspace" / "WZX" / "DataSet" / "DeepDRiD"  # DeepDRiD 官方仓库的根目录
FEATURE_DIR = ROOT / "features"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
RISE_DIR = RESULT_DIR / "rise"
