# src/utils.py
import os
from pathlib import Path

def get_output_dir():
    """
    智能获取和创建输出目录（返回绝对路径）。
    尝试以下位置（优先顺序）：
      1. <project_root>/data/example_output
      2. <current_working_dir>/data/example_output
      3. ./data/example_output
      4. 临时目录 / 当前工作目录
    说明：返回的是绝对路径，方便 notebook / 脚本都能写入同一位置。
    """
    # 以项目根（脚本文件所在上两级）为优先：假设 src/ 在项目根下
    try:
        # 当前文件的父目录（src），向上两级到项目根
        this_file = Path(__file__).resolve()
        project_root = this_file.parents[1]
    except Exception:
        project_root = Path.cwd()

    candidates = [
        project_root / "data" / "example_output",
        Path.cwd() / "data" / "example_output",
        Path("data") / "example_output",
    ]

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            # 测试写权限
            test_file = path / "._writetest.tmp"
            with open(test_file, "w") as f:
                f.write("ok")
            test_file.unlink()
            # 返回绝对路径字符串
            abs_path = str(path.resolve())
            print(f"✅ 使用输出目录: {abs_path}")
            return abs_path
        except (OSError, PermissionError) as e:
            # 无法写入，尝试下一个
            print(f"⚠️ 无法写入 {path}: {e}")

    # 退回到当前工作目录
    fallback = str(Path.cwd().resolve())
    print(f"⚠️ 所有候选路径不可写，使用当前工作目录: {fallback}")
    return fallback

# 只在模块导入时计算一次
OUTPUT_DIR = get_output_dir()