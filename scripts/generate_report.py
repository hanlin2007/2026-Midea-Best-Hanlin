#!/usr/bin/env python3
"""离线重新生成人类可读报告（pipeline 每次运行已自动生成，本脚本用于旧产物补生成）。

报告本体在 ``szy.reporting.generate_report``（逻辑收敛在包里，脚本只做
命令行薄壳——仓库的统一分工：逻辑进包，scripts 只做入口）。

用法：
    python scripts/generate_report.py outputs/<NAME>/<NAME>_detail.json
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本在"未安装包、未设 PYTHONPATH"时也能直接运行：把仓库 src/ 注入
# 模块搜索路径（与 separator.py 的 vendor 注入同一手法）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from szy.reporting import generate_report  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        detail_json = Path("outputs/detail.json")
    else:
        detail_json = Path(sys.argv[1])

    if not detail_json.is_file():
        print(f"Error: {detail_json} not found", file=sys.stderr)
        sys.exit(1)

    report_text = generate_report(detail_json)
    report_path = detail_json.with_name(detail_json.stem + "_report.txt")
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to: {report_path}")
    print(f"  {(report_path.stat().st_size / 1024):.1f} KB, {report_text.count(chr(10))} lines")


if __name__ == "__main__":
    main()
