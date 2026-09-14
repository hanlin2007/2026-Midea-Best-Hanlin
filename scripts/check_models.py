#!/usr/bin/env python3
"""模型资产完整性自检：按 model/MANIFEST.json 校验资产 + 发现 model/ 外残留权重。

校验内容：canonical 资产存在、兼容软链指向正确、目录字节数/文件数一致、
文件 sha256 一致、外部依赖软链存在；随后扫描工作区，报告未纳入 model/
管理的残留权重文件。哈希/目录统计/排除规则与资产搬迁工具
relocate_workspace_models.py 共用同一实现，避免双份口径漂移。

为什么需要这个脚本：搬迁工具 relocate_workspace_models.py 会把散落在工作区
各处的权重文件搬到 model/workspace/ 统一管理，在原位置留下软链接，并把每个
资产的字节数/文件数/sha256 指纹记进 model/MANIFEST.json。权重文件动辄几个 GB，
一旦搬运、换盘、清理过程中损坏或漏搬，训练脚本不会立刻报错——它会加载到坏的
（或旧的）权重，产出"看起来正常但结果不对"的实验。对竞赛复现来说，这种无声
腐败是最危险的。所以自检存在的价值是：实验前花十几秒对账，换掉"结果莫名其妙
变差却找不到原因"的几小时排查。

实现要点（初学可对照正文注释）：
  - 软链接只存"指针"不存数据，校验时用 Path.resolve() 解出真实目标再比较；
  - 大文件哈希必须分块读，否则 GB 级文件会撑爆内存（见 sha256 实现）；
  - 残留扫描与搬迁工具共用 WEIGHT_SCAN_ROOTS / is_excluded 常量，
    保证"搬迁时认定的权重"和"自检时认定的残留"是同一套口径。

用法（仓库根目录）：
  python scripts/check_models.py
退出码：0 = 全部通过；1 = 有错误（明细打印到 stderr，stdout 只保留一行
机器可读 JSON，方便脚本/CI 直接解析）。

依赖：标准库 json / sys；运行时直接 import 同目录的
relocate_workspace_models.py——直接执行脚本时 Python 会把脚本所在目录
放进 sys.path[0]，因此同目录模块无需额外配置即可导入。
"""
from __future__ import annotations

import json
import sys

# 从同目录脚本 import 公共实现，而不是把逻辑复制一份。
# 原因：如果两个脚本各维护一份"什么算权重文件 / 怎么算哈希"的代码，
# 迟早出现一边改了另一边没改的"口径漂移"——搬迁时认定为权重的文件，
# 自检时却被算作残留（或相反）。共享同一实现 = 定义只存在一处，
# 两个工具永远站在同一条线上。
from relocate_workspace_models import (
    WEIGHT_SCAN_ROOTS,
    WEIGHT_SUFFIXES,
    directory_stats,
    is_excluded,
    sha256,
    workspace_root,
)


def main() -> None:
    # workspace_root() 定义在 relocate_workspace_models.py 里，返回
    # Path(__file__).resolve().parents[2]——即仓库上级的"工作区根目录"
    # （model/、dataset/ 等资产目录和仓库本身都挂在这个根下）。
    root = workspace_root()
    manifest_path = root / "model" / "MANIFEST.json"
    # MANIFEST.json 是搬迁工具执行时写下的"资产台账"：每条记录包含
    # 原路径（软链位置）、规范路径（真实数据位置）、字节数/文件数/sha256。
    # 自检的本质 = 拿磁盘现状与这份台账逐项对账。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for asset in manifest["assets"]:
        # original_path 是搬迁后留在原位置的兼容软链接；
        # canonical_path 是 model/workspace/ 下的真实数据。
        # 校验要同时看"指针"（链接）和"数据"（本体）两端。
        original = root / asset["original_path"]
        canonical = root / asset["canonical_path"]
        # 最严重的错误：数据本体丢了。此时继续查链接指向没有意义，直接
        # continue 记录下一条（一条错误信息足以定位问题）。
        if not canonical.exists():
            errors.append(f"missing canonical asset: {canonical}")
            continue
        # 原位置必须是软链接。搬迁约定：数据挪走后留下"兼容链接"，
        # 让仍引用旧路径的脚本/配置无需修改也能继续工作。
        if not original.is_symlink():
            errors.append(f"original path is not a compatibility link: {original}")
        # resolve() 会把软链接一路解到底并归一化为绝对路径。
        # 注意不能直接拿 Path 字符串 == 比较：链接可能是以相对路径写的，
        # 两个字符串看起来不同但指向同一处；resolve 之后比较才可靠。
        elif original.resolve() != canonical.resolve():
            errors.append(f"compatibility link points elsewhere: {original}")
        if canonical.is_dir():
            # 目录型资产（如 LoRA adapter 目录）：只比对"总字节数 + 文件数"。
            # 目录里可能含数百个小文件，逐个哈希既慢又没必要——文件数与
            # 字节数双双吻合时，数据损坏的概率已经极低；且逐文件哈希会让
            # MANIFEST 膨胀到不可读。
            size, count = directory_stats(canonical)
            if size != asset["bytes"] or count != asset["file_count"]:
                errors.append(f"directory stat mismatch: {canonical}")
        else:
            # 单文件资产：先比大小（一次 stat，几乎免费），大小一致才做
            # sha256（要整读一遍文件，是重活）。先做廉价检查、命中才做
            # 昂贵检查，是工程里最常见的性能套路。
            if canonical.stat().st_size != asset["bytes"]:
                errors.append(f"file size mismatch: {canonical}")
            if asset["sha256"] and sha256(canonical) != asset["sha256"]:
                errors.append(f"file hash mismatch: {canonical}")
    # 外部依赖（FireRed 等）的数据不在本工作区，MANIFEST 只记录指向它们的软链。
    for dependency in manifest["external_dependencies"]:
        link = root / dependency["link"]
        # is_symlink() 只回答"是不是链接"；链接目标可能已被删除（悬空链接），
        # 所以还要 exists()——注意 exists() 会沿链接解析到目标，因此能
        # 同时覆盖"不是链接"和"是链接但目标没了"两种情况。
        if not link.is_symlink() or not link.exists():
            errors.append(f"missing external dependency link: {link}")
    # 残留扫描：搬迁工具只能搬"当时存在的"权重；之后如果有人往工作区新塞了
    # 权重（下载、新训练产物），它们不在 MANIFEST 台账里，不受自检保护。
    # 因此要主动扫描一遍，把"散落在 model/ 之外的权重"揪出来报告。
    residual = []
    for relative_root in WEIGHT_SCAN_ROOTS:
        scan_root = root / relative_root
        if not scan_root.exists():
            continue
        # 扫描根既可能是目录（大部分条目）也可能是单个文件
        # （如 sv_selector/selector.pt），所以先判断再选遍历方式。
        paths = (scan_root,) if scan_root.is_file() else scan_root.rglob("*")
        # rglob("*") 递归产出所有子孙路径。与 os.walk 相比，pathlib 的
        # rglob 直接返回 Path 对象且自带递归，写起来更顺；目录本身也在
        # 产出里，靠下面的 is_file() 过滤掉。
        for path in paths:
            # 三重过滤：
            # 1) 只要真实文件（目录、以及已搬到 model/ 的兼容软链接都跳过）；
            # 2) 跳过软链接——链接本身不占权重数据，其目标已在台账里查过；
            # 3) 只看权重后缀；suffix.lower() 小写化是兼容 .PT/.Pth 这类
            #    手滑的大小写。
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in WEIGHT_SUFFIXES:
                continue
            # relative_to(root) 把绝对路径裁成"相对工作区根"的形式，
            # 既供排除规则匹配（is_excluded 按相对路径段判断），
            # 也是最终报告给用户的可读路径。
            relative = path.relative_to(root)
            if not is_excluded(relative):
                residual.append(str(relative))
    if residual:
        errors.append(f"residual non-linked model files outside model/: {residual}")
    if errors:
        # 错误信息走 stderr：stdout 留给下面那行机器可读 JSON，
        # 这样 CI/脚本可以只解析 stdout 而不必过滤人类可读文本。
        print("\n".join(errors), file=sys.stderr)
        # 用 SystemExit(1) 而不是 sys.exit(1)：语义相同，但明确表达
        # "这是异常退出"。调用方（shell/CI）靠退出码判断成败。
        raise SystemExit(1)
    print(json.dumps({"asset_count": manifest["asset_count"], "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    # 直接执行才跑 main()；被 import（如单测想复用）时不执行，
    # 这是脚本和库两用的标准写法。
    main()
