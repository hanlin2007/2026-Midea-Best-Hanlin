#!/usr/bin/env python3

# shebang：这行在 Python 解释器眼里就是一行普通注释（以 # 开头）。它真正的读者是 Linux / macOS 内核，在 execve() 系统调用加载文件时读前两个字节：0x23 0x21  →  ASCII "#!" 
# 内核：哦，这是个脚本，不是 ELF 二进制
# 然后内核干的事等价于：
# /usr/bin/env python3 ./scripts/check_models.py 内核自动帮你找到解释器，把文件作为参数传给它去执行。

# 也意味着在 Linux 下可以：
# chmod + x scripts/check_models.py
# ./scripts/check_models.py

"""环境检查脚本

【用法】在仓库根目录执行：
python scripts/check_model.py

【功能】
    1. configs/ 正常加载
    2. 配置中的每一个模型文件是否存在
    3. 检查模型文件的指纹 sha256
    4. 数据集 jsonl 与 wav 的检查

【输出】
    1. 全部通过返回 0 输出下一步
    2. 任一失败返回 1 输出报错问题
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# 旧写法 使用 os.path，不推荐
# import os
# root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = Path(__file__).resolve().parents[1]
# __file__表示当前文件路径 .resolve()方法转换为绝对路径，跨文件系统友好  .parents[1]表示从[0]开始，向上推第二级目录

# /mnt/c/Users/lihanlin2007/Desktop/Github/midea_qiji_best_hanlin/scripts/check_models.py

EXPECTED_SHA256 = {
    "moss_model": "768f7a8aab85a8f75581bd296c9a53cd3d79d7d274d2a03383ab7b75a4748518",
    "dual_backend": "58eefbdaa75cf74d143cbb4d88af800c98b1ec1f487438b61928274f72d2ec6b",   # 尾逗号，python特性，git diff、粘贴修改友好
}
# python字典！来做这个配置文件，比json更友好！与json也可以转换

# 开始定义函数！用到参数和返回值的类型注释，注意冒号和返回类型注释的顺序
def sha256(path: Path) -> str:
    """这是一个多行的函数注释，这个函数的功能是计算路径文件的 SHA 256 指纹。这个描述会出现在函数描述标签上！ """
    digest = hashlib.sha256()
    with path.open("rb") as handle:   # 读二进制文本，算哈希：换行 ×
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# hash
# digest = hashlib.sha256()   # 1. 创建一个"计算器"
# digest.update(chunk)        # 2. 往里喂数据（可以喂很多次）
# digest.hexdigest()          # 3. 取最终结果（十六进制字符串）


# # 最终这两种 open 写法效果完全一样，Path 跨平台更友好、路径可字符拼接

# # 传统写法（内置函数 + 字符串路径）
# with open('data/foo.txt', 'r', encoding='utf-8') as f:
#     content = f.read()

# # pathlib 写法（Path 对象的方法）
# p = Path('da''ta') / 'docs' /'foo.txt'   # 前面是字符拼接，后面是重载的路径拼接，会自动进行跨平台转换 
# with p.open('r', encoding='utf-8') as f:
#     content = f.read()


# with 叫上下文管理器，进入缩进时打开资源，退出缩进时自动帮你关掉（即使中间报错了也会关） as handle 作为读取的处理对象


# 最绕的地方：迭代器读取 —— 等价 while 写法
# chunk_size = 1024 * 1024   # 1 MB
# while True:
#     chunk = handle.read(chunk_size)   # 读最多1MB
#     if chunk == b"":                  # 读到文件末尾，返回空 bytes
#         break
#     digest.update(chunk)

# 为什么要分块？​
# handle.read() 不加参数会一次读完整文件。如果文件 5GB，内存直接炸。
# 每次读 1MB，内存占用永远是 1MB 左右，无论文件多大。


def resolve(path_str:str) -> Path:
    """还需要定义一个工具函数，用于解析配置中的路径，把相对路径和绝对路径进行一个统一，便于后面的脚本检查。因为其实 config 写得很乱，后续需要多次复用这个路径处理工具！！"""
    p = Path(path_str).expanduser()
    return (p if p.is_absolute() else (REPO_ROOT / p)).resolve()
# !!!第一次能对 AI 生成的代码进行优化，哪怕只是一点小小的边界补充、细节规范
# 记住这个 resolve() 函数：使得跨系统路径更干净


def main() -> None:
    # 对配置 json 文件进行读取、建表，同时建立三个阶段的检查键列表
    config_path = REPO_ROOT / "configs" / "szy_v1_firered.json"
    config = json.loads(config_path.read_text(encoding="UTF-8"))

    step1_keys = ["dual_backend", "moss_model",]
    step2_keys = ["firered_model", "firered_repo", "firered_vendor",]
    step3_keys = ["sv_model", "cam_model",]
    
    # 初始化建表，错误列表、警告列表
    errors = []     # 出现错误先继续收集，到最后再一次性可视化输出
    warnings = []

    # 1. 第一步：检查文件型模型（双后端 MLP 和 MossFormer2）并校验 SHA256 【使用.is_file()方法】
    for key in step1_keys:
        # p = (REPO_ROOT / config[key]).resolve()
        # 终于想起来复用前面自己写的工具处理函数！！
        p = resolve(config[key]) 
        if p.is_file():
            real_sha256 = sha256(p)
            if(real_sha256 == EXPECTED_SHA256[key]):
                print(f"[Step1]: Model [{key}] sha256 check passed! Model is preparing well.")
            else:
                errors.append(
                    f"[Step1]: Model [{key}] sha256 check failed! Please reload the model again.")
        else:
            errors.append(f"[Step1]: Model [{key}] didn't exist.")


    # 2. 第二步：检查目录型模型（FireRED 三个文件目录） 【使用.is_dir()方法】
    for key in step2_keys:
        p = resolve(config[key])
        if p.is_dir():
            print(f"[Step2]: Model [{key}] is ready! Related to FireRed_{p.name}")
        else:
            errors.append(f"[Step2]: Model [{key}] (directory) didn't exist.")


    # 3. 第三步：检查魔塔社区来源，固定模型根目录 model_root 的两个声纹识别模型 【仍然使用.is_dir()方法】 
    # 其实到这里也能看出来，分步骤的主要依据就是代码和比较逻辑的复用性
    for key in step3_keys:
        p = resolve(Path(config["model_root"]) / config[key])
        if p.is_dir():
            print(f"[Step3]: Model [{key}] downloaded from ModelScope is ready!")
        else:
            errors.append(
                f"[Step3]: Model [{key}] downloaded from ModelScope didn't exist. Please run: python download_models.py")


    # 4. 第四步：仔细检查数据集中文件是否全部齐全，但这里其实并不足够严谨，【使用 any(p.glob("*.wav")) 无法报数量漏洞】 发现了 AI 写的代码的第三个追求体面问题！！
    dataset_path = REPO_ROOT / "dataset" /"dataset_test" / "datasetA" / "datasetA"
    for str in ["neg.jsonl","pos.jsonl"]:
        if (dataset_path / str).is_file():
            print(
                f"[Step4]: Jsonl files [{str}] is already in dataset!")
        else:
            errors.append(f"[Step4]: Jsonl files [{str}] didn't exists!")
    for key in ["neg","pos"]:
        p = dataset_path / key
        if not p.is_dir() or not any(p.glob("*.wav")):
            errors.append(f"[Step4]: Audio [{p.name}] directory didn't exists!")
        else:
            print(f"[Step4]: Audio [{p.name}] directory is ready!")


    # ============== 汇总测试结果，最终报告呈现 ==========================
    if errors:
        print(f"{len(errors)} error(s) found!")
        print(*errors,sep = "\n")
    else:
        print("MODEL CHECK PASS!! NEXT: SMOKE TEST. GOOD LUCK!!")


if __name__ == "__main__":
    main()
# ！！！！真的找到了第二个 AI 写的代码的 BUG ———— 这里的 warnings 完全是形式主义，目录型模型没有正确加载时，models_check 脚本会放过，而报检查全部通过，这是一处明显的自我感动


# 作用：设置唯一的运行开关:
# 单独当作脚本运行 python scripts/check_models.py 时，解释器把该文件 __name__ 设为 "__main__" ，才调用 main()

# 一个文件既是库又是脚本：
# 对仓库里别的 Python 文件：它可作为被导入的模块，提供 sha256 / resolve 的工具模块
# from scripts.check_models import sha256, resolve, REPO_ROOT
# print(sha256(Path("model.onnx")))   # 只借工具模块，不运行脚本
