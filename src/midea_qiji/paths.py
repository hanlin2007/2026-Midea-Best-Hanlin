"""midea_qiji.paths：仓库内模型路径常量与本地模型解析。

``resolve_model_path`` 是唯一合法的"模型 id → 本地路径"解析入口：优先直接
路径，其次 --model-root / MODELSCOPE_CACHE，最后 models/modelscope 与
modelscope hub 缓存，并兼容 ModelScope 的 ``. → ___`` 目录转义。

为什么要求"唯一入口"：同一个模型 id 可能落在多个位置（用户用
--model-root 指定的目录、modelscope 库自动下载的缓存、仓库自带目录……），
查找顺序与目录转义规则一旦分散在各调用方手里，就会出现"这边找得到、
那边找不到"的口径分裂，还改一处漏一处。收敛成单一函数后：查找规则只有
一份，改一处全仓库生效，还能单独写测试锁定行为。上层（recognizers 等）
需要解析模型路径时，一律 import 本函数，而不是自己拼路径。

PACKAGE_MODEL_ROOT 指向仓库内的 ``model/`` 目录，FireRed 三件套
（模型 / 源码仓库 / vendor）默认位于其下。

关键导出：``resolve_model_path``、``PACKAGE_MODEL_ROOT``、
``DEFAULT_FIRED_MODEL / DEFAULT_FIRED_REPO / DEFAULT_FIRED_VENDOR``。

依赖：仅标准库（json 不需要；os / pathlib）——零第三方依赖，任何模块
（包括只想拿个路径的脚本）都能放心 import。
"""
from __future__ import annotations  # 延迟求值注解（str | Path | None 等）

import os  # 仅用于读取 MODELSCOPE_CACHE 环境变量
from pathlib import Path  # 面向对象的路径类型：拼接、存在性判断、resolve 都比字符串操作安全

# PACKAGE_MODEL_ROOT：仓库内的模型资产根目录（<仓库根>/model），
# FireRed 三件套的默认挂载点。
# 为什么用 __file__ 推算而不是写死绝对路径：仓库可能被 clone 到任何位置，
# 写死 "E:/xxx" 在别人机器上必然失效；从本文件位置出发逐级向上找，则无论
# 从哪个工作目录启动 Python 都能定位到同一个目录。
# 逐级拆解（源码树布局为 <仓库根>/src/midea_qiji/paths.py）：
#   Path(__file__)     -> 本文件路径（可能是相对路径）
#   .resolve()         -> 绝对路径，同时展开符号链接
#   .parents           -> 逐级父目录序列：parents[0]=src/midea_qiji，
#                        parents[1]=src，parents[2]=仓库根
#   / "model"          -> Path 的 / 运算符拼接路径（自动带分隔符）
# 注意：该常量以"源码树方式运行"（PYTHONPATH=src）为前提；若把包 pip
# 安装进 site-packages，__file__ 指向安装目录，推算结果将失去意义——
# 这也是模型资产必须随仓库分发（而非随包分发）的原因之一。
PACKAGE_MODEL_ROOT = Path(__file__).resolve().parents[2] / "model"
# FireRedASR2 是语音识别模型，运行它需要三样东西（"三件套"）：
#   - 模型权重（DEFAULT_FIRED_MODEL）；
#   - 配套官方源码仓库（DEFAULT_FIRED_REPO，含模型定义/前处理代码）；
#   - vendor 依赖目录（DEFAULT_FIRED_VENDOR，kaldi_native_fbank 等原生库，
#     运行时需加入 sys.path / 环境变量）。
# 前两者转成 str，因为消费方（infer.py 的 CLI 默认值、模型加载 API）把
# 它们当字符串路径参数用；vendor 保留 Path 对象，方便继续做路径拼接。
DEFAULT_FIRED_MODEL = str(PACKAGE_MODEL_ROOT / "external" / "FireRedASR2-AED")
DEFAULT_FIRED_REPO = str(PACKAGE_MODEL_ROOT / "external" / "FireRedASR2S")
DEFAULT_FIRED_VENDOR = PACKAGE_MODEL_ROOT / "runtime" / "vendor"


def resolve_model_path(model: str | Path, model_root: str | Path | None) -> Path:
    """把模型 id（如 ``iic/speech_eres2netv2_sv_zh-cn_16k-common``）解析为本地路径。

    解析策略（按优先级从高到低）：
      1. model 本身就是存在的路径（绝对/相对都可以）——直接返回；
      2. 用户显式指定的 model_root（CLI 的 --model-root）下的候选拼法；
      3. 环境变量 MODELSCOPE_CACHE 下的候选拼法；
      4. 两个默认位置：工作目录下的 models/modelscope、家目录下的
         modelscope hub 缓存（modelscope 库的默认下载位置）。
    每个根目录下都尝试四种拼法：``<root>/<id>``、``<root>/<id 带 ___ 转义>``、
    ``<root>/models/<id>``、``<root>/models/<id 带 ___ 转义>``。
    "``.`` → ``___``" 是 ModelScope 的目录转义约定：模型 id 中的点
    （版本号/文件名里的 "."）在缓存目录里被替换成三个下划线。既然无法
    预知模型被下载成了哪种拼法，就把四种都试一遍，命中即返回。
    """
    direct = Path(model)  # 先当普通路径处理
    if direct.exists():
        return direct.resolve()  # resolve：绝对化 + 展开符号链接，保证调用方拿到可定位的真实位置
    roots = []  # 候选根目录列表：顺序即优先级（显式 > 环境 > 默认）
    if model_root is not None:
        roots.append(Path(model_root))  # 用户显式指定的目录最可信
    if os.environ.get("MODELSCOPE_CACHE"):
        roots.append(Path(os.environ["MODELSCOPE_CACHE"]))  # 环境变量次之
    roots.extend([Path("models/modelscope"), Path.home() / ".cache" / "modelscope" / "hub"])  # 默认位置兜底
    for root in roots:
        for candidate in (
            root / str(model),
            root / str(model).replace(".", "___"),  # ModelScope 的 "." -> "___" 目录转义拼法
            root / "models" / str(model),  # 部分缓存布局会多套一层 models/
            root / "models" / str(model).replace(".", "___"),
        ):
            if candidate.exists():
                return candidate.resolve()  # 命中即返回：先匹配到的根目录优先级更高
    raise FileNotFoundError(
        f"local model {model!r} was not found; set --model-root or MODELSCOPE_CACHE"
    )
