"""pipeline：目标说话人语音指令识别系统（生产管线包）。

这是一个 Python 包（package）：目录里放了 ``__init__.py``，``import pipeline``
时解释器就会先执行本文件。本文件只做"目录与说明"，不定义任何函数——
包的全部能力由下面列出的各模块提供。使用时按模块精确导入，例如
``from pipeline.engine import PipelineRunner``；精确导入的好处是只 import
自己需要的模块，不把用不到的依赖（尤其是 torch）拖进来。

目录分工（依赖只能向下：scripts/tests → pipeline → core → third_party）：

  - ``cli.py``            命令行入口（python -m pipeline.cli）
  - ``engine.py``         管线引擎：PipelineConfig / AsrCache / FusionGate / PipelineRunner
  - ``single_speaker.py`` 单人判别器与"双单人"原始音频余弦路由
  - ``separator.py``      dasynth 微调 MossFormer2 torch 分离器（音频前端）
  - ``metrics.py``        分离质量指标（SI-SDR / SI-SNR / PIT 损失）
  - ``train.py``          dasynth 域适配微调训练
  - ``helpers.py``        共享工具（哈希 / 音频读取 / 缓存路径）
  - ``reporting.py``      评测汇总（CER/PR/RR）、提交物写出与人类可读报告

"依赖只能向下"是分层原则：pipeline 只允许 import 它下层的包——core 是
自研算法内核（文本口径、声纹前端、识别器、样本数据模型），third_party
是外部 vendor 代码（如 mossformer2 模型定义，不在 pip 里，由代码手动
加入 sys.path 后 import）。反方向（core 里出现 ``import pipeline``）就是
分层错误。单向依赖让每一层都能被单独理解、测试和替换。

延迟导入约定：``separator.py`` 依赖 torch，管线把它延迟到 PipelineRunner
构造时才 import——于是没有 torch 的环境也能 ``python -m pipeline.cli --help``、
也能跑 tests/ 下的单测；只有真正跑分离推理时才需要 torch 在场。

运行入口：``bash scripts/run.sh``。
"""
