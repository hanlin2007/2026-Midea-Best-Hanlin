"""midea_qiji.recognizers：全部 ASR 识别后端（统一 transcribe 接口）。

四个后端按"转写口径 + 置信度语义"分工：

  - ``QwenRecognizer``           Qwen3-ASR-0.6B（baseline 默认后端，无置信度）；
  - ``FireRedRecognizer``        FireRedASR2-AED（SZY 系列主后端，带置信度）；
  - ``FireRedLLMRecognizer``     FireRedASR2-LLM（Qwen2-7B 解码，序列平均对数
                                 概率充当置信度，供音色触发器使用）；
  - ``ConditionedQwenRecognizer`` Qwen3-ASR + sv_asr 双音频条件 LoRA
                                 （[kws, cmd] → 目标说话人文本，实验后端）。

公共约定：``transcribe_many(paths) -> list[str]`` 返回归一化文本；管线内部
需要置信度时统一走 szy.helpers.transcribe_firered_rows（访问
``_model.transcribe`` 的原始行）。重型依赖（torch / peft / FireRed 包 /
modelscope）一律延迟导入，保证本模块可被轻量环境 import。

设计说明（面向初学者）：
- 四个后端没有共同的抽象基类，而是靠"鸭子类型"统一接口：只要对象提供
  transcribe_many(paths) 方法，infer.main 就能无差别调度它。这是 Python
  社区常见的"协议式多态"——不强制继承，只约定行为。FireRedLLMRecognizer
  更是把自己的 transcribe() 暴露成 ``_model.transcribe``，让上层能用与
  FireRedRecognizer 完全相同的代码访问（见该类的 self._model = self）。
- ``szy.helpers.transcribe_firered_rows`` 属于上层 szy 系统的入口：上层
  来调用本内核层的能力，内核层自身绝不 import 任何 szy/ 模块——依赖
  方向只能自下而上，这是分层架构的纪律。

功能：把音频路径列表转成归一化中文文本列表（各后端另按需提供置信度）。

关键导出：QwenRecognizer / FireRedRecognizer / FireRedLLMRecognizer /
ConditionedQwenRecognizer。

依赖：midea_qiji.text / .audio / .paths / .infer（suppress_model_output）。
重型依赖（torch / peft / qwen_asr / fireredasr2）全部在构造函数内延迟导入。

用法（通常由 infer.main 按 config.asr_backend 装配，不必手写）：
    recognizer = QwenRecognizer(model, device, model_root, batch_size)
    texts = recognizer.transcribe_many([Path("a.wav"), Path("b.wav")])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# 注意：这里顶部 import 了 .infer，但不会循环导入——infer 对 recognizers
# 的 import 全部放在 main() 函数内部（延迟导入），所以两个模块互相引用
# 时，import 顺序不会形成环。
from .audio import load_audio_mono_16k
from .infer import suppress_model_output
from .paths import DEFAULT_FIRED_VENDOR, resolve_model_path
from .text import extract_asr_text, normalize_transcript


class QwenRecognizer:
    """Qwen3-ASR-0.6B 后端（baseline 默认；transcribe 不输出置信度）。"""

    def __init__(
        self,
        model: str,
        device: str,
        model_root: str | Path | None,
        batch_size: int,
    ) -> None:
        # 延迟导入（模式说明见 gate.ModelScopeEmbedder 的注释）：torch 与
        # qwen_asr（本地封装包，运行环境已可导入）只在真正构造后端时才加载。
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            # 先按本地缓存解析模型路径，保证离线可复现。
            str(resolve_model_path(model, model_root)),
            dtype=torch.bfloat16,   # bf16 半精度：显存减半、速度更快，ASR 精度损失可忽略
            # Qwen3ASR 的 device_map 接口要具体设备名；用户填 "cuda" 时映射到 0 号卡。
            device_map="cuda:0" if device == "cuda" else device,
            # max_inference_batch_size 告诉模型单次最多吃多少条音频（内部据此
            # 预分配缓存）；至少为 1，防止 0 引发内部错误。
            max_inference_batch_size=max(1, batch_size),
            max_new_tokens=256,     # 解码长度上限（中文指令文本通常远小于 256 token）
        )

    def transcribe_many(self, audio_paths: list[Path]) -> list[str]:
        # suppress_model_output：模型加载/推理会向 stdout/stderr 打印大量
        # 日志，静默掉以免淹没我们自己的进度条与结果输出。
        with suppress_model_output():
            results = self.model.transcribe(
                audio=[str(path) for path in audio_paths],
                # Qwen3-ASR 支持语种提示；这里整批都是中文指令。
                language=["Chinese"] * len(audio_paths),
            )
        # 每个结果先抽出文本字段，再做统一归一化（全角/半角、标点等规则）。
        return [normalize_transcript(extract_asr_text(result)) for result in results]


class FireRedRecognizer:
    """FireRedASR2-AED 后端（zjm v2 实测 CER 比 Qwen3-ASR 好约 3.7pt）。

    AED = Attention Encoder-Decoder：端到端自回归解码，无 LLM 大解码器，
    速度与显存开销都比 LLM 版小；转写时输出带置信度的行。

    依赖：kaldi_native_fbank（装在 vendor 目录，不污染共享 conda）+
    FireRedASR2S 源码仓库（examples_infer/asr 下的 fireredasr2 包）。
    """

    def __init__(self, model_dir: str | Path, device: str, repo_root: str | Path,
                 vendor_root: str | Path = DEFAULT_FIRED_VENDOR) -> None:
        import sys

        # sys.path 注入：fireredasr2 包不在 pip 安装体系里（源码仓库 +
        # vendor 私有目录），import 前把这两个目录临时插到模块搜索路径
        # 最前面（insert(0) 保证优先级最高）。进程结束后环境自动复原，
        # 不影响其他项目/进程（比改 PYTHONPATH 环境变量更克制）。
        # vendor_root 装着 pip 安装到私有目录的依赖（如 kaldi_native_fbank），
        # 避免污染共享 conda 环境——多项目共用一台机器时的隔离手段。
        for path in (str(vendor_root), str(Path(repo_root) / "examples_infer" / "asr")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

        self._model = FireRedAsr2.from_pretrained(
            "aed",                            # 选 AED 版 checkpoint（而非 llm 版）
            str(Path(model_dir).resolve()),   # 绝对路径，避免工作目录变化导致找不到
            FireRedAsr2Config(
                # 下面一组是历史调参结果（zjm v2 实测的配置），勿动。
                use_gpu=device.startswith("cuda"),   # GPU 与否由 device 决定
                use_half=device.startswith("cuda"),  # GPU 上才用半精度
                beam_size=1,                  # 束宽 1 = 贪心解码（速度优先）
                nbest=1,                      # 只要 1-best 结果
                decode_max_len=0,             # 解码长度上限（0 = 默认行为）
                softmax_smoothing=1.25,       # 输出分布平滑系数（历史调参值）
                aed_length_penalty=0.6,       # AED 长度惩罚（历史调参值）
                eos_penalty=1.0,
                return_timestamp=False,       # 不需要时间戳，输出纯文本
                decode_min_len=0,
                repetition_penalty=1.0,       # 1.0 = 不惩罚重复
                llm_length_penalty=0.0,       # AED 版不用 LLM 相关参数
                temperature=1.0,              # 1.0 = 不缩放 logits
                elm_dir="",                   # 不使用 ELM 外部语言模型
                elm_weight=0.0,
            ),
        )

    def transcribe_many(self, audio_paths: list[Path]) -> list[str]:
        if not audio_paths:
            return []      # 空批次短路，避免底层对空输入行为不一致
        # FireRed 的 transcribe 接口要求每条音频配一个 utterance id
        # （转写结果按 id 回填），这里用下标生成唯一 id。
        utterance_ids = [f"utt_{index}" for index in range(len(audio_paths))]
        with suppress_model_output():
            # 底层返回"行"列表（dict，含 uttid/text/confidence 等字段），
            # 行序未必与输入一致，所以下面用 dict 重排。
            rows = self._model.transcribe(utterance_ids, [str(path) for path in audio_paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(audio_paths):
            # 数量对不上说明底层漏返回/重复返回，宁可报错也不要静默输出错位结果。
            raise RuntimeError("FireRed returned an unexpected result count")
        # 按输入顺序回填文本，并做统一归一化。
        return [normalize_transcript(str(mapped[uid].get("text", ""))) for uid in utterance_ids]


class FireRedLLMRecognizer:
    """FireRedASR2-LLM 后端（Conformer encoder + Adapter + Qwen2-7B 解码器）。

    与 AED 版相比：声学部分仍是 Conformer encoder，但解码器换成 Qwen2-7B
    大语言模型（通过 Adapter 把语音表示对齐到 LLM 的 embedding 空间），
    中文转写能力更强，代价是显存与速度。适合对准确率要求更高的场景。

    官方 LLM 推理不输出置信度；这里在 generate 上加 output_scores，
    以 beam 序列平均对数概率的指数（几何平均 token 概率，∈(0,1]）充当置信度，
    供音色触发器使用（触发阈值需在新置信度分布上重扫）。
    解码参数沿用官方 LLM 示例：beam_size=3, repetition_penalty=3.0,
    llm_length_penalty=1.0（=1 时 sequences_scores 即平均对数概率）。
    """

    def __init__(self, model_dir: str | Path, device: str, repo_root: str | Path,
                 vendor_root: str | Path = DEFAULT_FIRED_VENDOR,
                 batch_size: int = 4, beam_size: int = 3) -> None:
        import sys

        import torch

        # sys.path 注入 vendor 与源码仓库目录（原因见 FireRedRecognizer.__init__）。
        for path in (str(vendor_root), str(Path(repo_root) / "examples_infer" / "asr")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from fireredasr2.data.asr_feat import ASRFeatExtractor
        from fireredasr2.models.fireredasr_llm import FireRedAsrLlm
        from fireredasr2.tokenizer.llm_tokenizer import LlmTokenizerWrapper

        model_dir = Path(model_dir).resolve()
        # model.pth.tar 是 FireRed 官方打包格式：args（配置）+ 各组件权重。
        # weights_only=False：官方包内含普通 Python 对象，torch 新版的
        # weights_only=True 默认策略会拒绝；文件来自官方发布物（可信来源），
        # 关闭校验是可接受的（外部不可信文件切勿如此）。
        package = torch.load(model_dir / "model.pth.tar", map_location="cpu", weights_only=False)
        args = package["args"]
        # 官方包默认假设仓库内相对路径，这里逐项改写成本机绝对路径。
        args.encoder_path = str(model_dir / "asr_encoder.pth.tar")
        args.llm_dir = str(model_dir / "Qwen2-7B-Instruct")
        args.use_flash_attn = False  # yuyin 环境无 flash-attn，eager 即可
        args.use_fp16 = True         # 直接 bf16 加载（避免 fp32 30GB 峰值）
        model = FireRedAsrLlm.from_args(args)
        # strict=False：允许 checkpoint 与模型定义存在少量键名差异（官方包
        # 与代码版本漂移时的容错），能对上的权重全部加载。
        model.load_state_dict(package["model_state_dict"], strict=False)
        model.eval()
        del package   # 显式释放大 checkpoint 占用的内存（30GB 级，不能等 GC）
        if device.startswith("cuda"):
            # GPU 上按卡的能力选半精度：支持 bf16 用 bf16（数值更稳），
            # 否则退回 fp16；CPU 保持 fp32（CPU 半精度算子支持不全且无收益）。
            model = model.bfloat16() if torch.cuda.is_bf16_supported() else model.half()
            model = model.cuda()

        self._torch = torch      # 保存引用，后续方法直接用，避免重复 import
        self._llm = model
        self._model = self  # 与 FireRedRecognizer 对齐：管线走 asr._model.transcribe
        # Qwen2-7B 的 tokenizer 包装器与声学特征提取器（kaldi fbank + cmvn）。
        self._tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)
        self._feat_extractor = ASRFeatExtractor(str(model_dir / "cmvn.ark"))
        self._device = device
        self.batch_size = max(1, batch_size)  # 防 0/负数
        self.beam_size = max(1, beam_size)
        self.repetition_penalty = 3.0        # 强重复惩罚：LLM 解码器容易复读
        self.llm_length_penalty = 1.0        # =1 时 sequences_scores 即平均对数概率

    @property
    def _use_gpu(self) -> bool:
        """@property 把方法变成"只读属性"：self._use_gpu 不带括号调用。

        为什么要 property 而不是直接存一个 bool 字段：_device 是唯一事实
        来源（single source of truth），派生信息每次现算，就不会出现
        "device 改了但 use_gpu 没跟着改"的状态不同步 bug。下划线前缀
        表示"类内部使用，外部别碰"（Python 的约定式私有）。
        """
        return self._device.startswith("cuda")

    def _generate_with_conf(self, feats, lengths):
        """复刻 FireRedAsrLlm.transcribe，额外返回序列平均对数概率置信度。

        整体流程：声学特征 -> encoder -> projector 对齐到 LLM 空间 ->
        与占位文本 embedding 拼接 -> LLM.generate 输出 token 序列与分数。
        """
        torch = self._torch
        model = self._llm
        from fireredasr2.tokenizer.llm_tokenizer import LlmTokenizerWrapper

        # 占位文本：空字符串。语音 ASR 的文本输入只是"引子"（BOS），真正
        # 内容由语音特征注入；decode=True 表示按自回归解码方向准备。
        input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
            origin_texts=[""] * feats.size(0), tokenizer=self._tokenizer,
            max_len=128, decode=True)
        if self._use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        # encoder：Conformer 把 fbank 特征编码成高层声学表示。
        encoder_outs, enc_lengths, _ = model.encoder(feats, lengths)
        # projector（Adapter）：把声学表示投影/下采样到 LLM 的 embedding 空间。
        speech_features, speech_lens = model.encoder_projector(encoder_outs, enc_lengths)
        # 占位 token 换成 embedding 后，与语音 embedding 沿序列方向拼接
        # （注意力掩码同步调整），得到 LLM 的完整输入序列。
        inputs_embeds = model.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, _ = model._merge_input_ids_with_speech_features(
            speech_features.to(inputs_embeds.dtype), inputs_embeds, input_ids,
            attention_mask, speech_lens=speech_lens)
        # 生成上限对齐语音长度（ASR 输出一般不长于音频对应的 token 数），至少 1。
        max_new_tokens = max(1, speech_features.size(1))
        out = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=self.beam_size,     # 束搜索宽度
            do_sample=False,              # 关闭随机采样，用确定性的束搜索
            min_length=0,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.llm_length_penalty,
            bos_token_id=model.llm.config.bos_token_id,
            eos_token_id=model.llm.config.eos_token_id,
            pad_token_id=model.llm.config.pad_token_id,
            output_scores=True,           # 关键：让 generate 返回序列分数
            return_dict_in_generate=True, # 以对象形式返回，便于取 sequences_scores
        )
        # length_penalty=1.0 时 sequences_scores = 平均 token 对数概率（含重复惩罚扭曲）
        # 取指数 -> 几何平均 token 概率；clamp 保证落在 [0,1]。
        conf = torch.exp(out.sequences_scores.float()).clamp(0.0, 1.0)
        return out.sequences, conf

    def transcribe(self, batch_uttid: list[str], batch_wav_path: list[str]) -> list[dict[str, Any]]:
        """与 FireRedAsr2.transcribe 的 llm 分支同签名，返回带 confidence 的行。

        保持同签名是为了让上层 szy.helpers.transcribe_firered_rows 能用
        同一套代码访问本后端（上层只认 _model.transcribe(uttid, wav_path)）。
        """
        torch = self._torch
        rows: list[dict[str, Any]] = []
        with torch.no_grad():   # 纯推理，全程不建图
            for start in range(0, len(batch_wav_path), self.batch_size):
                # 内部分小批：batch_size 限制单次前向的显存峰值。
                paths = [str(p) for p in batch_wav_path[start:start + self.batch_size]]
                uttids = list(batch_uttid[start:start + self.batch_size])
                try:
                    # 特征提取（kaldi fbank + cmvn）。注意返回的 paths/uttids
                    # 会覆盖原变量——提取器会剔除提取失败的条目（官方实现口径）。
                    feats, lengths, _durs, paths, uttids = self._feat_extractor(paths, uttids)
                except Exception:
                    # 整批特征提取崩溃：给空文本 + 置信度 1.0 的占位行，
                    # 保证下游每一条输入都有对应输出行（失败也显式占位）。
                    rows += [{"uttid": u, "text": "", "confidence": 1.0} for u in uttids]
                    continue
                if feats is None:
                    rows += [{"uttid": u, "text": "", "confidence": 1.0} for u in uttids]
                    continue
                if self._use_gpu:
                    feats, lengths = feats.cuda(), lengths.cuda()
                    # 与加载时一致的半精度策略。
                    feats = feats.bfloat16() if torch.cuda.is_bf16_supported() else feats.half()
                try:
                    sequences, confs = self._generate_with_conf(feats, lengths)
                    texts = self._tokenizer.batch_decode(sequences, skip_special_tokens=True)
                except Exception:
                    # 解码失败的兜底：空文本 + 置信度 1.0（历史行为，上层
                    # 触发逻辑依赖"失败=空文本"这一约定）。
                    texts, confs = [""] * len(uttids), [1.0] * len(uttids)
                for uttid, wav, text, conf in zip(uttids, paths, texts, confs):
                    # 行格式与 FireRedAsr2 官方一致：uttid/text/confidence(+wav)，
                    # 保证上层按同一份字段代码读取。
                    row = {"uttid": uttid, "text": str(text).lower(),
                           "confidence": round(float(conf), 3)}
                    if isinstance(wav, str):
                        row["wav"] = wav
                    rows.append(row)
        return rows

    def transcribe_many(self, audio_paths: list[Path]) -> list[str]:
        """统一接口：与 FireRedRecognizer.transcribe_many 行为一致（鸭子类型）。"""
        if not audio_paths:
            return []
        utterance_ids = [f"utt_{index}" for index in range(len(audio_paths))]
        with suppress_model_output():
            # 走本类自己的 transcribe（带置信度的行），再抽取文本。
            rows = self.transcribe(utterance_ids, [str(path) for path in audio_paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(audio_paths):
            raise RuntimeError("FireRedLLM returned an unexpected result count")
        return [normalize_transcript(str(mapped[uid].get("text", ""))) for uid in utterance_ids]


class ConditionedQwenRecognizer:
    """sv_asr 双音频条件 LoRA 后端：输入 [kws(目标人参考), cmd] → 目标人文本。

    思路：给 Qwen3-ASR 同时喂两段音频——第一段是目标说话人的参考语音
    （kws/wake），第二段是待转写指令——让模型"学着"只输出第二段中目标
    说话人的内容，其余说话人一律忽略。转写由模型完成，拒识仍由门控负责。

    基座 Qwen3-ASR-0.6B + LoRA adapter（text decoder LoRA + audio projector
    全量，sv_asr/train_asr.py 训练，只学转写不学拒识，拒识仍由门控负责）。
    prompt / 解码 / max_new_tokens 与 train_asr.py 的 dev 评估完全一致。
    依赖：peft（vendor 目录）+ midea/src3 的 qwen_loader（只读复用）。
    """

    # 类属性标记：本后端是"条件后端"（需要 (wake, cmd) 成对输入）。
    # infer.main 用 getattr(recognizer, "conditioned", False) 探测它，
    # 决定走 transcribe_pairs 还是 transcribe_many——鸭子类型的轻量约定。
    conditioned = True

    def __init__(
        self,
        model: str,
        adapter: str | Path,
        device: str,
        model_root: str | Path | None,
        vendor_root: str | Path = "/home/hustdun/szy_workspace/vendor",
        src3_root: str | Path = "/home/hustdun/szy_workspace/midea/src3",
    ) -> None:
        import sys

        import torch

        # vendor 目录装 peft 等私有依赖；src3 是另一份代码库，本内核只
        # 只读复用其中的 qwen_loader（不改它、不拷它，避免两处维护）。
        for path in (str(vendor_root), str(src3_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from peft import PeftModel
        from qwen_loader import load_base_model, load_processor

        model_path = resolve_model_path(model, model_root)
        self.processor = load_processor(model_path)   # 基座模型配套的音频/文本处理器
        # device 参数统一转成具体设备名（"cuda" -> "cuda:0"）。
        self.device = "cuda:0" if device == "cuda" else device
        # GPU 用 bf16（省显存、速度快）；CPU 上 bf16 算子支持不全，回退 fp32。
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._torch = torch
        # device_map 以 dict 形式出现时表示"整个模型都放这个设备"；
        # 与 "auto"（自动切分多卡）相对。
        self.model = load_base_model(
            model_path, dtype=self.dtype, device_map={"": self.device}
        )
        # 把 LoRA adapter 挂到基座的 thinker（文本解码器）上：PeftModel
        # 只装载 adapter 增量权重，基座权重原地复用——所以 adapter 目录
        # 很小（几十 MB 级），可以为一个基座保存多个 adapter 快速切换。
        self.model.thinker = PeftModel.from_pretrained(
            self.model.thinker, str(Path(adapter).resolve())
        )
        self.model.eval()
        # 开启 KV cache：自回归解码时复用已算过的 key/value，大幅加速。
        self.model.thinker.config.use_cache = True
        self.max_new_tokens = 128

    def _build_prompt(self) -> str:
        """与 sv_asr/train_asr.py build_prompt 逐字一致。

        逐字一致是关键：LoRA 是在这套 prompt 格式下训练的，推理时 prompt
        变了，模型看到的输入分布就变了，效果会掉。两个 audio 占位符对应
        后面 processor(audio=[enroll, command]) 传入的两段音频。
        """
        messages = [
            # 系统提示词：规定任务语义（第一段=目标人参考，第二段=只转写目标人）。
            {
                "role": "system",
                "content": (
                    "你是目标说话人语音识别器。第一段音频是目标说话人的参考语音。"
                    "第二段音频中可能有多个人同时说话和背景噪声，"
                    "请只转写第二段音频中目标说话人"
                    "（与第一段参考语音是同一人）所说的内容，"
                    "忽略其他说话人。不要输出解释。"
                ),
            },
            # 用户消息：两个音频占位符（内容为空串，真正的音频由 processor
            # 在 audio= 参数里注入；Qwen3-ASR 的 chat template 会按此结构
            # 生成对应的 <audio> 标记）。
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": ""},
                    {"type": "audio", "audio": ""},
                ],
            },
        ]
        # apply_chat_template：把 messages 渲染成模型训练时用的字符串格式，
        # 并追加"生成提示"（告诉模型接下来该它说话了）。
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # 尾部后缀是 Qwen3-ASR 的输出格式约定：先输出语种标签，再输出
        # <asr_text> 包裹的转写文本（与训练时 dev 评估完全一致）。
        return prompt + "language Chinese<asr_text>"

    def transcribe_pairs(self, pairs: list[tuple[Path, Path]]) -> list[str]:
        """条件转写：pairs = [(wake, cmd), ...]，逐对输出目标人文本。

        逐对（而非批量）推理：两段音频要拼进同一个输入序列，长度组合
        各异，批量打包收益低且实现复杂，逐条处理更清晰可控。
        """
        torch = self._torch
        outputs: list[str] = []
        for wake_path, command_path in pairs:
            # 统一转成单声道 16kHz 波形（模型前置约定），两段分别是
            # 参考音频与待转写指令音频。
            enroll_audio = load_audio_mono_16k(wake_path)
            command_audio = load_audio_mono_16k(command_path)
            # processor 一次完成：prompt 分词 + 音频特征提取 + 组装输入。
            # return_tensors="pt" 让输出都是 torch 张量；padding=True 对齐
            # 序列长度。返回的是 BatchFeature（dict 的加强版），dict() 转普通 dict。
            inputs = dict(
                self.processor(
                    text=[self._build_prompt()],
                    audio=[enroll_audio, command_audio],
                    return_tensors="pt",
                    padding=True,
                )
            )
            # 逐字段搬设备/精度：浮点张量（音频特征等）要带 dtype 转 bf16；
            # 整型张量（input_ids/attention_mask）只搬设备不能转 dtype；
            # 非张量（list/str 等）原样保留。三层条件表达式用
            # hasattr(value, "to") 判断是否为张量。
            inputs = {
                key: (
                    value.to(device=self.device, dtype=self.dtype)
                    if hasattr(value, "is_floating_point") and value.is_floating_point()
                    else value.to(device=self.device)
                    if hasattr(value, "to")
                    else value
                )
                for key, value in inputs.items()
            }
            with torch.inference_mode():   # 推理上下文（比 no_grad 更快）
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )
            # 输出序列 = 输入 prompt + 新生成 token；切掉 prompt 部分
            # （inputs["input_ids"].shape[1] 即 prompt 长度），只留生成文本。
            sequence = generated.sequences[0, inputs["input_ids"].shape[1] :]
            text = self.processor.decode(
                sequence,
                skip_special_tokens=True,
                # 保留原始分词空格（与训练时 dev 评估的解码口径一致）。
                clean_up_tokenization_spaces=False,
            ).strip()
            outputs.append(normalize_transcript(text))
        return outputs
