"""core.firered：FireRedASR2 识别后端（AED 生产 / LLM 可选）。

两个后端按"转写口径 + 置信度语义"分工：

  - ``FireRedRecognizer``        FireRedASR2-AED（生产主后端，带置信度，
                                 支持 beam/N-best/ELM 与共享实例临时切换）；
  - ``FireRedLLMRecognizer``     FireRedASR2-LLM（Qwen2-7B 解码，序列平均对数
                                 概率充当置信度；上限更高但更慢，备选）。

公共约定：``transcribe_many(paths) -> list[str]`` 返回归一化文本；管线内部
需要置信度与 N-best 时统一走 ``_model.transcribe`` 的原始行（见
pipeline.engine.PipelineRunner._transcribe_with_asr）。重型依赖
（torch / fireredasr2 包）一律延迟导入，保证本模块可被轻量环境 import。

设计说明（面向初学者）：
- 两个后端没有共同的抽象基类，而是靠"鸭子类型"统一接口：只要对象提供
  transcribe_many(paths) 与 _model.transcribe，引擎就能无差别调度。这是
  Python 社区常见的"协议式多态"——不强制继承，只约定行为。
  FireRedLLMRecognizer 更是把自己的 transcribe() 暴露成
  ``_model.transcribe``，让上层能用与 FireRedRecognizer 完全相同的代码
  访问（见该类的 self._model = self）。
- 本模块在内核层，绝不 import 上层 pipeline/ 模块——依赖方向只能自下
  而上，这是分层架构的纪律。

功能：把音频路径列表转成归一化中文文本列表（另按需提供置信度与 N-best）。

关键导出：FireRedRecognizer / FireRedLLMRecognizer。

依赖：core.text / core.model_paths / core.samples（suppress_model_output）。
重型依赖（torch / fireredasr2）全部在构造函数内延迟导入。

用法（通常由 pipeline.engine 按配置装配，不必手写）：
    asr = FireRedRecognizer(model_dir, "cuda:0", repo_root, beam_size=3, nbest=3)
    texts = asr.transcribe_many([Path("a.wav"), Path("b.wav")])
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from .model_paths import DEFAULT_FIRED_VENDOR
from .samples import suppress_model_output
from .text import normalize_transcript


class FireRedRecognizer:
    """FireRedASR2-AED 后端（zjm v2 实测 CER 比 Qwen3-ASR 好约 3.7pt）。

    AED = Attention Encoder-Decoder：端到端自回归解码，无 LLM 大解码器，
    速度与显存开销都比 LLM 版小；转写时输出带置信度的行。

    生产上唤醒与命令通道**共用同一份 AED 权重**：唤醒侧需要稳定的
    beam=1 解码、命令侧需要 beam=3 + ELM，过去为此驻留两份模型
    （显存 ~7.4 GB），现在用 ``temporary_beam_size`` /
    ``temporary_elm_weight`` 两个上下文管理器在共享实例上临时改解码
    配置（峰值显存降到 4818 MiB），还顺带消除了两份模型同时在显存里
    造成的数值/内核选择抖动。

    依赖：kaldi_native_fbank（装在 vendor 目录，不污染共享 conda）+
    FireRedASR2S 源码仓库（examples_infer/asr 下的 fireredasr2 包）。
    """

    def __init__(self, model_dir: str | Path, device: str, repo_root: str | Path,
                 vendor_root: str | Path = DEFAULT_FIRED_VENDOR,
                 beam_size: int = 1, nbest: int = 1,
                 elm_dir: str | Path | None = None, elm_weight: float = 0.0) -> None:
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

        self._beam_size = beam_size
        self._nbest = nbest
        # elm_dir 在构造时一次性消费（ELM 权重常驻）；之后靠 elm_weight
        # 开关（见 temporary_elm_weight），不需要为"用/不用 ELM"准备两份模型
        resolved_elm_dir = "" if elm_dir is None else str(Path(elm_dir).resolve())
        self._model = FireRedAsr2.from_pretrained(
            "aed",                            # 选 AED 版 checkpoint（而非 llm 版）
            str(Path(model_dir).resolve()),   # 绝对路径，避免工作目录变化导致找不到
            FireRedAsr2Config(
                # 下面一组是历史调参结果（zjm v2 实测的配置），勿动。
                use_gpu=device.startswith("cuda"),   # GPU 与否由 device 决定
                use_half=device.startswith("cuda"),  # GPU 上才用半精度
                beam_size=beam_size,          # 束宽（1 = 贪心；生产命令通道用 3）
                nbest=nbest,                  # N-best 列表长度（<= beam_size）
                decode_max_len=0,             # 解码长度上限（0 = 默认行为）
                softmax_smoothing=1.25,       # 输出分布平滑系数（历史调参值）
                aed_length_penalty=0.6,       # AED 长度惩罚（历史调参值）
                eos_penalty=1.0,
                return_timestamp=False,       # 不需要时间戳，输出纯文本
                decode_min_len=0,
                repetition_penalty=1.0,       # 1.0 = 不惩罚重复
                llm_length_penalty=0.0,       # AED 版不用 LLM 相关参数
                temperature=1.0,              # 1.0 = 不缩放 logits
                elm_dir=resolved_elm_dir,     # ELM 外部语言模型目录（"" = 不加载）
                elm_weight=elm_weight,        # ELM 融合权重（0.0 = 解码时不参与）
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

    @contextlib.contextmanager
    def temporary_beam_size(self, beam_size: int):
        """Temporarily change the AED decoder beam for one sequential call.

        ``nbest`` is clamped to the new beam because FireRedAsr2 requires
        ``nbest <= beam_size``; both fields are restored on exit.

        （中文）上下文管理器（@contextlib.contextmanager 装饰器把"生成器
        函数"变成 with 语句可用的上下文）：with 块内把解码束宽临时改成
        指定值，退出时（无论正常结束还是异常）恢复原值。生产用法：共享
        AED 实例上，唤醒通道 ``temporary_beam_size(1)`` 求稳、命令通道
        用构造时的 beam=3。yield 之前的代码在进 with 时执行，之后的在
        退出时执行——try/finally 保证异常也一定能还原，不会把"临时"
        配置泄漏到后续解码。
        """
        if beam_size < 1:
            raise ValueError("FireRed beam size must be positive")
        previous_beam = self._model.config.beam_size
        previous_nbest = self._model.config.nbest
        self._model.config.beam_size = beam_size
        self._model.config.nbest = min(previous_nbest, beam_size)
        try:
            yield
        finally:
            self._model.config.beam_size = previous_beam
            self._model.config.nbest = previous_nbest

    @contextlib.contextmanager
    def temporary_elm_weight(self, elm_weight: float):
        """Temporarily enable/disable ELM on a shared AED instance.

        ``elm_dir`` is consumed once at construction (the ELM is already
        resident); FireRedAsr2 reads ``config.elm_weight`` at decode time, so
        toggling the weight is enough to share one AED across ELM/non-ELM
        decodes instead of loading a second copy of the weights.

        （中文）ELM 权重在构造时已随模型常驻显存，FireRedAsr2 解码时读取
        的是 config.elm_weight——所以"开/关 ELM"只需临时改这个标量，
        不必再加载一份模型。生产用法：所有参与门控/路由决策的转写一律
        ``temporary_elm_weight(0.0)``（保持 confidence 量程干净），只有
        路由选定后的最终通道才用配置的权重重解码一次。
        """
        previous = self._model.config.elm_weight
        self._model.config.elm_weight = float(elm_weight)
        try:
            yield
        finally:
            self._model.config.elm_weight = previous

    def transcribe_many_with_confidence(self, audio_paths: list[Path]) -> list[dict[str, Any]]:
        """Transcribe audio and return both text and confidence scores.

        Args:
            audio_paths: List of audio file paths

        Returns:
            List of {"text": str, "confidence": float}
        """
        if not audio_paths:
            return []
        utterance_ids = [f"utt_{index}" for index in range(len(audio_paths))]
        with suppress_model_output():
            rows = self._model.transcribe(utterance_ids, [str(path) for path in audio_paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(audio_paths):
            raise RuntimeError("FireRed returned an unexpected result count")

        results = []
        for uid in utterance_ids:
            row = mapped[uid]
            results.append({
                "text": normalize_transcript(str(row.get("text", ""))),
                "confidence": float(row.get("confidence", 0.0)),
            })
        return results

    def transcribe_many_nbest(self, audio_paths: list[Path]) -> list[list[dict[str, Any]]]:
        """Transcribe audio and return N-best hypotheses for each audio.

        Requires nbest > 1 in __init__. Each audio returns a list of hypotheses
        sorted by confidence (highest first).

        （中文）N-best 列表是 beam 搜索的副产品：beam=3 解码时模型本来就
        维护了多条候选，顺手取出（不再多花一次推理），供保守 MBR 重排序
        使用（见 core.firered_nbest）。构造时 nbest<2 则退化为单候选。

        Args:
            audio_paths: List of audio file paths

        Returns:
            List of hypothesis lists, where each hypothesis is {"text": str, "confidence": float}
        """
        if not audio_paths:
            return []
        if self._nbest < 2:
            # Fallback to single hypothesis if nbest not configured
            single_results = self.transcribe_many_with_confidence(audio_paths)
            return [[result] for result in single_results]

        utterance_ids = [f"utt_{index}" for index in range(len(audio_paths))]
        with suppress_model_output():
            rows = self._model.transcribe(utterance_ids, [str(path) for path in audio_paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(audio_paths):
            raise RuntimeError("FireRed returned an unexpected result count")

        all_nbest = []
        for uid in utterance_ids:
            row = mapped[uid]
            # Extract nbest list from row
            nbest_list = row.get("nbest", [])
            if not nbest_list:
                # Fallback: create single hypothesis from main result
                nbest_list = [{
                    "text": row.get("text", ""),
                    "confidence": row.get("confidence", 0.0),
                }]

            hypotheses = []
            for hyp in nbest_list:
                hypotheses.append({
                    "text": normalize_transcript(str(hyp.get("text", ""))),
                    "confidence": float(hyp.get("confidence", 0.0)),
                })
            all_nbest.append(hypotheses)
        return all_nbest


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

        保持同签名是为了让上层 pipeline.helpers.transcribe_firered_rows 能用
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
