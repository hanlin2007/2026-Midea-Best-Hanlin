"""midea_qiji.recognizers：全部 ASR 识别后端（统一 transcribe 接口）。

四个后端按"转写口径 + 置信度语义"分工：

  - ``QwenRecognizer``           Qwen3-ASR-0.6B（baseline 默认后端，无置信度）；
  - ``FireRedRecognizer``        FireRedASR2-AED（SZY 系列主后端，带置信度）；
  - ``FireRedLLMRecognizer``     FireRedASR2-LLM（Qwen2-7B 解码，序列平均对数
                                 概率充当置信度，供音色触发器使用）；
  - ``ConditionedQwenRecognizer`` Qwen3-ASR + sv_asr 双音频条件 LoRA
                                 （[kws, cmd] → 目标说话人文本，实验后端）。

公共约定：``transcribe_many(paths) -> list[str]`` 返回归一化文本；管线内部
需要置信度时统一走 szy_common.helpers.transcribe_firered_rows（访问
``_model.transcribe`` 的原始行）。重型依赖（torch / peft / FireRed 包 /
modelscope）一律延迟导入，保证本模块可被轻量环境 import。

依赖：midea_qiji.text / .audio / .paths / .infer（suppress_model_output）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

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
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            str(resolve_model_path(model, model_root)),
            dtype=torch.bfloat16,
            device_map="cuda:0" if device == "cuda" else device,
            max_inference_batch_size=max(1, batch_size),
            max_new_tokens=256,
        )

    def transcribe_many(self, audio_paths: list[Path]) -> list[str]:
        with suppress_model_output():
            results = self.model.transcribe(
                audio=[str(path) for path in audio_paths],
                language=["Chinese"] * len(audio_paths),
            )
        return [normalize_transcript(extract_asr_text(result)) for result in results]


class FireRedRecognizer:
    """FireRedASR2-AED 后端（zjm v2 实测 CER 比 Qwen3-ASR 好约 3.7pt）。

    依赖：kaldi_native_fbank（装在 vendor 目录，不污染共享 conda）+
    FireRedASR2S 源码仓库（examples_infer/asr 下的 fireredasr2 包）。
    """

    def __init__(self, model_dir: str | Path, device: str, repo_root: str | Path,
                 vendor_root: str | Path = DEFAULT_FIRED_VENDOR) -> None:
        import sys

        for path in (str(vendor_root), str(Path(repo_root) / "examples_infer" / "asr")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

        self._model = FireRedAsr2.from_pretrained(
            "aed",
            str(Path(model_dir).resolve()),
            FireRedAsr2Config(
                use_gpu=device.startswith("cuda"),
                use_half=device.startswith("cuda"),
                beam_size=1,
                nbest=1,
                decode_max_len=0,
                softmax_smoothing=1.25,
                aed_length_penalty=0.6,
                eos_penalty=1.0,
                return_timestamp=False,
                decode_min_len=0,
                repetition_penalty=1.0,
                llm_length_penalty=0.0,
                temperature=1.0,
                elm_dir="",
                elm_weight=0.0,
            ),
        )

    def transcribe_many(self, audio_paths: list[Path]) -> list[str]:
        if not audio_paths:
            return []
        utterance_ids = [f"utt_{index}" for index in range(len(audio_paths))]
        with suppress_model_output():
            rows = self._model.transcribe(utterance_ids, [str(path) for path in audio_paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(audio_paths):
            raise RuntimeError("FireRed returned an unexpected result count")
        return [normalize_transcript(str(mapped[uid].get("text", ""))) for uid in utterance_ids]


class FireRedLLMRecognizer:
    """FireRedASR2-LLM 后端（Conformer encoder + Adapter + Qwen2-7B 解码器）。

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

        for path in (str(vendor_root), str(Path(repo_root) / "examples_infer" / "asr")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from fireredasr2.data.asr_feat import ASRFeatExtractor
        from fireredasr2.models.fireredasr_llm import FireRedAsrLlm
        from fireredasr2.tokenizer.llm_tokenizer import LlmTokenizerWrapper

        model_dir = Path(model_dir).resolve()
        package = torch.load(model_dir / "model.pth.tar", map_location="cpu", weights_only=False)
        args = package["args"]
        args.encoder_path = str(model_dir / "asr_encoder.pth.tar")
        args.llm_dir = str(model_dir / "Qwen2-7B-Instruct")
        args.use_flash_attn = False  # yuyin 环境无 flash-attn，eager 即可
        args.use_fp16 = True         # 直接 bf16 加载（避免 fp32 30GB 峰值）
        model = FireRedAsrLlm.from_args(args)
        model.load_state_dict(package["model_state_dict"], strict=False)
        model.eval()
        del package
        if device.startswith("cuda"):
            model = model.bfloat16() if torch.cuda.is_bf16_supported() else model.half()
            model = model.cuda()

        self._torch = torch
        self._llm = model
        self._model = self  # 与 FireRedRecognizer 对齐：管线走 asr._model.transcribe
        self._tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)
        self._feat_extractor = ASRFeatExtractor(str(model_dir / "cmvn.ark"))
        self._device = device
        self.batch_size = max(1, batch_size)
        self.beam_size = max(1, beam_size)
        self.repetition_penalty = 3.0
        self.llm_length_penalty = 1.0

    @property
    def _use_gpu(self) -> bool:
        return self._device.startswith("cuda")

    def _generate_with_conf(self, feats, lengths):
        """复刻 FireRedAsrLlm.transcribe，额外返回序列平均对数概率置信度。"""
        torch = self._torch
        model = self._llm
        from fireredasr2.tokenizer.llm_tokenizer import LlmTokenizerWrapper

        input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
            origin_texts=[""] * feats.size(0), tokenizer=self._tokenizer,
            max_len=128, decode=True)
        if self._use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        encoder_outs, enc_lengths, _ = model.encoder(feats, lengths)
        speech_features, speech_lens = model.encoder_projector(encoder_outs, enc_lengths)
        inputs_embeds = model.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, _ = model._merge_input_ids_with_speech_features(
            speech_features.to(inputs_embeds.dtype), inputs_embeds, input_ids,
            attention_mask, speech_lens=speech_lens)
        max_new_tokens = max(1, speech_features.size(1))
        out = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=self.beam_size,
            do_sample=False,
            min_length=0,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.llm_length_penalty,
            bos_token_id=model.llm.config.bos_token_id,
            eos_token_id=model.llm.config.eos_token_id,
            pad_token_id=model.llm.config.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # length_penalty=1.0 时 sequences_scores = 平均 token 对数概率（含重复惩罚扭曲）
        conf = torch.exp(out.sequences_scores.float()).clamp(0.0, 1.0)
        return out.sequences, conf

    def transcribe(self, batch_uttid: list[str], batch_wav_path: list[str]) -> list[dict[str, Any]]:
        """与 FireRedAsr2.transcribe 的 llm 分支同签名，返回带 confidence 的行。"""
        torch = self._torch
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for start in range(0, len(batch_wav_path), self.batch_size):
                paths = [str(p) for p in batch_wav_path[start:start + self.batch_size]]
                uttids = list(batch_uttid[start:start + self.batch_size])
                try:
                    feats, lengths, _durs, paths, uttids = self._feat_extractor(paths, uttids)
                except Exception:
                    rows += [{"uttid": u, "text": "", "confidence": 1.0} for u in uttids]
                    continue
                if feats is None:
                    rows += [{"uttid": u, "text": "", "confidence": 1.0} for u in uttids]
                    continue
                if self._use_gpu:
                    feats, lengths = feats.cuda(), lengths.cuda()
                    feats = feats.bfloat16() if torch.cuda.is_bf16_supported() else feats.half()
                try:
                    sequences, confs = self._generate_with_conf(feats, lengths)
                    texts = self._tokenizer.batch_decode(sequences, skip_special_tokens=True)
                except Exception:
                    texts, confs = [""] * len(uttids), [1.0] * len(uttids)
                for uttid, wav, text, conf in zip(uttids, paths, texts, confs):
                    row = {"uttid": uttid, "text": str(text).lower(),
                           "confidence": round(float(conf), 3)}
                    if isinstance(wav, str):
                        row["wav"] = wav
                    rows.append(row)
        return rows

    def transcribe_many(self, audio_paths: list[Path]) -> list[str]:
        if not audio_paths:
            return []
        utterance_ids = [f"utt_{index}" for index in range(len(audio_paths))]
        with suppress_model_output():
            rows = self.transcribe(utterance_ids, [str(path) for path in audio_paths])
        mapped = {str(row.get("uttid")): row for row in rows}
        if len(mapped) != len(audio_paths):
            raise RuntimeError("FireRedLLM returned an unexpected result count")
        return [normalize_transcript(str(mapped[uid].get("text", ""))) for uid in utterance_ids]


class ConditionedQwenRecognizer:
    """sv_asr 双音频条件 LoRA 后端：输入 [kws(目标人参考), cmd] → 目标人文本。

    基座 Qwen3-ASR-0.6B + LoRA adapter（text decoder LoRA + audio projector
    全量，sv_asr/train_asr.py 训练，只学转写不学拒识，拒识仍由门控负责）。
    prompt / 解码 / max_new_tokens 与 train_asr.py 的 dev 评估完全一致。
    依赖：peft（vendor 目录）+ midea/src3 的 qwen_loader（只读复用）。
    """

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

        for path in (str(vendor_root), str(src3_root)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from peft import PeftModel
        from qwen_loader import load_base_model, load_processor

        model_path = resolve_model_path(model, model_root)
        self.processor = load_processor(model_path)
        self.device = "cuda:0" if device == "cuda" else device
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._torch = torch
        self.model = load_base_model(
            model_path, dtype=self.dtype, device_map={"": self.device}
        )
        self.model.thinker = PeftModel.from_pretrained(
            self.model.thinker, str(Path(adapter).resolve())
        )
        self.model.eval()
        self.model.thinker.config.use_cache = True
        self.max_new_tokens = 128

    def _build_prompt(self) -> str:
        """与 sv_asr/train_asr.py build_prompt 逐字一致。"""
        messages = [
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
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": ""},
                    {"type": "audio", "audio": ""},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return prompt + "language Chinese<asr_text>"

    def transcribe_pairs(self, pairs: list[tuple[Path, Path]]) -> list[str]:
        torch = self._torch
        outputs: list[str] = []
        for wake_path, command_path in pairs:
            enroll_audio = load_audio_mono_16k(wake_path)
            command_audio = load_audio_mono_16k(command_path)
            inputs = dict(
                self.processor(
                    text=[self._build_prompt()],
                    audio=[enroll_audio, command_audio],
                    return_tensors="pt",
                    padding=True,
                )
            )
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
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )
            sequence = generated.sequences[0, inputs["input_ids"].shape[1] :]
            text = self.processor.decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            outputs.append(normalize_transcript(text))
        return outputs
