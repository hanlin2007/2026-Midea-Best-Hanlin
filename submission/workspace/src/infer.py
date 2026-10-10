from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODEL_ROOT = PACKAGE_ROOT / "model"
DEFAULT_FIRED_MODEL = str(PACKAGE_MODEL_ROOT / "FireRedASR2-AED")
DEFAULT_FIRED_REPO = str(PACKAGE_ROOT / "third_party" / "FireRedASR2S")
DEFAULT_FIRED_VENDOR = PACKAGE_ROOT / "vendor"


@dataclass(frozen=True)
class Config:
    device: str = "cuda:0"
    sv_device: str | None = None
    asr_model: str = "Qwen/Qwen3-ASR-0.6B"
    sv_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    sv_model_revision: str | None = "v1.0.0"
    sv_accept_threshold: float = 0.27
    id_mode: str = "relative_stem"
    # 门控模式："cos" = ERes2NetV2 余弦阈值（baseline）；"dual_mlp" = 双前端 MLP 裁判
    gate_mode: str = "cos"
    cam_model: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    dual_backend: str | None = None
    dual_threshold: float = 0.5334
    # ASR 后端："qwen" = Qwen3-ASR-0.6B；"firered" = FireRedASR2-AED；
    # "qwen_lora" = sv_asr 双音频条件 LoRA 微调权重（[kws, cmd] → 目标人文本）
    asr_backend: str = "qwen"
    firered_model: str = DEFAULT_FIRED_MODEL
    firered_repo: str = DEFAULT_FIRED_REPO
    qwen_lora_adapter: str | None = None
    # Optional cache-driven second-stage target-presence gate.  It is disabled
    # for the production baseline until an external model is trained and
    # calibrated against the required PR/RR constraints.
    second_stage_enabled: bool = False
    second_stage_model: str | None = None
    second_stage_features: str | None = None
    second_stage_presence_threshold: float | None = None


@dataclass(frozen=True)
class Sample:
    raw_id: str
    wake_audio: Path
    command_audio: Path
    command_relative_path: str
    label: str | None
    wake_text: str | None = None


def load_config(path: str | Path) -> Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = Config.__dataclass_fields__
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return Config(**payload)


def resolve_model_path(model: str | Path, model_root: str | Path | None) -> Path:
    direct = Path(model)
    if direct.exists():
        return direct.resolve()
    roots = []
    if model_root is not None:
        roots.append(Path(model_root))
    if os.environ.get("MODELSCOPE_CACHE"):
        roots.append(Path(os.environ["MODELSCOPE_CACHE"]))
    roots.extend([Path("models/modelscope"), Path.home() / ".cache" / "modelscope" / "hub"])
    for root in roots:
        for candidate in (
            root / str(model),
            root / str(model).replace(".", "___"),
            root / "models" / str(model),
            root / "models" / str(model).replace(".", "___"),
        ):
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(
        f"local model {model!r} was not found; set --model-root or MODELSCOPE_CACHE"
    )


class SpeakerVerifier:
    def __init__(self, model: str, device: str, model_root: str | Path | None) -> None:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self.pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=str(resolve_model_path(model, model_root)),
            device=device,
        )

    def score(self, enrollment_audio: Path, command_audio: Path) -> float:
        return extract_speaker_score(self.pipeline([str(enrollment_audio), str(command_audio)]))


class QwenRecognizer:
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
                 vendor_root: str | Path = DEFAULT_FIRED_VENDOR,
                 beam_size: int = 1, nbest: int = 1,
                 elm_dir: str | Path | None = None, elm_weight: float = 0.0) -> None:
        import sys

        for path in (str(vendor_root), str(Path(repo_root) / "examples_infer" / "asr")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config

        self._beam_size = beam_size
        self._nbest = nbest
        resolved_elm_dir = "" if elm_dir is None else str(Path(elm_dir).resolve())
        self._model = FireRedAsr2.from_pretrained(
            "aed",
            str(Path(model_dir).resolve()),
            FireRedAsr2Config(
                use_gpu=device.startswith("cuda"),
                use_half=device.startswith("cuda"),
                beam_size=beam_size,
                nbest=nbest,
                decode_max_len=0,
                softmax_smoothing=1.25,
                aed_length_penalty=0.6,
                eos_penalty=1.0,
                return_timestamp=False,
                decode_min_len=0,
                repetition_penalty=1.0,
                llm_length_penalty=0.0,
                temperature=1.0,
                elm_dir=resolved_elm_dir,
                elm_weight=elm_weight,
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

    @contextlib.contextmanager
    def temporary_beam_size(self, beam_size: int):
        """Temporarily change the AED decoder beam for one sequential call.

        ``nbest`` is clamped to the new beam because FireRedAsr2 requires
        ``nbest <= beam_size``; both fields are restored on exit.
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


def load_audio_mono_16k(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """与 sv_asr/train_asr.py load_audio 一致：单声道 float32，必要时重采样。"""
    import soundfile as sf

    waveform, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = np.asarray(waveform, dtype=np.float32)
    if source_rate != sample_rate:
        import torch
        import torchaudio

        tensor = torch.from_numpy(waveform)
        waveform = torchaudio.functional.resample(tensor, source_rate, sample_rate).numpy()
    return waveform.astype(np.float32, copy=False)


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
        vendor_root: str | Path = PACKAGE_ROOT / "vendor",
        src3_root: str | Path = PACKAGE_ROOT / "src3",
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


def read_many_jsonl(paths: Iterable[str | Path], data_root: str | Path) -> list[Sample]:
    root = Path(data_root)
    samples: list[Sample] = []
    for jsonl_path in paths:
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    wake_relative_path = str(row["唤醒音频"])
                    command_relative_path = str(row["识别音频"])
                except KeyError as error:
                    raise KeyError(f"{jsonl_path}:{line_number} missing field {error}") from error
                samples.append(
                    Sample(
                        raw_id=str(row.get("id", "")),
                        wake_audio=root / wake_relative_path,
                        command_audio=root / command_relative_path,
                        command_relative_path=command_relative_path,
                        label=row.get("识别文本"),
                        wake_text=row.get("唤醒文本") or row.get("wake_text"),
                    )
                )
    return samples


def submission_id(sample: Sample, mode: str) -> str:
    if mode == "raw":
        return sample.raw_id
    if mode == "name":
        return sample.command_audio.name
    if mode == "path":
        return sample.command_audio.as_posix()
    if mode == "relative":
        return sample.command_relative_path
    if mode == "relative_stem":
        path = Path(sample.command_relative_path)
        return (path.parent / path.stem).as_posix()
    if mode == "stem":
        return sample.command_audio.stem
    raise ValueError(f"unsupported id mode: {mode}")


def extract_speaker_score(result: Any) -> float:
    if isinstance(result, dict):
        for key in ("score", "scores", "similarity", "output", "outputs"):
            if key in result:
                try:
                    return extract_speaker_score(result[key])
                except ValueError:
                    continue
        raise ValueError(f"speaker verification output has no score: {list(result)}")
    if isinstance(result, (list, tuple)):
        for value in result:
            try:
                return extract_speaker_score(value)
            except ValueError:
                continue
        raise ValueError("speaker verification output was empty")
    if isinstance(result, np.ndarray):
        if result.size == 0:
            raise ValueError("speaker verification score array was empty")
        return float(result.reshape(-1)[0])
    try:
        return float(result)
    except (TypeError, ValueError) as error:
        raise ValueError(f"cannot parse speaker score from {result!r}") from error


def extract_asr_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "sentence", "raw_text"):
            if key in result:
                return str(result[key])
        for value in result.values():
            try:
                return extract_asr_text(value)
            except ValueError:
                continue
    if isinstance(result, list):
        parts = []
        for value in result:
            try:
                parts.append(extract_asr_text(value))
            except ValueError:
                continue
        if parts:
            return "".join(parts)
    raise ValueError(f"cannot extract ASR text from {result!r}")


def normalize_transcript(text: str) -> str:
    digits = "零一二三四五六七八九"

    def number_to_chinese(value: int) -> str:
        if value < 10:
            return digits[value]
        if value < 20:
            return "十" + (digits[value % 10] if value % 10 else "")
        if value < 100:
            return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
        if value < 1000:
            remainder = value % 100
            prefix = digits[value // 100] + "百"
            if remainder == 0:
                return prefix
            if remainder < 10:
                return prefix + "零" + digits[remainder]
            return prefix + number_to_chinese(remainder)
        return str(value)

    def convert_number(match: re.Match[str]) -> str:
        return number_to_chinese(int(match.group(0)))

    normalized = re.sub(r"<\|[^|]+?\|>", "", str(text))
    normalized = re.sub(r"(\d+)\s*%", r"百分之\1", normalized)
    normalized = re.sub(r"百分之\s*(\d+)", lambda match: "百分之" + number_to_chinese(int(match.group(1))), normalized)
    normalized = re.sub(r"\d+", convert_number, normalized)
    return re.sub(r"[\s,，。.!！?？:：;；、】【、\"'“”‘’（）()\[\]<>《》|/\\_-]+", "", normalized.lower())


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_character != right_character)))
        previous = current
    return previous[-1]


def cer(hypothesis: str, label: str | None) -> float | None:
    if label is None:
        return None
    return levenshtein(hypothesis, label) / len(label) if label else float(hypothesis != "")


@contextlib.contextmanager
def suppress_model_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def write_reports(results: list[dict[str, Any]], duration: float, output: Path, final_output: Path) -> dict[str, float]:
    total_errors = 0
    total_characters = 0
    positive_nonempty = 0
    negative_total = 0
    negative_rejects = 0
    submission_results = []
    final_results = []
    for item in results:
        content = str(item["content"] or "")
        label = item["label"]
        positive = label is not None and label != ""
        item_cer = cer(content, label) if positive else None
        if positive:
            total_errors += levenshtein(content, str(label))
            total_characters += len(str(label))
            positive_nonempty += int(content != "")
        else:
            negative_total += 1
            negative_rejects += int(content == "")
        submission_results.append({"id": item["id"], "content": content, "label": label, "cer": "" if item_cer is None else f"{item_cer:.6f}"})
        final_item = {
            "id": item["id"],
            "content": content,
            "label": label,
            "sample_type": "positive" if positive else "negative",
            "cer": item_cer,
            "pr": float(content != "") if positive else None,
            "rr": None if positive else float(content == ""),
        }
        for key in (
            "sv_score",
            "accepted",
            "kept_segments",
            "wake_localized",
            "wake_candidate_count",
            "wake_transcript",
            "wake_text_score",
            "wake_selected_source",
            "wake_text_confirmed",
            "sv_accept_threshold",
            "item_duration",
            "second_stage_enabled",
            "second_stage_probability",
            "second_stage_accepted",
            "second_stage_reason",
        ):
            if key in item:
                final_item[key] = item[key]
        final_results.append(final_item)
    summary = {
        "final_cer": total_errors / total_characters if total_characters else 0.0,
        "final_pr": positive_nonempty / (len(results) - negative_total) if len(results) > negative_total else 0.0,
        "final_rr": negative_rejects / negative_total if negative_total else 0.0,
        "positive_count": len(results) - negative_total,
        "positive_nonempty": positive_nonempty,
        "negative_count": negative_total,
        "negative_rejects": negative_rejects,
        "duration": duration,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    final_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"result": {"results": submission_results, "final_cer": f"{summary['final_cer']:.6f}", "duration": f"{duration:.6f}"}}, ensure_ascii=False, indent=2), encoding="utf-8")
    final_output.write_text(json.dumps({"summary": summary, "results": final_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone ERes2NetV2 plus Qwen3-ASR DatasetA baseline")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--config", default="configs/eres2netv2_qwen3_raw_command.json")
    parser.add_argument("--model-root", help="ModelScope cache root containing iic/ and Qwen/")
    parser.add_argument("--output", default="outputs/submission.json")
    parser.add_argument("--final-output", default="outputs/detail.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--asr-batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-output")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.asr_batch_size < 1:
        raise ValueError("--asr-batch-size must be positive")
    config = load_config(args.config)
    samples = read_many_jsonl(args.jsonl, args.data_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else None
    results: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if args.resume:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires --checkpoint-output from a prior run")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results = checkpoint["results"]
        elapsed_before = float(checkpoint["duration"])
        completed = {item["id"] for item in results}
        samples = [sample for sample in samples if submission_id(sample, config.id_mode) not in completed]
    gate = None
    verifier = None
    second_stage = None
    if config.second_stage_enabled:
        if config.gate_mode != "dual_mlp":
            raise ValueError("second_stage_enabled requires gate_mode=dual_mlp")
        if not config.second_stage_model or not config.second_stage_features:
            raise ValueError(
                "second_stage_enabled requires second_stage_model and second_stage_features"
            )
        from moss_second_stage import CachedPresenceStage

        second_stage = CachedPresenceStage(
            config.second_stage_model,
            config.second_stage_features,
            threshold=config.second_stage_presence_threshold,
        )
    if config.gate_mode == "dual_mlp":
        from gate import DualMLPGate

        if config.dual_backend is None:
            raise ValueError("gate_mode=dual_mlp requires dual_backend in config")
        print(f"Loading dual MLP gate (ERes2NetV2 + CAM++) on {config.sv_device or config.device}...", flush=True)
        gate = DualMLPGate(
            config.sv_model, config.cam_model, config.dual_backend,
            config.dual_threshold, config.sv_device or config.device, args.model_root,
        )
        if second_stage is not None and second_stage.primary_threshold is not None:
            if not np.isclose(config.dual_threshold, second_stage.primary_threshold, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "dual_threshold must equal the primary_threshold stored in the "
                    "second-stage artifact; jointly calibrated cascade thresholds "
                    "cannot be mixed with another primary threshold"
                )
    else:
        print(f"Loading ERes2NetV2 on {config.sv_device or config.device}...", flush=True)
        verifier = SpeakerVerifier(config.sv_model, config.sv_device or config.device, args.model_root)
    if config.asr_backend == "firered":
        print(f"Loading FireRedASR2-AED on {config.device}...", flush=True)
        recognizer = FireRedRecognizer(config.firered_model, config.device, config.firered_repo)
    elif config.asr_backend == "qwen_lora":
        if config.qwen_lora_adapter is None:
            raise ValueError("asr_backend=qwen_lora requires qwen_lora_adapter in config")
        print(f"Loading Qwen3-ASR + LoRA adapter ({config.qwen_lora_adapter}) on {config.device}...", flush=True)
        recognizer = ConditionedQwenRecognizer(config.asr_model, config.qwen_lora_adapter, config.device, args.model_root)
    else:
        print(f"Loading Qwen3-ASR on {config.device}...", flush=True)
        recognizer = QwenRecognizer(config.asr_model, config.device, args.model_root, args.asr_batch_size)
    started = time.perf_counter()
    pending_pairs: list[tuple[Path, Path]] = []
    pending_indices: list[int] = []

    def save_checkpoint() -> None:
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps({"duration": elapsed_before + time.perf_counter() - started, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    def flush_asr() -> None:
        if not pending_pairs:
            return
        if getattr(recognizer, "conditioned", False):
            transcripts = recognizer.transcribe_pairs(pending_pairs)
        else:
            transcripts = recognizer.transcribe_many([command for _, command in pending_pairs])
        if len(transcripts) != len(pending_indices):
            raise RuntimeError("ASR returned an unexpected transcript count")
        completed_at = time.perf_counter()
        for index, transcript in zip(pending_indices, transcripts):
            results[index]["content"] = transcript
            if second_stage is not None:
                decision = second_stage.decide(results[index]["id"])
                results[index].update(
                    {
                        "second_stage_enabled": True,
                        "second_stage_probability": (
                            None if decision.probability is None else f"{decision.probability:.6f}"
                        ),
                        "second_stage_accepted": decision.accepted,
                        "second_stage_reason": decision.reason,
                    }
                )
                if not decision.accepted:
                    results[index]["content"] = ""
            results[index]["item_duration"] = f"{completed_at - results[index].pop('_started'):.6f}"
        pending_pairs.clear()
        pending_indices.clear()
        save_checkpoint()

    for sample in tqdm(samples, desc="infer", unit="utt"):
        item_started = time.perf_counter()
        item_id = submission_id(sample, config.id_mode)
        if not sample.wake_audio.is_file() or not sample.command_audio.is_file():
            raise FileNotFoundError(f"missing audio for {item_id}")
        if gate is not None:
            accepted, speaker_score = gate.accept(sample.wake_audio, sample.command_audio)
            active_threshold = config.dual_threshold
        else:
            speaker_score = verifier.score(sample.wake_audio, sample.command_audio)
            accepted = speaker_score >= config.sv_accept_threshold
            active_threshold = config.sv_accept_threshold
        item = {"id": item_id, "content": "", "label": sample.label, "sv_score": f"{speaker_score:.6f}", "accepted": accepted, "kept_segments": 0, "wake_localized": False, "wake_candidate_count": 0, "wake_transcript": "", "wake_text_score": "0.000000", "wake_selected_source": None, "wake_text_confirmed": True, "sv_accept_threshold": f"{active_threshold:.6f}", "item_duration": "", "second_stage_enabled": bool(second_stage is not None), "second_stage_probability": None, "second_stage_accepted": None, "second_stage_reason": "not_run_primary_reject" if second_stage is not None and not accepted else None}
        results.append(item)
        if accepted:
            item["_started"] = item_started
            pending_pairs.append((sample.wake_audio, sample.command_audio))
            pending_indices.append(len(results) - 1)
            if len(pending_pairs) >= args.asr_batch_size:
                flush_asr()
        else:
            item["item_duration"] = f"{time.perf_counter() - item_started:.6f}"
            if not pending_pairs:
                save_checkpoint()
    flush_asr()
    for item in results:
        item.pop("_started", None)
    duration = elapsed_before + time.perf_counter() - started
    summary = write_reports(results, duration, Path(args.output), Path(args.final_output))
    print(f"samples={len(results)} final_cer={summary['final_cer']:.6f} final_pr={summary['final_pr']:.6f} final_rr={summary['final_rr']:.6f} duration={duration:.3f}s", flush=True)


if __name__ == "__main__":
    main()
