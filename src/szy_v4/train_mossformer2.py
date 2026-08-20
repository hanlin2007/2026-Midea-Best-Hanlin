#!/usr/bin/env python3
"""在 dataset_dasynth 上微调 MossFormer2_SS_16K（ClearerVoice torch 权重起步）。

动机：离线 MossFormer2 在 DatasetA 风格腐蚀下崩塌（dasynth valid 实测：
reverb 子集 SI-SDR 比混合还差 6~14dB）。dasynth 提供 18k 条"腐蚀混合 +
腐蚀前对齐干净双源"，正好做联合 去混响+去噪+分离 的域适配微调。

训练：PIT SI-SNR，2s 随机裁剪（与官方预训练一致），bf16 autocast。
评估：每 epoch 在 valid 前 --valid-n 条上整波推理，oracle 排列 SI-SDR，
      报 ALL / clean / reverb / no-reverb / masked 分组，按 ALL mean 存 best。
方法学：训练/选模只用 dasynth train/valid，DatasetA 只做一次性应用。

用法（conda env: yuyin，仓库根目录）：
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
  python -m szy_v4.train_mossformer2 --run-name mossft_v1 \
    --data-root ../dataset_dasynth \
    --init ../exp_sep/clearvoice_ckpt/MossFormer2_SS_16K/last_best_checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WS = _REPO_ROOT.parent
DEFAULT_DATA_ROOT = _WS / "dataset_dasynth"
DEFAULT_INIT = _WS / "exp_sep" / "clearvoice_ckpt" / "MossFormer2_SS_16K" / "last_best_checkpoint.pt"
SR = 16000

sys.path.insert(0, str(_REPO_ROOT / "third_party"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from szy_v4.separator import load_mossformer2_torch  # noqa: E402


# ---------------- PIT SI-SNR ----------------

def si_snr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """est/ref: (B, T) -> (B,) SI-SNR"""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    denom = (ref ** 2).sum(dim=-1).clamp_min(1e-12)
    scale = (est * ref).sum(dim=-1) / denom
    target = scale.unsqueeze(-1) * ref
    noise = est - target
    return 10.0 * torch.log10(
        (target ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1).clamp_min(1e-12) + 1e-12
    )


def pit_sisnr_loss(ests: list[torch.Tensor], refs: torch.Tensor) -> torch.Tensor:
    """ests: list of 2 x (B,T); refs: (B,2,T)。返回最小排列的负 SI-SNR 均值。"""
    perms = list(permutations(range(2)))
    vals = []
    for p in perms:
        v = si_snr(ests[p[0]], refs[:, 0]) + si_snr(ests[p[1]], refs[:, 1])
        vals.append(v)
    stacked = torch.stack(vals, dim=0)  # (P, B)
    best = stacked.max(dim=0).values
    return -(best.mean() / 2.0)


def si_sdr_np(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    est = est[:n].astype(np.float64)
    ref = ref[:n].astype(np.float64)
    est -= est.mean()
    ref -= ref.mean()
    d = float(ref @ ref)
    if d < 1e-12:
        return float("nan")
    t = float(est @ ref) / d * ref
    no = est - t
    return 10.0 * np.log10(float(t @ t) / max(float(no @ no), 1e-18))


# ---------------- 数据 ----------------

class DasynthPairs(Dataset):
    def __init__(self, data_root: Path, meta_path: Path, crop_sec: float, train: bool):
        self.data_root = data_root
        self.rows = [json.loads(l) for l in meta_path.open(encoding="utf-8") if l.strip()]
        self.crop = int(crop_sec * SR)
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        mix, _ = sf.read(self.data_root / r["mixed_audio"], dtype="float32")
        s1, _ = sf.read(self.data_root / r["source_1_audio"], dtype="float32")
        s2, _ = sf.read(self.data_root / r["source_2_audio"], dtype="float32")
        n = min(len(mix), len(s1), len(s2))
        if n <= self.crop:
            st = 0
            pad = self.crop - n
        else:
            st = random.randint(0, n - self.crop) if self.train else (n - self.crop) // 2
            pad = 0
        mix = mix[st: st + self.crop]
        s1 = s1[st: st + self.crop]
        s2 = s2[st: st + self.crop]
        if pad > 0:
            mix = np.pad(mix, (0, pad))
            s1 = np.pad(s1, (0, pad))
            s2 = np.pad(s2, (0, pad))
        return (
            torch.from_numpy(mix.copy()),
            torch.from_numpy(np.stack([s1, s2]).copy()),
        )


# ---------------- 评估 ----------------

@torch.no_grad()
def evaluate(model, data_root: Path, valid_meta: list[dict], n: int, device: str) -> dict:
    model.eval()
    recs = []
    for m in valid_meta[:n]:
        mix, _ = sf.read(data_root / m["mixed_audio"], dtype="float32")
        refs = [
            sf.read(data_root / m["source_1_audio"], dtype="float32")[0],
            sf.read(data_root / m["source_2_audio"], dtype="float32")[0],
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(torch.from_numpy(mix).to(device)[None, :])
        ests = [o.squeeze(0).float().cpu().numpy() for o in out]
        best = max(
            np.mean([si_sdr_np(ests[p[0]], refs[0]), si_sdr_np(ests[p[1]], refs[1])])
            for p in permutations(range(2))
        )
        recs.append({
            "v": best,
            "reverb": any(m["corruption"].get("reverb", [False, False])),
            "masked": m["corruption"].get("masked", False),
            "clean": m["corruption"].get("clean", False),
        })
    model.train()

    def agg(sel):
        v = [r["v"] for r in sel if not np.isnan(r["v"])]
        return float(np.mean(v)) if v else float("nan")

    return {
        "all": agg(recs),
        "clean": agg([r for r in recs if r["clean"]]),
        "reverb": agg([r for r in recs if r["reverb"]]),
        "no_reverb": agg([r for r in recs if not r["reverb"]]),
        "masked": agg([r for r in recs if r["masked"]]),
        "n": len(recs),
    }


# ---------------- 主流程 ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--init", type=Path, default=DEFAULT_INIT)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="默认 <data-root>/out_mossft")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accu", type=int, default=2, help="梯度累积步数，等效 batch = batch*accu")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--crop-sec", type=float, default=2.0)
    ap.add_argument("--valid-n", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N 个 epoch 额外存一份 ep{epoch}.pt（0=关闭，用于 epoch 扫描）")
    ap.add_argument("--run-name", type=str, default="mossft")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.data_root / "out_mossft")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    import wandb
    wandb.init(project="mossformer2-dasynth", name=args.run_name, config=vars(args))

    model, device = load_mossformer2_torch(args.init, device)
    model.train()
    print(f"init from {args.init}", flush=True)

    train_ds = DasynthPairs(args.data_root, args.data_root / "metadata" / "train.jsonl",
                            args.crop_sec, train=True)
    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=6, drop_last=True, persistent_workers=True,
    )
    valid_meta = [json.loads(l)
                  for l in (args.data_root / "metadata" / "valid.jsonl").open(encoding="utf-8")]

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1e9
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        it = iter(loader)
        for _ in range(args.steps_per_epoch):
            try:
                mix, refs = next(it)
            except StopIteration:
                it = iter(loader)
                mix, refs = next(it)
            mix = mix.to(device, non_blocking=True)
            refs = refs.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(mix)
                loss = pit_sisnr_loss([o.float() for o in out], refs.float())
                loss = loss / args.accu
            loss.backward()
            if step % args.accu == args.accu - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
            losses.append(float(loss) * args.accu)
            step += 1
            if step % 250 == 0:
                wandb.log({"train/loss": np.mean(losses[-250:]), "step": step})
        metrics = evaluate(model, args.data_root, valid_meta, args.valid_n, device)
        log = {f"valid/{k}": v for k, v in metrics.items() if k != "n"}
        log["train/epoch_loss"] = float(np.mean(losses))
        log["epoch"] = epoch
        wandb.log(log)
        print(
            f"epoch {epoch:3d} loss {np.mean(losses):7.3f} "
            f"valid ALL {metrics['all']:6.2f} clean {metrics['clean']:6.2f} "
            f"reverb {metrics['reverb']:6.2f} noRev {metrics['no_reverb']:6.2f} "
            f"masked {metrics['masked']:6.2f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
        torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "last.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "valid_all": metrics["all"]}, out_dir / f"ep{epoch}.pt")
        if metrics["all"] > best:
            best = metrics["all"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_all": best},
                       out_dir / "best.pt")
            print(f"  -> new best {best:.2f} @epoch {epoch}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
