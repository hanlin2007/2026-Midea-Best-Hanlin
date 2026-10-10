#!/usr/bin/env python3
"""在 dataset_dasynth 上微调 MossFormer2_SS_16K（ClearerVoice torch 权重起步）。

动机：离线 MossFormer2 在 DatasetA 风格腐蚀下崩塌（dasynth valid 实测：
reverb 子集 SI-SDR 比混合还差 6~14dB）。dasynth 提供 18k 条"腐蚀混合 +
腐蚀前对齐干净双源"，正好做联合 去混响+去噪+分离 的域适配微调。

"域适配（domain adaptation）微调"的含义：模型已经会分离（官方预训练），
但没见过竞赛风格的腐蚀音频（混响/加噪/掩盖）；用目标域的数据再训练
一小段，让它把已有能力迁移到新域——起始点是官方权重而非随机初始化，
所以只需很少的 epoch 和较小的学习率。

训练：PIT SI-SNR，2s 随机裁剪（与官方预训练一致），bf16 autocast。
评估：每 epoch 在 valid 前 --valid-n 条上整波推理，oracle 排列 SI-SDR，
      报 ALL / clean / reverb / no-reverb / masked 分组，按 ALL mean 存 best。
方法学：训练/选模只用 dasynth train/valid，DatasetA 只做一次性应用。
（"训练/选模与最终评测数据隔离"是实验规范：若拿评测集挑模型，分数
会虚高，上线后必现落差；所以 valid 只来自 dasynth，DatasetA 仅最后验证一次。）

用法（conda env: yuyin，仓库根目录）：
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src \
  python -m pipeline.train --run-name mossft \
    --data-root ../dataset_dasynth \
    --init ../exp_sep/clearvoice_ckpt/MossFormer2_SS_16K/last_best_checkpoint.pt
"""
from __future__ import annotations

import argparse                                  # 命令行参数
import json                                      # dasynth 元数据（jsonl 行）
import random                                    # 随机种子（训练裁剪位置）
import sys                                       # sys.path 注入（third_party/src）
import time                                      # 计时
from pathlib import Path                         # 路径

import numpy as np                               # 波形数组运算
import soundfile as sf                           # 音频读取
import torch                                     # 深度学习框架本体
from torch.utils.data import DataLoader, Dataset # 数据管道：数据集协议 + 多进程批量加载

# 从本文件位置推导仓库根：parents[2] 上溯两级（parents[0]=src/pipeline，
# parents[1]=src，parents[2]=仓库根）。_WS 是仓库的上级目录（workspace），
# dasynth 数据与官方权重就放在那里
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WS = _REPO_ROOT.parent
DEFAULT_DATA_ROOT = _WS / "dataset_dasynth"
DEFAULT_INIT = _WS / "exp_sep" / "clearvoice_ckpt" / "MossFormer2_SS_16K" / "last_best_checkpoint.pt"
SR = 16000                                    # 采样率常量（与 core.audio.SAMPLE_RATE 同值）

# 先注入路径再 import pipeline：本脚本以 `python -m pipeline.train` 方式运行，此时
# src/ 已在搜索路径里；但直接 `python src/pipeline/train.py` 运行时则不在，
# 所以显式注入更稳。third_party 供 separator 里 mossformer2 模型定义使用。
# 与 pipeline 的延迟导入不同：训练脚本本来就是"必须装 torch 的环境"，
# 没有"无 torch 也能跑"的需求，所以顶层 import 即可
sys.path.insert(0, str(_REPO_ROOT / "third_party"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# noqa: E402 豁免 linter 的"import 不在顶部"——上面必须先改 sys.path
from pipeline.metrics import oracle_sisdr, pit_sisnr_loss  # noqa: E402
from pipeline.separator import load_mossformer2  # noqa: E402


# ---------------- 数据 ----------------

class DasynthPairs(Dataset):
    """dasynth 合成对数据集：混合音频 + 两路腐蚀前干净源。

    每条样本随机/固定裁剪 crop_sec 秒；不足则右侧补零（保持 batch 形状）。

    Dataset 是 torch 的数据协议：实现 __len__（多少条）与 __getitem__
    （取第 idx 条），DataLoader 就能自动完成批量组装、打乱、多进程
    预取。torch 只要求这两个方法——这就是"鸭子类型"式的接口约定。
    """

    def __init__(self, data_root: Path, meta_path: Path, crop_sec: float, train: bool):
        self.data_root = data_root
        # 读 jsonl 元数据：每行一个 JSON 对象，含 mixed_audio / source_1_audio /
        # source_2_audio 相对路径与 corruption（腐蚀类型）标注。
        # 过滤空行（文件末尾常见）；json.loads(l) 解析每行
        self.rows = [json.loads(l) for l in meta_path.open(encoding="utf-8") if l.strip()]
        # 裁剪长度换算成采样点数（秒 * 采样率）
        self.crop = int(crop_sec * SR)
        self.train = train

    def __len__(self) -> int:
        """数据集大小。DataLoader 据此计算"一个 epoch 有多少步"。"""
        return len(self.rows)

    def __getitem__(self, idx: int):
        """取第 idx 条样本，返回 (混合波形张量, 双源波形张量)。

        注意每条样本都是"读到哪算哪"地即时加载：元数据只有路径，音频
        不常驻内存（18k 条 * 若干秒音频全放内存要几十 GB）。多进程
        DataLoader 会并行调用本方法，音频解码开销被分摊到多个 worker。
        """
        r = self.rows[idx]
        # sf.read 返回 (波形, 采样率)；dtype="float32" 直接读浮点。
        # 只取波形（_ 丢弃采样率）——dasynth 全部 16k，不必校验
        mix, _ = sf.read(self.data_root / r["mixed_audio"], dtype="float32")
        s1, _ = sf.read(self.data_root / r["source_1_audio"], dtype="float32")
        s2, _ = sf.read(self.data_root / r["source_2_audio"], dtype="float32")
        # 三路取最短长度对齐（合成数据的混合与源应该等长，这里是防御性写法）
        n = min(len(mix), len(s1), len(s2))
        if n <= self.crop:
            # 音频比裁剪窗口短：从头取整段，右侧补零到固定长度。
            # 为什么必须固定长度：batch 内的张量要堆叠成 (B, T) 矩阵，
            # 每条必须等长；补零是最简单的对齐手段
            st = 0
            pad = self.crop - n
        else:
            # 训练集：随机起点裁剪（数据增强——同一段音频每次出现的位置
            # 不同，等效于扩充数据）；验证/评估：取中心段（确定性，
            # 保证每次评估结果可比、可复现）
            st = random.randint(0, n - self.crop) if self.train else (n - self.crop) // 2
            pad = 0
        # 三路用同一起点/长度切片——混合与两个源必须严格时间对齐，
        # 否则 SI-SNR 的"目标/噪声分解"会张冠李戴
        mix = mix[st: st + self.crop]
        s1 = s1[st: st + self.crop]
        s2 = s2[st: st + self.crop]
        if pad > 0:
            # np.pad((0, pad))：只在末尾（右侧）补零，左侧不动。
            # 补零不引入任何新内容，模型学到的"静音区"是中性信息
            mix = np.pad(mix, (0, pad))
            s1 = np.pad(s1, (0, pad))
            s2 = np.pad(s2, (0, pad))
        # torch.from_numpy：零拷贝包装（与 numpy 共享内存，省一次复制）；
        # .copy()：确保张量拥有自己连续的内存——np.pad/切片可能产生非连续
        # 视图，DataLoader 默认 collate 需要连续内存才能高效堆叠
        # np.stack([s1, s2])：沿新轴堆成 (2, T)——两个源合成一个张量，
        # 与模型两路输出、pit_sisnr_loss 的 refs (B, 2, T) 对齐
        return (
            torch.from_numpy(mix.copy()),
            torch.from_numpy(np.stack([s1, s2]).copy()),
        )


# ---------------- 评估 ----------------

@torch.no_grad()
def evaluate(model, data_root: Path, valid_meta: list[dict], n: int, device: str) -> dict:
    """valid 前 n 条整波推理 + oracle SI-SDR，按腐蚀类型分组汇总。

    @torch.no_grad() 是函数装饰器：整个函数体都在 no_grad 上下文里——
    评估纯推理，不建梯度图，省显存（训练时的梯度中间量可能比参数
    本身还大）也更快。评估期间不更新参数，梯度毫无用处。
    """
    # 切评估模式：关 dropout 等训练期随机行为，结果可复现
    model.eval()
    recs = []
    for m in valid_meta[:n]:
        # 整波推理（训练用 2s 裁剪，评估用完整音频）：测模型在真实
        # 输入长度下的表现，也避免裁剪把难例切掉
        mix, _ = sf.read(data_root / m["mixed_audio"], dtype="float32")
        refs = [
            sf.read(data_root / m["source_1_audio"], dtype="float32")[0],
            sf.read(data_root / m["source_2_audio"], dtype="float32")[0],
        ]
        # autocast：自动混合精度（AMP）。模型前向全程用 bfloat16 计算：
        # 数值动态范围与 float32 相同（指数位一样多）、精度略低，
        # 但显存减半、张量核速度快一倍以上，训练大模型的标准配置。
        # 只有支持 bf16 的 GPU（Ampere 及以后）才受益
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(torch.from_numpy(mix).to(device)[None, :])
        # 输出两路转 numpy（squeeze 去 batch 维、float() 回 float32——
        # autocast 里输出可能是 bf16，numpy 不支持）
        ests = [o.squeeze(0).float().cpu().numpy() for o in out]
        # oracle 排列 SI-SDR：自动对齐"估计↔干净源"的最优对应再打分，
        # 不因通道顺序冤枉模型（见 metrics.oracle_sisdr）
        best = oracle_sisdr(ests, refs)
        recs.append({
            "v": best,
            # 从元数据提取腐蚀标签：reverb 是 [bool, bool] 列表（两路是否
            # 各加混响），any 表示"至少一路加了混响"
            "reverb": any(m["corruption"].get("reverb", [False, False])),
            "masked": m["corruption"].get("masked", False),
            "clean": m["corruption"].get("clean", False),
        })
    # 评估完切回训练模式（调用方随后继续训练；忘切回来 dropout 会一直
    # 关着，等于悄悄改变了训练行为）
    model.train()

    def agg(sel):
        """分组取 SI-SDR 均值；NaN（静音参考等不可评样本）先过滤。

        闭包：定义在 evaluate 内部、捕获外层变量（recs）的小工具函数，
        避免为每个分组重复写同样的过滤逻辑。
        """
        v = [r["v"] for r in sel if not np.isnan(r["v"])]
        return float(np.mean(v)) if v else float("nan")

    # 按腐蚀类型分组报告：ALL 是选模指标（存 best），其余分组用于
    # 诊断"模型在哪类腐蚀上进步/退化"
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

    # args.out_dir 为 None 时用默认（or 短路：None 为假值）
    out_dir = args.out_dir or (args.data_root / "out_mossft")

    # 三处随机源全部固定种子：数据裁剪位置（random）、numpy 运算、
    # torch 的权重初始化/算子。种子相同 → 每次运行的数据流与初始化
    # 完全一致 → 结果可复现（论文/实验的基本要求）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    # wandb：训练过程可视化/日志平台（曲线、超参、系统指标）。
    # config=vars(args) 把命令行参数快照上传，事后可查"这组曲线是什么
    # 参数跑出来的"。import 放在函数内：wandb 是可选依赖，不登录也能
    # 跑（会提示离线模式），但 import 失败不应阻塞脚本的其他用途
    import wandb
    wandb.init(project="mossformer2-dasynth", name=args.run_name, config=vars(args))

    # 从官方/历史权重初始化（load_mossformer2 返回 (模型, 实际设备)，
    # 请求 cuda 但无 GPU 时会回落 cpu——返回值覆盖 device 变量）
    model, device = load_mossformer2(args.init, device)
    # 切训练模式：打开 dropout 等训练行为（eval 状态下训练会静默失效）
    model.train()
    print(f"init from {args.init}", flush=True)

    # 训练集（随机裁剪）；验证元数据单独读成 list 供 evaluate 用
    train_ds = DasynthPairs(args.data_root, args.data_root / "metadata" / "train.jsonl",
                            args.crop_sec, train=True)
    # DataLoader 参数逐项说明：
    #   batch_size     每个 batch 的样本数；
    #   shuffle=True   每 epoch 打乱顺序（训练必须；顺序喂会让模型
    #                  记住样本次序，泛化变差）；
    #   num_workers=6  用 6 个子进程并行读盘/解码音频。Python 的 GIL
    #                  让单进程多线程提速有限，多进程才是正解；
    #   drop_last=True 丢弃最后不足一个 batch 的尾巴。为什么丢：显存
    #                  紧张时 batch 大小是精打细算的，最后来个"半个
    #                  batch"形状不同，若配了 BatchNorm 或梯度累积
    #                  会出数值问题；丢弃的代价可忽略；
    #   persistent_workers=True 子进程不随 epoch 重建（否则每 epoch
    #                  都重新 fork 6 个进程，启动开销巨大）
    loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=6, drop_last=True, persistent_workers=True,
    )
    valid_meta = [json.loads(l)
                  for l in (args.data_root / "metadata" / "valid.jsonl").open(encoding="utf-8")]

    # Adam 优化器：weight_decay=1e-5 是 L2 正则（防过拟合），
    # lr=5e-5 是微调尺度的小学习率（从头训练通常是 1e-3 量级，
    # 微调太大 lr 会冲垮预训练权重）
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    out_dir.mkdir(parents=True, exist_ok=True)   # 输出目录先建好（写存档时不再操心）
    best = -1e9                                  # 历史最佳 valid ALL 分（初始极小值）
    step = 0                                     # 全局步数计数器（跨 epoch 累计）
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        # iter(loader) 拿到迭代器；下面手动 next——不用 for 循环是因为
        # 要精确控制"每 epoch 跑 steps_per_epoch 步"（而不是整个数据集
        # 一遍），且中途要处理"一轮耗尽"的情况
        it = iter(loader)
        for _ in range(args.steps_per_epoch):
            try:
                mix, refs = next(it)
            except StopIteration:
                # 数据集耗尽：重建迭代器继续取（shuffle 会重新打乱）。
                # 这样 epoch 的长度由 steps_per_epoch 决定，与数据集
                # 大小解耦（想多训几轮/少训几轮改一个参数即可）
                it = iter(loader)
                mix, refs = next(it)
            # non_blocking=True：异步拷贝到 GPU（CPU 继续取下一批数据，
            # 拷贝在后台进行，隐藏传输延迟）
            mix = mix.to(device, non_blocking=True)
            refs = refs.to(device, non_blocking=True)
            # 混合精度前向+损失（与评估相同的 bf16 autocast）
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(mix)
                # PIT SI-SNR 损失：自动挑"最优通道对应"算分再取负——
                # 模型只需学会把两人分开，不被通道顺序惩罚
                # （见 metrics.pit_sisnr_loss）。o.float()/refs.float()
                # 把 bf16 张量转回 float32 再进损失，保证数值精度
                loss = pit_sisnr_loss([o.float() for o in out], refs.float())
                # 梯度累积：loss 除以 accu。backward 累加的是"缩放后的
                # 梯度"，accu 步累加起来恰好等于一个大 batch 的梯度
                loss = loss / args.accu
            # 反向传播：计算梯度并累加到参数 .grad 上（不清零——
            # 这就是累积的关键）
            loss.backward()
            if step % args.accu == args.accu - 1:
                # 每 accu 步才真正更新一次参数。
                # 为什么要梯度累积：显存放不下大 batch（MossFormer2 不小，
                # 单卡可能只能 batch=2），但小 batch 梯度噪声大、训练不稳；
                # 累积 accu 步再更新，等效于用 batch*accu 的大 batch 训练，
                # 只花小 batch 的显存
                # clip_grad_norm_：梯度裁剪，总范数超过 10.0 就整体缩放。
                # 防止单条异常样本产生爆炸梯度把权重冲飞（训练崩溃的
                # 头号原因）。末尾下划线是 torch 惯例：原地修改
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optim.step()
                # 清空梯度，为下一轮累积做准备。set_to_none=True 把
                # .grad 置 None 而不是填 0：省一次显存清零写操作，
                # 且某些算子对 None 梯度有优化——微优化但免费
                optim.zero_grad(set_to_none=True)
            # 记录时乘回 accu：恢复"等效大 batch"的损失量级，
            # 不同 accu 设置的实验之间曲线才可比
            losses.append(float(loss) * args.accu)
            step += 1
            if step % 250 == 0:
                # 每 250 步向 wandb 报一次最近 250 步的平均损失（切片
                # losses[-250:]），平滑曲线便于观察趋势
                wandb.log({"train/loss": np.mean(losses[-250:]), "step": step})
        # 每个 epoch 结束：评估（自动切 eval/train）、上报、存档
        metrics = evaluate(model, args.data_root, valid_meta, args.valid_n, device)
        log = {f"valid/{k}": v for k, v in metrics.items() if k != "n"}  # "n" 只是计数，不上报
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
        # last.pt 每 epoch 覆盖：训练中断时从这里恢复最近的进度。
        # state_dict() 只含参数张量（不含结构/优化器），配合
        # separator.load_mossformer2 的两种格式兼容读回
        torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / "last.pt")
        if args.save_every and epoch % args.save_every == 0:
            # 周期快照 ep{N}.pt：用于"epoch 扫描"（挑哪个 epoch 权重
            # 最好），valid_all 一并存档免去事后重评
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "valid_all": metrics["all"]}, out_dir / f"ep{epoch}.pt")
        if metrics["all"] > best:
            # 按 valid ALL 选模：best.pt 永远保存"史上最好"那份。
            # 只信验证集、不信训练损失——训练损失可能一直降但模型
            # 已经过拟合
            best = metrics["all"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "valid_all": best},
                       out_dir / "best.pt")
            print(f"  -> new best {best:.2f} @epoch {epoch}", flush=True)
    wandb.finish()                                # 结束 wandb 会话（刷新剩余日志）


if __name__ == "__main__":
    main()
