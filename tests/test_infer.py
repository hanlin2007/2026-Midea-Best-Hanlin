"""midea_qiji 基线数据/文本/报告口径的单元测试（numpy-only，无需模型）。

这组测试锁死的是"口径"——文本归一化规则、id 生成规则、manifest 中文
字段、报告与竞赛提交物的 schema。这些规则一旦被"顺手改进"，官方评测
脚本可能直接判 0 分，或者 CER 统计口径悄悄漂移。所以它们存在的意义
不是"证明代码能跑"，而是"证明历史行为没有被改挂"——测试即文档：
读这些测试就能知道每个函数的约定是什么。

为什么这些测试不需要模型/GPU：被测函数（normalize_transcript、
submission_id、read_many_jsonl、write_reports、summarize、write_outputs）
全是纯数据/文本处理，测试里用手工构造的小数据覆盖全部边界。

unittest 基础（初学可对照正文）：
  - 继承 unittest.TestCase 的类就是一组测试；类里以 test_ 开头的方法
    会被自动发现并逐个执行（discover 机制），其他方法只是普通辅助；
  - 断言用 self.assertEqual/assertTrue/assertIsNone 等：失败时给出
    期望值 vs 实际值的清晰对比，比手写 if + raise 好用；
  - 每个测试方法独立运行、互不影响，一个挂了不影响其他。

运行（仓库根目录）：
  PYTHONPATH=src python -m unittest tests.test_infer -v
（PYTHONPATH=src 让 unittest 进程能 import 到 src/ 下的包——
测试文件自身在 tests/ 里，不在包内，和脚本一样需要路径配置。）
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from midea_qiji.infer import Sample, normalize_transcript, read_many_jsonl, submission_id, write_reports
from szy.reporting import summarize, write_outputs


class InferTest(unittest.TestCase):
    def test_normalization_matches_baseline_rules(self) -> None:
        # 归一化规则的两类核心转换各举一例锁死：
        # 数字转中文（25 -> 二十五、30 -> 三十、101 -> 一百零一）、
        # 百分号展开（30% -> 百分之三十）、标点/空格删除。
        # 这条测试的价值：normalize_transcript 是 CER 计算的前置步骤，
        # 任何"改进"（比如保留标点、改数字读法）都会让全仓库的 CER
        # 口径与官方评测错位——所以它必须被冻结。
        self.assertEqual(normalize_transcript("空调开到25度，风量30%"), "空调开到二十五度风量百分之三十")
        # 三位数带零的读法（101 -> 一百零一），补一个容易写错的边界。
        self.assertEqual(normalize_transcript("设置101分钟"), "设置一百零一分钟")

    def test_relative_stem_id_is_split_safe(self) -> None:
        # Sample 是数据类（dataclass），按位置传参依次是：
        # raw_id, wake_audio, command_audio, command_relative_path, label。
        # "relative_stem" 模式生成 id = 相对路径去掉扩展名（"pos/cmd_1"）。
        # 为什么测它：id 是提交物里标识样本的唯一键，必须与官方/历史
        # 口径一致；而且它不带扩展名、保留子目录，天然"拆分安全"
        # （数据集被拆到多个子目录时 id 不冲突、不随目录移动而变）。
        sample = Sample("1", Path("wake.wav"), Path("cmd.wav"), "pos/cmd_1.wav", None)
        self.assertEqual(submission_id(sample, "relative_stem"), "pos/cmd_1")

    def test_manifest_preserves_chinese_wake_text(self) -> None:
        # 测两点：1) read_many_jsonl 能正确读出中文键值（唤醒文本）；
        # 2) 写文件时 ensure_ascii=False 的重要性——JSON 里中文原样保存，
        # 读回来仍是"你好科慕"而不是乱码/转义。
        # tempfile.TemporaryDirectory() 作为上下文管理器：进入 with 时
        # 在系统临时目录建一个随机名字的空目录，退出 with 时连同内容
        # 自动删除。测试用它的原因：a) 不污染仓库目录；b) 不用自己写
        # 清理代码（try/finally）；c) 每次运行都是全新目录，测试之间
        # 不会因为残留文件互相干扰。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "items.jsonl"
            manifest.write_text(json.dumps({"id": 1, "唤醒音频": "pos/kws_1.wav", "唤醒文本": "你好科慕", "识别音频": "pos/cmd_1.wav", "识别文本": "开灯"}, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = read_many_jsonl([manifest], root)
        # 唤醒文本是下游"唤醒锚点选择"的关键输入：这里读丢/读错的话，
        # 后面的声纹注册会静默用错音频——这种 bug 不会报错只会掉分，
        # 所以要用测试提前锁死。
        self.assertEqual(samples[0].wake_text, "你好科慕")

    def test_report_metrics_and_schema(self) -> None:
        # 用两条手工数据（一正一负）测 write_reports 的完整行为：
        # 指标计算 + detail.json 的 schema。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = write_reports(
                [
                    {"id": "pos/a", "content": "开灯", "label": "开灯", "sv_score": "0.5"},
                    {"id": "neg/a", "content": "", "label": None, "sv_score": "0.1"},
                ],
                2.0,
                root / "submission.json",
                root / "detail.json",
            )
            payload = json.loads((root / "detail.json").read_text(encoding="utf-8"))
        # 正样本转写完全正确 -> CER=0；正样本有非空输出 -> PR=1；
        # 负样本空输出 -> 正确拒绝 -> RR=1。
        self.assertEqual(summary["final_cer"], 0.0)
        self.assertEqual(summary["final_pr"], 1.0)
        self.assertEqual(summary["final_rr"], 1.0)
        # detail 结果行必须带 sample_type 字段（下游 replay/消融脚本
        # 靠它区分正负样本）；负样本行的 rr 应为 1.0。
        self.assertEqual(payload["results"][0]["sample_type"], "positive")
        self.assertEqual(payload["results"][1]["rr"], 1.0)

    def test_szy_submission_matches_competition_schema(self) -> None:
        # SZY 提交物的竞赛 schema 契约测试。
        # schema = 文件结构约定。官方评分脚本按固定结构解析提交文件，
        # 少一个键/多一个键/类型不对都可能直接判 0 分——本地跑分再高
        # 也没用。所以用测试把结构锁死。
        rows = [
            {"id": "1", "content": "开灯", "label": "开灯", "route": "raw"},
            {"id": "2", "content": "", "label": None, "route": "fusion_reject"},
        ]
        summary = summarize(rows, 2.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "submission.json"
            write_outputs(rows, summary, output, root / "detail.json")
            result = json.loads(output.read_text(encoding="utf-8"))["result"]

        # 顶层键必须恰好是这四个：set 比较不关心顺序，只关心集合。
        self.assertEqual(set(result), {"results", "avg_cer", "avg_rr", "duration"})
        self.assertEqual(result["avg_cer"], 0.0)
        self.assertEqual(result["avg_rr"], 1.0)
        # duration 以毫秒为单位（2.5 秒 -> 2500），这是竞赛格式约定。
        self.assertEqual(result["duration"], 2500.0)
        self.assertEqual(result["results"][0]["cer"], 0.0)
        # 负样本没有 cer（无从计算错误率），必须是 JSON null（Python None）。
        # 用 assertIsNone 而不是 assertEqual(..., None)：前者失败信息
        # 直接打出"是 None 而不是 X"，更利于排查。
        self.assertIsNone(result["results"][1]["cer"])


if __name__ == "__main__":
    # 直接运行本文件时也能跑测试（python tests/test_infer.py），
    # 与 python -m unittest 等价；被 discover 导入时这行不执行。
    unittest.main()
