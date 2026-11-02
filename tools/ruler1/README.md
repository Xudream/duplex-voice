# 尺子1 评测工装（公开集）

把公开集映射成统一的 S0–S4 任务清单，跑被测系统、统一判分。
数据不进仓库（体积大），只提交代码与清单生成逻辑。

## 目录

```
tools/ruler1/
├── fetch.py             # hf-mirror 直连下载（跟分页、只下清单、抗僵死）
├── build_manifests.py   # 公开集 → manifests/*.jsonl（S0–S4 映射 + audio_ok 标记）
├── make_mini.py         # 分层抽最小集（冒烟/首轮选型，默认 380 条）
├── llm_judge.py         # 被测系统适配器：openai_compat / jev / dryrun
├── rule_baseline.py     # A0 基线：字符 n-gram + 逻辑回归（传统分类器）
├── baselines.py         # 平凡基线 all_silent / all_respond（定地板与天花板）
├── score.py             # 分场景拒识率 / FAR_w / FRR / clarify / invalid / 误执行 + Wilson CI + 工作点扫描
└── stats.py             # McNemar、Bootstrap AUC 差、最小可分辨差
```

数据默认放在 `~/.hermes/workspace/ruler1/data/`（本地临时目录，不入库）：

```
data/
├── chorus/      BiggestWave/ChorusDuplex-SideTalk（中文 TTS，1.0 GB）
├── wearvox/     zlinao/WearVox（英文眼镜，non-assistant-directed 1,082 条）
└── humdial/     ASLP-lab/HumDial-FDBench（中文真实，只解 cn_test_nondev）
```

## 用法

```bash
# 1) 下载
#    - fetch.py 会跟 Link: rel="next" 翻页（HF tree API 每页最多 1,000 条，不翻页会静默截断）
#    - curl 带 --speed-limit/--speed-time，hf-mirror 偶发"连上不传数据"时会重试而不是卡死
python3 fetch.py BiggestWave/ChorusDuplex-SideTalk data/chorus  --jobs 6
python3 fetch.py zlinao/WearVox                    data/wearvox --jobs 6
python3 fetch.py ASLP-lab/HumDial-FDBench          data/humdial_dl --jobs 1
# 只补清单里点名的文件（WearVox 全量 59.45 GB，用到的 wav 只有 ~310 MB）
python3 fetch.py zlinao/WearVox data/wearvox --only-file needed.txt --jobs 4
unzip -oq data/humdial_dl/Humdial-Track2-Test.zip "test/cn_test_nondev/*" -d data/humdial

# 2) 生成清单（含 segments/target_text/audio_ok）
python3 build_manifests.py --dataset all --data ~/.hermes/workspace/ruler1/data --out manifests

# 3) 抽最小集（380 条：S1 120 / S2 90 / S3 60 / S0 60 / S0_guest 20 / REG 30）
python3 make_mini.py --manifests manifests --out manifests/mini.jsonl

# 4) 平凡基线 + A0 传统分类器基线
python3 baselines.py --manifest manifests/chorus.jsonl --mode silent --out predictions/silent__chorus.jsonl
python3 rule_baseline.py --manifest manifests/mini.jsonl --out predictions/a0__mini.jsonl   # 5 折 CV

# 5) 接被测模型（跑道 A）
#    --text-mode last（默认）：只判当前 VAD 段（HumDial 这类多说话人样本必需）
#    --prompt-version v1|v2|v3：v2 加多说话人规则，v3 加 p_silent 连续分
python3 llm_judge.py --manifest manifests/mini.jsonl --backend openai_compat \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen-plus --text-mode last --prompt-version v3 \
  --out predictions/qwen-plus-last-v3__mini.jsonl

# 6) 判分（--sweep 出工作点曲线；--baseline-pred 出 McNemar；子集跑分加 --only-predicted）
python3 score.py --manifest manifests/mini.jsonl --pred predictions/qwen-plus-last-v3__mini.jsonl \
  --only-predicted --sweep --report reports/qwen-plus-last-v3__mini.md
```

## 各集映射规则（v1.2 实际实现）

| 集 | 规则 | 场景 |
|---|---|---|
| ChorusDuplex | `kind ∈ {backchannel, interruption}` | REG |
| ChorusDuplex | 说话人=主用户（该对话中对助手说话最多者）且对助手说 | S0 |
| ChorusDuplex | **非**主用户对助手说 | S0_guest（访客指令，不计入 FRR） |
| ChorusDuplex | 主用户 → 旁人 | S1 |
| ChorusDuplex | 旁人 → 主用户 | S2 |
| ChorusDuplex | 旁人 → 旁人 | S3 |
| WearVox | `ground_truth=[Active]` | S0 |
| WearVox | `[Mute]` 且 `unrelated_talk` | S1 |
| WearVox | `[Mute]` 且 `bystander_only` | S3 |
| HumDial | `talk_to_others` | S1 |
| HumDial | `others_talk_to_user_before` / `others_talk_to_user_after` | S2 |
| HumDial | `backchannel` / `pause` | REG |

> HumDial 每个样本有 `X.wav` 与 `clean_X.wav` 两版，**只收 raw**（否则清单虚高 62%）。

## 指标口径（与设计文档 §1.3 一致）

- 分场景拒识率 `Rej_s = 正确静默 / 应静默`
- `FAR_w = Σ w_s(1−Rej_s)`，w 默认 S1 0.40 / S2 0.15 / S3 0.25 / S4 0.20，缺场景自动归一
- `FRR = S0 被置静默的比例`（S0 应响应）
- `clarify` 单列，不计入拒识率分子、不计入 FAR
- `invalid/缺失` 单独计数，>1% 该模型不可用
- 误执行一票否决（预测里带 `tool_call: true` 且动作属于响应类）
- 工作点扫描：有 `p_silent`（0–100）时按 `p_silent ≥ t` 判静默，否则退回置信度回退

## 已知边界

- 公开集没有音箱远场/手机近场的声学，也没有 DoA/近讲真值 → 尺子1 只能选"裁判"，不能选"前端"
- HumDial SDO 训练集为 0 条 → S1 的训练数据只能来自自建集
- **S4 = 0 条**：三套公开集都没有电视/媒体人声 → 见 `docs/S4媒体人声冒烟方案.md`
- HumDial 英文部分未解压（需 6.9 GB）
- CoDeTT（51 GB，选型主分辨器）本机磁盘放不下，需外接盘或跳过
- 置信度限制：qwen 的 `confidence` 只落在 0.95/0.98，必须用 v3 的 `p_silent` 才能扫工作点
