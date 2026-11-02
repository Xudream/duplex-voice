# 尺子1 评测工装（公开集）

把公开集映射成统一的 S0–S4 任务清单，跑被测系统、统一判分。
数据不进仓库（体积大），只提交代码与清单生成逻辑。

## 目录

```
tools/ruler1/
├── build_manifests.py   # 公开集 → manifests/*.jsonl（S0–S4 映射）
├── fetch.py             # 从 hf-mirror 直连下载（绕开 hf CLI 的 metadata 校验问题）
├── baselines.py         # 平凡基线 all_silent / all_respond / oracle（定地板与天花板）
├── llm_judge.py         # 被测系统适配器：openai_compat / jev / dryrun 三种后端
├── score.py             # 分场景拒识率 / FAR_w / FRR / clarify / invalid / 误执行 + Wilson CI + 工作点扫描
└── stats.py             # McNemar、Bootstrap AUC 差、最小可分辨差
```

数据默认放在 `~/.hermes/workspace/ruler1/data/`（本地临时目录，不入库）：

```
data/
├── chorus/      BiggestWave/ChorusDuplex-SideTalk（中文 TTS，clean + babble）
├── wearvox/     zlinao/WearVox（英文眼镜，non-assistant-directed 1,082 条）
└── humdial/     ASLP-lab/HumDial-FDBench（中英真实，解压后）
```

## 用法

```bash
# 1) 下载（网络差时支持断点续传，脚本会跳过已完成文件）
python3 fetch.py BiggestWave/ChorusDuplex-SideTalk data/chorus --jobs 6
python3 fetch.py zlinao/WearVox data/wearvox --jobs 6
python3 fetch.py ASLP-lab/HumDial-FDBench data/humdial_dl --jobs 1
unzip -q data/humdial_dl/Humdial-Track2-Test.zip -d data/humdial && rm data/humdial_dl/*.zip

# 2) 生成清单
python3 build_manifests.py --dataset all --data ~/.hermes/workspace/ruler1/data --out manifests

# 3) 参照基线（定地板/天花板）
python3 baselines.py --manifest manifests/chorus.jsonl --mode silent  --out predictions/silent__chorus.jsonl
python3 baselines.py --manifest manifests/chorus.jsonl --mode respond --out predictions/respond__chorus.jsonl

# 4) 接被测模型（跑道 A：用集自带转写；--backend dryrun 用于链路自检）
python3 llm_judge.py --manifest manifests/chorus.jsonl --backend dryrun --limit 120 \
  --out predictions/dryrun__chorus.jsonl
DASHSCOPE_API_KEY=... python3 llm_judge.py --manifest manifests/chorus.jsonl \
  --backend openai_compat --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen-plus --out predictions/qwen-plus__chorus.jsonl

# 5) 判分（--sweep 出工作点曲线；--baseline-pred 出 McNemar；子集跑分加 --only-predicted）
python3 score.py --manifest manifests/chorus.jsonl \
  --pred predictions/silent__chorus.jsonl --sweep --report reports/silent__chorus.md
```

## 各集映射规则

| 集 | 规则 | 场景 |
|---|---|---|
| ChorusDuplex | `kind ∈ {backchannel, interruption}` → 仅回归（REG） | REG |
| ChorusDuplex | `addressee` 含 `assistant`，且说话人是主用户（该对话中对助手说话最多者） | S0 |
| ChorusDuplex | 主用户 → 旁人（addressee 不含 assistant） | S1 |
| ChorusDuplex | 旁人 → 主用户 | S2 |
| ChorusDuplex | 旁人 → 旁人 | S3 |
| WearVox | `ground_truth=[Active]` | S0 |
| WearVox | `ground_truth=[Mute]` 且 id 含 `unrelated_talk` | S1 |
| WearVox | `ground_truth=[Mute]` 且 id 含 `bystander_only` | S3 |
| HumDial | 目录/文件名含 `sdo` → S1；`tps` → S2；`urb`/`ph` → 仅回归 | 待解压后按实际标注确认 |

## 指标口径（与设计文档 §1.3 一致）

- 分场景拒识率 `Rej_s = 正确静默 / 应静默`
- `FAR_w = Σ w_s(1−Rej_s)`，w 默认 S1 0.40 / S2 0.15 / S3 0.25 / S4 0.20，缺场景自动归一
- `FRR = S0 被置静默的比例`（S0 应响应）
- `clarify` 单列，不计入拒识率分子、不计入 FAR
- `invalid/缺失` 单独计数，>1% 该模型不可用
- 误执行一票否决（预测里带 `tool_call: true` 且动作属于响应类）

## 已知边界

- 公开集没有音箱远场/手机近场的声学，也没有 DoA/近讲真值 → 尺子1 只能选"裁判"，不能选"前端"
- HumDial SDO 训练集为 0 条 → S1 的训练数据只能来自自建集
- CoDeTT（51 GB，选型主分辨器）本机磁盘放不下，需外接盘或跳过
