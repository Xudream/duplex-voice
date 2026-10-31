# sideTalk 拒识评测集构建方案

> 配套文档：《拒识sideTalk设计文档.md》（场景定义、模块边界、指标口径以其 §1.3 为准）
> 版本：v1.0　｜　更新：2026-10-31
> 用途：**模型选型（淘汰赛）+ 立项验收**两件事共用一套尺子
> 口径纪律：本文所有外部数字均标注来源；未标注者不得进入评审材料

---

## 1. 目的与边界

### 1.1 这份文档解决什么

《拒识sideTalk设计文档》定义了"要判什么"（S1–S4 归属问题）和"判到什么程度"（分场景目标），但没有回答三个工程问题：

| 问题 | 本文给出的答案 |
|---|---|
| 用哪些数据、各多少条？ | §2 资源清单 + §3 公开集映射 + §4 自建集配额 |
| 模型之间怎么比才可信？ | §6 统计口径（CI、McNemar、工作点曲线） |
| 谁来跑、跑出什么？ | §5 harness 规格 + §8 排期与交付物 |

### 1.2 两把尺子的分工

| 尺子 | 数据 | 用途 | 决策权 |
|---|---|---|---|
| 尺子 1（公开集） | HumDial-FDBench、CoDeTT、WearVox、ChorusDuplex | **淘汰赛**：把 8–10 个候选模型筛到 2 个 | 只做淘汰，不做最终定选 |
| 尺子 2（自建集） | 音箱远场 + 手机近场实测 | **定选 + 验收**：中文真实声学、S1 主场景 | 最终结论以尺子 2 为准 |

理由：公开集覆盖面广但形态不对（英文/眼镜/合成），自建集贴产品但规模小、拿不到业界坐标。两者互补，不可互相替代。

### 1.3 三条口径纪律

1. **dev/test 隔离**：任何 prompt、阈值、规则调参只能碰 dev；test 只在最终表报一次，触碰即该版本 test 作废。
2. **只报分布不报单点**：所有指标给点值 + 95% 置信区间；差异小于 CI 宽度时不写"优于"。
3. **一票否决不参与加权**：误执行（应静默却触发工具）恒为 0，不做加权平均，出现即淘汰。

---

## 2. 评测集总览

### 2.1 公开集清单

| 集 | 语言 | 真实/合成 | 可用条数（进 sideTalk 指标） | 获取 | 许可 |
|---|---|---|---|---|---|
| HumDial-FDBench（Track II） | 中/英双语 | 真实录制（双通道） | 拒识 test 2,000（SDO 200 / TPS 600）+ dev 800 | HuggingFace `ASLP-lab/HumDial-FDBench`，仓库含 `Humdial-Track2-Test.zip` | Apache-2.0 |
| CoDeTT | 中/英 | 真实+合成 | SideTalk 子集约 2,000（待解压核对） | HuggingFace `YingaoWang-casia/CoDeTT`（单包 `CoDeTT.lz4`） | 见仓库 |
| WearVox | 英 | 真实（AI 眼镜，多通道） | 582 负例 + 500 正例 | HuggingFace `zlinao/WearVox` | 见仓库 |
| ChorusDuplex-SideTalk | 中文 | 合成（TTS，clean + babble 双版） | side_talk 134 轮 + third_party 轮 | HuggingFace `BiggestWave/ChorusDuplex-SideTalk`（CC BY 4.0，babble 版需保留 VOiCES 归属） | CC BY 4.0 |
| VGBench | 英 | 合成+受控 | **0（数据未公开）** | 仅论文 arXiv:2609.32536 | — |
| SLT 2026 SmartGlasses | 中文 | 真实（四通道） | Dev 抽样 50 段（仅诊断） | 注册表单 → 邮件发放 | 挑战赛协议 |

### 2.2 合计规模与统计功效

| 用途 | 样本量 | 指标假设 | 95% CI 半宽 | 能分辨的最小模型差异 |
|---|---|---|---|---|
| CoDeTT SideTalk（选型主分辨器） | ~2,000 | p=0.90 | ±1.3 pt | ~3 pt |
| HumDial 拒识 test（对外坐标） | 2,000（SDO 仅 200 / TPS 600） | p=0.90 | SDO ±4.2 pt / TPS ±2.4 pt | SDO 上 ~8 pt |
| WearVox（眼镜参考） | 1,082 | p=0.90 | ±1.8 pt | ~4 pt |
| ChorusDuplex（CI 冒烟） | 134 | p=0.90 | ±5.1 pt | 只能抓大退化 |
| 自建集 v1.0 S1 | ≥600 | p=0.90 | ±2.4 pt | ~5 pt |
| 自建集 S0（FRR） | ≥600 | p=0.03 | ±1.4 pt | 只能判"≤3%" |

> 结论：**HumDial 的 SDO 子集（200 条）不能用来选型**——两个模型差 4 个点在统计上就是噪声。它只做对外坐标与回归；选型分辨器必须是 CoDeTT + 自建集。

### 2.3 一个必须先知道的事实：SDO 没有训练数据

HumDial-FDBench README 的分布表（Train / Dev / Test）：

| 子类 | Train | Dev | Test |
|---|---|---|---|
| URB（用户附和） | 1,211 | 200 | 600 |
| PH（停顿处理） | 1,211 | 200 | 600 |
| TPS（第三方说话） | 120 | 200 | 600 |
| **SDO（对他人说，= S1）** | **0** | 200 | 200 |

影响三条：

1. 想在 S1 上达标，**不存在可从公开集直接取用的训练样本**，只能靠自建集或跨集迁移
2. TPS 训练样本也只有 120 条，跨集迁移的价值有限
3. 自建集不是"补强项"，而是 **S1 的唯一训练来源**——排期必须按训练+评测双用途规划

---

## 3. 尺子 1：公开集构建

### 3.1 获取与解压

```bash
# 1) HumDial-FDBench（test 包）
huggingface-cli download ASLP-lab/HumDial-FDBench --repo-type dataset --local-dir data/humdial
unzip data/humdial/Humdial-Track2-Test.zip -d data/humdial/test

# 2) CoDeTT（单包 lz4）
huggingface-cli download YingaoWang-casia/CoDeTT --repo-type dataset --local-dir data/codett
lz4 -dc data/codett/CoDeTT.lz4 | tar -x -C data/codett/

# 3) WearVox
huggingface-cli download zlinao/WearVox --repo-type dataset --local-dir data/wearvox

# 4) ChorusDuplex-SideTalk
huggingface-cli download BiggestWave/ChorusDuplex-SideTalk --repo-type dataset --local-dir data/chorus
```

解压后第一件事是**核对清单**（写进 `data/MANIFEST.sha256`）：文件数、每条样本的音频可读性、标注字段齐备性。缺一条就停，不要带着半份数据开跑。

### 3.2 子集抽取与 S0–S4 映射

| 集 | 取哪些 | 字段/规则 | 映射 | 进指标 |
|---|---|---|---|---|
| HumDial | test 的 Rejection 四子类 | 按标注 `rejection_type` 分桶 | SDO→S1；TPS→S2/S3；URB/PH→仅回归 | 2,000（S1 200 / S2-S3 600） |
| HumDial | test 的 Interruption 五子类 | 同上 | 不进 sideTalk 指标，仅回归 | 0（回归 3,000） |
| HumDial | dev（800） | 同上 | 调参专用，不报数 | 0 |
| CoDeTT | `SystemIdle`→`Dismissal`→`SideTalk`、`SystemSpeaking`→`Maintain`→`SideTalk` | 目录路径即标签 | →S1 | ~2,000（待核对） |
| CoDeTT | 其余 12 类意图 | — | 仅回归 | 0 |
| WearVox | Side-Talk Rejection 任务 | `audio_query`（单通道）/ `audio_query_mc`（多通道） | 582 side-talk→S2/S3；500 valid query→S0 | 1,082 |
| ChorusDuplex | 逐轮 `kind` / `speaker` | `annotations.jsonl` | `kind=side_talk`→S1；`speaker=third_party`→S2/S3；assistant 轮→S0；backchannel/interruption→仅回归 | 134 + 若干 |

### 3.3 五个已知的坑

| # | 坑 | 后果 | 处理 |
|---|---|---|---|
| 1 | HumDial test 的 SDO 只有 200 条 | 无法区分 <4pt 的模型差异 | 只做坐标；选型靠 CoDeTT + 自建 |
| 2 | SDO/TPS 训练样本为 0/120 | 不能拿公开集训练 S1 判别器 | 自建集承担训练；公开集只评测 |
| 3 | WearVox 的 1,082 里有 500 条是从 tool-calling 复制的正例 | 正负比例失真、同一音频多任务复用 | 报数时分开统计；不要把这 500 当独立正例做 FRR 主结论 |
| 4 | ChorusDuplex 是 TTS 合成、只有 134 条 side_talk | 只能做链路冒烟，不能当验收 | CI 必跑、不作为定选依据 |
| 5 | CoDeTT 子集条数未在 README 给出（仅给 18,000 总量 / 14 类） | ±1.3pt 的精度声明可能不成立 | 解压后核对；若为 1,000 条则改为 ±1.8pt、"可分辨 ~4pt" |

### 3.4 输入构造三要素（跨集统一，缺一条结论就不可比）

| 要素 | 规格 | 各集取值 |
|---|---|---|
| 音频 | 按各集原始声道布局，48kHz 或原采样率，不做重采样以外的处理 | HumDial 双通道；Chorus 立体声 L=非助手/R=助手；WearVox 单通道版与 mc 版各跑一遍 |
| 系统态 | `idle` / `speaking` 二值，显式给定 | HumDial、CoDeTT 由标注给；Chorus 由 `trigger` 推 |
| 多轮历史 | 最近 5 轮转写，纯文本 | 各集自带标注；不足补空；**不得含 gold 的 addressee 标签** |

### 3.5 判分规则

```
输入：模型输出 JSON {"action":..., "confidence":..., "reason":...}
规则优先级：
1. 解析失败 / 动作不在允许集合 → 记 invalid，单独计数（>1% 视为该模型不可用于产品）
2. action = silent | silent_record → 判"静默"
3. action = clarify → 记"非静默"，单列，不计入拒识率分子、不计入 FAR
4. action = respond | ack | wait → 判"响应"（应静默样本即算误响应；S0 样本算正确）
5. 应静默样本上 action 触发工具调用 → 记"误执行"（一票否决）
```

公开集的 gold 只有"该静默 / 该响应 + 理由类别"，因此：

- CoDeTT 四动作映射：`Dismiss`→silent、`Takeover`→respond、`Stop&Listen`/`Maintain`→仅回归（语义 VAD 域）
- clarify 在 CoDeTT 上**不计入 4 动作 ACC**，单列 clarify 率

### 3.6 各集门槛与 sanity check

| 集 | 必须先复现的基线 | 允许偏差 | 复现失败怎么办 |
|---|---|---|---|
| HumDial | 官方 Baseline 拒识 35.2、冠军 72.2 | ±3 pt | 判 harness bug（时序对齐/RS 计算），修完再跑 |
| WearVox | 两段式基线 85.6%（单通道） | ±3 pt | 检查声道选择与重采样 |
| CoDeTT | 仓库给出/可复现的四动作 baseline | ±3 pt | 检查系统态与 5 轮历史拼装 |
| ChorusDuplex | 无官方基线 | — | 用规则基线（全 respond）与全 silent 定位下限/上限 |

---

## 4. 尺子 2：自建集构建

### 4.1 场景矩阵

| 维度 | 音箱（远场） | 手机（近场） |
|---|---|---|
| 距离 | 1 / 2 / 4 m | 手持 0.2 m / 桌放 0.5 m / 车载 0.6 m |
| 说话人 | 1–4 人（家庭多成员） | 主用户 1 人 + 0–2 旁人 |
| 朝向 | 对设备 / 对家人 / 背对 | 对屏幕 / 对旁人 / 侧头 |
| 噪声 | 电视开 / 音乐外放 / 厨房 / 安静 | 街道 / 车内 / 办公室 / 外放短视频 |
| 语言 | 普通话 + 15% 中英混说 | 同左 |

### 4.2 规模与每格配额（v1.0，3,000 段）

| 场景 | 音箱 | 手机 | 合计 | 统计理由 |
|---|---|---|---|---|
| S0 真指令/答句 | 400 | 300 | **700** | FRR 需 ≥600 才能给 ±1.4pt（p=3%） |
| S1 主用户→旁人 | 400 | 300 | **700** | 核心场景，且是 S1 唯一训练来源，需 ≥600 |
| S2 旁人→主用户 | 250 | 150 | 400 | p=0.95 → ±1.7pt |
| S3 旁人→旁人 | 250 | 150 | 400 | 同上 |
| S4 媒体/电子人声 | 250 | 150 | 400 | 同上 |
| self-talk（单列） | 100 | 100 | 200 | 只观测基线，不设门槛 |
| speaker-switch 对抗 | 100 | 100 | 200 | 同上 |
| **合计** | **1,750** | **1,250** | **3,000** | — |

两种设备**分开出指标**，不混池；S1 的 700 段里，至少 200 段含"帮我/给我"等请求句式（对抗子集）。

### 4.3 采集 SOP

1. 演员照脚本演（对齐 HumDial 的"LLM 脚本 + 演员表演"），打断/重叠发生在语义接缝（迟疑、换气）而非随机时间戳
2. side-talk 脚本**不得**含唤醒词；对抗子集单独成轨，故意含"帮我/小X"但交流对象明确是旁人
3. 每段 10–30 s，同步记录：SNR、混响估计（RT60）、DoA 真值、距离真值、设备型号与摆位
4. 同一场景至少 3 个 take，事后按自然度挑选，未选中 take 保留为"边缘样本"备查
5. 现场即校验：波形不削波、双通道无串音、标注文件与音频文件名一一对应

### 4.4 标注 schema 与一致性

| 层级 | 字段 |
|---|---|
| 段级 | `speaker_role`（primary/other/media）、`directed_to`（assistant/companion/none）、`gold_action`（respond/ack/wait/clarify/silent/silent_record）、`scene_tags[]`、`is_adversarial` |
| 轮级 | 词级时间戳、转写、意图标签、与上一系统轮的衔接类型（答句/打断/无关/附和） |

- 双人独立标注 + 第三人仲裁；**Cohen's κ ≥ 0.8 才入库**
- 每版本抽 10% 复标，κ 掉到 0.7 以下整批返工
- 标注指引单独成文（含 20 条边界样例），随版本号管理

### 4.5 切分与防泄漏

| 措施 | 做法 |
|---|---|
| 说话人隔离 | train/dev/test 三集说话人不重叠（按人切，不按段切） |
| 场景隔离 | 同一房间/同一噪声组合不出现在两个 split 的同一场景里 |
| 去重 | 音频指纹（chromaprint）查重，同源脚本的不同 take 不跨 split |
| 防泄漏 | 脚本文本、录音日期、噪声源名不出现在 prompt 中；评估时不向模型透露"这是测试" |
| 训练用途 | 只有 train 可用于训练 L2 判别器；test 只评不训 |

### 4.6 成本与排期【估算】

| 项 | 数量 | 说明 |
|---|---|---|
| 有效音频 | 3,000 × 20 s ≈ 16.7 h | 拍摄比按 1:3，现场约 50 h |
| 录音 | 6 个工作日 | 2 演员 + 1 录音师 + 2 种设备 |
| 标注 | 约 750 人时（≈94 人日） | 3,000 段 × 双标 + 仲裁 |
| 合计 | 6–8 周 | 与公开集评测并行，不串行 |

---

## 5. 统一评测协议（harness）

### 5.1 目录结构

```
ruler1/
├── data/                    # 各集原始数据（不入 git）
├── manifests/
│   ├── humdial_test.jsonl
│   ├── codett_sidetalk.jsonl
│   ├── wearvox_str.jsonl
│   └── chorus.jsonl
├── prompts/                 # 冻结的提示词，含版本号
├── runners/                 # 各模型适配器（jev / hf / api / e2e）
├── predictions/             # <model>__<manifest>__<runhash>.jsonl
├── score.py
└── reports/                 # 生成的表与图
```

### 5.2 manifest 字段

```json
{"id":"humdial-test-SDO-000123","dataset":"humdial","split":"test",
 "audio":"data/humdial/test/xxx.wav","channels":2,
 "system_state":"speaking","history":[{"role":"user","text":"..."}],
 "gold_action":"silent","gold_scene":"S1","gold_lang":"zh",
 "meta":{"rejection_type":"SDO","snr_db":12.5,"source":"real"}}
```

### 5.3 predictions 字段

```json
{"id":"humdial-test-SDO-000123","model":"qwen3-0.6b-npu","run":"2026-11-03T10:00Z",
 "action":"silent","confidence":0.87,"reason":"...","latency_ms":42,
 "raw":"<原始输出，便于回溯>"}
```

### 5.4 解析与容错

- 动作不在白名单 → `invalid`（单独计数，>1% 该模型直接出局）
- 缺 `confidence` → 按 1.0 计，但记 `warn_missing_conf`
- 分类模型（非生成式）输出概率向量 → 取 argmax，`confidence` 取该概率

### 5.5 score.py 输出

```
reports/<model>__<manifest>.md
  ├─ 分场景表：S1/S2/S3/S4 拒识率（点值 + 95%CI）
  ├─ 汇总：FAR_w、FRR、误执行、clarify 率、invalid 率
  ├─ 延迟：P50/P90（决策延迟、响应首延迟）
  └─ 错误归因：ACC 错 / 应静默却 respond / 蒙对但 SMR 命中
```

---

## 6. 指标与统计

### 6.1 指标定义

| 指标 | 公式 | 备注 |
|---|---|---|
| 分场景拒识率 | `Rej_s = 正确静默数_s / 应静默数_s` | 按 S1–S4 分别出 |
| 加权误响应率 | `FAR_w = Σ w_s × (1 − Rej_s)` | w 按产品流量占比（起始建议 S1 0.40 / S2 0.15 / S3 0.25 / S4 0.20）；**不设独立常数门槛** |
| 漏响应率 | `FRR = S0 被误吞数 / S0 总数` | 承诺 ≤3%，内部目标 2% 待 S0≥3,000 |
| 误执行 | 计数 | 一票否决，报 `0/N` 与 95% 上界 `≤3/N` |
| clarify 率 | `clarify 次数 / 总样本` | 监控 ≤2%，不计入 FAR |
| 决策延迟 | 语音结束 → silent 落定 | P50/P90，窗 ≤500 ms |
| 响应首延迟增量 | 含裁判 vs 不含裁判 | P90 增量 ≤100 ms |

### 6.2 工作点而不是单点

对每个模型扫 `confidence` 阈值 0.05–0.95（步长 0.05），报三条：

1. 固定 `FAR_w = 5%` 处的 FRR
2. 固定 `FRR = 3%` 处的 FAR_w
3. FAR–FRR 曲线的 AUC（AUC 越大越好，可用梯形法）

只报"拒识率 92%"是无意义的——不同阈值下可以差 15 个点。

### 6.3 置信区间与显著性

| 场景 | 方法 |
|---|---|
| 单模型单指标 | Wilson 区间（不用正态近似，p 接近 0/1 时更稳） |
| 两模型同批样本 | McNemar 检验（配对，只看不一致样本） |
| AUC/曲线差异 | Bootstrap 1,000 次，报差值的 95%CI |
| 多模型比较 | Holm 校正，避免多重比较假阳性 |

### 6.4 报数纪律

- 差异 < CI 宽度 → 写"不可区分"，不写"略优"
- 每次结果必须带：模型版本、prompt 版本、manifest hash、运行时间
- 同一张表里不允许混用不同 split 的数据

---

## 7. 模型选型打分卡

### 7.1 硬淘汰线（任一触发即出局）

| 条件 | 阈值 |
|---|---|
| 误执行 | > 0 |
| invalid 输出率 | > 1% |
| S1 拒识率 | < 80% |
| 决策延迟 P90 | > 800 ms |
| 部署约束 | 需出境且无替代方案 |

### 7.2 加权打分（通过硬淘汰后）

| 维度 | 权重 | 计分方式 |
|---|---|---|
| S1 拒识率（CoDeTT + 自建 dev） | 30 | 线性映射，80%→0 分，95%→满分 |
| FAR_w @ FRR≤3% | 25 | 越低越好，5%→满分，15%→0 分 |
| 误执行余量 | 15 | 0 次满分，出现即淘汰（本项用于排名区分） |
| 决策延迟 P90 | 10 | ≤300ms 满分，800ms→0 分 |
| 可部署性（端侧/私有化） | 10 | 端侧满分，私有化 7，仅 API 3 |
| 成本（每千次决策） | 10 | ≤¥1 满分，¥10→0 分 |

### 7.3 两条跑道与对照

| 跑道 | 输入 | 候选 | 作用 |
|---|---|---|---|
| A 文本裁判 | 转写 + 系统态 + 5 轮历史 | Qwen3-0.6B/1.7B/4B、Qwen3.5-4B/9B、DeepSeek-V4-Flash、Jev（L3 快分支）、**规则+BERT 分类器（必跑基线）** | 主选型跑道 |
| B 端到端 | 原始音频 | Qwen-Audio-3.1-Realtime、Gander、Moshi | 给出"模型自带拒识"上界 |

A 跑道必须跑两版：**gold 转写** 与 **真实 ASR 转写**。两版差值 = ASR 误差传导，这是选型结论里最有价值的一列（决定"值不值得为 ASR 升级付费"）。

---

## 8. 排期与交付物

### 8.1 十二天计划

| 天 | 任务 | 产出 | Go/No-Go |
|---|---|---|---|
| D1 | harness 骨架 + 4 个 manifest 生成器 | 能跑通一条假样本 | 解析链路无报错 |
| D2 | Chorus + WearVox 跑通 | 首张表 + sanity check | 基线复现偏差 ≤3pt |
| D3 | CoDeTT 解压、子集抽取 | SideTalk 子集条数确认 | 条数 ≥1,500，否则降级为诊断集 |
| D4 | CoDeTT 全量跑（2,000） | 主分辨表初版 | — |
| D5 | HumDial test 解压 + 四子类映射 | 对外坐标表 | SDO=200 / TPS=600 核对一致 |
| D6 | HumDial dev（800）调参 | prompt/阈值冻结版 v1 | dev 上 S1 ≥85% |
| D7 | test 一次性评估 | 定稿表（test 只此一次） | 与 dev 差距 ≤5pt |
| D8 | 跑道 B（端到端 3 个） | 上界对照 | — |
| D9 | 统计（CI / McNemar / 工作点） | 曲线图 + 显著性表 | — |
| D10 | 打分卡 + 提名 | 2 个模型进尺子 2 | — |
| D11–D12 | 尺子 2 先导批（每类 50 条） | 穿刺报告 | S1 ≥85% 且误执行 0 |

### 8.2 交付物

1. `reports/选型总表.md`：8–10 个模型 × 6 个指标 + CI + 排名
2. `reports/工作点曲线.png`：FAR–FRR 曲线（每模型一条）
3. `reports/错误归因.md`：三桶错误（没识别 / 不肯闭嘴 / 蒙对理由错）分布
4. `manifests/`：可复现的样本清单（含 hash）
5. `predictions/`：全部原始输出（含 latency）
6. 一页结论：两个提名模型 + 淘汰原因 + 下一步

### 8.3 成本

| 项 | 估算 |
|---|---|
| API 调用（Jev + LLM，约 5,500 样本 × 3 遍） | < ¥50 |
| 本地推理（Qwen3 系列，单卡） | 约 20 机时 |
| 人力 | 1 人 × 12 天 |

---

## 9. 风险与回退

### 9.1 数据侧风险

| 风险 | 影响 | 回退 |
|---|---|---|
| CoDeTT SideTalk 实际条数远少于 2,000 | 选型分辨力下降 | 用"CoDeTT + Chorus + 自建先导批"合并判定；把差异门槛放宽到 5pt |
| HumDial test 包标签缺失/语言分桶不可用 | 拿不到对外坐标 | 邮件申请（wangshuiyuan@mail.nwpu.edu.cn）；期间只报内部指标，不宣称对标 |
| SDO 无训练数据导致 S1 上不去 | 选型结论失真 | L2 判别器只用规则+声纹先跑；S1 的提升全部押在自建 train 上 |
| 自建集采集延期 | 验收无数据 | 先用公开集 + 线上 badcase 回流做灰度，不承诺验收 |

### 9.2 方法侧风险

| 风险 | 影响 | 回退 |
|---|---|---|
| 判分器版本漂移 | 跨版本不可比 | 判分 prompt + 模型 id 入仓，写进报告头 |
| 过拟合公开集 | test 数字虚高 | dev/test 隔离 + 每季度换一批自建 test |
| 阈值调优引入乐观偏差 | 工作点曲线虚高 | 阈值只在 dev 上选，test 只验证 |

---

## 10. 附录

### 10.1 各集字段速查

| 集 | 关键字段 | 备注 |
|---|---|---|
| HumDial | `rejection_type`（URB/PH/TPS/SDO）、`lang`、双通道音频 | dev/test 分目录 |
| CoDeTT | 目录路径即动作与意图；jsonl 含系统态与 5 轮历史 | 单包解压 |
| WearVox | `audio_query`、`audio_query_mc`、`task` | S-T Rejection 取 1,082 |
| ChorusDuplex | `annotations.jsonl`（`kind`/`speaker`/`trigger`）、`*.timeline.json` | clean 与 `_babble` 各 100 |

### 10.2 常用命令

```bash
# 看 manifest 条数与场景分布
python3 -c "import json,collections;rows=[json.loads(l) for l in open('manifests/humdial_test.jsonl')];\
print(len(rows));print(collections.Counter(r['gold_scene'] for r in rows))"

# 跑一个模型（示例：本地 vLLM）
python3 runners/hf_runner.py --manifest manifests/codett_sidetalk.jsonl \
  --model Qwen/Qwen3-1.7B --out predictions/qwen3-1.7b__codett__$(date +%s).jsonl

# 打分
python3 score.py --pred predictions/qwen3-1.7b__codett__*.jsonl \
  --manifest manifests/codett_sidetalk.jsonl --report reports/qwen3-1.7b__codett.md
```

### 10.3 指标公式汇总

```
Rej_s      = 正确静默_s / 应静默_s
FAR_w      = Σ_s w_s (1 − Rej_s)
FRR        = 误吞数_S0 / 总数_S0
clarify率  = clarify数 / 总样本
决策延迟    = t(decision) − t(speech_end)          # P50/P90
CI 半宽     = z·√(p(1−p)/n),  z=1.96
最小可分辨差 ≈ 2 × CI 半宽
```

### 10.4 数据合规

| 集 | 许可 | 义务 |
|---|---|---|
| HumDial-FDBench | Apache-2.0 | 保留声明 |
| ChorusDuplex-SideTalk | CC BY 4.0 | babble 变体再分发须保留 VOiCES 出处 |
| WearVox | 见仓库 | 眼镜第一视角，勿用于人脸/身份推断 |
| 自建集 | 内部 | 声纹与旁听缓存按隐私策略留存，标注含个人信息需脱敏 |

---

*本文与《拒识sideTalk设计文档》同源；指标口径冲突时以设计文档 §1.3 为准，选型流程以本文为准。*
