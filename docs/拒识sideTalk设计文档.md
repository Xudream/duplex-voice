# 语音 sideTalk 拒识 · 设计文档

---

## 1. 拒识的场景分析

### 1.1 HumDial 口径

ICASSP 2026 HumDial Challenge 把 Track II 的拒识拆成四类：

| 缩写 | 全称 | 中文 | 系统应有动作 |
|---|---|---|---|
| URB | user real-time backchannels | 用户实时附和 | 静默继续（不打断） |
| PH | pause handling | 停顿处理 | 等待，不抢话 |
| TPS | third-party speech | 第三方在说话 | 静默 |
| SDO | speech directed at others | 对他人说的话（sidetalk） | 静默 + 记上下文 |

sidetalk ≈ SDO（主用户转向旁人），TPS 是它的相邻场景（背景里别人在聊，主用户没参与）。

**来源**：[HumDial Challenge 官网](https://aslp-lab.github.io/HumDial-Challenge/) · [论文 arXiv:2601.05564](https://arxiv.org/html/2601.05564v2) · [GitHub ASLP-lab/Hum-Dial](https://github.com/ASLP-lab/Hum-Dial)。论文原文口径：Rejection 考察 "robustness against non-instructional speech such as user real-time backchannels (URB), pause handling (PH), third-party speech (TPS), and speech directed at others (SDO)"；Track II 总分 = 0.4×Interruption + 0.4×Rejection + 0.2×First Response Delay，拒识相关指标为 Rejection Rate、Early Interrupt Rate。主办方含西北工业大学、南大、清华、剑桥、华为、AISHELL 等；数据为 LLM 生成脚本（DeepSeek）+ 专业演员真实录制。**测试集构成（论文 Table 1，与 FDBench GitHub 仓库 README 分布表一致；HF 数据集页 README 为空、仅 license 行）**：拒识 2,000 = URB 600 / PH 600 / TPS 600 / SDO 200，打断 3,000；**SDO 测试仅 200 条**（Wilson 95%CI ≈ ±4.2pt，子类差异 <4pt 统计不可区分，解读时注意）；SDO 定义（FDBench GitHub 仓库 README 原文）："the user temporarily addresses another person, often on an unrelated topic, and the system is expected to detect and ignore such speech"。**语言口径（已定论）**：Track II 为中英双语——官方 Track II 榜单将 Interruption / Rejection 分 CN、EN 两列分别计分，双语结构由榜单本身确证（解压测试包后按语言分桶即可）。

### 1.2 VGBench 口径

| 类型 | 定义 | 例子 | 文本层是否可分 |
|---|---|---|---|
| side-talk | 主用户对旁人说话 | "帮我把这个放桌上"（对同事） | 不可分（与指令同形） |
| self-talk | 自言自语 | "嗯……这个放哪来着" | 部分可分（无请求对象） |
| speaker-switch | 同一句话换说话人/距离/时间边界 | 同一句指令由旁人说 vs 穿戴者说 | 完全不可分 |

speaker-switch 是三者中最难的：词不变、只有来源变，任何纯文本模型都无能为力。

> 口径提示：三分类同属"该不该响应"（addressedness）问题家族；self-talk 与 side-talk 的区分维度是**有无交流对象**。注意：VGBench 三分类 ≠ 本模块支持范围——side-talk 对应本模块 S1，self-talk 不进本模块 v1 支持清单（判定复用本模块裁判、评测单列），speaker-switch 由身份层信号保障，详见 1.3.0/1.3.1。

**来源**：[VGBench 论文 arXiv:2609.32536](https://arxiv.org/pdf/2609.32536)——"a 1,018-item diagnostic benchmark for action-level addressedness across side-talk, self-talk, and speaker-switch scenarios"（1,018 条动作级 addressedness 诊断集）。

### 1.3 我们对 sideTalk 场景的定义

#### 1.3.0 总定义与模块边界

**产品形态与场景范围**：本方案的主场景为**家庭智能音箱**（固定放置、远场拾音、家庭多成员环境）与**手持设备**（手机，近场为主、随人移动、常伴随外放媒体）；**智能眼镜为补充场景**（v2 预留），其多通道空间信号与结论仅作参考，不计入 v1 验收。

**定义**：sideTalk 指"语音在拾音范围内，但不是对系统说的"。sideTalk 拒识模块的职责是：对这类语音**保持静默、不产生任何响应**，同时**不破坏正在进行的人机对话状态**（不终止会话、不打断当前播报、不吞掉随后的真实指令）。

**模块边界**：本模块解决的是**归属问题**（谁在说、对谁说），不解决**时机问题**（该不该现在开口）。因此：

| 问题 | 举例 | 归属模块 |
|---|---|---|
| 时机问题：这话该不该现在响应？ | 用户附和（"嗯/好的"）、用户停顿思考（"那个……"） | **语义 VAD / 轮次决策模块**（本模块不涉及） |
| 归属问题的特殊子类：指向无人 | 自言自语（self-talk，无交流对象） | 复用本模块裁判、评测单列（v1 场景清单外，见下） |
| 归属问题：这话是不是对我说的？ | 主用户转向旁人、旁人闲聊、媒体人声 | **sideTalk 拒识模块（本文档）** |

对齐公开口径：HumDial 四类中，**SDO、TPS 属本模块**，URB、PH 属语义 VAD；VGBench 三类中，**side-talk 属本模块**，self-talk（自言自语）在业界是与 side-talk **并列的独立 addressedness 子场景**（VGBench 三分类、DUPLEX-MPE 负例 N3 均将其单列，区分维度是"有无交流对象"），不属于本模块 v1 支持清单，但其判定**复用本模块裁判**（见 1.3.3），评测单列；speaker-switch 靠本模块的身份层信号（声纹/DoA）保障。

#### 1.3.1 场景定义与清单

sideTalk 模块支持 **4 个场景（S1–S4）**，另设 1 个**对照场景（S0）**——S0 不属于本模块，列出它是为了度量拒识的代价：

| 编号 | 场景 | 说话人 | 交流对象 | 典型样例 | 系统动作 |
|---|---|---|---|---|---|
| S0 | 真指令 / 答句（对照场景） | 主用户 | 系统 | "今天天气怎么样" | 响应（正常链路；本模块不得误吞） |
| S1 | **主用户转向旁人（SDO，核心场景）** | 主用户 | 第三方 | 在家对家人说"帮我把酱油递一下"；办公室对同事说"帮我把这个放桌上" | 静默 + 记上下文 |
| S2 | 旁人对主用户说话（B1） | 旁人 | 主用户 | "你手机响没响？" | 静默（可选摘要级记录） |
| S3 | 旁人之间的对话（B2） | 旁人 | 旁人 | 家人之间闲聊（音箱场景）、咖啡馆邻桌（手机场景） | 静默 |
| S4 | 媒体 / 电子人声（B3） | 电子源 | — | 电视台词"把枪放下"（音箱场景高频）、手机外放短视频/语音消息 | 静默，防指令式台词带偏 |

**核心难点在 S1**：它与 S0 **文本同形**——"帮我把这个放桌上"既可能是对助手下指令，也可能是对旁人说话，纯文本不可分，必须依赖带外信号（见 1.3.2）。S1 的语音进入旁听缓存（带时间戳与说话人标签），**默认不参与回答**，仅当用户显式引用时才可使用——这是隐私与体验的双重边界。

**范围外（明确不做，防止口径漂移）**：

- 附和（URB）、停顿（PH）：时机问题 → 语义 VAD；
- 自言自语（self-talk）：主用户、无交流对象——业界（VGBench、DUPLEX-MPE）将其列为与 side-talk 并列的独立 addressedness 子场景（非时机问题），判定复用本模块裁判，但 v1 不列入支持场景清单，评测单列（自建集覆盖）；
- 访客指令（旁人对设备说"小X，放首歌"）：涉及声纹白名单与隐私策略 → v1 不做，留开关；
- speaker-switch 作为独立类型不单列：其判定 = "旁人说出的指令"，由 S2–S4 的身份层信号覆盖（旁人**对设备**下指令的访客指令除外，见上条），并在自建对抗集中强化测试（4.4 阶段 2）。

#### 1.3.2 判定信号（四层，供 4.3 裁判使用）

| 层 | 信号 | 强度 | 对哪个场景关键 |
|---|---|---|---|
| ① 说话人身份层 | 主用户声纹（闭集判定"是否机主"）；距离/朝向声学特征——音箱：环形麦阵 DoA + 远场波束（区分室内谁在说话、是否对设备）；手机：近讲检测（近场能量比、持握姿态对麦克风的声道差） | 最强，文本无关 | S2/S3/S4 的主判据；S1 的第一步分流 |
| ② 语义层 | 请求句式（祈使句、"帮我/给我"）、称谓（唤醒词/助手名）、与当前对话的衔接度（承接系统上一句 → 大概率答句） | 中，可被同形文本欺骗 | 仅对 S1 关键（S1 与 S0 同形，靠语义做最后分辨） |
| ③ 时序层 | 出现在系统提问之后（答句）vs 播报之中 vs 与当前对话无关 | 中 | S1 的辅助判据 |
| ④ 多模态层（可选） | 音箱：设备态（是否正被使用、媒体是否外放中）、房间活动感知；手机：屏幕朝向/握持态、前台应用态（导航/通话中默认更保守）、是否外放中。眼镜（补充场景）另有注视/朝向可用 | 增强 | S1 与 S4（外放中 → 媒体人声概率大增）最有用 |

#### 1.3.3 决策逻辑

本模块对每段语音按两步判定：

**第一步：说话人是谁**（身份层，文本无关）

- 是主用户（声纹命中） → 进入第二步；
- 不是主用户 → 是媒体源吗？（声学事件检测/设备状态）是 → **S4 静默**；否 → 与主用户有交流关系吗？（朝向、距离、语义衔接）有 → **S2 静默**；无 → **S3 静默**。

**第二步：主用户的话指向谁**（语义层 + 时序层 + 多模态层）

- 指向系统（承接系统提问、含唤醒词/请求句式、朝向设备） → 放行给正常链路（S0）；
- 指向第三方（朝向旁人、交流对象是具体的人、与当前对话无关） → **S1 静默 + 记上下文**；
- 指向无人（无交流对象、无话题衔接） → self-talk：动作同为静默（**不记上下文**）——判定复用本模块裁判，但不属 v1 支持清单（见 1.3.1 范围外），评测单列；
- 无法判定（低置信度） → 按语义信号分两路，避免一刀切静默吞掉真指令：
  - 含请求句式或承接系统上一句 → **clarify 轻确认**（如"您是说把空调调到 26 度吗？"——出声但不执行、不计为误响应；单列 clarify 率监控，目标 ≤ 2%）；
  - 否则 → **静默 + 记上下文**（宁可少说，不可说错；旁听缓存保留后续引用的可能）。

**与相邻模块的边界**：

| 相邻场景 | 判别特征 | 归属 |
|---|---|---|
| 附和（URB）/ 停顿（PH） | 主用户、短促反馈或话轮未完 | 语义 VAD / 轮次决策模块 |
| 自言自语（self-talk） | 主用户、无交流对象 | **独立 addressedness 子场景**：复用本模块裁判（第二步"指向无人"分支），v1 场景清单外、评测单列 |
| 真实打断（barge-in） | 主用户、有请求、发生在系统播报中 | 主链路（sideTalk 的相反面：明确接受） |
| 唤醒词出现 | 任何说话人说出唤醒词/助手名 | v1 仅主用户声纹唤醒生效；对 S3/S4 是兜底规则 |

#### 1.3.4 指标定义（与 4.2 对齐）

拒识类指标（拒识率 / FAR / 误执行）只在 sideTalk 场景（S1–S4）的样本上统计；FRR 在 S0（真指令）样本上统计：

- **拒识率**（Rejection Rate）：应静默样本中正确静默的比例，**按 S1–S4 分别统计**；
- **误响应率**（FAR）：应静默样本上产生**完整响应或执行工具**的比例，全场景加权聚合 **FAR = Σ w_s × (1 − Rej_s)**（w_s 按产品真实流量占比，看板按实算）——**它是分场景拒识率的联动视图，不设独立常数门槛**（否则与分场景目标可能数学上互斥）；clarify 轻确认不计入 FAR，单列监控；
- **误执行**：应静默样本上触发工具调用等高危动作的次数——**一票否决项**，目标恒为 0，不设阈值；
- **漏响应率**（FRR）：真指令（S0）被本模块误吞的比例——拒识的代价项，与语义 VAD 模块共同承担；**承诺值 ≤ 3%**（当前 S0 可用样本 n≈700–800 时 Wilson 95% 半宽约 ±1.0pt，3% 是可统计证明的上界；≤ 2% 为内部目标，待尺子2 S0 扩充至 ≥ 3,000 段后生效）；
- **clarify 率**：clarify 动作发生的频率（监控指标，目标 ≤ 2%；clarify 出声但不执行，用户可感知但无害）；
- **静默决策延迟**：从应静默语音结束到系统确认不响应的判定窗。**闭合条件**：拒识判定基于 **partial ASR**（流式部分识别结果，不等 final），由 L1+L2 在窗内完成（30 + 300 + 10 + 50 ≈ 390 ms < 500 ms）；L3 复判**异步**进行（快分支 ~300 ms，超时回退 L2 结果），不阻塞静默判定与正常响应链路。

#### 1.3.5 总结：支持场景 × 指标 × 目标

| 场景 | 系统动作 | 核心指标 | v1.0 目标 | 目标依据 |
|---|---|---|---|---|
| S0 真指令/答句（对照） | 响应 | 漏响应率 FRR | 承诺 ≤ 3%（内部目标 ≤ 2%，待 S0 扩至 ≥ 3,000 段） | 拒识不能以吞掉真指令为代价；样本量约束见 1.3.4 |
| S1 主用户→旁人（SDO） | 静默 + 记上下文 | 拒识率、误执行 | 拒识率 ≥ 90%，误执行 = 0 | 文本同形最难场景；WearVox 多通道 93.9% 为眼镜形态上界（参考值，音箱/手机按自建集实测） |
| S2 旁人↔主用户（B1） | 静默 | 拒识率 | ≥ 95% | 声纹闭集判定为主信号 |
| S3 旁人↔旁人（B2） | 静默 | 拒识率 | ≥ 98% | 纯背景人声，声纹/DoA 最易判 |
| S4 媒体/电子人声（B3） | 静默 | 拒识率、误执行 | 拒识率 ≥ 95%，误执行 = 0 | 媒体源标识 + 声学事件检测（SenseVoice AED）兜底 |

**综合目标（sideTalk 模块整体）**：

- 各场景分项达标（上表 S1–S4）即为综合达标；**误执行 = 0**（一票否决）；
- FAR 为加权监控视图（公式见 1.3.4），随看板实算，**不设独立门槛**；
- **FRR 承诺 ≤ 3%**（统计可证明上界；内部目标 ≤ 2%，待 S0 扩充后生效）；
- HumDial Track II：**SDO 子集拒识率 ≥ 90%、TPS ≥ 95% 为承诺目标**——但 HumDial 仅作**对外坐标 + 回归尺子**（SDO 200 条 ±4.2pt / TPS 600 条 ±1.8pt，模型间 <4pt 差异统计不可区分），**选型主分辨器为 CoDeTT SideTalk 子集（2,000 条，±1.3pt）+ 自建集**；官方榜单 Rejection Total Score（Baseline 35.2 / 冠军 72.2）为整系统参照坐标，不作为承诺目标；
- 静默决策延迟 ≤ 500 ms（闭合条件见 1.3.4：partial ASR + L1/L2 窗内完成 + L3 异步）；正常响应首延迟 P90 ≤ 800 ms（不因本模块变慢）；
- self-talk 作为独立 addressedness 子场景单列评测（不进 S1–S4 目标表；VGBench 实证其为高难场景，原始模型静音率仅约 52%），v1 先观测基线、目标随数据补充后另定；
- 时机类指标（误打断率 ≤ 8%，参照 Gander FD-Bench v3 抢话率）由语义 VAD 模块承担，本项目只做回归验证、不设开发目标；
- 以上指标在 4.2.1 公开集与 4.2.2 自建集同口径考核，4.4 穿刺各阶段以本表为 Go/No-Go 依据。

---

## 2. 评测集洞察

| 资源 | 语言 | 真实/合成 | 规模 | 提供什么指标 | 可用性 | 获取方式 / 下载链接 |
|---|---|---|---|---|---|---|
| HumDial Track II | 中/英（官方榜单 CN/EN 分列计分确证；解压后按语言分桶） | 真实录制 | 测试 5,000 条：拒识 2,000（URB 600 / PH 600 / TPS 600 / SDO 200）+ 打断 3,000（README 分布表已确认） | 拒识率、早期打断率、首响应延迟 | ✅ 公开（测试包在列） | 全双工数据已公开：[ASLP-lab/HumDial-FDBench](https://huggingface.co/datasets/ASLP-lab/HumDial-FDBench)（双通道真实录制；仓库含 `Humdial-Track2-Test.zip`——拒识 2,000 = URB/PH/TPS 各 600 + SDO 200、打断 3,000，README 分布表已确认，解压核对子类标签即可开跑）；[官方评测脚本](https://github.com/ASLP-lab/Hum-Dial/tree/main/Full-Duplex_Interaction/evaluation)公开；备选邮件申请（wangshuiyuan@mail.nwpu.edu.cn）；[官方榜单](https://aslp-lab.github.io/HumDial-Challenge/track2/results/)已公布：冠军拒识 72.2 / Baseline 35.2 |
| CoDeTT | 中/英 | 真实+合成 | 18,000 决策 | 4 动作准确率、SMR（语义错配率） | ✅ 公开 | [HuggingFace YingaoWang-casia/CoDeTT](https://huggingface.co/datasets/YingaoWang-casia/CoDeTT)，`.lz4` 包解压即用；4 动作空间（Maintain / Stop & Listen / Takeover / Dismiss）+ 14 细粒度意图，SystemSpeaking / SystemIdle 两系统态下均含 SideTalk 负例 |
| WearVox | 英 | 真实（AI 眼镜） | 3,842 条多通道 | 各任务准确率（含 Side-Talk Rejection） | ✅ 公开 | [HuggingFace zlinao/WearVox](https://huggingface.co/datasets/zlinao/WearVox)：Side-Talk Rejection 1,082 条（500 条由 tool-calling 复制构成配对）；每条含单通道 `audio_query` + 多通道 `audio_query_mc` 两个字段；Search-Grounded QA 需与 [CRAG](https://github.com/facebookresearch/CRAG) 按 `id` join 补搜索结果 |
| VGBench | 英 | 合成+受控 | 1,018 条 | 静默率、工具选择、回答质量 | ❌ 数据未公开 | [论文 arXiv:2609.32536](https://arxiv.org/abs/2609.32536)（2026-09-26）未随附数据/代码；github.com/vgbench/VGBench 是同名**向量图形**基准（2024），无关勿混；论文结论可作方法参考：原始模型最高 switch 静音率 14%，VoxGate 后训练 91.3% |
| ChorusDuplex-SideTalk | 中文 | 合成（TTS） | 1,667 轮 | 逐轮 speaker attribution + 是否该静默 | ✅ 公开（CC BY 4.0） | [HuggingFace BiggestWave/ChorusDuplex-SideTalk](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)（v2，2026-09）：100 段 / 1,667 轮 / 200 条 48kHz 立体声（clean + babble 10dB 背景人声双版本）；逐轮 `addressee` / `kind` / `trigger` 标注，side_talk 134 轮、含第三方说话的对话 98/100；babble 变体再分发须保留 VOiCES（CC BY 4.0）出处 |
| SLT 2026 SmartGlasses | 中文 | 真实（四通道） | 106h / 714 段 | TSA-ASR + SLU | ⚠️ 注册后邮件发放 | [注册表单](https://docs.google.com/forms/d/e/1FAIpQLSfReBQPIFSNJ_pJkpyPJ_icfYxHznTD5y6WtOk5goC-BLPcOQ/viewform)（国内可选[腾讯问卷](https://docs.qq.com/form/page/DVWV6ZGJoSFBnd3Zi)）→ 1 个工作日内确认邮件附下载链接；官方统计 714 段 / 106.98h（双人对话 518 段 + 多方会议 196 段）；四麦几何坐标公开（右腿 47mm / 左腿 50mm / 前跨 145mm / 后跨 146mm；来源：挑战赛官网 Data Description 节，论文正文未列）；[评测工具包](https://github.com/ASLP-lab/Smart-Glass-Challenge)（MeetEval）与 [Codabench 榜单](https://www.codabench.org/competitions/17161/)公开 |

> 背景：HumDial Track I 情绪智能集 [ASLP-lab/HumDial-EIBench](https://huggingface.co/datasets/ASLP-lab/HumDial-EIBench)（1,077 条中英，Apache-2.0）也已公开，与拒识无关，仅作背景参考。

| 角色 | 用哪个 | 用法 |
|---|---|---|
| 对标 | HumDial Track II（官方榜单：冠军拒识 72.2 / Baseline 35.2，仅作参照坐标）、ChorusDuplex-SideTalk、CoDeTT（SideTalk 子集） | 判断自家系统在真实录制上是否明显落后 |
| 诊断 | CoDeTT（SMR）、VGBench（方法论参考，数据未公开） | 判断错在"没识别出"还是"识别了但不肯闭嘴" |
| 补充参考 | WearVox（AI 眼镜形态）、SLT 2026 SmartGlasses（眼镜四通道） | 眼镜补充场景的参考结论：多通道空间增益、说话人重叠诊断；**不作为主场景验收依据** |
| 训练数据 | ChorusDuplex-SideTalk（已公开，CC BY 4.0）+ 自建 | 中文合成集只够跑通链路，上产品需补真实录音 |

> 口径注（对齐 1.3）：公开集的时机类子集（HumDial URB/PH、CoDeTT Backchannel/Distraction/Interruption、ChorusDuplex backchannel/interruption 轮）不进 sideTalk 指标，仅做回归；各集具体子集映射与测评方法见 4.2.1。

**可执行结论**：主场景尺子（HumDial/FDBench、CoDeTT、ChorusDuplex-SideTalk）公开可下载、当天可开工（HumDial 拒识子集覆盖待下载确认）；WearVox、SmartGlasses 为眼镜补充场景参考，可延后获取；VGBench 数据未公开，其 speaker-switch 维度由 4.2.2 自建对抗集承接。

---

## 3. 业界方案洞察

### 3.1 外挂LLM，系统判断

**模式定义**：拒识与时机决策不训进语音模型，而是架设在传统语音链路之上，由一个（可外挂的）LLM 担任"裁判"，输出结构化决策，由系统执行。拒识策略以提示词/规则形式存在，可热更新、可审计。

**通用链路**：

```
音频前端（AEC/NS/波束成形）
  → VAD
  → 说话人模块（声纹 + DoA/近讲检测）
  → 流式 ASR（增量转写 + 词级时间戳）
  → 拒识裁判【多级：L1 规则/信号 → L2 端侧小模型 → L3 云端 LLM 复判】
  → 决策动作 { respond / acknowledge(附和) / wait(等待) / silent+record(静默+记上下文) / silent }
  → 对话与任务执行（回复 TTS / 工具调用 / 旁听缓存写入）
```

**三级拒识漏斗**（延迟从低到高）：

| 级 | 判断依据 | 延迟 | 挡掉什么 |
|---|---|---|---|
| L1 信号/规则层 | 声纹白名单、唤醒词、能量、DoA、媒体源标识 | < 10 ms（端侧） | 大部分 TPS/B3：非主用户声纹直接静默 |
| L2 端侧有界决策 | 端侧 NPU 小模型（Qwen3-0.6B/1.7B 量化）做动作级判定（Choice + escape） | 10–50 ms | self-talk、明显闲聊、大部分 SDO 歧义 |
| L3 云端复判（异步） | **快分支**：Jev 类决策模型（有界动作，~300 ms【第三方实测】，带校准置信度）；**强分支**：带多轮上下文与场景态的 LLM，输出结构化 JSON（action/reason/confidence/record_flag），100–500 ms | 快分支 ~300 ms；强分支 100–500 ms（云/边） | L1/L2 无法判定的文本同形 SDO 歧义 case；L2 低置信度分流 |

**优点**：决策可审计（每条拒识留 JSON 依据）；策略可热更新（改提示词/规则不发版）；无需重训端到端语音模型；隐私可分级（声纹/DoA 不出端，语义模糊 case 才上云）。

**缺点**：链路长、时延叠加；ASR 错 → 裁判错判的错误级联；旁听缓存与主对话状态机的上下文管理复杂（需隔离）。

**业界代表**：

- 华为小艺（见 3.1.1）；
- Meta 在 WearVox 上验证的"Whisper ASR + 文本 LLM"两段式 pipeline——单通道下 side-talk 拒识 85.6%，接入多通道空间信息后升至 93.9%：说明外挂路线可达标，瓶颈在前端空间信息而非裁判本身（[WearVox 论文](https://arxiv.org/html/2601.02391v1)）；
- 阿里 Qwen-Audio-Agent 的 Frontend Agent（harness 模式）：前端 Agent 解释流式输入、维持全双工对话、在直调工具与后台委派之间选择——本质也是"系统判断 + 外挂大脑"（[arXiv:2609.25195](https://arxiv.org/html/2609.25195v1)）；
- TypeSafe Jev（System One 决策模型，2026-09-15 发布）：无 LLM 的语音 Agent 参考栈（Deepgram Flux STT → Jev → TTS），每决策约 300ms、带校准置信度、0% 类型错误——证明"有界转录判断"可以不依赖 LLM（[Evalgent 解析](https://www.evalgent.com/blog/jev-for-voice-agents)）。

### 3.1.1 小艺方案

华为未发表小艺拒识链路的论文级细节，以下为**公开资料还原**（官方支持文档、产品页、公开报道），供对标；与本团队"外挂 LLM、系统判断"的路线判断一致：

1. **系统级"对话中控"，先决策、后内容。** 华为云盘古 Bot 公开架构中，"对话中控（多轮问答理解、意图识别、分发）"被定位为"整个对话系统的核心枢纽"，之下才是助手编排与插件调用（[盘古 Bot 架构](https://www.huaweicloud.com/product/cbs/pangubot.html)）。小艺在终端侧同理：语音先被中控判断"是什么、归谁管"，再分发执行——判断与内容生成解耦。
2. **全双工连续对话 + 拒识过滤。** 小艺支持免唤醒连续对话；公开科普口径中，拒识的功能即"把无效的语音过滤掉"（周围人说话、背景噪音），多轮对话依赖连续会话的上下文缓存（[全双工连续对话与拒识](https://zhuanlan.zhihu.com/p/161563225)）。
3. **声纹私语——身份层的产品化。** 小艺提供"小艺私语"开关（设置 > 小艺 > 唤醒/唤醒与对话）：基于声纹识别，仅对已注册机主的语音做出响应（[官方说明](https://consumer.huawei.com/cn/support/content/zh-cn16010150/)）——正是 1.3 中"说话人身份层（最强信号）"的落地：闭集声纹判定在端侧完成，非机主直接静默。
4. **多模态场景态辅助判断。** "边聊边看"模式下摄像头与语音同时输入（[官方支持页](https://consumer.huawei.com/cn/support/content/zh-cn16053374/)）；HarmonyOS 系统级感知（前台应用、屏幕状态、位置、是否通话中等 200+ 数据点）可作为裁判的场景特征。
5. **分级隐私。** 声纹、唤醒等敏感信号不出端，语义模糊 case 才上云（盘古大模型，Celia 自 2023 年起接入 [Pangu 3.0](https://voicebot.ai/2023/08/08/huawei-adds-generative-ai-powered-voice-assistant-to-smartphones-beating-apple-and-google/)）复判。

**对齐到 1.3 的四层信号**：小艺 = 身份层（声纹私语）+ 语义层（中控意图识别）+ 多模态层（系统感知/边聊边看），时序层内化在连续对话状态机中。

**补充（业界专利）**：语义层面拒识判断模型（[CN107665708B 智能语音交互方法及系统](https://patents.google.com/patent/CN107665708B/zh)）、含唤醒词的全双工语音拒识方法（[聚好看专利申请报道](https://www.163.com/dy/article/L8QIFP0E0519QIKK.html)）。

### 3.2 语音模型的两类形态

按"语音是否直接进入模型"，业界语音模型分两类：

- **语音直入型（端到端）**：音频直接进模型、无 ASR 文本瓶颈，拒识与时机可训进模型，但策略固化、不可热更；
- **基于文本型（级联）**：语音先转文本（ASR）再交给文本 LLM，拒识依赖外挂判断（3.1 模式），策略可热更、可审计，但受 ASR 正确率制约。

#### 3.2.1 语音直入型（端到端）

| # | 模型 / 产品 | 厂商 | 拒识与时机机制 | 公开数据 | 证据 |
|---|---|---|---|---|---|
| 1 | GPT-Live-1 / 1 mini / 1.1 | OpenAI（2026-07-08 起） | 全双工；会用 "mhmm / yeah" 附和，也会在用户思考时保持安静；复杂任务后台委派给前沿模型（发布时 GPT-5.5），等待期间继续对话 | 官方描述，未公开静默率 | 官方发布（2026-07-08；1.1 增 13 音色、减少 "checking…"；07-31 起加 SynthID 水印） |
| 2 | GPT-Realtime-2 | OpenAI | Realtime API：server_vad / semantic_vad、interrupt_response | 作为基线出现在 Qwen 报告（OAB 87.31 / VoiceBench 83.37 / AMC 50.33） | 官方文档 + 论文 |
| 3 | Gemini Live（Gemini 3.x Flash / for Home） | Google | 支持打断；Continued Conversation 免唤醒续聊 | 未公开拒识率 | 官方支持页 |
| 4 | Qwen-Audio-3.1-Realtime | 阿里 | smart_turn 语义轮次（"嗯/啊"不打断）+ 说话人增强屏蔽旁人 | FD-Bench v1.5 对背景语音响应率 73%→13% | 官方文档 + arXiv:2609.25176 |
| 5 | Qwen3.5-Omni / Qwen3.8 Omni Flash Realtime | 阿里 | 端到端，支持视频输入（眼镜/座舱形态） | 未公开 | 官方文档 + qwen-audio-agent 前端支持 |
| 6 | Seeduplex 1.2.6.1 / 3.0 | 字节 | 端到端全双工 | Qwen 报告基线：OAB 85.38 / VoiceBench 87.43 / AMC 41.91；人格 20+ 轮 3.75（高于 Qwen 3.1 的 3.46） | 论文（Qwen 报告 Table 2/8） |
| 7 | Gander | 腾讯混元（Apache-2.0 开源） | "先决策、后内容"解耦，时机训进模型 | FD-Bench v3：接管 100% / 抢话 8.0%（对照 GPT-Realtime 96%/13.5%、Gemini Live 3.1 接管 78%、级联抢话 33%） | 项目报告 |
| 8 | Step-Audio 2 / 2 mini | 阶跃星辰（开源） | 端到端语音理解 + 生成，支持中英日与粤语、情感 | URO Bench 开源端到端最高（自报） | 官方仓库 |
| 9 | GLM-Realtime | 智谱 | 音视频通话，2 分钟记忆，函数调用 | 音频 0.18 元/分钟、视频 1.2 元/分钟 | 官方文档 |
| 10 | MiniCPM-o 4.5 | OpenBMB | 全模态；Gander 的基座模型 | — | 项目报告 |
| 11 | Moshi | Kyutai（开源） | 全双工、可打断，本地部署 | 未公开拒识指标 | 官方仓库 |
| 12 | Baichuan-Audio | 百川（开源） | 端到端中英双语实时对话 | 未单独披露拒识/打断指标 | 官方技术文 |
| 13 | JoyAI-Talker | 京东探索研究院（2026 WAIC 发布） | Joy-Duplex 框架：实时解码路径交织双工状态标记，作为对话轮转控制的语义门控引擎；官方自报附和误打断率 0.01、旁人误触发率 0.17、全链路时延 1s 以内 | 官方自报，未见第三方评测 | [官方页](https://research.joyai.com/joyai-talker) + arXiv:2608.01119 |

来源：[GPT-Live 发布页](https://openai.com/index/introducing-gpt-live/) / [GPT-Live-1 API](https://openai.com/index/introducing-gpt-live-1-in-the-api/) / [System Card](https://deploymentsafety.openai.com/gpt-live) / [1.1 社区公告](https://community.openai.com/t/gpt-live-1-1-is-rolling-out-with-new-voices-and-better-accents/1406704) · [Qwen-Audio-3.1-Realtime](https://arxiv.org/abs/2609.25176) · [Gander 技术博客](https://omni-interaction-gander.github.io/Omni-Interaction-Agent/) / [arXiv:2609.08977](https://arxiv.org/html/2609.08977v3) · [Seeduplex](https://baike.baidu.com/item/Seeduplex/67596638) / [3.0 API 口径](https://developer.cloud.tencent.com/news/4112309) · [Step-Audio 2 mini](https://www.oschina.net/news/369623) · [GLM-Realtime 文档](https://docs.bigmodel.cn/cn/guide/models/sound-and-video/glm-realtime) / [定价](https://docs.bigmodel.cn/cn/guide/start/pricing) · [MiniCPM-o 4.5](https://openbmb.cn/model/minicpm-o-4-5) / [arXiv:2604.27393](https://arxiv.org/html/2604.27393v1) · [JoyAI-Talker](https://research.joyai.com/joyai-talker) / [arXiv:2608.01119](https://arxiv.org/abs/2608.01119)。

#### 3.2.2 基于文本型（级联）

| # | 模型 / 方案 | 厂商 | 机制 | 拒识相关证据 |
|---|---|---|---|---|
| 1 | 传统 ChatGPT Voice（Standard） | OpenAI | STT→LLM→TTS 三模型级联 | 官方发布页：信息跨模型丢失、慢而僵（已被 GPT-Live 替代） |
| 2 | Whisper + GPT-5 两段式 | Meta（WearVox 基线） | ASR + 文本 LLM | [WearVox](https://arxiv.org/html/2601.02391v1)：单通道 side-talk 拒识 85.6%，多通道空间信息升至 93.9% |
| 3 | Cascade 参赛系统 | HumDial Track II 参赛队 | 级联（**注：HumDial Track II 官方榜单为整系统评测，该行仅作参照**） | [官方榜单](https://aslp-lab.github.io/HumDial-Challenge/track2/results/)：拒识总分 30.9（九队垫底，对照冠军 72.2）——纯文本级联对拒识天然吃亏 |
| 4 | DuplexCascade | 开源论文 | VAD-free 级联流式 pipeline | [arXiv:2603.09180](https://www.alphaxiv.org/abs/2603.09180) |
| 5 | MMW | Meta | 多麦 Whisper 前端增强（服务文本链路） | [arXiv:2507.05609](https://arxiv.org/html/2507.05609v1)：噪声下 WER 相对下降 4.95% |
| 6 | 小艺类外挂判断 | 华为 | 3.1 模式（见 3.1.1） | 官方口径 |
| 7 | Jev 决策栈 | TypeSafe AI | STT → System One 决策模型 → TTS（轮环内无 LLM，应答由人工预写、模型只选择） | [Evalgent 解析](https://www.evalgent.com/blog/jev-for-voice-agents)：【第三方数据：Evalgent】端到端 70–500ms/决策、输入 $0.042/MTok、0% 类型错误（数学保证）；Pipecat 230-case 实测比 LLM 多恢复 46% 乱码指令、延迟低 73%；**部署形态为托管 API（无开放权重）**；[Jev 1.14](https://www.ithome.com/1/013/794.htm)（2026-10-14）支持图像输入、更长上下文；一天后 Liquid AI 发布 d1-ultra（自称八类任务优于 Jev 1.14） |

**两类分野决定拒识设计**：语音直入型把时机/拒识训进模型（Gander 小脑、Joy-Duplex 双工状态标记、Qwen smart_turn），上限高但策略固化、不可热更；基于文本型的拒识是外挂判断，可热更可审计，但 HumDial 官方榜单上纯级联系统拒识总分仅 30.9（对照冠军 72.2）——文本级联必须补声纹/空间信号才能达标（见 3.1 三级漏斗）。我们的路线（4.1）= **基于文本型链路 + 外挂裁判 + Jev 类决策模型**。

---

## 4. 模型选型及技术穿刺

### 4.1 方案建议

**类似小艺，外挂 LLM 判断。** 落到本系统（**家庭智能音箱 + 手机**为主场景，免唤醒连续对话、中文为主；智能眼镜为补充）：

- **不做**端到端重训：复用现有语音链路，新增"拒识裁判"模块（L1 规则/声纹 → L2 端侧小模型 → L3 云端 LLM），策略全部可热更。
- **统一决策协议**：裁判输出 `{"action": "respond|ack|wait|clarify|silent|silent_record", "who": "primary|other", "reason": "...", "confidence": 0~1}`；`silent_record` 时附带旁听缓存写入（默认不参与回答）；`clarify` 为低置信度兜底的轻确认动作（出声不执行，见 1.3.3）。按 1.3 的模块边界，sideTalk 模块只消费归属类动作（respond 放行 / silent / silent_record）；ack、wait 供语义 VAD 模块消费——同一裁判、两个下游，评测时 sideTalk 指标只看归属三类。
- **前端按设备形态补空间/距离信息**——音箱：环形麦阵 DoA + 远场波束（家人在不同方位、是否朝设备说话可分）；手机：近讲检测 + 持握/放置姿态。WearVox 官方案例研究证明空间信息是 side-talk 拒识最大增益（单→多通道 85.6%→93.9%、整体 61.9%→66.4%），该结论跨形态方向一致，具体增益量级由尺子2 在音箱/手机上实测。
- **保留端到端模型作对照组与兜底**（见 4.3/4.4）。

### 4.2 评测集构建

**原则**：公开集用来对齐业界、防止自嗨（4.2.1）；自建集用来贴近产品——中文、真实的音箱远场与手机近场声学、1.3 场景完整覆盖（4.2.2）。两把尺子同口径出数，进同一张"拒识看板"。

**与 1.3 的对齐约定（适用两把尺子）**：

- 公开集的**时机类子集**（HumDial 的 URB/PH、CoDeTT 的 Backchannel/Distraction/Interruption、ChorusDuplex 的 backchannel/interruption 轮）**不进 sideTalk 指标**，只做"本模块不引入退化"的回归验证（时机判定归语义 VAD 模块）；
- self-talk 单列评测（复用本模块裁判、不进 S1–S4 目标表，见 1.3.0）；
- 每条样本先按 1.3.1 映射到 S0–S4 再计指标；映射表随各集构建方案给出。

#### 4.2.1 尺子1 公开测评集——构建方案与测评方法

**构建总流程（六步，所有公开集统一）**：

```
① 获取下载（HF/注册） → ② 子集抽取与 S0–S4 映射 → ③ 输入构造（音频+系统态+多轮历史）
→ ④ 被测系统推理（统一动作协议，见 4.1） → ⑤ 动作解析与判分 → ⑥ 指标计算与看板汇总
```

**各集概览（选集理由 + S 映射 + 门槛）**：

| 公开集 | 角色与理由 | 有效子集 → S 映射 | 排除/降级 | 通过门槛 |
|---|---|---|---|---|
| HumDial Track II / FDBench | 对外坐标 + 回归尺子：唯一中英真实录制、官方榜单可对标；但 SDO 测试仅 200 条（±4.2pt），**不作选型主分辨器** | SDO 子集→S1；TPS 子集→S2/S3；dev 拒识侧 4×200=800 条→调参专用 | URB/PH 子集→仅回归（语义 VAD 域）；test 集只在最终表报一次 | SDO 拒识率 ≥ 90%、TPS ≥ 95%；榜单对照（冠军 72.2 / Baseline 35.2） |
| CoDeTT | **选型主分辨器**：SideTalk 子集约 2,000 条（±1.3pt，足以区分 <4pt 的模型差异）+ SMR 抓"动作对理由错"；18,000 决策带 5 轮历史 | SystemIdle_Dismiss_SideTalk + SystemSpeaking_Maintain_SideTalk（中英×真实/合成）→S1 | Backchannel/Distraction/Interruption 等 12 子集→仅回归 | 先复现官方 baseline，门槛 = baseline + 5 pt（首跑后核定） |
| WearVox（补充） | 眼镜形态参考尺子：唯一真机多通道 Side-Talk Rejection；**主场景外**，用于验证空间信息增益方向与裁判复判能力 | Side-Talk Rejection：582 side-talk（旁人语音）→S2/S3；500 valid queries→S0 对照 | 全英文且眼镜形态 → 不进主场景验收，单列参考 | 参考：side-talk 拒识 ≥ 90%（上界 93.9%）；S0 侧 FRR 承诺 ≤ 3%（内部 ≤ 2%） |
| ChorusDuplex-SideTalk | 中文冒烟尺子：CI 必过；clean+babble 抗噪对比 | kind=side_talk（134 轮）→S1；third_party 轮→S2/S3；assistant 轮→S0 对照 | backchannel/interruption 轮→仅回归 | 两版全绿：S1 拒识率 ≥ 90%、S2/S3 ≥ 95% |
| VGBench | 方法论参考（数据未公开） | — | 全部降级；speaker-switch 由尺子2 承接 | — |
| SLT 2026 SmartGlasses | 前端诊断：说话人重叠对声纹/ASR 的影响 | Dev 集抽样 50 段 | 不进 sideTalk 主指标 | 诊断参考值，无门槛 |

**① HumDial Track II（对外坐标 + 回归尺子）——具体做法**

- **获取**：[FDBench](https://huggingface.co/datasets/ASLP-lab/HumDial-FDBench)（GitHub 仓库含 `Humdial-Track2-Test.zip`）——README 分布表确认测试集：拒识 2,000 = URB 600 / PH 600 / TPS 600 / **SDO 200**，打断 3,000；解压核对子类标签后即可开跑，邮件申请仅作备份（wangshuiyuan@mail.nwpu.edu.cn）。官方[评测脚本](https://github.com/ASLP-lab/Hum-Dial/tree/main/Full-Duplex_Interaction/evaluation)直接复用其指标实现，保证与榜单同口径。
- **子集抽取**：按标注把拒识子集拆为 SDO/TPS/URB/PH 四桶（test：600/600/600/200；**dev：每子任务 200 条、拒识侧合计 800 条**——README 分布表 Dev 列）；SDO→S1、TPS→S2/S3 计入 sideTalk 指标，URB/PH 仅回归。
- **调参纪律（防 test 污染）**：**一切 prompt/阈值/规则调参只在 dev（800 条）上进行；test 集只在最终表报一次**——每条 test 结果仅允许出现在定稿评测表中，调参过程中的 test 触碰即视为该版本 test 作废。
- **输入构造**：对话式推入——系统态（idle/speaking）+ 双通道音频流 + 既有对话历史转写；复现官方评测环境（标准化 Docker + RTX A6000，延迟分取官方脚本口径）。
- **判分**：以官方脚本输出为准（Rejection Rate / Early Interrupt Rate / First Response Delay）；本模块侧另计"每子类拒识率"与误执行（0 容忍）。
- **对标**：SDO 子集拒识率 ≥ 90%、TPS ≥ 95% 为承诺目标；官方 Rejection Total Score（Baseline 35.2 / 冠军 72.2）为整系统参照坐标（总分含打断与延迟分量，不作为本模块承诺）。

**② CoDeTT（诊断尺子）——具体做法**

- **获取**：[HF 数据包](https://huggingface.co/datasets/YingaoWang-casia/CoDeTT)，`lz4 -dc CoDeTT.lz4 | tar -x` 解压即得中英 × 真实/合成 × 系统态的完整目录（jsonl + 音频）。
- **子集抽取**：仅 `SystemIdle_Dismiss_SideTalk.jsonl` 与 `SystemSpeaking_Maintain_SideTalk.jsonl` 两类进 sideTalk 指标（→S1）；其余 12 类意图子集全量跑但不计指标。
- **输入构造**：每实例 = 系统态 + 5 轮对话历史 + 当前目标语音；统一问"此刻应执行哪个动作"。
- **判分**：官方两阶段协议——4 动作（Maintain/Stop&Listen/Takeover/Dismiss）ACC；对动作正确的样本再核意图是否同向算 SMR（语义错配率）。映射到我们的协议：Dismiss→silent、Takeover→respond、Stop&Listen/Maintain→语义 VAD 域动作（仅回归核对"本模块未把它判成归属错误"）。
- **诊断用法**：把错误分三桶——"没识别出归属"（ACC 错）、"识别了不肯闭嘴"（应 Dismiss 给了 Takeover）、"蒙对理由错"（ACC 对但 SMR 命中）——分别反哺 L1/L2/L3。

**③ WearVox（补充尺子，眼镜形态）——具体做法**

- **获取**：[HF 数据集](https://huggingface.co/datasets/zlinao/WearVox)，取 Side-Talk Rejection 任务 1,082 条（582 side-talk + 500 有效 query）。
- **子集抽取与映射**：side-talk 条目（旁人语音）→S2/S3；valid queries（真指令）→S0 对照，构成"拒识率 × FRR"的配对检验。
- **输入构造**：先单通道 `audio_query` 跑全量；再对多通道版用 `audio_query_mc` 复跑——**两轮差值即空间信息增益**（官方参考：85.6%→93.9%）。注意数据为眼镜第一视角录制，与音箱远场/手机近场的声学几何不同，**增益方向可参考、绝对值不迁移**——主场景增益以尺子2 实测为准。
- **判分**：二分类（silent=拒识正确 / 非 silent=失误）+ S0 侧答对率；与官方两段式基线（Whisper+GPT-5）和业界区间（论文口径：实时 SLLM 整体准确率 29%–59%）对照。
- **定位声明**：仅英文、仅眼镜形态——不进主场景验收；用于三点参考：裁判复判能力迁移性、空间信息增益方向、WearVox 论文口径（实时 SLLM 整体准确率 29%–59%）的难度标定。

**④ ChorusDuplex-SideTalk（中文冒烟尺子）——具体做法**

- **获取**：[HF 数据集](https://huggingface.co/datasets/BiggestWave/ChorusDuplex-SideTalk)（CC BY 4.0），clean 100 段 + babble 100 段；`timeline.json` 提供逐轮绝对时间戳。
- **子集抽取与映射**：按 `annotations.jsonl` 逐轮判——`kind=side_talk`（134 轮）→S1；`speaker=third_party` 的轮→S2/S3；assistant 轮→S0 对照；`kind=backchannel/interruption` 的轮→仅回归（语义 VAD 域）。
- **输入构造**：按 timeline 切段推入，stereo 布局（L=非助手声道、R=助手声道）与我们的双流前端直接对齐。
- **判分**：逐轮 gold = `addressee + kind` 推出的应静默/应响应标签；clean 与 babble 两版分别出数，差值即背景人声鲁棒性。
- **用法**：CI 冒烟——每次模型/规则/提示词变更必跑必过；也是尺子2 采集前的链路试跑场。

**统一测评方法（跨集通用规则）**：

1. **输入构造三要素固定**：音频（按各集声道布局）+ 系统态（idle/speaking，CoDeTT 与 HumDial 显式给定）+ 多轮历史（转写文本，取各集自带标注）；半双工实现只测 idle 态并在看板标注。
2. **判分以规则为主**：动作级 gold 直接判对错，无需 LLM judge；仅 respond 的内容质量用冻结版 LLM 判分器——**与官方 scoring judge 完全一致：`Qwen/Qwen3-Omni-30B-A3B-Instruct`**（HumDial 官方 GitHub 确认），判分 prompt 与版本号入仓锁定。
3. **指标公式**（与 1.3.4 同口径）：子类拒识率 = 该子类正确静默数 / 应静默总数（S1–S4 分别出）；FAR = 1 − 加权拒识率；误执行单独计数（0 容忍）；FRR = S0 被误吞数 / S0 总数；静默决策延迟 P50/P90。
4. **Sanity check**：每接入一集，先复现已公布基线（WearVox 两段式 85.6% 一线、CoDeTT 官方基线、HumDial Baseline 35.2），偏差 > 3 pt 判 harness bug，修完才开跑被测系统。
5. **回归纪律**：时机类子集（URB/PH/Backchannel 等）每次变更必跑，验证"sideTalk 模块的变更不引起语义 VAD 域指标退化"；退化即回滚。
6. **clarify 的判分落点（跨集统一）**：被测系统输出 clarify（轻确认）时记"非静默"，**单列统计、不计入拒识率分子、不计入 FAR**（与 1.3.4 口径一致）；CoDeTT 映射中 clarify→不计入 4 动作 ACC，单列"clarify 率"列监控——否则公开集会把 clarify 误算成误响应，与 1.3 口径打架。
7. **看板汇总**：列 = 数据集 × S 子类（S1/S2/S3/S4/self-talk 单列），行 = 被测系统/版本；与 4.2.2 自建集同表、分列来源。

**运行机制**：

- 统一离线评测 harness（Docker + 固定 GPU RTX A6000，对齐 HumDial 官方评测环境规范）；每次模型/规则/提示词变更全量回归，周度出趋势。
- 对标坐标（HumDial Track II 官方榜单，整系统口径）：冠军 cookie_asr（Transsion）Rejection Total Score 72.2 / 打断 79.3 / 总分 76.6；官方 Baseline 拒识 35.2 / 总分 56.4。**本模块承诺目标为子集级**（SDO ≥ 90% / TPS ≥ 95%，见 1.3.5）；榜单分数仅作参照坐标（总分 60% 分量不由本模块决定），不写承诺值。

#### 4.2.2 尺子2 自构建测评集

**目标**：公开集补不齐的四件事——中文真实场景、**音箱远场声学**（电视开着的客厅、家人分坐、厨房噪声）与**手机近场声学**（手持/桌放/车载支架、外放媒体、户外噪声）、1.3 场景完整覆盖（访客指令、S4 媒体人声、对抗样本）、speaker-switch 对抗强化。

**场景矩阵（分别对两种主设备采集）**：

| 维度 | 音箱（远场） | 手机（近场/手持） |
|---|---|---|
| 设备位置与用户距离 | 1–4 m 固定（客厅/厨房/卧室） | 手持 0.2 m / 桌放 0.5 m / 车载支架 0.6 m |
| 说话人构成 | 1–4 人（家庭多成员为主） | 主用户 1 人 + 0–2 旁人 |
| 主用户朝向 | 对设备 / 对家人 / 背对（做饭等） | 对屏幕 / 对旁人 / 侧头 |
| 声学环境 | 电视开着 / 音乐外放 / 厨房 / 安静 | 街道 / 车内 / 办公室 / 外放短视频 |
| 典型 S4 源 | 电视、智能电视投屏 | 手机自身外放、旁人手机 |
| 语言 | 普通话为主 + 15% 中英混说（两设备同） | 普通话为主 + 15% 中英混说 |
| 采集语音类型 | S0 真指令/答句（对照）/ S1 SDO / S2 B1 / S3 B2 / S4 B3 媒体 / self-talk（单列）/ speaker-switch 对抗（旁人说指令；不含访客指令，见 1.3.1 范围外）。URB/PH **不采**（语义 VAD 域，用公开集回归） | 同左 |

**规模建议**：v1.0 采集 3,000 有效段（每段 10–30 s）：音箱 1,800 段（设备自身麦阵原样录制，含全部可用通道）+ 手机 1,200 段（手机双麦 + 环境参考麦同步录制）；其中 sideTalk 负例（S1–S4）≥ 1,200、self-talk 单列段 ≥ 150、speaker-switch 对抗段 ≥ 150。两种设备**分开出指标**，不混池。

**采集规范（防语义泄漏）**：

- 演员照脚本演，对齐 HumDial "LLM 脚本 + 演员表演"的做法——打断/重叠发生在语义接缝（迟疑、换气处）而非随机时间戳，保留自然韵律；
- side-talk 脚本**不得**含唤醒词/请求句式；对抗样本单独成轨，故意含"帮我/小X"字样但交流对象明确是旁人；
- 说话人不重叠划分（train/dev/test 说话人隔离）；同步记录场景元数据（SNR、混响估计、DoA 真值、距离真值）。

**标注 schema（段级 + 轮级）**：

- 段级：`{speaker_role: primary|other|media, directed_to: assistant|companion|none, gold_action: respond|ack|wait|silent|silent_record, scene_tags[]}`
- 轮级：词级时间戳 + 转写 + 意图标签 + 与上一系统轮的衔接类型（答句/打断/无关/附和）
- 双人独立标注 + 第三人仲裁，Cohen's κ ≥ 0.8 方可入库；每版本 10% 抽检复标。

**指标**（与 1.3.4 及尺子1 同口径）：S1–S4 分子类拒识率、全场景 FAR、误执行（0 容忍）、S0 侧 FRR、self-talk 单列静音率、静默决策延迟 P50/P90；另录 URB/PH 域回归指标（仅验不退化）。

**发布与迭代**：版本化（v1.0 起），季度增量补采；线上 badcase（用户投诉"抢话/插嘴/没听到"）经脱敏回流为新场景。

### 4.3 模型选型

按 4.1 架构逐模块选型。选型原则：**前端信号宁多勿少、裁判分级、先开源后自研**。类型定位：主链路为**基于文本型**（ASR→LLM 级联 + 外挂裁判），端到端模型（语音直入型）仅作对照组与兜底。

| 模块 | 推荐 | 备选 | 选型理由 |
|---|---|---|---|
| 声纹/说话人 | CAM++（3D-Speaker）声纹嵌入 + 余弦阈值做**闭集**判定"是否机主"；重叠场景加帧级 diarization（参考 MMW 的 Frame Diarization Mamba Layer） | pyannote、SpeakerLM（AAAI 2026，端到端 diarization+识别） | 闭集声纹判定比开集分离简单可靠、端侧可跑；SmartGlasses 挑战赛结论：说话人重叠是 TSA-ASR 主要误差源，必须帧级处理 |
| VAD | Silero VAD v5 | FunASR FSMN-VAD | 帧级 <30 ms、流式、误触发低 |
| ASR | FunASR Paraformer 流式（中文主）+ 词级时间戳；方言/中英混说补 Fun-ASR-Nano | SenseVoice-Small（非流式，情感/声学事件检测并联）；Qwen3-ASR（已有端侧适配先例） | 流式低延迟、中文强、开源可私有化；SenseVoice 的情感/事件输出可作为 L1 信号（笑声/音乐 → 疑似媒体或附和） |
| 空间前端 | 按形态分置：**音箱**——环形麦阵 GCC-PHAT/DoA + 波束成形（远场指向性，判"声源在设备哪个方位、是否朝设备"）；**手机**——双麦近讲检测（近场能量比、左右麦声道差随持握姿态变化）+ 放置状态识别（手持/桌放） | MMW 框架（Tri-Mamba Mix Block + 帧级 diarization + 多尺度 GRPO，端到端微调 Whisper，噪声下 WER 相对下降 4.95%）——多麦融合通用思路，音箱环形阵可直接借鉴 | WearVox 实证空间信息是 side-talk 拒识最大增益（+8.3 pt，眼镜形态）；音箱环形麦阵的 DoA 能力强于眼镜双麦（基线更长），方向一致 |
| L2 裁判（端侧，有界决策） | **Qwen3-0.6B / 1.7B（NPU 量化）**：以"转录 + 场景态"为输入，动作集合 `respond / silent / silent_record / clarify / not_enough_information` 为约束输出（有界决策），端侧 10–50 ms；动作协议与评测口径与具体模型解耦 | d1-ultra（Liquid AI，同类决策模型思路）；Qwen3-4B 升档 | 拒识是**有界转录判断**，适合有界决策范式：schema 约束输出、不可幻觉出新动作、带置信度（天然支持低置信度分流）。**部署约束**：必须端侧（10–50 ms 预算 + 内网/隐私场景），因此排除托管 API。**工程注意**（对齐 System One 模型设计经验）：动作集合必须显式加 `not_enough_information` escape 选项，其前加一条判定（"该段语音是否存在指向系统的请求"）分离"无证据"与"有证据但选 B"；看置信度分布而不只看标签；无内置弃权是此类模型的通病 |
| L3 裁判（云/边，异步复判） | **快分支：Jev（TypeSafe AI，System One 决策模型，托管 API）**——动作集合为 Choice 选项、~300 ms/决策【第三方实测 p50 ≈ 312 ms】，带校准置信度，承担 L2 低置信度分流与歧义 case 初判；**强分支**：与业务主 LLM 一致（DeepSeek-V4.1-Flash / V4-Pro、Qwen3.5、GLM、豆包任选），输出结构化 JSON + 理由 | 快分支备选 d1-ultra（Liquid AI）；强分支备选用尺子1/尺子2 数据蒸馏的专用判别模型 | System One 模型把"该不该响应"从自由生成降为有界决策——快、不可幻觉出新动作、置信度可校准【第三方：Evalgent 分析】；但 Jev 是托管 API（无开放权重、云端处理），**不能做端侧 L2**，隐私敏感场景需评估数据出境合规；强分支处理快分支接不住的开放推理 case，CoDeTT 的 SMR 指标要求给出理由 |
| 端到端对照组（语音直入型） | Qwen-Audio-3.1-Realtime（API）：官方证据链 FD-Bench v1.5 背景语音响应率 73%→13% | Seeduplex 3.0 API（动态判停）、JoyAI-Talker（京东，官方页自报附和误打断 0.01 / 旁人误触发 0.17 / 全链路 <1s；无第三方验证）、GLM-Realtime（音频 0.18 元/分钟） | 作为"模型自带拒识"的对照基线，量化外挂路线的相对收益 |

**关键判断**：外挂路线与端到端不是二选一。Qwen-Audio-3.1-Realtime 的官方架构本身就是"全双工决策模型 + S2T 响应模型"双路——决策与内容已内部分离；Gander 更进一步把时机训进"小脑"、把任务外挂给"大脑"。我们的差异化定位：**在不重训语音模型的前提下**，用外挂裁判（L1 信号 → L2 Jev 类有界决策 → L3 LLM 复判）+ 按形态分置的前端（音箱 DoA/波束、手机近讲检测）达到同类拒识水平，并保持策略可热更、决策可审计。裁判分级：端侧 L2（Qwen3 小模型）执行有界决策保证 10–50 ms 窗口，云端 L3 快分支（Jev）以 ~300 ms 处理低置信度分流、强分支（LLM）处理开放推理——把"该不该响应"从自由生成降为有界决策，更快、不可幻觉、置信度可校准；L3 整体异步，不阻塞静默判定。

### 4.4 方案穿刺

"穿刺"＝在正式立项前，用最小代价把方案的关键不确定性逐个扎穿。四个阶段，每阶段带 Go/No-Go 门槛：

**阶段 0：公开集基线（0a + 0b，合计约 1 周）**

- **0a（3–4 天，三套公开集先行）**：把候选链路（Silero VAD + CAM++ 声纹 + Paraformer 流式 + L2 端侧裁判）串成离线 harness，接入 CoDeTT + WearVox + ChorusDuplex（三套 HuggingFace 公开、无申请依赖），跑通判分与看板。
- **0b（2–3 天，HumDial 到位后）**：下载 FDBench `Humdial-Track2-Test.zip`（README 分布表已确认：拒识 2,000 = URB/PH/TPS 各 600 + SDO 200），解压核对子类标签后复用官方评测脚本跑分；标签缺失再走邮件申请（备份路径）。
- 产出：基线表（S1–S4 分子类拒识率 / FAR / SMR / self-talk 静音率），对照 HumDial 官方榜单（Baseline 35.2 / 冠军 72.2）与 WearVox 论文口径（整体准确率 29%–59%）定位。
- No-Go：公开集拒识率 < 70% 且误差无法归因到具体层 → 先补前端（空间/声纹）再谈裁判。

**阶段 1：延迟穿刺（约 1 周）**

- 全链路时延分解实测：VAD（<30 ms）+ **partial ASR** 首词（200–300 ms，不等 final）+ L1（<10 ms）+ L2 端侧裁判（10–50 ms）→ **静默判定合计 ≤ 500 ms 闭合**；L3 快分支（Jev，~300 ms）与强分支（LLM，100–500 ms）均**异步**。
- 门槛：正常响应首延迟 P90 ≤ 800 ms（不因裁判显著变慢）；应静默样本的静默决策窗 ≤ 500 ms；S0 真指令不被误吞（FRR 承诺 ≤ 3%，内部目标 ≤ 2%）；URB/PH 域回归无退化（时机判定归语义 VAD，本模块只验证不引入打断）。
- 穿刺点：L3 **不阻塞**静默判定与正常链路——静默/放行由 L1+L2 主判；L3 异步复判：快分支（Jev）~300 ms 出结果，强分支（LLM）超时 300 ms 即回退快分支结果（不回退端侧 L2，避免重复判定）。

**阶段 2：对抗穿刺（约 2 周）**

- 8 类困难 case 逐项验证（用尺子2 先导批，每类 ≥ 50 条）：
  1. side-talk 含"帮我/唤醒词"字样（对象是旁人）；
  2. 旁人喊主用户名字后继续说话；
  3. 主用户背对/侧身对旁人说话；
  4. 电视/播客中的指令式台词（B3）；
  5. 双说话人抢话 + 主用户附和（URB 嵌套；时机归语义 VAD，本模块验证不误判为归属问题）；
  6. self-talk 带完整请求句式（"这个放哪来着"）；
  7. 系统提问后的旁人抢答；
  8. 中英混说的 SDO。
- 每条错误归因到层（身份层错 / 语义层错 / 时序层错 / 延迟超限），反哺 L1 阈值与 L3 提示词。

**阶段 3：真机 A/B（约 2 周）**

- 真机 A/B 分两路：**音箱**（≥ 10 户家庭，含电视开/关两态，每户 ≥ 3 h）与**手机**（≥ 15 人，街道/车内/办公室，每人 ≥ 2 h）；眼镜真机为补充可选。
- 指标：主观打扰率（用户报告"抢话/插嘴"次数/小时）、误执行次数（高危，必须为 0）、公开集回归无退化。
- Go 门槛：S1–S4 分场景拒识率达标（1.3.5 目标）且**误执行 = 0**；S0 侧 FRR 承诺 ≤ 3%（内部 ≤ 2%）；FAR 按加权公式实算（监控，不设独立门槛）；主观打扰报告 ≤ 0.5 次/小时；时机类（误打断率 ≤ 8%，参照 Gander FD-Bench v3 抢话 8.0%）由语义 VAD 承担、仅回归。

**风险与回退**：

| 风险 | 缓解 | 回退 |
|---|---|---|
| L3 裁判延迟超标 | L3 整体异步（快分支 ~300 ms；强分支超时 300 ms 回退快分支结果）；静默判定由 L1+L2 窗内完成，不受 L3 影响 | 常驻 L1+L2 最小闭环 |
| ASR 错 → 裁判错 | 声纹/DoA 信号不依赖 ASR；低置信度默认静默+记录 | 端到端对照组 API 兜底（Qwen-Audio-3.1-Realtime） |
| L2 端侧小模型效果不足 | Qwen3-0.6B→1.7B→4B 升档；动作协议与评测口径与模型解耦 | 低置信度分流交 L3 快分支（Jev）兜底 |
| 中文真实数据不足（合成只够跑链路） | 尺子2 v1.0 先行 3,000 段；HumDial 中英真实数据注册申请 | 先英文公开集对齐，中文灰度放量 |
| 旁听缓存隐私合规 | 默认不参与回答、用户可查可删、端侧留存时限 | 关闭 silent_record，仅静默 |

**产出物**：穿刺报告（基线表、延迟分解表、对抗 case 错误归因、真机 A/B 数据）→ 作为 4.3 选型定稿与正式立项依据。
