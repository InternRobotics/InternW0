# W0 总体设计与实现说明

本文按模块说明当前代码的模型结构、训练计算图、数据接口与推理机制。

系统采用一个共享 VideoDiT、一个共享 ActionDiT 和观测引导的上下文路由模块（OVCR）。预训练时联合学习视频与动作；推理时复用视频计划，由动作模型结合最新观测分块执行。不同数据集和机器人本体通过动作侧的参数 bank 适配。

具体数据准备和域表命令见[多域预训练](PRETRAINING.md)；prompt 的位置、注意力与训练约束见[soft prompt 说明](SOFT_PROMPT_PRETRAIN_POSTTRAIN.md)。后续更新应直接修改对应模块说明，使本文与代码保持一致。

## 1. 模型组成

| 模块 | 输入与输出 | 训练方式 |
| --- | --- | --- |
| Wan VAE | 视频 RGB 与视频 latent 之间的转换 | 冻结 |
| VideoDiT | 根据图像、语言和视频噪声状态预测视频 flow，并提供各层视频 K/V | 联合训练 |
| DINOv3 | 当前 chunk 起点 RGB → patch features | 冻结，保持 eval |
| 观测投影与 query encoder | DINO features → OVCR queries | 联合训练 |
| OVCR | 最新视觉观测 queries 与基础视频 K/V → 当前 chunk 的视频 K/V | 联合训练 |
| ActionDiT | noisy action、视频 K/V、语言与 proprio → action flow | 联合训练 |
| 动作域 bank | 按域选择 soft prompt、action encoder/decoder、proprio encoder | 联合训练 |
| 文本 encoder | 指令 → text context；训练可预先缓存 | 冻结 |

默认 VideoDiT 为 30 层、hidden size 3072、FFN 14336；ActionDiT 为 30 层、hidden size 1024、FFN 4096。两者注意力接口均为 24 heads × 128。OVCR 默认每 chunk 使用 32 个 query。

**域条件只进入动作侧。** VideoDiT 不接收 `domain_id`，不添加域 embedding，也不读取 action token。域对应的 proprio token 进入 ActionDiT 的 context，视觉观测用于构造 OVCR query。

## 2. 联合预训练计算图

当前多域预训练配置使用 `future_training_mode: idm`。同一个 VideoDiT 分别处理视频 flow-matching 流与视频条件流，二者共享参数，但各自维护 hidden state、timestep 和噪声。

```text
示范视频 → frozen Wan VAE → z_gt
                           ├─ 视频噪声 / t_v → VideoDiT FM 流 → L_video
                           └─ 条件噪声 / t_c → VideoDiT 条件流 → 各层完整 K/V
                                                                    ↓
chunk 起点 RGB → frozen DINO → 观测投影 / queries ─────────────────→ OVCR
                                                                    ↓
原生动作 → 分域归一化 → 动作噪声 / t_a → 分域 action encoder → ActionDiT
                                                       ↑            ↓
                                               分域 soft prompt   分域 decoder
                                                                    ↓
                                                                 L_action

当前 proprio → 分域归一化 → 分域 proprio encoder → ActionDiT context
语言指令 → 冻结文本编码 / cache → VideoDiT 和 ActionDiT context
```

总损失为：

```text
L_total = lambda_video * L_video + lambda_action_prior * L_action
```

`L_video` 更新 VideoDiT；`L_action` 更新动作主干、动作域 bank、OVCR、观测投影，以及条件流中的共享 VideoDiT。完整条件 K/V 保留在 autograd 图中，所以动作监督能够影响视频表征。VAE/DINO 的冻结边界位于输入编码阶段。

### 2.1 独立的条件噪声

每个样本以 `idm_video_cond_noise_prob=0.5` 决定视频条件是否加噪。选中的样本使用独立 timestep 和 Gaussian noise，其余样本使用 clean video latent：

```text
z_fm   = (1 - sigma_v) * z_gt + sigma_v * noise_v
z_cond = (1 - sigma_c) * z_gt + sigma_c * noise_c
video target = noise_v - z_gt
action target = noise_a - action_gt
```

两条视频流的首帧均恢复为干净观测 latent，首帧 timestep 为 0。视频 FM 噪声、条件噪声与动作噪声分别采样；动作 timestep 支持按 chunk 采样或整段共享。

噪声在 activation checkpoint 外采样，重算复用相同状态。视频条件为 clean 时仍计算视频 FM loss。条件流的监督来自动作损失，不另加一份条件流的视频重建 loss。

### 2.2 逐层计算与注意力可见性

主要调用链为：

```text
W0.training_loss / _training_loss_action_offset
  → FutureVideoMixin._forward_training_video_action
  → MoT.forward_idm_action_with_chunk_updated_kv
  → MoT._forward_idm_layer
```

| Query 来源 | 视频 FM 流 | 视频条件流 | action / prompt |
| --- | --- | --- | --- |
| 视频 FM 流 | 视频 mask | 不可见 | 不可见 |
| 视频条件流 | 不可见 | 视频 mask | 不可见 |
| action / prompt | 不可见 | OVCR 更新后的完整有效 K/V | 当前 chunk 内可见 |

视频采用 `first_frame_causal`：首帧不读未来，未来部分按视频 mask 交互。动作侧按 chunk 独立路由，防止后续 noisy action 或后续观测经本地 token 泄漏到早期 chunk。

每层把该层**输入状态投影出的条件 K/V**交给 OVCR 和 ActionDiT，再推进下一层。训练逐层消费这些张量，不构造跨所有流的大型全局注意力矩阵。

## 3. 动作域 Bank 与 Soft Prompt

### 3.1 域身份与独立接口

域键为 `dataset/embodiment`。Registry 的有序 `domain_configs` 确定整数 ID；数据、模型、统计和 checkpoint 使用相同的域名、顺序、字段 schema 与维数。

每个域有四类参数：

| 参数 | 作用 |
| --- | --- |
| `soft_prompt_hub` | 为动作生成提供可学习的域条件 token |
| `action_encoder` | 原生动作空间 → 共享动作 hidden space |
| `head` | 共享动作 hidden space → 原生动作空间 |
| `proprio_encoder` | 原生 proprio → 动作 context token |

线性接口采用 dense 参数 bank，通过样本 ID gather 后进行批量矩阵运算。同一 batch 可含不同域，不同 rank 也可访问不同域，同时保持固定的参数图。

### 3.2 每个 Chunk 内插入 Prompt

默认 `K=8`、动作 horizon=64、chunk size=16，内部布局为：

```text
[P_d:8 | A_0:16] [P_d:8 | A_1:16] [P_d:8 | A_2:16] [P_d:8 | A_3:16]
```

四份 prompt 激活查同一行参数，进入 Transformer 后在各自 chunk 内独立演化。每域 prompt 参数量为 `8 × 1024 = 8192`。

Prompt 参与 Q/K/V 投影、自注意力、cross-attention、FFN 和残差计算；使用当前 chunk 的 timestep 调制与 context 可见性。它们使用 identity RoPE，不占动作的物理时间位置，不添加 diffusion noise，不进入 flow target。

内部序列由 64 个动作 token 变为 96 个 token；公开动作长度仍为 64。输出先按 chunk 去掉 prompt，再调用所选域的 decoder。物理 action step、内部 token 数和 cache slot 分开传递，非零 action offset 仍沿用真实动作位置。

### 3.3 不同维数的混合训练

当前 registry 包含 InternData-A1 的 4 个域和 RoboMIND2 的 7 个域，共 11 域。原生 action/proprio 维数为 7～26，具体字段见 [域注册表](../configs/data/domain_banks/internw0_domains.json)。

各域先按自身统计归一化，再右侧补零到最大维数。Encoder 入口屏蔽无效维度，decoder 的无效输出与推理动作 padding 置零。动作 loss 先按该样本的有效动作维数平均，再结合有效时间步和 scheduler weighting 计算。

`K=0` 且保留 bank 可作为无 prompt 的接口消融；`K=0` 且无 bank 使用原有共享接口路径。

## 4. 视频计划、动作时间与 Padding

设视频计划覆盖 H 个动作时间步，动作／视频采样比为 r，VAE 时间压缩因子为 f：

```text
sampled RGB frames = H / r + 1
latent frames      = H / (r * f) + 1
action_offset + action_horizon <= H
```

默认 H=64、r=8、f=4，对应 9 张 RGB、3 个 latent frame。对于 `[384,320]` 图像，每个 latent frame 有 120 个 patch token，完整视频计划共 360 tokens。四个动作 chunk 均能读取完整有效计划。

Action RoPE 使用真实动作时间。Offset=15 时，四个 chunk 起点为 `[15,31,47,63]`。Cache slot 只表示存储位置，部分执行后可以复用 slot，但必须更新动作位置。启用 history 或显式视频位置时，视频位置使用 sampled-frame 单位，与无 history 的默认 latent-frame 编号区分。

布局检查同时验证 horizon、stride、latent 数和 offset 覆盖。视频首帧必须有效；其余每 4 个 sampled frame 全部为 padding 时，才把对应 latent 标为 padding。有效 key mask 用于视频注意力、OVCR 和动作读取，动作时间 padding 用于 loss。

## 5. DINO 观测与 OVCR

当前 chunk 起点 RGB 从 `[-1,1]` 转为 DINO 所需的 ImageNet 标准化输入。默认冻结的 DINOv3 ViT-S/16 在 `[384,320]` 分辨率下输出 480 个 384 维 patch features，不使用 CLS token，也不做 feature L2 normalization。

训练中的每个 chunk 使用自身起点观测；推理使用最新观测。视频 latent 和生成仍由 Wan VAE 处理。DINO 权重按本地文件严格加载，模型身份与预处理配置参与 checkpoint 兼容检查。

观测特征经可训练投影和 query encoder 进入 OVCR。每层编辑过程为：

```text
queries 读取 base K/V → routed features
base tokens 读取 routed features → decoded features
delta projection → delta_K, delta_V
K_chunk = K_base + sigmoid(gate) * delta_K
V_chunk = V_base + sigmoid(gate) * delta_V
```

每个 chunk 都从同一份基础计划 K/V 出发，更新不累积回基础计划。32 个 query 用于路由，输出仍保留完整的 360 个视频 token。Delta 输出层零初始化，因此最初一步中部分观测上游参数的梯度可能为零；随着 delta 分支更新，观测路径参与学习。

## 6. 数据读取与归一化

多域 reader 使用用户选择的原生关节动作，加上夹爪／手部关节，读取记录的 `action(t)` command。Registry 明确记录有序 action/proprio 字段，避免用错误的 `robot_type` 标签或字段名称猜测动作含义。

天工 xsens 数据中的 12 维手指字段与 14 维关节动作拼成 26 维。InternData 与 RoboMIND 的相同本体仍按不同数据集分域，franka 的 1rgb/3rgb 采集构型也分别注册。

原生动作和 proprio 线性重采样至 50 Hz；视频按 PTS 选择最近源帧并校验时间。三相机画布上方 2/3 为主视角，下方为左右腕部视角，缺失槽填黑。默认样本形状为：

```text
action:  [64,26]
proprio: [65,26]
video:   [3,9,384,320]
context: [128,4096]
domain_id: scalar integer
```

默认先等概率选域，再在域内选择轨迹和窗口。Epoch/index 决定可复现的窗口采样，每轮更新窗口；域权重仅用于采样，不重复乘入 loss。归一化统计只从训练 split 聚合，每域独立保存 mean/std/count，并记录 registry 指纹与 split 参数。

当前多域 reader 支持 history=0、action offset=0、单任务 episode。RoboTwin
LeRobot reader 使用记录的原生 command 与 state，并支持 25 Hz full-window
Clean2Random 采样；多域预训练 reader 使用各数据源的原生命令语义。

## 7. 推理与异步缓存

慢系统根据当前图像和语言生成未来视频，发布完整的各层 K/V。缓存取自最后一次视频去噪 forward；`video_plan_latents` 则是 scheduler 更新后的 latent。

快系统每个 chunk 用最新视觉观测经 OVCR 更新视频上下文，并结合当前 proprio 和域参数运行 ActionDiT。`prefill_video(..., domain_id=...)` 把域固定到本次 inference state；切换域需要重新 prefill，VideoDiT 本身不读取该 ID。

推理接口按 `phase='video'` 与 `phase='action'` 暴露两阶段执行。视频阶段发布可复用的各层 K/V；动作阶段使用最新图像和 proprio 生成当前 chunk。RoboTwin / LIBERO 评测适配器负责 session reset、观测传输与控制节拍，实际控制率、相机时间、动作单位必须与训练定义一致。

## 8. Checkpoint 加载与续训

| 操作 | 入口 | 语义 |
| --- | --- | --- |
| 随机 DiT 预训练 | `skip_dit_load_from_pretrain: true`，`init_checkpoint: null` | DiT 从随机参数开始，冻结编码器使用预训练权重 |
| 原生权重初始化 | `init_checkpoint=/path/to/weights.pt` | 加载模型权重，以新 optimizer 和训练进度开始 |
| 增加 prompt | `init_soft_prompts=true` 配合 `init_checkpoint` | K=0 → K>0，其余接口和行为必须一致 |
| 完整续训 | `resume=/path/to/checkpoints/state/step_*` | 在兼容配置下恢复 optimizer、scheduler、RNG 和训练进度 |

权重加载发生在 `accelerator.prepare()` 之前，使 ZeRO FP32 master weights 从正确的模型参数建立。

`behavior_config` schema 3 记录视频读取／缓存模式、mask、horizon、时间单位、观测编码器，以及 K、有序域名、原生 schema 和 prompt 布局。`training_recipe` 记录训练模式、scheduler、loss、数据／统计指纹、split、采样和 LR 配方。加载先检查语义，再检查权重；仅有相同 shape 不足以允许恢复。

Released base, RoboTwin-clean, and LIBERO checkpoints are native `.pt` files
and are loaded directly.

## 9. 训练配置与执行入口

主配置为 [pretrain.yaml](../configs/pretrain.yaml)：

| 设置 | 当前默认值 |
| --- | --- |
| future training mode | IDM |
| action horizon / chunk size | 64 / 16 |
| video plan horizon / stride | 64 / 8 |
| history / action offset | 0 / 0 |
| prompt length / init std | 8 / 0.02 |
| 主干 LR / prompt LR 倍率 | `5e-5` / `0.1` |
| weight decay | `0.01` |
| optimizer | fused AdamW，按约 100M 元素分组 |
| precision | BF16 |
| condition noise probability | 0.5 |
| activation checkpoint | 整层重算，默认 stride=1 |

Prompt 与主干使用独立 optimizer 参数组，cosine 对各自初始 LR 乘相同因子，使 LR 比例保持不变。参数分组限制更新阶段的临时显存；activation checkpoint stride 控制存储和重算，不跳过模型层。

代码复用两视频流相同的文本投影，并先按帧计算相同 timestep 的投影再展开到空间 token。完整条件 K/V 和动作到视频的梯度路径保持不变。吞吐比较应使用包含数据、编码器、通信和 optimizer 的完整训练窗口。

在准备好第 6 节的数据统计和文本 cache 后，从新仓库目录启动：

```bash
cd <w0-release>
PYTHONPATH=src accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 2 scripts/train.py \
  --config-name pretrain batch_size=1
```

环境、模型资产与完整数据准备命令见[资产说明](ASSETS_AND_DATA.md)和[多域预训练](PRETRAINING.md)。机器路径由配置或环境变量覆盖；正式输出目录应单独指定。

## 10. 代码位置与验证范围

| 文件 | 职责 |
| --- | --- |
| [runtime.py](../src/w0/runtime.py) | registry 驱动模型构造、数据集和训练／推理入口 |
| [future_video.py](../src/w0/models/wan22/future_video.py) | 计划布局、条件噪声、训练分流和 recipe |
| [w0.py](../src/w0/models/wan22/w0.py) | 普通／offset loss、视觉与 proprio 路由、prefill |
| [mot.py](../src/w0/models/wan22/mot.py) | 双视频状态逐层计算、OVCR、chunk attention |
| [action_dit.py](../src/w0/models/wan22/action_dit.py) | 动作接口 bank 与 action forward |
| [soft_prompt.py](../src/w0/models/wan22/helpers/soft_prompt.py) | 域选择、prompt packing、RoPE、时间与 mask |
| [domain_bank.py](../src/w0/models/wan22/helpers/domain_bank.py) | dense 域线性接口 |
| [w0_chunk_base.py](../src/w0/models/wan22/w0_chunk_base.py) | 观测编码、分块推理和 checkpoint 保存加载 |
| [checkpoint_behavior.py](../src/w0/models/wan22/helpers/checkpoint_behavior.py) | 模型行为兼容检查 |
| [domain_video_dataset.py](../src/w0/datasets/domain_video_dataset.py) | 原生命令读取、视频时间对齐、归一化和采样 |
| [trainer.py](../src/w0/trainer.py) | 训练参数组、调度、分布式与恢复 |
| [RobotWin adapter](../experiments/robotwin/w0_policy/deploy_policy.py) | RoboTwin 固定时钟与训练统计／动作契约适配 |
| [LIBERO adapter](../experiments/libero/w0_policy/deploy_policy.py) | LIBERO EEF10 观测、动作和 session 适配 |

当前 release 回归覆盖 prompt packing、domain bank、IDM、checkpoint 行为、RoboTwin/LIBERO 数据契约和分布式恢复。另完成 8 卡 BF16 DDP 与 ZeRO-2 的 prompt／bank 同步检查，以及真实 8.143B target checkpoint 的完整加载和一step 视频／动作前向 smoke。

这些结果验证实现的连通性、契约和短步训练能力。完整预训练收敛、跨域迁移收益与 RoboTwin/LIBERO 闭环成功率仍需对应实验。
