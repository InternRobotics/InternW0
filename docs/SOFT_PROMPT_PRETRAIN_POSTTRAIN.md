# w0 Soft Prompt：预训练、后训练与 X-VLA 对比

本文整理 w0 codebase 中 action soft prompt 的实际实现，重点回答三个问题：

1. 多数据集 / 多 domain 预训练时，soft prompt 和动作侧参数 bank 如何维护；
2. RoboTwin / LIBERO 等单目标域后训练时，目标域 bank 如何组织；
3. 该逻辑与参考 X-VLA implementation 的关系是否一致。

## 结论

w0 的 soft prompt 实现明确区分“预训练多域 bank”和“后训练目标域 bank”两种使用方式，但它不是两套完全独立的模型代码。更准确地说：

> w0 使用同一套 `ActionDiT + DomainLinear + ActionSoftPromptMixin` 实现，通过不同的 data registry、Hydra config、native checkpoint contract 和 `init_checkpoint / resume` 语义，区分预训练和后训练。

实际逻辑是：

- **多数据集 / 多 domain 预训练**：每个 `dataset × embodiment` domain 维护自己的：
  - soft prompt row
  - action encoder
  - action decoder / head
  - proprio encoder
  - normalization / schema / valid slots
- **RoboTwin / LIBERO 等单域后训练**：目标域通常只有 1 个 domain；
  - released RoboTwin-clean / LIBERO `.pt` checkpoint 已包含单域 action encoder、action head、soft prompt 和 proprio encoder；
  - 训练时通过 `init_checkpoint` 直接加载；
  - 之后在这个单域数据上训练。

这和 X-VLA 的核心思想是“一致来源、明显借鉴”，但不是逐行或逐语义复刻。w0 是把 X-VLA 式 domain-conditioned action head / soft prompt 机制迁移到 W0 的 chunked ActionDiT 和异步 world-action 架构里，并额外做了更适合 w0 的时序、mask、checkpoint 和 posttraining 语义。

---

## 1. w0 中 soft prompt 的参数结构

核心实现集中在这些文件：

```text
src/w0/models/wan22/helpers/soft_prompt.py
src/w0/models/wan22/helpers/domain_bank.py
src/w0/models/wan22/action_dit.py
src/w0/models/wan22/base_wam.py
src/w0/models/wan22/helpers/checkpoint_behavior.py
```

### 1.1 每个 domain 的 soft prompt

在 `ActionSoftPromptMixin._init_soft_prompts()` 中：

```python
self.soft_prompt_hub = nn.Embedding(
    len(self.soft_prompt_domains),
    self.num_soft_prompts * self.hidden_dim,
)
nn.init.normal_(self.soft_prompt_hub.weight, std=init_std)
```

参数含义是：

```text
soft_prompt_hub.weight: [num_domains, K * hidden_dim]
```

当前常用：

```text
K = num_soft_prompts = 8
hidden_dim = 1024
```

所以每个 domain 有：

```text
8 × 1024 = 8192 个 prompt 参数
```

如果是 11 个 domain：

```text
11 × 8192 = 90,112 个 prompt 参数
```

初始化方式是：

```text
Normal(mean=0, std=0.02)
```

这一点与 X-VLA 的 `soft_prompt_hub` 初始化方式一致。

### 1.2 不只有 soft prompt：动作侧也是一个 domain bank

w0 的 domain conditioning 不只是 prompt。

`ActionDiT.__init__()` 中，如果存在 `domain_configs`：

```python
self.action_encoder = DomainLinear(action_dim, hidden_dim, len(names))
self.head = DomainLinear(hidden_dim, action_dim, len(names))
```

也就是说每个 domain 有：

```text
action_encoder[domain]
head / action_decoder[domain]
soft_prompt[domain]
```

另外，`BaseWAM` 中如果有 domain bank，proprio encoder 也变成：

```python
self.proprio_encoder = DomainLinear(
    self.proprio_dim,
    self.text_dim,
    len(action_expert.domain_configs),
)
```

所以完整的动作侧 domain bank 是：

```text
sample.domain_id
  ├─ soft_prompt_hub[domain]
  ├─ action_encoder[domain]
  ├─ action_head / decoder[domain]
  └─ proprio_encoder[domain]
```

共享的部分是：

```text
VideoDiT 主干
ActionDiT Transformer 主干
OVCR / chunk KV editor
observation projector
chunk query encoder
文本编码器
VAE / DINO encoder
```

这符合设计目标：

> domain 只 condition ActionDiT，VideoDiT 不接收 domain ID，也不为每个 domain 复制一个世界模型。

---

## 2. soft prompt 在 ActionDiT 里怎么插入

w0 的默认动作时间设定是：

```text
action_horizon = 64
action_chunk_size = 16
num_soft_prompts = 8
```

因此 ActionDiT 内部变成：

```text
[P_d:8 | A_0:16]
[P_d:8 | A_1:16]
[P_d:8 | A_2:16]
[P_d:8 | A_3:16]
```

也就是：

```text
4 chunks × (8 prompt tokens + 16 action tokens)
= 4 × 24
= 96 内部 token
```

但模型最终输出仍然是：

```text
64 个物理 action step
```

prompt token 会在 `post_dit()` 前被剥离，不直接产生 action loss。

### 2.1 prompt 参数共享方式

四个 chunk 使用的是同一行 prompt 参数：

```python
prompt = self.soft_prompt_hub(ids).reshape(
    b, 1, k, self.hidden_dim
)
prompt.expand(b, n, k, self.hidden_dim)
```

也就是说：

```text
domain d 的第 1 个 chunk prompt 参数
domain d 的第 2 个 chunk prompt 参数
domain d 的第 3 个 chunk prompt 参数
domain d 的第 4 个 chunk prompt 参数
```

初始参数相同。

但因为它们在不同 chunk 中与不同的 action tokens、观测 context、timestep 一起过 Transformer，所以四个 chunk 里的 prompt activations 会独立演化。

这不是四份参数，而是：

```text
同一行参数，四个 chunk 各用一次
```

### 2.2 时间、mask 和 RoPE 约束

`_pack_soft_prompt_state()` 做了几件关键事情。

#### prompt 使用 chunk timestep

ActionDiT 的 diffusion timestep 是 per-chunk 的。w0 要求：

```text
同一个 chunk 内 action tokens 共享 timestep
```

然后 prompt 复制该 chunk 的 timestep。

因此 prompt 的 timestep mode 是：

```text
follow_chunk
```

#### prompt 使用 identity RoPE

真实 action token 保留原来的 action-step RoPE 位置。

prompt token 不占用物理 action 位置，而是使用 identity RoPE：

```text
prompt RoPE = 1 + 0i
```

这样插入 prompt 不会改变真实 action token 的物理位置，也不会破坏 offset 训练里的 action position semantics。

#### chunk-local self-attention

soft prompt 插入后，自注意力 mask 变成块对角：

```text
chunk 0 的 action + prompt 只能看 chunk 0
chunk 1 的 action + prompt 只能看 chunk 1
chunk 2 的 action + prompt 只能看 chunk 2
chunk 3 的 action + prompt 只能看 chunk 3
```

不会让第 1 个 chunk 的 prompt hidden state 直接读到未来 chunk 的 action token。

这是 w0 和 X-VLA 一个非常关键的差异，因为 W0 / w0 是 chunked action executor，不同 chunk 可能有不同观测和 diffusion timestep。

---

## 3. 多 domain 预训练时怎么维护和选择 prompt

多域预训练入口主要是：

```text
configs/pretrain.yaml
configs/data/pretrain.yaml
configs/data/domain_banks/internw0_domains.json
src/w0/datasets/domain_video_dataset.py
```

### 3.1 domain 身份

domain key 是：

```text
dataset / embodiment
```

例如：

```text
InternData-A1/franka
InternData-A1/genie1
RoboMIND2/franka_1rgb
RoboMIND2/tienkung_xsens_1rgb
...
```

当前 registry 中有 11 个 domain。

`domain_id` 是 registry 顺序下的整数 ID。为了避免“同一个数字在不同 registry 版本里含义不同”，w0 保存并校验：

```text
domain_configs
soft_prompt_domains
registry fingerprint
stats fingerprint
checkpoint behavior config
```

所以它不是简单相信 shape，而是要求语义一致。

### 3.2 数据侧返回 `domain_id`

`DomainVideoDataset.__getitem__()` 会返回：

```python
{
    "video": ...,
    "action": ...,
    "proprio": ...,
    "domain_id": torch.tensor(domain, dtype=torch.long),
    ...
}
```

模型训练时：

```python
domain_id = sample.get("domain_id")
```

然后传给：

```text
ActionDiT.pre_dit(..., domain_id=...)
BaseWAM._encode_domain_proprio(..., domain_id=...)
```

因此 mixed batch 中每个 sample 可以选择自己的 prompt row、action encoder row、head row 和 proprio encoder row。

### 3.3 `DomainLinear` 的实现

`DomainLinear` 不是为每个 domain 动态创建一个 `nn.Linear` module，而是保存 dense parameter bank：

```python
self.weight = nn.Parameter(
    torch.empty(num_domains, out_features, in_features)
)
self.bias = nn.Parameter(
    torch.empty(num_domains, out_features)
)
```

forward 时按 sample 的 `domain_id` gather：

```python
weights = self.weight[domain_id]
bias = self.bias[domain_id]
out = torch.bmm(x, weights.transpose(1, 2)) + bias
```

这样做的优点是：

- 参数图固定，适合 DDP / DeepSpeed ZeRO
- mixed-domain batch 可用一次 bmm
- 不需要 dynamic module switch
- checkpoint 里 bank 顺序和 schema 可以显式记录

需要注意一个细节：

> 某些 domain row 在当前 batch 中没被选中时，由于 dense parameter bank 的梯度是整块参数图，AdamW weight decay 或历史 momentum 仍可能影响未选中 row。

这不是 soft prompt 独有的问题，而是 dense bank + optimizer 的通用细节。开源文档里应该明确说明；如果未来要“只训新 domain、严格冻结旧 domain”，需要 optimizer-level row mask，而不是只靠梯度为 0。

---

## 4. w0 是否区分预训练和后训练？

**区分，但不是通过两套 soft prompt 模块区分，而是通过 config + data + checkpoint 语义区分。**

可以概括为：

```text
同一套 ActionDiT soft prompt 实现
        ↓
预训练：N-domain registry，N 行 prompt / encoder / decoder / proprio bank
        ↓
后训练：目标域 config N=1，直接加载 released target-domain `.pt` checkpoint
后训练：目标域 config N=1，直接加载 released target-domain `.pt` checkpoint
```

## 4.1 预训练阶段：多 domain，全 bank 训练

典型 config：

```text
--config-name pretrain
```

模型构建时通过：

```python
domain_registry_path = ...
```

读取 registry，然后 `runtime.create_w0()` 会覆盖 action config：

```python
action_dit_config = dict(
    action_dit_config,
    domain_configs=domain_configs,
    soft_prompt_domains=[d["name"] for d in domain_configs],
    action_dim=max(...),
)
```

因此模型里有 N 个 domain 的：

```text
soft prompt rows
action encoder rows
action head rows
proprio encoder rows
```

训练时每个 batch 根据 `sample.domain_id` 选择对应 row。

预训练阶段通常：

```text
init_checkpoint = null
resume = null
```

从随机或指定初始化开始，联合训练：

```text
VideoDiT
ActionDiT
OVCR
action encoder / decoder bank
proprio encoder bank
soft prompt bank
```

多域任务里 prompt 学习率通常更低：

```text
learning_rate = 5e-5
soft_prompt_lr_scale = 0.1
```

所以：

```text
backbone LR = 5e-5
soft prompt LR = 5e-6
```

`trainer.py` 中有专门分组：

```python
group_action_prompt_parameters(...)
```

把：

```text
soft_prompt_hub.parameters()
```

单独作为 optimizer group，并乘 `soft_prompt_lr_scale`。

## 4.2 后训练阶段：目标域单 bank checkpoint

RoboTwin 后训练入口例如：

```text
configs/posttrain_robotwin.yaml
```

目标域 config 中只有一个 domain：

```yaml
action_dit_config:
  action_dim: 37
  num_soft_prompts: 8
  soft_prompt_domains:
    - robotwin_wam/aloha
  domain_configs:
      - name: robotwin_wam/aloha
        dataset: robotwin_wam
      embodiment: aloha
      action_dim: 37
      proprio_dim: 37
      action_valid_indices: [...]
      proprio_valid_indices: [...]
```

数据 reader 也始终返回：

```python
domain_id = 0
```

因为目标 bank 只有这一行。

Released RoboTwin-clean and LIBERO checkpoints are already native `.pt` files
with this one-row target bank. They are loaded directly through
`init_checkpoint`.

---

## 5. 和 X-VLA 的逻辑是否一致？

结论：

> 核心思想一致，w0 明显借鉴了 X-VLA 的 domain soft prompt + domain-specific action interface 设计；但 token 放置、时序约束、domain registry、后训练迁移逻辑都不同。

参考 X-VLA revision 是：

```text
commit 6bc2513f5f1cbec715cc668b414392a6cae5c671
```

关键文件：

```text
X-VLA/models/transformer.py
X-VLA/models/modeling_xvla.py
X-VLA/models/configuration_xvla.py
X-VLA/datasets/domain_config.py
X-VLA/peft_train.py
X-VLA/train.py
```

## 5.1 一致的部分

### 1. 都是 domain-specific soft prompt bank

X-VLA：

```python
self.soft_prompt_hub = nn.Embedding(
    num_domains,
    len_soft_prompts * hidden_size
)
nn.init.normal_(self.soft_prompt_hub.weight, std=0.02)
```

w0：

```python
self.soft_prompt_hub = nn.Embedding(
    len(self.soft_prompt_domains),
    self.num_soft_prompts * self.hidden_dim
)
nn.init.normal_(self.soft_prompt_hub.weight, std=init_std)
```

基本结构一致：

```text
[num_domains, K * hidden_dim]
```

初始化 std 也都是 0.02。

### 2. 都有 domain-specific action encoder / decoder

X-VLA：

```python
self.action_encoder = DomainAwareLinear(...)
self.action_decoder = DomainAwareLinear(...)
```

w0：

```python
self.action_encoder = DomainLinear(...)
self.head = DomainLinear(...)
```

两者本质都是：

```text
按 domain_id 选择不同的线性输入 / 输出接口
```

只是实现细节不同。

### 3. 数据侧都传 `domain_id`

X-VLA dataset：

```python
sample["domain_id"] = torch.tensor(
    DATA_DOMAIN_ID.get(robot_type, 0)
)
```

w0 dataset：

```python
sample["domain_id"] = torch.tensor(domain)
```

### 4. 都给 soft prompt 单独学习率或学习率系数

X-VLA：

```python
soft_prompt_params = list(
    model.transformer.soft_prompt_hub.parameters()
)

param_groups = [
    ...,
    {
        "name": "soft_prompts",
        "params": soft_prompt_params,
        "lr": lr * lr_coef_soft,
    },
]
```

w0：

```python
group_action_prompt_parameters(
    trainable_params,
    prompt_hub.parameters(),
    lr=self.learning_rate,
    lr_scale=soft_prompt_lr_scale,
)
```

这也是一致的。

## 5.2 不一致的部分

| 项目 | X-VLA | w0 |
|---|---|---|
| 架构位置 | VLM + 单个 policy Transformer head | W0 的 VideoDiT + ActionDiT + OVCR |
| prompt 插入位置 | 整个序列后面追加一次 | 每个 action chunk 内前置 |
| 默认 prompt 长度 | 32 | 8 |
| 注意力 | 无 w0 这种 chunk-local block mask | chunk 内 action/prompt 互读，跨 chunk 隔离 |
| RoPE / 位置 | 普通 positional embedding | prompt identity RoPE，真实 action 位置不变 |
| timestep | 全局 action diffusion timestep | per-chunk timestep，prompt follow chunk |
| domain key | 主要是 hardcoded robot/dataset mapping | `dataset/embodiment` registry + schema |
| 有效维度 | action space / max dim padding | explicit valid slots + mask |
| proprio | action encoder 输入里拼 action+proprio+time | 单独 domain proprio encoder bank |
| Video / world model | 共享 VLM 表征 | VideoDiT 不接收 domain ID，ActionDiT 接收 |
| 后训练 | 通常加载固定 num_domains 的 pretrained bank，再 full/LoRA 微调 | 加载已包含目标单域 bank 的 native `.pt` |
| checkpoint 语义 | HF config / pretrained model | behavior_config + domain schema |

## 5.3 X-VLA 的 prompt 放置方式

X-VLA `SoftPromptedTransformer.forward()` 是：

```python
x = self.action_encoder(action_tokens, domain_id)

x = torch.cat(
    [x, vlm_features, aux_visual_inputs],
    dim=1,
)

x = x + self.pos_emb[:, :seq_len, :]

soft_prompts = self.soft_prompt_hub(domain_id).view(
    B, self.len_soft_prompts, self.hidden_size
)

x = torch.cat([x, soft_prompts], dim=1)

# through Transformer

return self.action_decoder(
    self.norm(x[:, :num_actions]),
    domain_id,
)
```

也就是：

```text
[action tokens | visual/language tokens | soft prompts]
```

prompt 追加在整条序列尾部。

w0 是：

```text
[P | chunk 0 actions]
[P | chunk 1 actions]
[P | chunk 2 actions]
[P | chunk 3 actions]
```

并且块间隔离。

这个差异不是小实现细节，而是由任务结构决定的：

> X-VLA 是一个整段 action sequence 的 policy head；w0 是 chunked asynchronous action executor，不同 chunk 有不同观测路由和 diffusion timestep，不能让 prompt hidden state 跨 chunk 全局互读。

## 5.4 X-VLA 后训练逻辑和 w0 不一致

X-VLA 的 fine-tuning 通常是：

```python
model = XVLA.from_pretrained(args.models)
```

也就是说加载一个已经固定 `num_domains` 的 pretrained bank。

然后通过：

```text
train.py
或
peft_train.py
```

训练：

```text
soft_prompt_hub
action_encoder
action_decoder
LoRA / backbone 部分
```

`peft_train.py` 里：

```python
modules_to_save = [
    "transformer.soft_prompt_hub",
    "transformer.action_encoder",
    "transformer.action_decoder",
]
```

这更像是：

```text
加载已有 domain row，然后微调
```

而 w0 的 RoboTwin posttraining 是：

```text
加载 released target-domain checkpoint
  * 单行 action encoder / head / prompt / proprio bank
  * 共享 world/action 主干
再在目标域数据上 posttrain
```

所以两者在“预训练 domain adapter 如何迁移到下游”这个层面并不等价。

尤其是 X-VLA 的 domain mapping 里存在：

```python
DATA_DOMAIN_ID.get(robot_type, 0)
```

未注册 robot type 会默认到 domain 0。对开源复现来说，这种隐式 fallback 容易造成语义混乱。w0 这方面更严格：registry、schema、checkpoint behavior 不匹配会直接报错。

---

## 6. 直接判断

当前理解是正确的：

```text
预训练：
  多 domain，每 domain 维护 soft prompt / action encoder / decoder / proprio encoder。

后训练：
  目标域单 bank；
  released target checkpoint 已包含单域 action-side 参数；
  然后单域 posttraining。
```

与 X-VLA 的关系应表述为：

> w0 采用了 X-VLA 式的 domain soft prompt 与 domain-specific action interface 思想，但为了适配 W0 的 chunked ActionDiT、OVCR 和异步 action generation，重新设计了 prompt 的插入位置、时序、RoPE、mask 和 native checkpoint contract。

这句话建议直接写进开源文档，避免读者误以为 w0 是 X-VLA 的直接复制，或者误以为两边 checkpoint / domain 语义可以直接互换。
