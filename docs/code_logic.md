# GDN SpectralShift 代码逻辑

本仓库的交付主体是 [megatron-spectralshift.patch](../patches/megatron-spectralshift.patch)。本文解释补丁涉及的计算与运行时调用链，不收录参数注册、配方和可运行训练示例。

## 1. 方法与调用顺序

论文 §4.1–4.3、Algorithm 1 将方法分为两部分：加载参考权重后，一次性对 alpha 投影做中心化缩放；继续预训练期间，为 alpha 投影维持相对基础 LR 的倍率。

```text
GDN 构造：共享 input norm + q/k/v/z/b/a 六个独立参数
  ↓
optimizer 构造：LR callback → 数值倍率 → parameter groups
  ↓
checkpoint 加载：恢复权重、optimizer 和已变换 marker
  ↓
post-load：检查 marker / self-resume → alpha 中心化缩放
  ↓
optimizer.reload_model_params() → 同步 FP32 主权重 → 设置 marker
  ↓
训练：split 或 fused 前向 → 独立参数梯度 → base_lr(t) × lr_mult
  ↓
checkpoint 保存：持久化 gdn_len_ext_postinit_applied
```

权重缩放不进入每次 forward 的计算图，不是每个 step 重新施加的约束。论文取 `s = sqrt(L_ref / L_tar)`，补丁只消费传入的数值，不包含从训练长度推导参数的配置代码。

## 2. `gdn_alpha_weight_scale` 与 `gdn_alpha_weight_scale_mode`

原实现：`megatron/core/ssm/gated_delta_net.py::_apply_centered_weight_scale_`（检查时约 639 行）、`apply_split_alpha_weight_scale`（约 663 行）。补丁保留这两个方法的实现。

对 `a_proj.weight` 执行：先取得 FP32 权重；求均值；计算 `center + scale * (weight - center)`；转回原 dtype 后原地写入。Parameter 对象不更换。

| 模式 | 均值范围 | 论文对应 |
| --- | --- | --- |
| `matrix` | 每层完整 alpha 投影矩阵的一个标量均值 | 论文主方法 |
| `row` | `mean(dim=1, keepdim=True)`，每个输出行单独中心化 | 原实现的额外变体 |

scale=1 时直接返回；有限非负数均可传入，scale=0 会将偏差压到零，大于 1 则放大偏差。论文扩窗使用 `(0,1]`。原实现即使输入为 FP64，也先转 FP32 计算。

只处理 `a_proj.weight`，不处理 alpha projection bias、`dt_bias` 或 `A_log`。原方法要求 split 模式且 TP=1；在多个 TP shard 上各自求局部均值不能等价替代完整矩阵的均值。

GDN 的 retention gate 为 `exp(-exp(A_log) * softplus(a + dt_bias))`，其中 `a=W_alpha h`。缩小中心化偏差对不同符号的输入方向产生相反的门变化，因此不能将方法解释为“所有 alpha 都变大”或“直接缩放 alpha”。应用阶段也不需要 SVD。

原训练仓库另外存在 `gdn_postinit_alpha_scale` 所代表的整矩阵除法，以及 key 缩放和 A_log 平移。它们不属于本补丁；不要把这些旧干预当作 centered weight scale。

## 3. post-load 与恢复

原 `_maybe_apply_gdn_len_ext_postinit` 位于 `megatron/training/training.py` 约 1753 行。补丁将其中 SpectralShift 专用部分整理为 `megatron/training/gdn_spectralshift.py::apply_gdn_spectralshift_postload`，在 checkpoint 加载后、checkpoint 格式转换前调用。

跳过条件包括：恒等缩放、checkpoint 的 `gdn_len_ext_postinit_applied=True`，以及 `realpath(load)` 与输出 checkpoint namespace 相同。后者检查 `save`、`save/non_persistent` 和自定义非持久目录，用于兼容没有 marker 的旧 checkpoint。

真正的变换遍历 model shards 中类名为 `GatedDeltaNet` 的模块。变换结束后调用 `optimizer.reload_model_params()`，将已修改模型权重同步到普通 mixed-precision optimizer 的主权重。centered scale 方法本身没有直接修改 `.main_param`。优化器动量统计不在此重置。

补丁通过 common checkpoint state 保存和恢复 marker，并兼容旧 checkpoint args 中的同名字段。相较原 wrapper，提取版会在没有本地 GDN 的 pipeline stage 上也设置 marker，保证各 stage 表达的是同一个训练阶段状态。

开始新一轮扩窗与恢复同一轮训练必须区分：同一轮恢复保留 marker；有意开始下一阶段时，在 checkpoint 准备阶段清除上轮 marker，并切换输出 namespace。补丁不会自动推断多阶段扩窗意图。

## 4. `gdn_lr_mult_{q,k,v,alpha,beta}`

这部分有三个实际生效的位置：

| 原位置 | 职责 |
| --- | --- |
| `training/training.py` 约 1523 行 | 为五个 weight 后缀建立倍率回调 |
| `core/optimizer/__init__.py::_get_param_groups` 约 157 行 | 把 callback 结果解释为 bool 或数值并按倍率分组 |
| `core/optimizer_param_scheduler.py` 约 228 行 | `group['lr'] = new_lr * group.get('lr_mult', 1.0)` |

补丁将回调整理为 `gdn_projection_lr_condition`，在 optimizer 构造前接入；基线 scheduler 已有最后一步，无需修改。

| 原参数 | 匹配的 Parameter 后缀 |
| --- | --- |
| `gdn_lr_mult_q` | `.q_proj.weight` |
| `gdn_lr_mult_k` | `.k_proj.weight` |
| `gdn_lr_mult_v` | `.v_proj.weight` |
| `gdn_lr_mult_alpha` | `.a_proj.weight` |
| `gdn_lr_mult_beta` | `.b_proj.weight` |

非 1 的 GDN 倍率直接返回，覆盖已有 LR 规则；倍率为 1 的项回退到之前的 callback。它不是与已有的 µP 等倍率再次相乘。保留原代码的后缀匹配语义，不额外按类名限定，所以其他同名 projection 也可能被匹配。

bool 与数值必须区分：`True` 使用旧接口的通用 `lr_mult`，`False`/`None` 使用 1.0，数值直接用作倍率。特别是浮点数 0.0 不能因为 falsy 而回退到默认值。parameter group 同时保留 weight decay、expert 等原有分组维度。

z projection、各 projection bias、`A_log`、`dt_bias`、norm、卷积和输出投影不接受这五个专用倍率，但已有 callback 仍可能影响它们。补丁也不会自动冻结 `A_log` 和 `dt_bias`；论文实验中对这些参数的固定处理属于训练入口职责。

这是学习率缩放，不能用梯度乘系数替代。Adam 类优化器会对梯度做动量归一化，而 LR 控制最终更新。AdamW 的 group LR 也影响 decoupled weight decay 的步长。LR=0 可使参数不动，但 optimizer moments 仍可能更新。

## 5. `gdn_split_in_proj`

原构造位于 GDN 文件约 394 行，前向入口 `_prepare_gdn_input_projection` 约 1134 行。补丁将这部分移植到公开基线的 `__init__` 与 `forward`。

| 投影 | 总输出维度 | checkpoint section |
| --- | --- | --- |
| `q_proj` | `qk_dim` | `query` |
| `k_proj` | `qk_dim` | `key` |
| `v_proj` | `v_dim` | `value` |
| `z_proj` | `v_dim` | `z` |
| `b_proj` | `num_value_heads` | `beta` |
| `a_proj` | `num_value_heads` | `alpha` |

六个 ColumnParallelLinear 各自拥有独立 Parameter，顺序保持 q/k/v/z/b/a，所以能够分别进入 optimizer groups。非 split 模式仍走原单一 `in_proj`。原实现断言 TP=1，不能直接推定六段各自的 TP sharding 与原 fused matrix 的 TP sharding 相同。

输入 norm 必须只做一次。为避免为了选择线性层类型而引入一整套配置转发代码，补丁让 `GatedDeltaNetSubmodules` 同时提供：原 `in_proj`（内含 LayerNorm）、`split_in_proj`（plain linear）和 `in_proj_layer_norm`。构造阶段根据 split 开关选择。

split 分支共享一次 `in_proj_layer_norm`，六个 plain linear 接收同一个归一化结果；non-split 分支仍使用原融合 norm 的线性层。外层 `fuse_input_layernorm=True` 元信息保持原义，避免 Transformer 层重复做 norm。这是对公开基线的接口适配，计算含义与提取源相同。

## 6. `gdn_fused_split_forward`

补丁保留原 `_FusedSplitLinearFunc` 和 `_FusedSplitLinearBoundary` 的核心实现。

前向沿输出行拼接六份 weight 和可选 bias，做一次 `F.linear`。只有 split 分支会使用这个逻辑。bias 必须六个都有或六个都没有；零长度的 TE bias sentinel 按无 bias 处理。

直接将 `torch.cat(weights)` 交给普通 autograd 会保留那份拼接矩阵直到 backward。自定义 Function 保存 input 和原始权重的引用，在反向重新拼接，所以避免每层长期额外保留一份投影矩阵。临时拼接和反向重建仍有成本，补丁不宣称固定的加速比例。

反向将 leading dimensions 展平为二维，计算 `dX=dY@W`、`dW=dY.T@X`，再按六段输出行宽拆分 dW。bias 梯度是展平 dY 的列和。展平尤其能避免 `[tokens,1,width]` 布局触发广播 batched GEMM。

boundary 自己不注册参数或子模块，通过普通引用读取真正 owner 的 Parameter，因此 optimizer 与 checkpoint 中的参数归属不变。它每次读取当前 Parameter，支持 owner 替换权重对象后的正确引用。

有 `main_grad` 时，真实梯度累计到该 buffer，并设置 `grad_added_to_main_grad=True`；返回同 dtype 的 dummy gradient 以触发 DDP hooks。`zero_out_wgrad=True` 时返回零 dummy，否则可为未初始化张量。该协议要求 Megatron DDP 识别这些标记，不能把 dummy 当作普通 optimizer 的真实梯度。

## 7. DDP、delayed wgrad 与 checkpoint 键

fusion 绕过了六个 projection 自己的 `forward`。如果只在这些模块上挂参数 gather hook，forward 将错过等待参数同步的时机。补丁保留 boundary 的 `get_extra_ddp_param_gather_params`，由 DDP 映射到连续的 gather bucket chain，并在 boundary forward 前完成同步。

`backward_dw` 在普通 split 分支倒序调用六个 projection 的 delayed wgrad。在 fused split 分支，autograd 已经计算了 wgrad，不应再向 Transformer Engine 的 delayed store 取一次；只调用相应的 `wgrad_accumulation_and_reduce_hooks`。融合路径的 optimizer overlap 仍限定为 delay_wgrad_compute=False。

`sharded_state_dict` 将 split 权重 remap 为 `in_proj.weight.{query,key,value,z,beta,alpha}`，bias 使用同样的 section 名。输入 norm 的 weight remap 为 `in_proj.layer_norm_weight`，对应源实现的 RMSNorm 风格。映射通过原 ShardedTensor 保留 Parameter 身份，而不是普通复制字典值，否则 distributed optimizer 可能无法匹配。

公开基线原先只有 fused weight 的 section factory；补丁给可选 fused bias 也补齐相同 section factory。带输入 LayerNorm bias 的模型需要额外处理 norm bias remap，不能把这个 weight-only 接口视作通用 LayerNorm checkpoint converter。

## 8. 边界与来源

[base_manifest.json](base_manifest.json) 给出可公开获取的补丁基线及原文件 SHA-256。原逻辑提取自检查时的工作树，其 HEAD 为 `7d6b60342797e9ac808d9089608ee912915194ed`，包含本地修改；[source_manifest.json](source_manifest.json) 记录检查文件的 SHA-256，因此本文的原始行号以检查快照为准。

补丁没有纳入参数注册、实验配方、旧的 key/alpha 整体除法、A_log 平移、其他优化器、动态 CP/MoE dispatcher、训练日志或论文 PDF。它也没有为旧公开基线补齐现代 GDN 的 CP、packed sequence 或推理功能。

本地验证直接从应用后的 patch 文件中提取代码进行，发布目录不附带独立包或可运行示例。已检查补丁应用/撤销、全部文件语法、核心方法与提取源的 AST 一致性，以及 51 项 CPU 数值/接口检查。完整 GPU 训练和分布式 checkpoint round trip 尚未验证。
