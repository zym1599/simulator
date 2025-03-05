################这个是decode阶段的代码


import math
from dataclasses import dataclass

@dataclass
class HardwareSpec:
    """硬件规格（支持混合算力单位）"""
    name: str                # 设备名称
    tflops: float = 0        # TFLOPS（用于非全连接部分）
    tops: float = 0          # TOPS（用于全连接/MLP/MoE）
    mem_bw: float  = 200          # 显存带宽（GB/s）
    link_bw: float = 50      # 跨设备带宽（GB/s）
    world_size: int = 1      # 本地并行规模

class SplitModelConfig:
    def __init__(
        self,
        ffn_on_device: str = "A",  # FFN部署的设备
        hidden_size: int = 7168,
        n_heads: int = 128,
        n_routed_experts: int = 256,
        experts_per_tok: int = 8,
        seq_len: int = 163840,
        inter_dim: int = 18432,      # MLP中间维度
        moe_inter_dim: int = 2048,   # MoE专家中间维度
        bitwidth: int = 8,            # 量化位宽（4/8/16）
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_dense_layers: int = 3

    ):
        self.ffn_on_device = ffn_on_device
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_routed_experts = n_routed_experts
        self.experts_per_tok = experts_per_tok
        self.seq_len = seq_len
        self.inter_dim = inter_dim
        self.moe_inter_dim = moe_inter_dim
        self.bitwidth = bitwidth
        self.dtype_bytes = bitwidth // 8
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_dense_layers = n_dense_layers
def estimate_cross_device_time(
    hw_a: HardwareSpec, 
    hw_b: HardwareSpec, 
    cfg: SplitModelConfig
) -> dict:
    """跨设备推理时间估算（支持TOPS/TFLOPS混合，覆盖所有全连接层）"""
    # ---------------------- MLA注意力层 ----------------------
    q_proj_flops = cfg.hidden_size * 1536 * (128*128 + 128*64) * 2  # Q低秩投影
    kv_proj_flops = cfg.hidden_size * 512 * 2                       # KV投影
    attn_flops = cfg.seq_len * cfg.n_heads * 128 * 2                # 注意力计算
    rope_flops = cfg.seq_len * cfg.hidden_size * 10                 # RoPE旋转
    mla_total_tflops = (q_proj_flops + kv_proj_flops + attn_flops + rope_flops) / 1e12

    # ---------------------- MLP前馈层 ----------------------
    mlp_flops = cfg.hidden_size * cfg.inter_dim * 2 * 3  # w1, w2, w3
    mlp_total_tflops = mlp_flops / 1e12

    # ---------------------- MoE专家层 ----------------------
    expert_flops_per_token = cfg.hidden_size * cfg.moe_inter_dim * 2 * 3  # 每个专家的三层
    moe_flops = expert_flops_per_token * cfg.experts_per_tok * cfg.seq_len / 1e12
    gate_flops = cfg.hidden_size * cfg.n_routed_experts * 2 / 1e12
    moe_total_tflops = (gate_flops + moe_flops) / hw_a.world_size

    # ---------------------- 共享专家层 ----------------------
    shared_expert_flops = cfg.hidden_size * cfg.moe_inter_dim * 2 * 3 / 1e12  # 3层MLP
    shared_total_tflops = shared_expert_flops

    # ---------------------- 全连接层显存访问 ----------------------
    ffn_weight_size = (cfg.hidden_size * cfg.inter_dim * 3) * cfg.dtype_bytes
    moe_weight_size = (cfg.n_routed_experts * cfg.hidden_size * cfg.moe_inter_dim * 3) * cfg.dtype_bytes
    mem_time = (ffn_weight_size + moe_weight_size) / (hw_a.mem_bw * 1e9) * 1e3  # ms

    # ---------------------- 设备分配与时间计算 ----------------------
    if cfg.ffn_on_device == "A":
        device_mla, device_ffn = hw_b, hw_a
    else:
        device_mla, device_ffn = hw_a, hw_b

    # MLA时间（基于TFLOPS）
    mla_time = mla_total_tflops / (device_mla.tflops * 0.8) * 1e3  # ms

    # FFN时间（基于TOPS）
    ffn_total_tops = (mlp_total_tflops + moe_total_tflops + shared_total_tflops)
    ffn_time = ffn_total_tops / (device_ffn.tops * 0.8) * 1e3  # ms

    # ---------------------- 跨设备通信 ----------------------
    transfer_data = cfg.hidden_size * cfg.seq_len * cfg.dtype_bytes  # 单向数据量
    transfer_time = (2 * transfer_data) / (min(hw_a.link_bw, hw_b.link_bw) * 1e9) * 1e3  # 双向传输

    # ---------------------- 总时间 ----------------------
    total_time = max(mla_time, ffn_time) + transfer_time + mem_time

    return {
        "MLA_device": device_mla.name,
        "FFN_device": device_ffn.name,
        "MLA_time(ms)": round(mla_time, 4),
        "MLP_time(ms)": round(mlp_total_tflops / (device_ffn.tops * 0.8) * 1e3, 4),
        "MoE_time(ms)": round(moe_total_tflops / (device_ffn.tops * 0.8) * 1e3, 4),
        "SharedExpert_time(ms)": round(shared_total_tflops / (device_ffn.tops * 0.8) * 1e3, 4),
        "Transfer_time(ms)": round(transfer_time, 4),
        "MemLoad_time(ms)": round(mem_time, 4),
        "Total_time(ms)": round(total_time, 4)
    }

def estimate_cross_device_time_decode(
    hw_a: HardwareSpec, 
    hw_b: HardwareSpec, 
    cfg: SplitModelConfig,
    n_layers: int = 61,         # 网络层数
    n_new_tokens: int = 128,    # 要生成的新token数
    avg_context_len: int = 1024 # decode时，注意力所面对的平均上下文长度
) -> dict:
    """
    估算decode阶段(逐token)的推理时间。
    :param hw_a: 硬件A，假设放置FFN
    :param hw_b: 硬件B，假设放置MLA(或反过来)
    :param cfg:  模型配置, hidden_size, n_heads, ...
    :param n_layers: 模型Block层数
    :param n_new_tokens: 要decode生成的token总数
    :param avg_context_len: 每次decode时的平均上下文序列长度(包含历史token数)
    :return: 字典，包含decode阶段各部分耗时(ms)及总耗时
    """
    # ------------------------------------------------------------------------
    # STEP 1: 选定哪块硬件跑注意力(MLA)，哪块硬件跑FFN(MLP/MoE)
    # ------------------------------------------------------------------------
    if cfg.ffn_on_device == "A":
        device_mla, device_ffn = hw_b, hw_a
    else:
        device_mla, device_ffn = hw_a, hw_b

    # ------------------------------------------------------------------------
    # STEP 2: 计算 "单步" MLA(多头注意力) FLOPs
    #   - 只对 1 个新token 做 Q投影 + (K/V 也可能只需对该token)
    #   - Attention点积规模: n_heads * head_dim * avg_context_len
    # ------------------------------------------------------------------------
    # 注意: 这里给出了一个简化示例，可根据实际rank/头维度再做更精细拆分

    # Q投影(低秩Q)大约:
    q_proj_flops_single = cfg.hidden_size * 1 * (cfg.q_lora_rank or 1) * 2  
    # 其中 "* 1"表示只对1个token, (cfg.q_lora_rank or 1)表示若q_lora_rank=0就用1(避免乘0)
    
    # K/V投影(针对1个token)，简化近似:
    kv_proj_flops_single = cfg.hidden_size * (cfg.kv_lora_rank or 1) * 2

    # QK点积 & Softmax： n_heads * avg_context_len * head_dim * 2
    # 这里 head_dim ~ (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) = 128+64=192(示例)
    attn_flops_single = cfg.n_heads * avg_context_len * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) * 2

    # RoPE旋转，对1个token做一些旋转，量级很小，这里给个固定常数近似:
    rope_flops_single = cfg.hidden_size * 10

    mla_flops_single = (q_proj_flops_single 
                        + kv_proj_flops_single
                        + attn_flops_single
                        + rope_flops_single)

    # 注意，每层都要做一次 MLA:
    mla_total_flops_single_layer = mla_flops_single
    # 全部n_layers:
    mla_total_flops_all_layers = mla_total_flops_single_layer * n_layers
    # 转TFlops:
    mla_total_tflops_single_step = mla_total_flops_all_layers / 1e12

    # ------------------------------------------------------------------------
    # STEP 3: 计算 "单步" FFN(MLP + MoE + shared_expert) FLOPs
    # ------------------------------------------------------------------------
    # 3.1  MLP 部分 (若layer是普通 dense层)
    mlp_flops_single = cfg.hidden_size * cfg.inter_dim * 2 * 3

    # 3.2  MoE部分 (若layer是MoE层, 也做简化处理, 在这里直接加总)
    #      experts_per_tok, n_routed_experts, moe_inter_dim等
    expert_flops_per_token = cfg.hidden_size * cfg.moe_inter_dim * 2 * 3
    moe_flops_single = expert_flops_per_token * cfg.experts_per_tok  # 对1个token
    gate_flops_single = cfg.hidden_size * cfg.n_routed_experts * 2
    # shared专家
    shared_expert_flops_single = cfg.hidden_size * cfg.moe_inter_dim * 2 * 3

    # 如果我们把n_dense_layers层都用mlp_flops, 其余(n_layers-n_dense_layers)层都用moe
    # 这是一种近似写法
    mlp_layers = cfg.n_dense_layers
    moe_layers = n_layers - mlp_layers

    mlp_total_flops = mlp_flops_single * mlp_layers
    moe_total_flops = (moe_flops_single + gate_flops_single + shared_expert_flops_single) * moe_layers

    ffn_total_flops_single_step = mlp_total_flops + moe_total_flops
    ffn_total_tflops_single_step = ffn_total_flops_single_step / 1e12

    # ------------------------------------------------------------------------
    # STEP 4: 分配到硬件 & 计算单步耗时 (ms)
    # ------------------------------------------------------------------------
    # MLA 由 device_mla (TFLOPS) 处理:
    mla_time_single_step = mla_total_tflops_single_step / (device_mla.tflops * 0.8) * 1e3

    # FFN 由 device_ffn (TOPS) 处理:
    # 这里把 FFN FLOPs 当做 "int8" 之类 => TOPS
    ffn_time_single_step = ffn_total_tflops_single_step / (device_ffn.tops * 0.8) * 1e3

    # 若能并行跑, 则单步计算时间 = max(mla_time_single_step, ffn_time_single_step)
    single_step_compute_time = max(mla_time_single_step, ffn_time_single_step)

    # ------------------------------------------------------------------------
    # STEP 5: 通信时间(每一步都要在设备间传递1个token的激活)
    # ------------------------------------------------------------------------
    # decode时, 假设激活是 hidden_size * dtype_bytes, 双向2倍
    transfer_data = cfg.hidden_size * cfg.dtype_bytes
    transfer_time_single_step = (2 * transfer_data) / (min(hw_a.link_bw, hw_b.link_bw) * 1e9) * 1e3

    # ------------------------------------------------------------------------
    # STEP 6: 其他(如内存加载)可以简化
    # decode过程通常权重不会反复加载, 已经驻留在显存中, 所以可忽略
    mem_time_single_step = 0.0

    # 如果需要输出词表投影
    #   hidden->vocab_size   roughly: hidden_size * cfg.vocab_size * 2
    #   这里演示, 如需可加:
    # vocab_flops_single_layer = cfg.hidden_size * cfg.vocab_size * 2
    # vocab_tflops_single_layer = vocab_flops_single_layer / 1e12
    # vocab_time_single_step = vocab_tflops_single_layer / (device_ffn.tops*0.8) * 1e3
    # single_step_compute_time += vocab_time_single_step

    # ------------------------------------------------------------------------
    # STEP 7: 汇总"单步"decode时间
    # ------------------------------------------------------------------------
    single_step_time_ms = single_step_compute_time + transfer_time_single_step + mem_time_single_step

    # ------------------------------------------------------------------------
    # STEP 8: 生成 n_new_tokens => 乘 n_new_tokens
    # ------------------------------------------------------------------------
    total_time_ms = single_step_time_ms * n_new_tokens

    return {
        "MLA_device": device_mla.name,
        "FFN_device": device_ffn.name,
        "MLA_time_single_step(ms)": round(mla_time_single_step, 4),
        "FFN_time_single_step(ms)": round(ffn_time_single_step, 4),
        "Transfer_time_single_step(ms)": round(transfer_time_single_step, 4),
        "Single_step_time(ms)": round(single_step_time_ms, 4),
        "Decode_steps": n_new_tokens,
        "Total_decode_time(ms)": round(total_time_ms, 4)
    }

# 示例：FPGA（TOPS）+ GPU（TFLOPS）异构部署
if __name__ == "__main__":
    # 硬件A（FPGA，专注全连接计算）
    hw_a = HardwareSpec(
        name="RRAM",
        tops=100,          # 250 TOPS（8bit量化）
        mem_bw=20000,
        link_bw=100
    )

    # 硬件B（GPU，负责注意力计算）
    hw_b = HardwareSpec(
        name="GPU",
        tflops=150,        # 150 TFLOPS（FP16）
        mem_bw=800,
        link_bw=100
    )

    # 模型配置（8bit量化，FFN部署在FPGA）
    model_cfg = SplitModelConfig(
        ffn_on_device="A",
        seq_len=1024,
        bitwidth=8
    )
    # 先估算 prefill
    prefill_result = estimate_cross_device_time(hw_a, hw_b, model_cfg)
    # 再估算 decode
    decode_result = estimate_cross_device_time_decode(hw_a, hw_b, model_cfg,
                                                  n_layers=61,
                                                  n_new_tokens=128,
                                                  avg_context_len=1024,
    )
    print("=== Prefill阶段 ===", prefill_result)
    print("=== Decode阶段 ===", decode_result)
    total_time = prefill_result["Total_time(ms)"] + decode_result["Total_decode_time(ms)"]
    print("推理总时间估算(ms):", total_time)

    # 执行估算
    result = estimate_cross_device_time(hw_a, hw_b, model_cfg)
    print("跨设备推理时间估算：")
    for k, v in result.items():
        print(f"{k:25}: {v}")