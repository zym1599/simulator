######这是prefill阶段的####



import math
from dataclasses import dataclass
import matplotlib.pyplot as plt
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
        bitwidth: int = 8            # 量化位宽（4/8/16）
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
    fc_time = ffn_time  #  ms
    # ---------------------- 跨设备通信 ----------------------
    transfer_data = cfg.hidden_size * cfg.seq_len * cfg.dtype_bytes  # 单向数据量
    transfer_time = (2 * transfer_data) / (min(hw_a.link_bw, hw_b.link_bw) * 1e9) * 1e3  # 双向传输

    # ---------------------- 总时间 ----------------------
    total_time = mla_time+ffn_time+ transfer_time + mem_time

    return {
        "MLA_device": device_mla.name,
        "FFN_device": device_ffn.name,
        "MLA_time(ms)": round(mla_time, 4),
        "MLP_time(ms)": round(mlp_total_tflops / (device_ffn.tops * 0.8) * 1e3, 4),
        "MoE_time(ms)": round(moe_total_tflops / (device_ffn.tops * 0.8) * 1e3, 4),
        "SharedExpert_time(ms)": round(shared_total_tflops / (device_ffn.tops * 0.8) * 1e3, 4),
        "FC_time(ms)": round(fc_time, 4),
        "Transfer_time(ms)": round(transfer_time, 4),
        "MemLoad_time(ms)": round(mem_time, 4),
        "Total_time(ms)": round(total_time, 4),
        "FC_time(ms)":round((mlp_total_tflops / (device_ffn.tops * 0.8) * 1e3)+(moe_total_tflops / (device_ffn.tops * 0.8) * 1e3), 4)
    }

# RRAM（TOPS）+ GPU（TFLOPS）异构部署
if __name__ == "__main__":
    # 硬件A（FPGA，专注全连接计算）
    hw_a = HardwareSpec(
        name="RRAM",
        tops=100,      # 这里初始写 100，后面会在循环里改
        mem_bw=20000,
        link_bw=100
    )

    # 硬件B（GPU，负责注意力计算）
    hw_b = HardwareSpec(
        name="GPU",
        tflops=150,    # 150 TFLOPS
        mem_bw=800,
        link_bw=100
    )

    # 模型配置
    model_cfg = SplitModelConfig(
        ffn_on_device="A",
        seq_len=1024,
        bitwidth=8
    )

    # 准备记录 x轴(tops) 和 y轴(total_time)
    tops_values = []
    fc_time_values = []

    # 在 50 ~ 200 的范围内遍历 TOPS，每步+1
    for tops in range(50, 201):
        # 更新 hw_a 的TOPS
        hw_a.tops = tops
        
        # 调用估算函数
        result = estimate_cross_device_time(hw_a, hw_b, model_cfg)
        
        # 记录对应的 (x, y)
        tops_values.append(tops)
        fc_time_values.append(result["FC_time(ms)"])

    # 输出部分查看
    print("Tops -> Total_time(ms):")
    for i in range(len(tops_values)):
        print(f"{tops_values[i]}  =>  {fc_time_values[i]}")

    # 画折线图
    plt.plot(range(50, 201), fc_time_values)
    plt.xlabel("TOPS (Hardware A)")
    plt.ylabel("FC_time(ms)")
    plt.title("Fully-Connected Time vs. TOPS")
    plt.show()