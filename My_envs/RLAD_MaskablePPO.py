"""
快速修复版本 - 包含最常见问题的解决方案
使用方法: python RLAD_MaskablePPO.py 1-1
"""
import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from my_envs.envs import RLADEnv
from my_envs.lib.parameter import parameters
import sys
import torch
import traceback

def find_env_with_attr(env, attr='valid_action_mask'):
    """
    穿透 wrappers，返回第一个实现 attr 的底层 env。
    如果找不到则抛出 AttributeError。
    这个函数比直接使用 env.unwrapped 更稳健，能处理各种 wrapper 链。
    """
    current = env
    visited = set()
    while True:
        if hasattr(current, attr):
            return current
        # 防止无限循环（某些 wrapper 可能循环引用）
        if id(current) in visited:
            break
        visited.add(id(current))
        # 常见的 wrapper 链字段：env, envs, unwrapped
        # 先尝试 .env（多数 wrapper 有）
        if hasattr(current, 'env') and current.env is not current:
            current = current.env
            continue
        # gymnasium 的 unwrapped
        if hasattr(current, 'unwrapped') and current.unwrapped is not current:
            # 注意：unwrapped 可能直接返回底层 env
            current = current.unwrapped
            continue
        # 有些自定义 wrapper 可能把原始 env 放在 .venv / .envs / .envs attribute
        if hasattr(current, 'venv') and current.venv is not current:
            current = current.venv
            continue
        if hasattr(current, 'envs') and current.envs is not current:
            # envs 可能是 VecEnv 等，不能直接索引 - 停止查找
            break
        break
    raise AttributeError(f"没有找到实现 '{attr}' 的底层 env。最外层对象类型: {type(env)}")

def mask_fn(env) -> np.ndarray:
    """ ActionMasker 所需的 mask 函数：返回 1D bool numpy 数组 """
    base = find_env_with_attr(env, 'valid_action_mask')
    mask = base.valid_action_mask()
    mask = np.asarray(mask)
    # 确保是 1D
    mask = mask.ravel()
    if mask.dtype != np.bool_:
        try:
            mask = mask.astype(bool)
        except Exception:
            # 如果无法转换，抛出有用的错误供诊断
            raise TypeError(f"valid_action_mask 返回值不能转换为 bool 数组，dtype={mask.dtype}, sample={mask[:20]}")
    return mask

def compute_expected_mask_length(action_space, fallback_len=None):
    """ 根据 action_space 计算期望的 mask 长度（支持 MultiDiscrete） """
    try:
        if hasattr(action_space, 'nvec'):  # MultiDiscrete
            return int(np.sum(action_space.nvec))
        # 离散单一动作
        if hasattr(action_space, 'n'):
            # 离散类的 n 表示动作数量
            return int(action_space.n)
        # 连续或其他，尝试用 shape 展平
        if hasattr(action_space, 'shape') and action_space.shape is not None:
            prod = 1
            for s in action_space.shape:
                prod *= int(s)
            return int(prod)
    except Exception:
        pass
    # 回退值
    return int(fallback_len) if fallback_len is not None else None

def print_wrapper_chain(env):
    """ 打印 wrapper 链（用于诊断）"""
    cur = env
    depth = 0
    visited = set()
    while True:
        if id(cur) in visited:
            print(f"    [loop detected] depth {depth}: {type(cur)}")
            break
        visited.add(id(cur))
        print(f"    depth {depth}: {type(cur)}")
        # 尝试下一个常见字段
        if hasattr(cur, 'env') and cur.env is not cur:
            cur = cur.env
            depth += 1
            continue
        if hasattr(cur, 'unwrapped') and cur.unwrapped is not cur:
            cur = cur.unwrapped
            depth += 1
            continue
        break

def train_model(machine_num):
    """ 训练 MaskablePPO 模型 """
    if not machine_num:
        print("错误：machine_num 为空！")
        return

    print(f"使用的 machine_num: {machine_num}")

    # === 新增：检查 CUDA 状态 ===
    print(f"\n[系统信息]")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU 内存: {memory_gb:.2f} GB")
        except Exception:
            pass
    print()

    datasets = {
        "machine_num": machine_num,
        "dataset_Omni": f"--dataset machine-{machine_num}",
        "dataset_mtad": f"--dataset SMD --group {machine_num}",
        "dataset_InterFusion": f"--dataset=machine-{machine_num}",

        # 数据就在 SDFVAE 项目内，用绝对路径
        "dataset_SDFVAE_train": f"--dataset_path C:\\Projects\\RL-ADS\\space\\model\\SDFVAE\\data_preprocess\\data_processed\\machine-{machine_num}-train "
                                f"--log_path C:\\Projects\\RL-ADS\\space\\model\\SDFVAE\\log_trainer\\machine-{machine_num} "
                                f"--checkpoints_path C:\\Projects\\RL-ADS\\space\\model\\SDFVAE\\model\\machine-{machine_num}",
        "dataset_SDFVAE_test": f"--dataset_path C:\\Projects\\RL-ADS\\space\\model\\SDFVAE\\data_preprocess\\data_processed\\machine-{machine_num}-test "
                               f"--log_path C:\\Projects\\RL-ADS\\space\\model\\SDFVAE\\log_tester\\machine-{machine_num} "
                               f"--checkpoints_path C:\\Projects\\RL-ADS\\space\\model\\SDFVAE\\model\\machine-{machine_num}",
        "dataset_SDFVAE_evaluation": f"--llh_path C:\\Projects\\RL-ADS\\space\\model\\SDFVAE\\log_tester\\machine-{machine_num} "
                                     f"--log_path C:\\Projects\\RL-ADS\\space\\model\\SDFVAE\\log_evaluator\\machine-{machine_num}",
    }

    # **创建环境**
    print("[1/4] 创建环境...")
    env = gym.make('my_envs/RLAD-v0', parameters=parameters, datasets=datasets)
    print("      ✓ 环境创建完成")

    # === 新增：验证 mask 形状 ===
    print("[2/4] 验证 action mask...")
    try:
        # 在验证时穿透 wrapper 找到实现 valid_action_mask 的底层 env
        base_env = find_env_with_attr(env, 'valid_action_mask')
        print(f"      ✓ 在底层 env 找到 valid_action_mask，类型: {type(base_env)}")
        test_mask = base_env.valid_action_mask()
        test_mask = np.asarray(test_mask).ravel()
        expected_len = compute_expected_mask_length(env.action_space, fallback_len=test_mask.size)
        if expected_len is None:
            print("      ⚠ 无法计算期望的 mask 长度（action_space 未识别），将使用 mask 的实际长度作为期望值。")
            expected_len = int(test_mask.size)

        if test_mask.size != expected_len:
            print(f"      ✗ 错误: Mask 长度 {test_mask.size} != 期望长度 {expected_len}")
            print(f"      - action_space: {env.action_space}")
            print(f"      - mask dtype: {test_mask.dtype}")
            print(f"      - mask sample (前20): {test_mask[:min(20, test_mask.size)]}")
            print("      - wrapper 链 (最外层 -> 底层):")
            print_wrapper_chain(env)
            return

        if test_mask.dtype != np.bool_:
            print(f"      ⚠ mask dtype 不是 bool（{test_mask.dtype}），脚本会尝试在运行时转换为 bool。")

        print(f"      ✓ Mask 验证通过 (长度: {test_mask.size})")
    except AttributeError as e:
        print(f"      ✗ Mask 验证失败: {e}")
        print("      - wrapper 链 (最外层 -> 尝试穿透):")
        print_wrapper_chain(env)
        return
    except Exception as e:
        print(f"      ✗ Mask 验证失败: {e}")
        traceback.print_exc()
        return

    print("[3/4] 应用 ActionMasker...")
    # 当 ActionMasker 在运行时调用 mask_fn 时，mask_fn 会再次穿透 wrapper 去调用底层实现
    env = ActionMasker(env, mask_fn)
    print("      ✓ ActionMasker 应用完成")

    # **训练 MaskablePPO 模型**
    print("[4/4] 初始化 MaskablePPO...")
    print("      （这可能需要 10-30 秒，请耐心等待...）")

    # === 修复：强制使用 CPU，避免 GPU 初始化问题（如需要使用 GPU，请改为 device='cuda' 并确认版本兼容） ===
    try:
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            env,
            verbose=1,
            device='cuda',  # 关键修复：强制使用 GPU，避免某些 GPU/驱动/库 的初始化错误
            # === 可选：减少内存使用 ===
            n_steps=128,  # 减少步数（默认 2048）
            batch_size=64,  # 减少批次大小（默认 64）
            n_epochs=5,  # 减少训练轮数（默认 10）
        )
        print("      ✓ MaskablePPO 初始化完成！")
    except Exception as e:
        print(f"      ✗ 初始化失败: {e}")
        traceback.print_exc()
        return

    print("\n开始训练...")
    print("=" * 60)
    try:
        model.learn(total_timesteps=500)
    except Exception as e:
        print("训练过程中出现异常:", e)
        traceback.print_exc()
    print("=" * 60)

    print(f"\n训练完成！Machine: {machine_num}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("错误：缺少 machine_num 参数")
        print("使用方法: python RLAD_MaskablePPO_fixed.py 1-1")
        sys.exit(1)

    machine_num = sys.argv[1]

    print("=" * 60)
    print("MaskablePPO 训练 - 修复版本")
    print("=" * 60)
    print()

    try:
        train_model(machine_num)
        print("\n运行完成")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n程序异常: {e}")
        traceback.print_exc()