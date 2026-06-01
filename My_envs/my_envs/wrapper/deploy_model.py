import os
import re
import shlex
import subprocess
import sys
import threading
import queue
import time
import glob
from datetime import datetime


# ============================================================
# 通用工具函数
# ============================================================

def _now_ts():
    return datetime.now().strftime("%H:%M:%S")


def run_subprocess_stream(command, env=None, prefix="PROC", heartbeat_secs=15):
    """
    实时运行子进程 + 流式输出 + 心跳提示
    返回 (returncode, full_output_text)
    """
    print("=" * 60)
    print(f"{_now_ts()} [{prefix}] 启动进程")
    print(f"{_now_ts()} [{prefix}] COMMAND:\n{command}")
    print("=" * 60, flush=True)

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=env
    )

    q = queue.Queue()
    output_lines = []
    last_output_time = time.time()

    def reader_thread(pipe, q):
        try:
            for line in iter(pipe.readline, ''):
                q.put(line)
        finally:
            pipe.close()

    reader = threading.Thread(target=reader_thread, args=(process.stdout, q), daemon=True)
    reader.start()

    while True:
        try:
            line = q.get(timeout=heartbeat_secs)
            last_output_time = time.time()
            output_lines.append(line)
            print(f"{_now_ts()} [{prefix}] {line}", end='', flush=True)
        except queue.Empty:
            if process.poll() is None:
                idle = int(time.time() - last_output_time)
                print(f"{_now_ts()} [{prefix}] ...still running (no output {idle}s)", flush=True)
            else:
                break

    reader.join(timeout=2)

    process.wait()
    ret = process.returncode
    full_output = ''.join(output_lines)

    print(f"\n{_now_ts()} [{prefix}] 进程结束 (exit code={ret})", flush=True)
    print("=" * 60)

    return ret, full_output


# ============================================================
# 主部署函数
# ============================================================

def deployment_func(selected_model_name, selected_model_params, datasets):
    machine_num = datasets["machine_num"]
    dataset_mtad = datasets["dataset_mtad"]
    dataset_InterFusion = datasets["dataset_InterFusion"]
    dataset_SDFVAE_train = datasets["dataset_SDFVAE_train"]
    dataset_SDFVAE_test = datasets["dataset_SDFVAE_test"]
    dataset_SDFVAE_evaluation = datasets["dataset_SDFVAE_evaluation"]

    deployment_model = selected_model_name

    print("\n" + "=" * 60)
    print(f"{_now_ts()} [DEPLOY] 本次部署模型: {deployment_model}")
    print(f"{_now_ts()} [DEPLOY] 参数配置:")
    for k, v in selected_model_params.items():
        print(f"    {k}: {v}")
    print("=" * 60)

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['TQDM_DISABLE'] = '1'

    # ========================================================
    # Omni
    # ========================================================
    if deployment_model == "Omni":
        param_names = ["z_dim", "window_length", "rnn_num_hidden",
                       "dense_dim", "batch_size", "level"]

        args = []
        for param_name in param_names:
            param_value = selected_model_params[param_name + "_Omni"]
            args.append(f"--{param_name}={shlex.quote(str(param_value))}")

        Omni_work_dir = r"C:\Projects\RL-ADS\space\model\OmniAnomaly"

        command = (
            f'conda run -n Omni cmd /c "cd /d {Omni_work_dir} '
            f'&& python -u main.py --dataset=machine-{machine_num} {" ".join(args)}"'
        )

        ret, output = run_subprocess_stream(command, env=env, prefix="Omni")

        if ret == 0:
            print(f"{_now_ts()} [Omni] 训练成功，解析 best-f1")
            f1_result = re.search(r"'best-f1':\s*([0-9]+\.[0-9]+)", output)
            if f1_result:
                f1 = float(f1_result.group(1))
                print(f"{_now_ts()} [Omni] best-f1 = {f1}")
                return f1
            else:
                print(f"{_now_ts()} [Omni] 未找到 best-f1")
                return 0
        else:
            print(f"{_now_ts()} [Omni] 训练失败")
            return 0

    # ========================================================
    # mtad-gat
    # ========================================================
    elif deployment_model == "mtad-gat":
        param_names = ["lookback", "epochs", "gru_n_layers",
                       "gru_hid_dim", "bs", "init_lr"]

        args = []
        for param_name in param_names:
            param_value = selected_model_params[param_name + "_mtad"]
            args.append(f"--{param_name}={shlex.quote(str(param_value))}")

        mtad_work_dir = r"C:\Projects\RL-ADS\space\model\mtad-gat-pytorch"

        command = (
            f'conda run -n mtad cmd /c "cd /d {mtad_work_dir} '
            f'&& python -u train.py {dataset_mtad} {" ".join(args)}"'
        )

        ret, output = run_subprocess_stream(command, env=env, prefix="MTAD")

        if ret == 0:
            f1_match = re.search(r"'f1':\s*([0-9]+\.[0-9]+)", output)
            if f1_match:
                f1 = float(f1_match.group(1))
                print(f"{_now_ts()} [MTAD] f1 = {f1}")
                return f1
            else:
                print(f"{_now_ts()} [MTAD] 未找到 f1")
                return 0
        else:
            print(f"{_now_ts()} [MTAD] 训练失败")
            return 0

    # ========================================================
    # InterFusion
    # ========================================================
    elif deployment_model == "InterFusion":
        param_names = ["model.window_length", "model.z_dim",
                       "model.z2_dim", "model.l2_reg",
                       "train.batch_size", "train.max_epoch"]

        args = []
        for param_name in param_names:
            param_value = selected_model_params[param_name + "_InterFusion"]
            args.append(f"--{param_name}={shlex.quote(str(param_value))}")

        InterFusion_work_dir = r"C:\Projects\RL-ADS\space\model\InterFusion"

        # ===== 步骤1: 训练 =====
        train_command = (
            f'conda run -n InterFusion cmd /c "cd /d {InterFusion_work_dir} '
            f'&& python -u -m algorithm.stack_train {dataset_InterFusion} {" ".join(args)}"'
        )

        ret, train_output = run_subprocess_stream(train_command, env=env, prefix="InterFusion-TRAIN")

        if ret != 0:
            print(f"{_now_ts()} [InterFusion] 训练失败")
            return 0

        print(f"{_now_ts()} [InterFusion] 训练完成，查找模型目录...")

        # ===== 步骤2: 查找最新的模型目录 =====
        result_dirs = glob.glob(os.path.join(InterFusion_work_dir, "results", "stack_train_*"))
        if not result_dirs:
            print(f"{_now_ts()} [InterFusion] 未找到训练结果目录")
            return 0

        latest_model_dir = max(result_dirs, key=os.path.getctime)
        model_dir_name = os.path.basename(latest_model_dir)
        print(f"{_now_ts()} [InterFusion] 使用模型目录: {model_dir_name}")

        # ===== 步骤3: 预测/评估 =====
        predict_command = (
            f'conda run -n InterFusion cmd /c "cd /d {InterFusion_work_dir} '
            f'&& python -u -m algorithm.stack_predict '
            f'--load_model_dir=./results/{model_dir_name} '
            f'--output-dir=./results/stack_predict/"'
        )

        ret, predict_output = run_subprocess_stream(predict_command, env=env, prefix="InterFusion-PREDICT")

        if ret != 0:
            print(f"{_now_ts()} [InterFusion] 预测失败")
            return 0

        # ===== 步骤4: 解析结果 =====
        # 从输出中查找 best-f1
        best_f1_match = re.search(r'best-f1[:\s]*([0-9.]+)', predict_output)
        if best_f1_match:
            f1 = float(best_f1_match.group(1))
            print(f"{_now_ts()} [InterFusion] best-f1 = {f1}")
            return f1

        # 尝试从结果文件读取
        result_json_path = os.path.join(InterFusion_work_dir, "results", "stack_predict", "result.json")
        if os.path.exists(result_json_path):
            try:
                import json
                with open(result_json_path, 'r') as f:
                    result_data = json.load(f)
                    if 'best-f1' in result_data:
                        f1 = float(result_data['best-f1'])
                        print(f"{_now_ts()} [InterFusion] 从文件读取 best-f1 = {f1}")
                        return f1
            except Exception as e:
                print(f"{_now_ts()} [InterFusion] 读取结果文件失败: {e}")

        print(f"{_now_ts()} [InterFusion] 未找到 best-f1")
        return 0

    # ========================================================
    # SDFVAE
    # ========================================================
    elif deployment_model == "SDFVAE":
        SDFVAE_work_dir = r"C:\Projects\RL-ADS\space\model\SDFVAE"

        train_epochs = selected_model_params["epochs_SDFVAE"]
        start_epoch = selected_model_params["epochs_SDFVAE"]

        commands = [
            # 训练
            (f'conda run -n SDFVAE cmd /c "cd /d {SDFVAE_work_dir} '
             f'&& python -u sdfvae\\trainer.py {dataset_SDFVAE_train} '
             f'--n 38 --gpu_id 0 --epochs {train_epochs}"',
             "TRAIN"),

            # 测试
            (f'conda run -n SDFVAE cmd /c "cd /d {SDFVAE_work_dir} '
             f'&& python -u sdfvae\\tester.py {dataset_SDFVAE_test} '
             f'--n 38 --gpu_id 0 --start_epoch {start_epoch}"',
             "TEST"),

            # 评估 - 使用 --llh_path 指向 tester 的输出目录
            (f'conda run -n SDFVAE cmd /c "cd /d {SDFVAE_work_dir} '
             f'&& python -u sdfvae\\evaluation.py '
             f'--llh_path {SDFVAE_work_dir}\\log_tester\\machine-{machine_num} '
             f'--log_path {SDFVAE_work_dir}\\log_evaluator\\machine-{machine_num} '
             f'--n 38 --start_epoch {start_epoch}"',
             "EVAL"),
        ]

        for cmd, stage in commands:
            ret, output = run_subprocess_stream(cmd, env=env, prefix=f"SDFVAE-{stage}")
            if ret != 0:
                print(f"{_now_ts()} [SDFVAE] {stage} 失败")
                return 0

        print(f"{_now_ts()} [SDFVAE] 全流程完成，解析 F1 分数...")

        # 读取评估结果文件
        evaluator_log_dir = os.path.join(SDFVAE_work_dir, f"log_evaluator\\machine-{machine_num}")

        if os.path.exists(evaluator_log_dir):
            eval_files = [f for f in os.listdir(evaluator_log_dir)
                         if os.path.isfile(os.path.join(evaluator_log_dir, f))]
            if eval_files:
                # 获取最新的评估结果文件
                latest_file = max(eval_files,
                                key=lambda f: os.path.getmtime(os.path.join(evaluator_log_dir, f)))
                latest_file_path = os.path.join(evaluator_log_dir, latest_file)

                print(f"{_now_ts()} [SDFVAE] 读取评估文件: {latest_file}")

                try:
                    with open(latest_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 匹配 "f1-best is: 0.xxxx"
                        f1_match = re.search(r'f1-best is:\s*([0-9.]+)', content)
                        if f1_match:
                            f1 = float(f1_match.group(1))
                            print(f"{_now_ts()} [SDFVAE] f1-best = {f1}")
                            return f1
                        else:
                            print(f"{_now_ts()} [SDFVAE] 未在文件中找到 f1-best")
                            print(f"{_now_ts()} [SDFVAE] 文件内容预览:\n{content[:500]}")
                            return 0
                except Exception as e:
                    print(f"{_now_ts()} [SDFVAE] 读取文件失败: {e}")
                    return 0
            else:
                print(f"{_now_ts()} [SDFVAE] 评估目录为空")
                return 0
        else:
            print(f"{_now_ts()} [SDFVAE] 评估目录不存在: {evaluator_log_dir}")
            return 0

    else:
        print(f"{_now_ts()} [DEPLOY] 未知模型: {deployment_model}")
        return 0