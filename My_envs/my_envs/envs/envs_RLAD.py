import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import MultiDiscrete

from my_envs.lib import mask as model_des
from my_envs.wrapper.deploy_model import deployment_func


class RLADEnv(gym.Env):
    def __init__(self, parameters, datasets):
        self.parameters = parameters  # 获取参数列表
        self.datasets = datasets  # 部署的数据集

        self.reset_num = 0  # 记录重置的次数，也就是部署的次数

        self.step_num = 0  # 记录交互的次数，也就是部署的次数
        self.negative_num = 0  # 记录不利探索的次数

        self.last_select_model_name = ""  # 上一次选择的模型
        self.select_model_name = ""  # 这一次选择的模型

        self.model_equal = 0  # 用来记录上一次选择的模型和当前模型是否相同，0与上次不同，1与上次相同，2与上次相同且与上上次也相同
        self.model_delete_num = 0  # 用来记录裁剪的模型数量

        self.flunk = 0  # 模型是否不及格，小于等于0.65为不及格，设为1，及格为0

        # 0表示无效性能指标
        self.first_performance = 0.0  # 记录第一次交互的性能表现
        self.last_performance = 0.0  # 记录上一次交互的性能表现

        self.last_reward = 0  # 记录上一次的奖励

        # 用来记录模型的掩码
        self.mask_Omni_model = model_des.model_Omni_mask_Y
        self.mask_Omni = model_des.Omni_mask_Y

        self.mask_mtad_model = model_des.model_mtad_mask_Y
        self.mask_mtad = model_des.mtad_mask_Y

        self.mask_InterFusion_model = model_des.model_InterFusion_mask_Y
        self.mask_InterFusion = model_des.InterFusion_mask_Y

        self.mask_SDFVAE_model = model_des.model_SDFVAE_mask_Y
        self.mask_SDFVAE = model_des.SDFVAE_mask_Y
        # 初始重置mask
        self.mask = (self.mask_Omni_model + self.mask_mtad_model + self.mask_InterFusion_model + self.mask_SDFVAE_model
                     + self.mask_Omni + self.mask_mtad + self.mask_InterFusion + self.mask_SDFVAE)

        self.models = ["Omni", "mtad-gat", "InterFusion", "SDFVAE"]

        self.parameter1_1 = "z_dim_Omni"
        self.parameter1_2 = "window_length_Omni"
        self.parameter1_3 = "rnn_num_hidden_Omni"
        self.parameter1_4 = "dense_dim_Omni"
        self.parameter1_5 = "batch_size_Omni"
        self.parameter1_6 = "level_Omni"

        self.parameter2_1 = "lookback_mtad"
        self.parameter2_2 = "epochs_mtad"
        self.parameter2_3 = "gru_n_layers_mtad"
        self.parameter2_4 = "gru_hid_dim_mtad"
        self.parameter2_5 = "bs_mtad"
        self.parameter2_6 = "init_lr_mtad"

        self.parameter3_1 = "model.window_length_InterFusion"
        self.parameter3_2 = "model.z_dim_InterFusion"
        self.parameter3_3 = "model.z2_dim_InterFusion"
        self.parameter3_4 = "model.l2_reg_InterFusion"
        self.parameter3_5 = "train.batch_size_InterFusion"
        self.parameter3_6 = "train.max_epoch_InterFusion"

        self.parameter4_1 = "learning_rate_SDFVAE"
        self.parameter4_2 = "s_dims_SDFVAE"
        self.parameter4_3 = "d_dims_SDFVAE"
        self.parameter4_4 = "conv_dims_SDFVAE"
        self.parameter4_5 = "hidden_dims_SDFVAE"
        self.parameter4_6 = "epochs_SDFVAE"

        # 构建观察空间
        self.observation_space = spaces.Dict({
            # models
            "model": spaces.Discrete(len(self.models)),  # 0->Omni , 1->mtad-gat , 2->InterFusion, 3->SDFVAE
            # parameters
            "parameter1": spaces.Discrete(6),
            "parameter2": spaces.Discrete(6),
            "parameter3": spaces.Discrete(6),
            "parameter4": spaces.Discrete(6),
            "parameter5": spaces.Discrete(6),
            "parameter6": spaces.Discrete(6),
            # performance
            "performance": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        # 构建动作空间
        discrete_ranges = [
            len(self.models),  # 0->Omni , 1->mtad-gat , 2->InterFusion, 3->SDFVAE
            # Omni
            len(parameters[0][self.parameter1_1]),
            len(parameters[0][self.parameter1_2]),
            len(parameters[0][self.parameter1_3]),
            len(parameters[0][self.parameter1_4]),
            len(parameters[0][self.parameter1_5]),
            len(parameters[0][self.parameter1_6]),
            # mtad-gat
            len(parameters[1][self.parameter2_1]),
            len(parameters[1][self.parameter2_2]),
            len(parameters[1][self.parameter2_3]),
            len(parameters[1][self.parameter2_4]),
            len(parameters[1][self.parameter2_5]),
            len(parameters[1][self.parameter2_6]),
            # InterFusion
            len(parameters[2][self.parameter3_1]),
            len(parameters[2][self.parameter3_2]),
            len(parameters[2][self.parameter3_3]),
            len(parameters[2][self.parameter3_4]),
            len(parameters[2][self.parameter3_5]),
            len(parameters[2][self.parameter3_6]),
            # SDFVAE
            len(parameters[3][self.parameter4_1]),
            len(parameters[3][self.parameter4_2]),
            len(parameters[3][self.parameter4_3]),
            len(parameters[3][self.parameter4_4]),
            len(parameters[3][self.parameter4_5]),
            len(parameters[3][self.parameter4_6]),
        ]
        self.action_space = MultiDiscrete(discrete_ranges)  # 构建动作空间
        print("模型数量", len(self.models))
        print("观察空间：\n", self.observation_space)
        print("动作空间：\n", self.action_space)

    def get_model_info(self, action):
        # 根据动作获得具体的参数值
        if action[0] == 0:
            self.select_model_name = "Omni"
        elif action[0] == 1:
            self.select_model_name = "mtad-gat"
        elif action[0] == 2:
            self.select_model_name = "InterFusion"
        elif action[0] == 3:
            self.select_model_name = "SDFVAE"
        # 得到抽样的参数
        selected_model_params = {}
        if self.select_model_name == "Omni":
            selected_model_params = {
                self.parameter1_1: self.parameters[0][self.parameter1_1][action[1]],
                self.parameter1_2: self.parameters[0][self.parameter1_2][action[2]],
                self.parameter1_3: self.parameters[0][self.parameter1_3][action[3]],
                self.parameter1_4: self.parameters[0][self.parameter1_4][action[4]],
                self.parameter1_5: self.parameters[0][self.parameter1_5][action[5]],
                self.parameter1_6: self.parameters[0][self.parameter1_6][action[6]],
            }
        elif self.select_model_name == "mtad-gat":
            selected_model_params = {
                self.parameter2_1: self.parameters[1][self.parameter2_1][action[7]],
                self.parameter2_2: self.parameters[1][self.parameter2_2][action[8]],
                self.parameter2_3: self.parameters[1][self.parameter2_3][action[9]],
                self.parameter2_4: self.parameters[1][self.parameter2_4][action[10]],
                self.parameter2_5: self.parameters[1][self.parameter2_5][action[11]],
                self.parameter2_6: self.parameters[1][self.parameter2_6][action[12]],
            }
        elif self.select_model_name == "InterFusion":
            selected_model_params = {
                self.parameter3_1: self.parameters[2][self.parameter3_1][action[13]],
                self.parameter3_2: self.parameters[2][self.parameter3_2][action[14]],
                self.parameter3_3: self.parameters[2][self.parameter3_3][action[15]],
                self.parameter3_4: self.parameters[2][self.parameter3_4][action[16]],
                self.parameter3_5: self.parameters[2][self.parameter3_5][action[17]],
                self.parameter3_6: self.parameters[2][self.parameter3_6][action[18]],
            }
        elif self.select_model_name == "SDFVAE":
            selected_model_params = {
                self.parameter4_1: self.parameters[3][self.parameter4_1][action[19]],
                self.parameter4_2: self.parameters[3][self.parameter4_2][action[20]],
                self.parameter4_3: self.parameters[3][self.parameter4_3][action[21]],
                self.parameter4_4: self.parameters[3][self.parameter4_4][action[22]],
                self.parameter4_5: self.parameters[3][self.parameter4_5][action[23]],
                self.parameter4_6: self.parameters[3][self.parameter4_6][action[24]],
            }
        return self.select_model_name, selected_model_params

    def get_obs(self, action, performance):
        # 【修复】将 performance 统一转为 np.array，满足 Box 空间要求
        perf_array = np.array([performance], dtype=np.float32)
        # 获取观察
        observation = {}
        if action[0] == 0:
            observation = {
                "model": action[0],
                "parameter1": action[1],
                "parameter2": action[2],
                "parameter3": action[3],
                "parameter4": action[4],
                "parameter5": action[5],
                "parameter6": action[6],
                "performance": perf_array
            }
        elif action[0] == 1:
            observation = {
                "model": action[0],
                "parameter1": action[7],
                "parameter2": action[8],
                "parameter3": action[9],
                "parameter4": action[10],
                "parameter5": action[11],
                "parameter6": action[12],
                "performance": perf_array
            }
        elif action[0] == 2:
            observation = {
                "model": action[0],
                "parameter1": action[13],
                "parameter2": action[14],
                "parameter3": action[15],
                "parameter4": action[16],
                "parameter5": action[17],
                "parameter6": action[18],
                "performance": perf_array
            }
        elif action[0] == 3:
            observation = {
                "model": action[0],
                "parameter1": action[19],
                "parameter2": action[20],
                "parameter3": action[21],
                "parameter4": action[22],
                "parameter5": action[23],
                "parameter6": action[24],
                "performance": perf_array
            }
        return observation

    def get_info(self, performance_metric):
        return {
            "performance": performance_metric
        }

    def reset(self, seed=None, options=None):
        self.reset_num += 1
        print("============================================重置环境============================================\n",
              "第", self.reset_num, "次重置")
        # 重置相关参数
        self.mask_Omni_model = model_des.model_Omni_mask_Y
        self.mask_Omni = model_des.Omni_mask_Y
        self.mask_mtad_model = model_des.model_mtad_mask_Y
        self.mask_mtad = model_des.mtad_mask_Y
        self.mask_InterFusion_model = model_des.model_InterFusion_mask_Y
        self.mask_InterFusion = model_des.InterFusion_mask_Y
        self.mask_SDFVAE_model = model_des.model_SDFVAE_mask_Y
        self.mask_SDFVAE = model_des.SDFVAE_mask_Y

        self.step_num = 0  # 记录交互的次数，也就是部署的次数
        self.negative_num = 0  # 记录不利探索的次数

        self.last_select_model_name = ""  # 上一次选择的模型
        self.select_model_name = ""  # 这一次选择的模型

        self.model_equal = 0  # 用来记录上一次选择的模型和当前模型是否相同，0与上次不同，1与上次相同，2与上次相同且与上上次也相同
        self.model_delete_num = 0  # 用来记录裁剪的模型数量

        self.first_performance = 0.0  # 记录第一次交互的性能表现
        self.last_performance = 0.0  # 记录上一次交互的性能表现

        # 需要以下行来设置 self.np_random 的种子
        super().reset(seed=seed)
        # 从动作空间中随机采样一个动作
        action = self.action_space.sample()
        # 输出动作抽样的实际信息
        selected_model_name, selected_model_params = self.get_model_info(action)

        print("model:", selected_model_name)
        print("params:", selected_model_params)

        # 返回与选定模型和参数相关的观察和信息
        observation = self.get_obs(action, self.first_performance)
        info = self.get_info(self.first_performance)
        print("重置得到的观察为：", observation)
        return observation, info

    def step(self, action):
        terminated = False  # 终止状态
        truncated = False  # 截断条件
        self.flunk = 0  # 重置及格标志
        self.step_num = self.step_num + 1  # 更新交互的次数
        print("这是第", self.step_num, "次交互，采取的action为\n", action)

        if self.step_num == 40:  # 最多交互40次
            sys.exit()

        # 得到部署的模型和参数
        selected_model_name, selected_model_params = self.get_model_info(action)
        # 更新这一次选择的模型
        self.select_model_name = selected_model_name

        # 得到部署模型和参数，部署后计算性能指标
        print("部署模型：", selected_model_name)
        print("部署参数：", selected_model_params)
        print("============================================部署中============================================")
        performance_metric = deployment_func(selected_model_name, selected_model_params, datasets=self.datasets)
        print("返回性能表现指标：", performance_metric)
        # 部署后检测性能指标为0，发生异常，提前结束
        if performance_metric <= 0:
            print(
                "============================================模型部署出现问题============================================")
            print("返回性能表现指标：", 0)
            sys.exit()

        # 判断是不是第一次交互，是的话更新self.first_performance
        if self.first_performance == 0 and self.last_performance == 0:
            self.first_performance = performance_metric

        para_a = 0.5
        if self.first_performance >= 0.90:
            para_a = 0.5
        elif 0.80 <= self.first_performance < 0.9:
            para_a = 0.3
        elif 0.70 <= self.first_performance < 0.8:
            para_a = 0.1
        elif self.first_performance < 0.7:
            para_a = 0

        reward = 0  # 初始化奖励
        # 根据初始性能表现和上一次性能表现计算奖励,如果是第一次部署且性能表现>=0.85,给予默认奖励1,否则默认为0
        if self.first_performance != 0 and self.last_performance != 0:  # 不是第一次交互的情况
            reward = 100 * (para_a * (performance_metric - self.first_performance) + (1 - para_a) * (
                    performance_metric - self.last_performance))
        elif self.first_performance != 0 and self.last_performance == 0:  # 第一次交互的情况
            if performance_metric >= 0.85:
                reward = 1
        else:
            print("第一次交互发生错误")
            sys.exit()
        print("reward:", reward)

        # 更新上一次性能指标
        self.last_performance = performance_metric

        observation = self.get_obs(action, performance_metric)
        print("观察：", observation)
        info = self.get_info(performance_metric)

        if performance_metric >= 0.95:
            print("完成任务")
            sys.exit()

        # 进行配置空间裁剪判断
        if performance_metric <= 0.65:  # 判断当前模型是否及格
            self.flunk = 1
            print("模型不合格")

        # 这次模型等于上次模型就让model_equal += 1，如果不相同就清零，记录连续出现相同模型的次数
        if self.select_model_name == self.last_select_model_name:
            self.model_equal += 1
        else:
            self.model_equal = 0
        self.last_select_model_name = selected_model_name  # 更新last_select_model_name
        print("模型连续出现次数：", self.model_equal)

        # 奖励小于0并且模型和上次模型相同就记录为一次不利探索
        # 切换模型或者出现正奖励就清零负面探索的次数
        if self.model_equal == 0:
            self.negative_num = 0

        # 连续探索过程中，负面奖励即为不利探索，奖励为正清零负面探索次数
        if reward < 0:
            self.negative_num += 1
        if reward > 0:
            self.negative_num = 0
        print("不利探索的次数：", self.negative_num)

        self.last_reward = reward  # 更新last_reward

        # 对动作空间调整
        # 1.连续两次都是对同一个模型的不利探索
        # 2.连续对同一个模型探索三次
        # 3.模型表现不合格
        if self.negative_num == 2 or self.model_equal == 2 or self.flunk == 1:
            print("===========裁剪动作空间===========")
            if self.select_model_name == "Omni":
                self.mask_Omni_model = model_des.model_Omni_mask_N
                self.mask_Omni = model_des.Omni_mask_N
                self.model_delete_num += 1

            elif self.select_model_name == "mtad-gat":
                self.mask_mtad_model = model_des.model_mtad_mask_N
                self.mask_mtad = model_des.mtad_mask_N
                self.model_delete_num += 1

            elif self.select_model_name == "InterFusion":
                self.mask_InterFusion_model = model_des.model_InterFusion_mask_N
                self.mask_InterFusion = model_des.InterFusion_mask_N
                self.model_delete_num += 1

            elif self.select_model_name == "SDFVAE":
                self.mask_SDFVAE_model = model_des.model_SDFVAE_mask_N
                self.mask_SDFVAE = model_des.SDFVAE_mask_N
                self.model_delete_num += 1

        # 当配置空间为空，则终止交互
        if self.model_delete_num == len(self.models):
            print("配置空间没有合适配置")
            sys.exit()

        print("\n")
        # 返回观察结果、奖励、是否结束、截断信号、信息
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def valid_action_mask(self):
        mask_init = (self.mask_Omni_model + self.mask_mtad_model + self.mask_InterFusion_model + self.mask_SDFVAE_model
                     + self.mask_Omni + self.mask_mtad + self.mask_InterFusion + self.mask_SDFVAE)
        mask = mask_init
        print("mask:", mask[0:4])
        return mask