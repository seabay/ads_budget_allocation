
# 每条 rollout 样本是：(state_0, action_0, discounted_r_total, state_n)
# 你可以将这个数据集作为额外样本加入到原始 offline dataset 中供 CQL 训练。

假设：
    已有一个 环境模拟器（如 LSTM），其 predict(state, action) 输出 next_state 和 reward
    有一个 策略 π(a|s)（如行为策略或当前 actor），可以给定状态采样动作
想实现 n-step TD rollout：从离线数据开始，扩展未来 n 步，生成新的 (s, a, r, s') 样本

import numpy as np

def rollout_n_step(env_model, policy_model, init_dataset, n_step=3, gamma=0.99):
    """
    env_model: 环境模型，输入 state 和 action，输出 next_state, reward
    policy_model: 策略模型 π(a|s)，输入 state，输出 action
    init_dataset: 初始离线数据集，包含 list of (s, a)
    n_step: rollout 步数
    gamma: 折扣因子
    """
    rollouts = []

    for (s0, a0) in init_dataset:
        s = s0
        a = a0
        states = [s]
        actions = [a]
        rewards = []

        for t in range(n_step):
            # 使用环境模拟器获取下一个状态和奖励
            s_next, r = env_model.predict(s, a)
            rewards.append(r)
            states.append(s_next)

            # 使用策略采样下一个动作
            a = policy_model.sample_action(s_next)
            actions.append(a)
            s = s_next

        # 计算 n-step 总 reward
        total_reward = sum((gamma ** i) * rewards[i] for i in range(len(rewards)))

        # 存储 (s0, a0, total_reward, s_n)
        rollout_sample = (states[0], actions[0], total_reward, states[-1])
        rollouts.append(rollout_sample)

    return rollouts
