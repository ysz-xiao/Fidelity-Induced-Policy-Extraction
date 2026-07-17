# --map=3m --alg=qmix --evaluate=True --load_model=True

from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args
import _YSZ_ as ysz
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import os
import gc

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns


# python 20220903_test_main_mine_SVM.py --map=2s_vs_1sc --alg=qmix --evaluate=True --load_model=True
def choose_model(model_name):
    if model_name == "DT(Gini)_D3":
        return ysz.agent.DT_Gini.DT_Gini(max_depth=3)
    if model_name == "DT(Gini)_D5":
        return ysz.agent.DT_Gini.DT_Gini(max_depth=5)
    if model_name == "DT(Gini)_D10":
        return ysz.agent.DT_Gini.DT_Gini(max_depth=10)
    if model_name == "DT(Gini)_D12":
        return ysz.agent.DT_Gini.DT_Gini(max_depth=12)
    if model_name == "DT(Gini)_D24":
        return ysz.agent.DT_Gini.DT_Gini(max_depth=24)
    if model_name == "KNN(Ball)_N1":
        return ysz.agent.KNN_Ball.KNN_Ball(n_neighbors=1)
    if model_name == "KNN(Ball)_N10":
        return ysz.agent.KNN_Ball.KNN_Ball(n_neighbors=10)
    if model_name == "KNN(Ball)_N100":
        return ysz.agent.KNN_Ball.KNN_Ball(n_neighbors=100)
    if model_name == "KNN(Brute)_N1":
        return ysz.agent.KNN_Brute.KNN_Brute(n_neighbors=1)
    if model_name == "KNN(Brute)_N10":
        return ysz.agent.KNN_Brute.KNN_Brute(n_neighbors=10)
    if model_name == "KNN(Brute)_N100":
        return ysz.agent.KNN_Brute.KNN_Brute(n_neighbors=100)
    if model_name == "GBDT":
        return ysz.agent.DT_GBDT.DT_GBDT()
    if model_name == "GP":
        return ysz.agent.GP.GP(random_state=123)
    if model_name == "SVM_SVC":
        return ysz.agent.SVM_SVC.SVM_SVC(probability=True)
    if model_name == "SVM_LinearSVC":
        return ysz.agent.SVM_LinearSVC.LinearSVC(probability=True)
    return



def model_selection(model_list, max_model_count):
    if len(model_list) <= max_model_count:
        return model_list
    # 竞争
    win_rate_list, reward_list, fidelity_list, performance_list = [], [], [], []
    for best_model in model_list:  # 评估库中所有模型
        win_rate, reward, fidelity = runner.evaluate_tree(tree_models=best_model)
        performance = win_rate + fidelity
        win_rate_list.append(win_rate)
        reward_list.append(reward)
        fidelity_list.append(fidelity)
        performance_list.append(performance)
    # 淘汰
    while len(model_list) > max_model_count:
        min_idx = np.argmin(performance_list)
        del model_list[min_idx]  # 删除最差的一个
        del performance_list[min_idx]
    # 最优的一个前移一位
    max_idx = np.argmax(performance_list)
    if max_idx > 0:
        model_list[max_idx], model_list[max_idx - 1] = model_list[max_idx - 1], model_list[
            max_idx]
    return model_list


def OUR_trainer(xai_model_name, RESULT_PATH, MAX_EXP_COUNT_EACH):
    # 创建路径
    if os.path.exists(RESULT_PATH) is False:
        os.makedirs(RESULT_PATH)

    # 加载现有经验库
    best_exp_path = f"{RESULT_PATH}/our_{xai_model_name}_exp.npy"
    test_data_path = f"{RESULT_PATH}/our_{xai_model_name}.npy"

    if os.path.exists(best_exp_path) is True:
        exp_pool = np.load(best_exp_path, allow_pickle=True)
        result_list = list(np.load(test_data_path, allow_pickle=True))
        print(f"\tload data--- exp number: {len(exp_pool[0])}")
    else:
        # 收集一轮初始经验
        exp_pool = runner.collect_exps_saq_OUR([[] for i in range(args.n_agents)], episode_num=1)
        result_list = []

    mix_rate = 1
    current_model = build_xai_model(exp_pool, xai_model_name)
    exp_size_each = len(exp_pool[0])
    best_model_pool = []
    best_model_pool.append(current_model)
    while exp_size_each < MAX_EXP_COUNT_EACH:
        gc.collect()
        mix_rate = mix_rate * 0.95
        # 混合模型采集经验
        exp_pool = runner.collect_exps_saq_OUR(exp_pool, current_model, episode_num=20, mix_rate=mix_rate)
        # 训练决策树
        # current_model, idxs = build_model_OUR(exp_pool, xai_model_name,remain_rate=0.5)
        current_model = build_xai_model(exp_pool, xai_model_name)
        # 新成员
        best_model_pool.append(current_model)
        # 模型选择淘汰一个
        best_model_pool = model_selection(best_model_pool, 3)
        # 测试、输出
        exp_size_each = len(exp_pool[0])
        win_rate, reward, fidelity = runner.evaluate_tree(tree_models=best_model_pool[0],
                                                          max_epoch=args.evaluate_epoch)
        print("\t[exp_number:{}],best_win_rate:{:.3f},best_reward:{:.3f},best_fidelity:{:.3f}"
              .format(exp_size_each, win_rate, reward, fidelity))
        # 存储
        result_list.append([exp_size_each, win_rate, reward, fidelity])
        np.save(best_exp_path, exp_pool, allow_pickle=True)
        np.save(test_data_path, result_list, allow_pickle=True)

    # 竞争直到最后一个
    while len(best_model_pool) > 1:
        best_model_pool = model_selection(best_model_pool, len(best_model_pool) - 1)
    # 最终测试
    win_rate, reward, fidelity = runner.evaluate_tree(tree_models=best_model_pool[0], max_epoch=8)
    print("\t[best]exp_number:{},best_win_rate:{:.3f},best_reward:{:.3f},best_fidelity:{:.3f}"
          .format(exp_size_each, win_rate, reward, fidelity))

def build_xai_model(exp_pool, model_name):
    result_models = []
    for agent_i in range(len(exp_pool)):  # 对于每一个智能体
        cur_exp_pool = exp_pool[agent_i]  # 当前的经验库
        while True:
            X = [cur_exp_pool[i][0] for i in range(len(cur_exp_pool))]
            Y = [cur_exp_pool[i][1] for i in range(len(cur_exp_pool))]
            if len(set(Y)) != 1:
                break
        tree = choose_model(model_name)
        tree.fit(X, Y)
        result_models.append(tree)
    return result_models

def XAI_main():
    """
    执行XAI相关的代码
    """
    XAI_models_names = [  # 待训练的XAI模型-名字
        "DT(Gini)_D5",
        "DT(Gini)_D12",
        "KNN(Brute)_N1",
        "KNN(Brute)_N10",
        "SVM_SVC"
    ]
    args.evaluate_epoch = 8
    MAX_EXP_COUNT_EACH = 10000
    RESULT_PATH = "./XAI_results/test_result/{}".format(args.map)
    EXP_PATH = "./XAI_results/exp_files/{}".format(args.map)
    ysz.trainsform.data_storage.create_dir(EXP_PATH)  # 创建exp存储路径
    # 开始执行本方法
    for xai_model_name in XAI_models_names:
        print("———————————————start {}———————————————".format(xai_model_name))
        # Step 0: Setup
        # DAGGER_trainer(xai_model_name)
        # VIPER_trainer(xai_model_name)
        OUR_trainer(xai_model_name, RESULT_PATH, MAX_EXP_COUNT_EACH)


if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)

        # Explain
        XAI_main()

        env.close()
