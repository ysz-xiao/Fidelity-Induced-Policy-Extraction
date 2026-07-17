from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args
import numpy as np
import copy

import os


def construct_exp_pool(Total_Episodes=100000):
    """
    构建一个总的经验库
    """
    Map_Name = args.map
    MA_Alg = args.alg
    # 数据准备
    EXP_Pool = [[] for i in range(args.n_agents)]
    for eps in range(Total_Episodes):
        # 收集经验
        Episode_EXP = [[] for i in range(args.n_agents)]
        win_tag = False
        while win_tag is False:
            Episode_EXP, win_tag = runner.collect_exps(copy.deepcopy(Episode_EXP))
        EXP_Pool = combine_exp_pool(EXP_Pool, Episode_EXP)
        # 经验路径
        EXP_path = "./EXPs/{}/{}".format(MA_Alg, Map_Name)
        EXP_name = "exp_full.npy"
        if not os.path.exists(EXP_path):  # 文件夹不存在时创建
            os.makedirs(EXP_path)
        # 保存经验
        np.save(EXP_path + "/" + EXP_name, EXP_Pool)
        if eps % 1000 == 0:
            np.save("{}/ep{}_{}".format(EXP_path, eps, EXP_name), EXP_Pool)
        EXP_size = sum([len(EXP_Pool[i]) for i in range(len(EXP_Pool))])
        print("EXP_size:{}, eps:{}".format(EXP_size, eps))


def combine_exp_pool(EXP_pool, combine_pool):
    """
    合并两个经验库
    """
    for i in range(len(EXP_pool)):
        EXP_pool[i].extend(combine_pool[i])
    return EXP_pool


def get_all_exps(args, ep=-1):
    """
    读取max_count个经验
    """
    Map_Name, MA_Alg = args.map, args.alg
    EXP_output = [[] for i in range(args.n_agents)]
    # 经验路径
    EXP_path = "./EXPs/{}/{}".format(MA_Alg, Map_Name)
    if ep == -1:
        EXP_name = "exp_full.npy"
    else:
        EXP_name = "ep{}_exp_full.npy".format(ep)
    # 读取经验
    if os.path.exists(EXP_path + "/" + EXP_name):  # 文件夹不存在时
        Episode_EXP = np.load(EXP_path + "/" + EXP_name, allow_pickle=True).tolist()
        combine_exp_pool(EXP_output, Episode_EXP)
    return EXP_output


if __name__ == '__main__':
    for i in range(1):
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

        # 收集经验
        construct_exp_pool()

        env.close()
