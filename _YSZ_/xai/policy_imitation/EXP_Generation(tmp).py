import numpy as np
import Exp_Process.Exp_Elimination

from sklearn.neighbors import KNeighborsClassifier


# 计算所有经验点的最近邻异分类点
def get_dis_list(EXP):
    dis_list = []
    state_dim = len(EXP[0]) - 1
    for i in range(len(EXP)):  # 遍历所有点
        # 找到最近的异分类点
        nearest_dis_all = float('inf')
        for j in range(len(EXP)):
            if i == j: continue
            dis = distance(EXP[i][0:state_dim], EXP[j][0:state_dim])
            if dis < nearest_dis_all and EXP[i][state_dim] != EXP[j][state_dim]:
                nearest_dis_all = dis
        dis_list.append(nearest_dis_all)  # 最近的点


def pointSplit(EXP, mid_dis, split_time, agent, mutation_rate=0.3):
    split_EXPS = []
    state_dim = len(EXP[0]) - 1
    exp_max, exp_min = [], []
    for i in range(state_dim):
        exp_max.append(np.max(EXP[:][i]))
        exp_min.append(np.min(EXP[:][i]))
    # 经验分裂
    for i in range(len(EXP)):  # 遍历所有点
        # 经验分裂
        for k in range(int((split_time))):
            if np.random.rand() > mutation_rate:  # 正常分裂
                rand = np.random.rand(len(EXP[0]))
                rand = rand / np.linalg.norm(rand[0:state_dim]) * mid_dis
                split_e = EXP[i] + rand
                split_e[state_dim] = agent.forward(split_e[0:state_dim], train=False)
                new_exp = split_e
            else:  # 突变分裂
                new_exp = []
                for j in range(state_dim):
                    rand = np.random.rand() * (exp_max[j] - exp_min[j]) + exp_min[j]
                    new_exp.append(rand)
                new_exp.append(agent.forward(new_exp, train=False))
            split_EXPS.append(new_exp)
    # print("\t generate split_EXPS: " + str(i) + "->" + str(
    #    len(EXP)) + "             ", end='\r')
    split_EXPS = np.array(split_EXPS)
    return split_EXPS


# 经验淘汰
def nature_selection(split_EXPS, EXP, mid_dis, max_exp_count):
    state_dim = len(EXP[0]) - 1
    for i in range(len(split_EXPS)):
        new_exp = split_EXPS[i]
        # EXP中最近的点
        nearest_dis = float('inf')
        for j in range(len(EXP)):
            cur_dis = distance(new_exp[0:state_dim], EXP[j][0:state_dim])
            if cur_dis < nearest_dis:
                nearest_dis = cur_dis
        # 添加
        if nearest_dis >= mid_dis:
            EXP.append(split_EXPS[i])
            if len(EXP) >= max_exp_count:
                return EXP
            # print("\t delete split_EXPS: " + str(i) + "->" + str(
            #    len(split_EXPS)) + "                                      ", end='\r')
    return EXP
def is_surrunded(EXP, exp, dim):
    """判断当前点是否被相关经验包裹"""
    # step1:找到p最近的异分类点p'
    exp_, dis_ = EXP[0], float('inf')  # 最近异分类点，距离
    for i in range(len(EXP)):
        if (exp[dim] != EXP[i][0:dim]).any():
            dis_tmp = distance(EXP[i][0:dim], exp[0:dim])
            if dis_tmp < dis_:
                exp_ = EXP[i]
                dis_ = dis_tmp
    # step2:找出P中与当前点p距离不超过dis的点的集合
    for i in range(len(EXP)):
        if exp[dim] == EXP[i][dim]:
            dis = distance(exp[0:dim], EXP[i][0:dim])
            if dis >= dis_:
                continue
            dis2 = distance(exp_[0:dim], EXP[i][0:dim])
            if dis2 >= dis_:
                continue
            return True  # 判断为中间点
    return False  # 判定为边界点
def distance(x1, x2):
    """计算两点距离"""
    return np.linalg.norm(np.array(x1) - np.array(x2))

def exp_elimination(EXP):
    """健壮性判定"""
    if len(EXP) <= 0: return False
    """公共数据"""
    input_dim = len(EXP[0]) - 1
    EXP_DEL = []
    """mid_count: 中间点数量最大值，即边界厚度"""
    oldSize = len(EXP)
    EXP = np.unique(EXP, axis=0)  # 去除重复元素
    for i in range(len(EXP)):
        if is_surrunded(EXP, EXP[i], input_dim) == False:
            EXP_DEL.append(EXP[i])
        print("\texp remove:" + str(oldSize) + "->" + str(len(EXP)) + "               ", end="\r")
    return EXP_DEL

def unInteract_optimize(EXP, agent, episode=-1, split_time=-1, mutation_rate=0.5, max_exp_count=-1):
    state_dim = len(EXP[0]) - 1
    preLen = len(EXP)
    # 参数自适应
    if max_exp_count == -1:
        max_exp_count = len(EXP)
    if split_time == -1:
        split_time = state_dim + 1
    if episode == -1:
        episode = split_time * split_time
    # 执行一次经验选择
    EXP = exp_elimination(EXP)
    print("\t optimize episode:" + str(0) + "/" + str(episode) + "--- exp#:" + str(preLen) + "->" + str(
        len(EXP)) + "                            ")
    # 开始迭代
    for e in range(episode):
        # 0.初始化
        EXP = list(EXP)
        # 1.计算最近邻点
        dis_list = get_dis_list(EXP)
        dis_list = np.sort(np.array(dis_list))
        mid_dis = dis_list[int(len(dis_list) / 2)]

        # 2.生成分裂点:以一定概率生成，
        split_EXPS = pointSplit(EXP, mid_dis, split_time, agent, mutation_rate=mutation_rate)

        # 3.经验淘汰：分裂经验生存和死亡，最近异分类点距离小于最小距离则死亡
        EXP = nature_selection(split_EXPS, EXP, mid_dis, max_exp_count)

        # 4.执行一次经验选择
        EXP = exp_elimination(EXP)

        # 5.判断结束条件:a）达到设定的最大值
        if len(EXP) >= max_exp_count:
            break

        print("\t optimize episode:" + str(e + 1) + "/" + str(episode) + "--- exp#:" + str(preLen) + "->" + str(
            len(EXP)) + "                            ")
    return list(EXP)


def unInteract_optimize_ablation(EXP, agent, episode=-1, split_time=-1, mutation_rate=0.3, max_exp_count=-1, mode="ab"):
    state_dim = len(EXP[0]) - 1
    preLen = len(EXP)
    # 参数自适应
    if max_exp_count == -1:
        max_exp_count = len(EXP)
    if split_time == -1:
        split_time = state_dim + 1
    if episode == -1:
        episode = split_time * split_time
    # 执行一次经验选择
    EXP = exp_elimination(EXP)
    # 开始迭代
    for e in range(episode):
        # 0.初始化
        EXP = list(EXP)
        # 1.计算最近邻点
        dis_list = get_dis_list(EXP)
        dis_list = np.sort(np.array(dis_list))
        mid_dis = dis_list[int(len(dis_list) / 2)]
        if mode == "a":
            # 2.生成分裂点:以一定概率生成，
            split_EXPS = pointSplit(EXP, mid_dis, split_time, agent, mutation_rate=mutation_rate)

            # 3.经验淘汰：分裂经验生存和死亡，最近异分类点距离小于最小距离则死亡
            EXP = nature_selection(split_EXPS, EXP, mid_dis)
        elif mode == "b":
            # 4.执行一次经验选择
            EXP = exp_elimination(EXP)
        else:
            # 2.生成分裂点:以一定概率生成，
            split_EXPS = pointSplit(EXP, mid_dis, split_time, agent, mutation_rate=mutation_rate)
            # 3.经验淘汰：分裂经验生存和死亡，最近异分类点距离小于最小距离则死亡
            EXP = nature_selection(split_EXPS, EXP, mid_dis)
            # 4.执行一次经验选择
            EXP = exp_elimination(EXP)

        # 5.判断结束条件:a）达到设定的最大值
        if len(EXP) >= max_exp_count:
            break

        print("\t optimize episode:" + str(e + 1) + "/" + str(episode) + "--- exp#:" + str(preLen) + "->" + str(
            len(EXP)) + "                            ", end='\r')
    return list(EXP)
