import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time
import random
import _YSZ_ as ysz
import copy


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def test_similarity(self, tree_models, episode_num=None, evaluate=False, savereplay=False):
        same_count = 0
        total_count = 0
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        step = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                if self.args.alg == 'maven':
                    DRL_action, exp = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                avail_action, epsilon, maven_z, evaluate)
                else:
                    DRL_action, exp = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                                avail_action, epsilon, evaluate)

                ##### 生成XAI action #####
                input = np.array(obs[agent_id].tolist(), dtype=float)
                action_probs = tree_models[agent_id].forward_proba(input)
                action_classes = tree_models[agent_id].model.classes_
                sorted_action_idx = sorted(range(len(action_probs)), key=lambda k: action_probs[k], reverse=True)

                action_cando = [i for i, x in enumerate(avail_action) if x]
                sorted_action = []
                for i in sorted_action_idx:
                    if action_probs[i] < 0:
                        continue
                    sorted_action.append(action_classes[i])
                # action = action_cando[random.randint(0, len(action_cando) - 1)]  # 初始化一个随机动作
                action = action_cando[-1]  # 初始动作为最后一个可行动作
                for i in range(len(sorted_action)):
                    cur_action = int(sorted_action[i])
                    if avail_action[cur_action] == 1:
                        action = cur_action
                        break
                #####           #####
                total_count += 1
                if DRL_action == action:
                    same_count += 1
                ###

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int32(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # print(actions)
            reward, terminated, info = self.env.step(actions)

        return same_count, total_count

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        EXPs = [[] for i in range(self.n_agents)]
        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                if self.args.alg == 'maven':
                    action_DRL, exp = self.agents.choose_action_get_exp(obs[agent_id], last_action[agent_id], agent_id,
                                                                        avail_action, epsilon, maven_z, evaluate)
                else:
                    action_DRL, exp = self.agents.choose_action_get_exp(obs[agent_id], last_action[agent_id], agent_id,
                                                                        avail_action, epsilon, evaluate)

                # 执行动作
                action = action_DRL

                EXPs[agent_id].append(exp)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int32(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       last_action=last_action.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()

        return episode, episode_reward, win_tag, step, EXPs

    def xy_seperate(self, EXP):
        Xs, Ys = [], []
        for i in range(len(EXP)):
            X, Y = np.array(EXP[i])[:, 0:len(EXP[i][0]) - 1], np.array(EXP[i])[:, len(EXP[i][0]) - 1]  # 数据分割
            Xs.append(X)
            Ys.append(Y)
        return Xs, Ys

    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    def generate_episode_get_frames(self, tree_models, episode_num=None, evaluate=False, savereplay=True):
        if self.args.replay_dir != '' and evaluate and episode_num == 0 and savereplay == True:  # prepare for save replay of evaluation
            self.env.close()
        same_count = 0
        total_count = 0
        my_obs, my_act = [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                # DRL action
                if self.args.alg == 'maven':
                    DRL_action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                           avail_action, epsilon, maven_z, evaluate)
                else:
                    DRL_action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                           avail_action, epsilon, evaluate)

                # XAI action
                input = np.array(obs[agent_id].tolist(), dtype=float)
                action_probs = tree_models[agent_id].forward_proba(input)
                action_classes = tree_models[agent_id].model.classes_
                sorted_action_idx = sorted(range(len(action_probs)), key=lambda k: action_probs[k], reverse=True)

                action_cando = [i for i, x in enumerate(avail_action) if x]
                sorted_action = []
                for i in sorted_action_idx:
                    if action_probs[i] < 0:
                        continue
                    sorted_action.append(action_classes[i])
                # action = action_cando[random.randint(0, len(action_cando) - 1)]  # 初始化一个随机动作
                action = action_cando[-1]  # 初始动作为最后一个可行动作
                for i in range(len(sorted_action)):
                    cur_action = int(sorted_action[i])
                    if avail_action[cur_action] == 1:
                        action = cur_action
                        break

                total_count += 1
                # if action == DRL_action:
                #     same_count += 1
                #####           #####
                if agent_id == 0:
                    my_obs_tmp = obs[agent_id].tolist()
                    my_obs_tmp.append(action)
                    my_obs.append(my_obs_tmp)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int32(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # print(actions)
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False

            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)

        self.env.save_replay()
        self.env.close()
        import random
        import string
        import time
        salt = ''.join(random.sample(string.ascii_letters + string.digits, 8)).upper()
        rt = time.strftime("%Y%m%d%H%M%S%MS", time.localtime(time.time()))
        ysz.trainsform.data_storage.write_csv(range(30), my_obs, "C:\\Program Files (x86)\\StarCraft II\\Replays",
                                              rt + salt, auto_cal=False)
        time.sleep(60)
        return my_obs, my_act, win_tag

    def generate_episode_tree(self, tree_models, episode_num=None, mix_rate=1, evaluate=False, savereplay=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0 and savereplay == True:  # prepare for save replay of evaluation
            self.env.close()
        same_count = 0
        total_count = 0
        EXPs = [[] for i in range(self.n_agents)]
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                # DRL action
                if self.args.alg == 'maven':
                    action_DRL, exp = self.agents.choose_action_get_exp(obs[agent_id], last_action[agent_id], agent_id,
                                                                        avail_action, epsilon, maven_z, evaluate)
                else:
                    action_DRL, exp = self.agents.choose_action_get_exp(obs[agent_id], last_action[agent_id], agent_id,
                                                                        avail_action, epsilon, evaluate)
                # XAI action
                inputs = obs[agent_id].copy()
                # transform agent_num to onehot vector # ····················VIPER-OUR
                agent_ids = np.zeros(self.n_agents)
                agent_ids[agent_id] = 1.
                if self.args.last_action:
                    inputs = np.hstack((inputs, last_action[agent_id]))
                if self.args.reuse_network:
                    inputs = np.hstack((inputs, agent_ids))
                inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
                inputs = inputs.tolist()[0] + avail_action

                action_probs = tree_models[agent_id].forward_proba(inputs)
                action_classes = tree_models[agent_id].model.classes_
                sorted_action_idx = sorted(range(len(action_probs)), key=lambda k: action_probs[k], reverse=True)

                action_cando = [i for i, x in enumerate(avail_action) if x]
                sorted_action = []
                for i in sorted_action_idx:
                    if action_probs[i] < 0:
                        continue
                    sorted_action.append(action_classes[i])
                # action = action_cando[random.randint(0, len(action_cando) - 1)]  # 初始化一个随机动作
                action_xai = action_cando[-1]  # 初始动作为最后一个可行动作
                for i in range(len(sorted_action)):
                    cur_action = int(sorted_action[i])
                    if avail_action[cur_action] == 1:
                        action_xai = cur_action
                        break
                total_count += 1
                if action_xai == action_DRL:
                    same_count += 1
                else: # ····················VIPER-OUR
                    EXPs[agent_id].append(exp)#..................our
                # EXPs[agent_id].append(exp)  # ..................viper_dagger

                #####           #####
                # 如果满足条件，则用tree的动作执行
                if random.random() <= mix_rate:
                    action = action_DRL
                else:
                    action = action_xai

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int32(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # print(actions)
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and savereplay == True:
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step, same_count, total_count, EXPs

    def generate_episode_tree_OUR(self, tree_models, episode_num=None, mix_rate=1, evaluate=False, savereplay=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0 and savereplay == True:  # prepare for save replay of evaluation
            self.env.close()
        same_count = 0
        total_count = 0
        EXPs = [[] for i in range(self.n_agents)]
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                # DRL action
                if self.args.alg == 'maven':
                    action_DRL, exp = self.agents.choose_action_get_exp(obs[agent_id], last_action[agent_id], agent_id,
                                                                        avail_action, epsilon, maven_z, evaluate)
                else:
                    action_DRL, exp = self.agents.choose_action_get_exp(obs[agent_id], last_action[agent_id], agent_id,
                                                                        avail_action, epsilon, evaluate)
                # XAI action
                inputs = obs[agent_id].copy()
                # transform agent_num to onehot vector # ····················VIPER-OUR
                agent_ids = np.zeros(self.n_agents)
                agent_ids[agent_id] = 1.
                if self.args.last_action:
                    inputs = np.hstack((inputs, last_action[agent_id]))
                if self.args.reuse_network:
                    inputs = np.hstack((inputs, agent_ids))
                inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
                inputs = inputs.tolist()[0] + avail_action

                action_probs = tree_models[agent_id].forward_proba(inputs)
                action_classes = tree_models[agent_id].model.classes_
                sorted_action_idx = sorted(range(len(action_probs)), key=lambda k: action_probs[k], reverse=True)

                action_cando = [i for i, x in enumerate(avail_action) if x]
                sorted_action = []
                for i in sorted_action_idx:
                    if action_probs[i] < 0:
                        continue
                    sorted_action.append(action_classes[i])
                # action = action_cando[random.randint(0, len(action_cando) - 1)]  # 初始化一个随机动作
                action_xai = action_cando[-1]  # 初始动作为最后一个可行动作
                for i in range(len(sorted_action)):
                    cur_action = int(sorted_action[i])
                    if avail_action[cur_action] == 1:
                        action_xai = cur_action
                        break
                total_count += 1
                if action_xai == action_DRL:
                    same_count += 1
                else: # ····················VIPER-OUR
                    EXPs[agent_id].append(exp)
                # EXPs[agent_id].append(exp)

                #####           #####
                # 如果满足条件，则用tree的动作执行
                if random.random() <= mix_rate:
                    action = action_DRL
                else:
                    action = action_xai

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int32(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # print(actions)
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == 'maven':
            episode['z'] = np.array([maven_z.copy()])
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and savereplay == True:
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step, same_count, total_count, EXPs


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init CommRolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if self.args.replay_dir != '' and evaluate and episode_num == 0:  # prepare for save replay
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(np.array(obs), last_action)

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(weights[agent_id], avail_action, epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int32(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            # if terminated:
            #     time.sleep(1)
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
            # print('Epsilon is ', self.epsilon)
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
            self.env.close()
        return episode, episode_reward, win_tag, step
