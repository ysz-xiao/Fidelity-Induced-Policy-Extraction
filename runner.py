import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import copy
import _YSZ_ as ysz


class Runner:
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find(
                'reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while time_steps < self.args.n_steps:
            print('Run {}, time_steps {}'.format(num, time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find(
                    'reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    # def close(self):
    #     self.env.close()

    def evaluate_tree(self, tree_models, max_epoch=8):
        win_number = 0
        episode_rewards = 0
        total_same_count = 0
        total_total_count = 0
        for epoch in range(max_epoch):
            _, episode_reward, win_tag, _, same_count, total_count, _ = self.rolloutWorker.generate_episode_tree(
                tree_models, epoch,
                savereplay=False, evaluate=True, mix_rate=0)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
            total_same_count += same_count
            total_total_count += total_count
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch, total_same_count / total_total_count

    def evaluate_tree_get_y_and_prey(self, tree_models):
        win_number = 0
        episode_rewards = 0
        total_same_count = 0
        total_total_count = 0
        real_ys_list, pre_ys_list = [], []
        for epoch in range(self.args.evaluate_epoch):
            real_ys, pre_ys, episode, episode_reward, win_tag, step, same_count, total_count = self.rolloutWorker.generate_episode_tree_get_y_and_prey(
                tree_models, epoch,
                savereplay=False, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
            total_same_count += same_count
            total_total_count += total_count
            real_ys_list.extend(real_ys)
            pre_ys_list.extend(pre_ys)
        return real_ys_list, pre_ys_list

    def get_decision_frams(self, model):
        my_obs, my_act, win_tag = self.rolloutWorker.generate_episode_get_frames(
            model, 10, evaluate=True)
        return my_obs, my_act, win_tag

    def collect_exps(self, EXP_POOL, episode_num=1):
        for i in range(episode_num):
            episode, reward, win_tag, step = self.rolloutWorker.generate_episode(1, evaluate=True)
            o = episode['o'][0]
            u = episode['u'][0]
            padded = episode['padded'][0]
            for step_i in range(len(u)):
                if padded[step_i] == 1:
                    break
                for agent_i in range(len(u[0])):
                    exp = o[step_i][agent_i].tolist()
                    exp.append(u[step_i][agent_i].tolist()[0])
                    EXP_POOL[agent_i].append(exp)
        return EXP_POOL

    def collect_exps_saq_OUR(self, EXP_POOL, tree_model=None, episode_num=1, mix_rate=1):
        for i in range(episode_num):
            if tree_model != None:
                episode, reward, win_tag, step, same_count, total_count, EXPs \
                    = self.rolloutWorker.generate_episode_tree_OUR(tree_model, episode_num=1, mix_rate=mix_rate,
                                                                   evaluate=False,
                                                                   savereplay=False)
            else:
                episode, reward, win_tag, step, EXPs = self.rolloutWorker.generate_episode(1, evaluate=True)
            for agent_i in range(len(EXPs)):  # 每一个智能体
                EXP_POOL[agent_i].extend(EXPs[agent_i])
        return EXP_POOL

    def seperate_EXP(self, EXP):
        EXP_list = []
        for j in range(len(EXP[0][-1])):
            EXP_list.append([])
        for i in range(len(EXP)):
            cur_state = EXP[i]
            cur_state = cur_state[0:len(EXP[0]) - 1]
            cur_Qs = EXP[i][-1]
            for j in range(len(cur_Qs)):
                cur_exp = copy.deepcopy(cur_state)
                cur_exp.append(cur_Qs[j])
                EXP_list[j].append(cur_exp)
        return EXP_list

    def extend_exps_qs(self, EXP_POOL, model):
        new_EXP = self.rolloutWorker.generate_episode_qs(EXP_POOL, evaluate=True)
        for i in range(len(EXP_POOL)):
            new_EXP[i] = self.seperate_EXP(new_EXP[i])
        if EXP_POOL[0] == []:
            EXP_POOL = new_EXP
        else:
            for i in range(len(new_EXP)):
                for j in range(len(new_EXP[i])):
                    EXP_POOL[i][j].extend(new_EXP[i][j])
        return EXP_POOL

    def collect_exps_time(self, EXP_POOL):
        episode, _, _, _ = self.rolloutWorker.generate_episode(1, evaluate=True)
        o = episode['o'][0]
        u = episode['u'][0]
        padded = episode['padded'][0]
        last_exp = None
        for step_i in range(len(u)):
            if padded[step_i] == 1:
                break
            for agent_i in range(len(u[0])):
                exp = None
                if step_i == 0:
                    exp_c = o[step_i][agent_i].tolist()
                    exp_change = [exp_c[i] - 0 for i in range(len(exp_c))]
                    exp = exp_c + exp_change
                    exp.append(u[step_i][agent_i].tolist()[0])
                    EXP_POOL[agent_i].append(exp)
                    last_exp = exp_c
                else:
                    exp_c = o[step_i][agent_i].tolist()
                    exp_change = [exp_c[i] - last_exp[i] for i in range(len(exp_c))]
                    exp = exp_c + exp_change
                    exp.append(u[step_i][agent_i].tolist()[0])
                    EXP_POOL[agent_i].append(exp)
                    last_exp = exp_c
        return EXP_POOL

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        # plt.close()
