import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import socket
import queue
import sys
import os
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, NormalizeObservation
from sympy.physics.units import action
import gym_examples
import torch.optim as optim
import torch.nn.functional as F
from helpers import ExponentialMovingAverage, ACKTimeoutEstimator, LinkQualityEstimator, Packet, DQN, quantize_to_grid, NormalizeFloatObservation, schedule_on_off, build_belief_nodes, NetworkRegistry

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Sensor:
    def __init__(self, tl_option, N, augmentation_type=None, threshold=None, network=None, controller_type=None, env=None, rl_model=None):
        self.state = 0
        self.u = 0
        self.A = 1.2
        if controller_type == 'LQG':
            self.B = 1
        else:
            self.B = 0.5#0.5#1
        self.k = 0
        self.state_list = np.zeros(N)

        self.threshold = threshold #add if method
        self.tl_option = tl_option
        self.augmentation_type = augmentation_type

        self.op_max = 5

        self.network = network
        self.last_ack_time = None


        if self.tl_option == "random":
            self.trigger_prob = 10


        if tl_option == "acp":
            self.start_ts = None
            self.t_last_sent = None
            self.aoi_list_last_epoch = []
            self.op_list_last_epoch = []
            self.upd_period = 100
            self.epoch_start = 1#5
            self.epoch_end = None
            self.prev_op_mean = None
          #  self.inter_ack_estimator = None
            
            self.flag = False
            self.gamma = 0
            self.b_star = None
            self.age_estimate = np.random.randint(10)
            self.step_size = 0.1
            self.period_multiplier = 50

        if tl_option == "tcp_tahoe":
            self.acked_packets_in_row = 0
            self.phase = 'slow start'
            self.op_threshold = 10000

        if tl_option == "tcp_vegas":
            self.op_threshold = 10000
            self.phase = 'slow start'
            self.base_rtt = None
            self.al = 0.15
            self.be = 0.3
            self.rtt = None
            self.last_adj = 0

        if tl_option == "rl":
            self.env = env
            self.n_actions = self.env.action_space.n
            state, info = self.env.reset()
            self.n_observations = len(state)
            self.rl_model = rl_model

        if tl_option == 'vou_inst':
            self.belief_nodes = []
            for i in range(1,6):
                self.belief_nodes.append(build_belief_nodes(i))
            self.netw_registry = NetworkRegistry()

        if augmentation_type == 'twin':
            self.controller_twin = Controller(controller_type, N)

    def associate_network(self, network):
        self.network = network


    def observe(self, control_input):
        if self.augmentation_type == 'twin':
            self.controller_twin.step()
        #    print(self.k, 'augmentation ', self.controller_twin.x_est)
        self.state = max(min(round(self.A * self.state + self.B * control_input + np.random.randn(), 5), 100000),
                              -100000)
        self.state_list[self.k] = self.state


        self.k += 1


    def trigger(self):
      #  if time.perf_counter() - self.t_last_sent >= 0.2:
       #     return 1
        if self.tl_option == "random":
            if not self.k % 100:
                self.trigger_prob = np.random.randint(1, 100)
            r = np.random.randint(100)
            return r < self.trigger_prob
        if self.tl_option == 'tcp_vegas':
            if self.rtt is not None and time.perf_counter() - self.last_adj > self.rtt:
                self.TCP_vegas_adjust()
        if self.tl_option == "acp":

            if self.epoch_end is None and self.epoch_start is not None and time.perf_counter() >= self.start_ts + self.epoch_start:
                self.initialize_first_epoch()
            elif self.epoch_end is not None and time.perf_counter() >= self.epoch_end:
                self.initialize_next_epoch()
            self.aoi_list_last_epoch.append(self.age_estimate)
            self.age_estimate += 1
            self.op_list_last_epoch.append(self.network.get_num_ops())
            if 1000 * (time.perf_counter() - self.t_last_sent) > np.round(self.upd_period / 10) * 10:
                self.t_last_sent = time.perf_counter()
                return 1
            else:
                return 0
        if self.tl_option == "udp":
            return 1
        if self.tl_option == "rl":
            action = self.get_rl_decision()
            return action
        if self.tl_option == "vou_inst":
            if self.network.get_num_ops() >= self.op_max:
                print("hit op max")
                return 0
            else:
                effectiveness = self.get_effectiveness()
            delay_scaled, delay = self.netw_registry.get_exp_delay(1000*(time.perf_counter() - self.t_last_sent))
           # print(self.k, self.state, effectiveness, delay_scaled)
           # print("current est ", self.get_augmented_state())
            if effectiveness > self.threshold * delay_scaled:
          #      print("trigger")
           #     print(self.k, self.state, effectiveness, delay_scaled)
                return 1
            else:
                return 0
        else:
            print("current state and est ", self.state, self.get_augmented_state())
            if abs(self.state - self.get_augmented_state()) > self.threshold and self.network.get_num_ops() < self.op_max:
                return 1
            else:
                return 0
            
    def get_effectiveness(self):
        op_cnt_bn = min(self.network.get_num_ops(), 3)
        if op_cnt_bn == 0:
            return abs(self.state - self.get_augmented_state())
        op_bn = self.network.in_network[self.network.get_num_ops() - op_cnt_bn::].copy()

        if not self.controller_twin.k_last_rec:
            return abs(self.state - self.get_augmented_state())
        
        current_t = time.perf_counter()
        
        times_since_tx = [(time.perf_counter() - pkt.generation_ts) for pkt in op_bn]

        vals = [self.state_list[pkt.generation_id] for pkt in op_bn] 

        nodes = np.array(self.belief_nodes[op_cnt_bn-1])

        probs_in_queue = np.array([self.netw_registry.get_in_service_probability(1000*(times_since_tx[i])) for i in range(op_cnt_bn)])


        probs_will_be_lost = np.array([self.network.link_qual.get_prob_lost_after(1000*(times_since_tx[i]), self.netw_registry.max_tx_delay)
                                       for i in range(op_cnt_bn)])
        probs_will_be_rec = np.array([1 - self.network.link_qual.get_prob_lost_after(1000*(times_since_tx[i]), self.netw_registry.max_tx_delay)
                                      for i in range(op_cnt_bn)])
        

        probs_rec = np.array([1 - self.network.link_qual.get_prob_lost_before(1000*(times_since_tx[i]), self.netw_registry.max_tx_delay)
                              for i in range(op_cnt_bn)])
        probs_lost = np.array([self.network.link_qual.get_prob_lost_before(1000*(times_since_tx[i]), self.netw_registry.max_tx_delay) for i in range(op_cnt_bn)])

        
        real_augm = self.get_augmented_state()
            
        delays_will_be_received = np.array([1000*times_since_tx[i] for i in range(op_cnt_bn)])
        delays_received = np.array([self.netw_registry.get_tx_delay_before(1000*(times_since_tx[i])) for i in range(op_cnt_bn)])
        augm_will_be_received = np.array([self.controller_twin.get_x_est_with_rx(op_bn[i].generation_id, op_bn[i].generation_id
                                                                    + int(np.ceil(delays_will_be_received[i] / 10)), self.state_list[op_bn[i].generation_id]) for i in range(op_cnt_bn)])
        augm_received = np.array([self.controller_twin.get_x_est_with_rx(op_bn[i].generation_id, op_bn[i].generation_id 
                                                                    + int(np.ceil(delays_received[i] / 10)), self.state_list[op_bn[i].generation_id]) for i in range(op_cnt_bn)])

        probs_of_nodes = np.zeros(len(nodes))
        values_for_nodes = np.zeros(len(nodes))
        for t in range(len(nodes)):
            current_prob = 1
            for i in range(op_cnt_bn):
                if nodes[t][i] == 1:
                    current_prob *= probs_will_be_lost[i] * probs_in_queue[i]
                elif nodes[t][i] == 2:
                    current_prob *= probs_will_be_rec[i] * probs_in_queue[i]
                elif nodes[t][i] == 3:
                    current_prob *= probs_rec[i] * (1 - probs_in_queue[i])
                else:
                    current_prob *= probs_lost[i] * (1 - probs_in_queue[i])
                if current_prob < 0.01:
                    current_prob = 0
                    break
                if current_prob < 0.01:
                    continue
            if current_prob < 0.01:
                continue

            probs_of_nodes[t] = current_prob
            pos = None
            if 2 in nodes[t]:
                pos = op_cnt_bn - list(nodes[t][::-1]).index(2) - 1
                if self.augmentation_type == "none":
                    values_for_nodes[t] = (abs(self.state - vals[pos]))
                elif self.augmentation_type == "twin":
                    if abs(augm_will_be_received[pos]) > abs(self.state):
                        values_for_nodes[t] = abs(self.state)
                    else:
                        values_for_nodes[t] = abs(self.state - augm_will_be_received[pos])
            elif 3 in nodes[t] and 2 not in nodes[t]:
                pos = op_cnt_bn - list(nodes[t][::-1]).index(3) - 1
                if self.augmentation_type == "none":
                    values_for_nodes[t] = (abs(self.state - vals[pos]))
                elif self.augmentation_type == "twin":
                    if abs(augm_received[pos]) > abs(self.state):
                        values_for_nodes[t] = abs(self.state)
                    else:
                        values_for_nodes[t] = (abs(self.state - augm_received[pos]))
            if pos is None:
                if abs(real_augm) > abs(self.state):
                    values_for_nodes[t] = abs(self.state)
                else:
                    values_for_nodes[t] = (abs(self.state - real_augm))
      #  print('_____________________________') # 2-WR 3-R
      #  print('state and augm ', self.state, real_augm)
      #  for t in range(len(nodes)):
      #      print('node ',nodes[t], probs_of_nodes[t], values_for_nodes[t])
      #  print('_____________________________')

        return np.sum(np.array(probs_of_nodes) * np.array(values_for_nodes))

    def TCP_vegas_adjust(self):
        self.base_rtt = min([self.network.rtt_list[i] for i in range(len(self.network.rtt_list)) if self.network.rtt_list[i] > 0])
        expected_rate = self.op_max / self.base_rtt
        actual_rate = self.op_max / (1000*self.rtt)
        diff = (expected_rate - actual_rate) * self.base_rtt
        if self.phase == 'slow start' and diff > 0.05:
            self.phase = 'additive increase'
        elif self.phase == 'slow start':
            self.op_max *= 2
        if self.phase == 'additive increase':
            if diff < self.al:
                self.op_max += 1
            if diff > self.be:
                self.op_max -= 1
        self.last_adj = time.perf_counter()

    def get_augmented_state(self):
        if self.augmentation_type == "none":
            return 0
        if self.augmentation_type == "twin":
            return self.controller_twin.get_x_est()

    def TCP_ack(self):
        if self.tl_option == "tcp_tahoe":
            self.acked_packets_in_row += 1
            if self.op_max >= self.op_threshold:
                self.phase = 'additive increase'
            if self.phase == 'slow start':
                self.op_max *= 2
            elif self.phase == 'additive increase':
                self.op_max += 1

    def TCP_timeout(self):
        if self.tl_option == "tcp_tahoe" or self.tl_option == "tcp_vegas":
            self.op_threshold = max(1, self.op_max / 2)
            self.acked_packets_in_row = 0
            self.phase = 'slow start'
            self.op_max = 1

    def initialize_first_epoch(self):
        self.upd_period = self.network.ack_estimator.rtt_estimator.get_average()
        self.upd_period = 100
        self.epoch_end = time.perf_counter() + 1 / 1000 * self.period_multiplier * self.upd_period #min(
        self.prev_aoi_mean = np.mean(self.aoi_list_last_epoch)
        self.epoch_start = None
        self.prev_op_mean = np.mean(self.op_list_last_epoch)
        self.op_list_last_epoch = []
        self.aoi_list_last_epoch = []
        self.target_rate = 10 / self.upd_period

    def initialize_next_epoch(self):
        new_aoi_mean = np.median(self.aoi_list_last_epoch[10:])#np.meadian
        delta = new_aoi_mean - self.prev_aoi_mean
        if self.network.loop_id == 1:
            print(self.network.loop_id, 'age went from ', self.prev_aoi_mean, 'to ', new_aoi_mean)
        self.prev_aoi_mean = new_aoi_mean
        self.aoi_list_last_epoch = []

        new_op_mean = np.mean(self.op_list_last_epoch[10:])
        b = new_op_mean - self.prev_op_mean 
        if self.network.loop_id == 1:
            print(self.network.loop_id, 'ops went from  ', self.prev_op_mean, 'to ', new_op_mean)
            print(self.op_list_last_epoch[10:])
        self.prev_op_mean = new_op_mean
        self.op_list_last_epoch = []

        if abs(b) < 0.05 * new_op_mean and False:
            pass
        else:
            if delta > 0 and b > 0:
            #    print('age increase op increase')
                if self.flag == False:
                    self.b_star = -self.step_size
                    self.flag = True
                else:
                    self.gamma += 1
                    self.b_star = -(1 - 2 ** (-self.gamma)) * new_op_mean
            elif delta > 0 and b <= 0:
             #   print('age increase op decrease')
                self.flag = False
                self.gamma = 0
                self.b_star = self.step_size
            elif delta <= 0 and b >= 0:
             #   print('age decrease op increase')
                self.flag = False
                self.gamma = 0
                self.b_star = self.step_size
            else:
             #   print('age decrease op decrease')
                if self.flag and self.gamma > 0:
                    self.b_star = -(1 - 2 ** (-self.gamma)) * new_op_mean
                else:
                    self.b_star = -self.step_size
                    self.flag = False
                    self.gamma = 0
            if self.network.loop_id == 1:
                print('inter ack ', self.network.inter_ack_estimator.get_average(), 'rtt ', self.network.ack_estimator.rtt_estimator.get_average())
            new_rate = max((1 / self.network.inter_ack_estimator.get_average() + self.b_star / ( self.network.ack_estimator.rtt_estimator.get_average())), 0.005)
            if new_rate < 0.75 * 1 / self.upd_period:
                self.upd_period = 4 / 3 * self.upd_period
            elif new_rate > 1.25 * 1 / self.upd_period:
                self.upd_period = 4 / 5 * self.upd_period
            else:
                self.upd_period = 1 / new_rate

        self.epoch_end += 1 / 1000 * self.period_multiplier * self.upd_period
        if self.network.loop_id == 1:
            print(self.network.loop_id, 'new upd period ', self.upd_period)


        self.target_rate = 10 / self.upd_period

    def get_state(self, k):
        return self.state_list[k]

    def get_rl_decision(self):
        acc_hist, bw = self.get_obs_hist()
        action = self.obs_to_action(acc_hist, quantize_to_grid(self.state), bw)
        return action

    def obs_to_action(self, acc_hist, val, bw):
        state = {
            "val": np.array([val / (4000 + 1e-8)], dtype=np.float32),
            "acc_history": np.array(acc_hist, dtype=np.int32),
            "bw": np.array([(bw - 5) / (10 + 1e-8)], dtype=np.float32),
        }

        state = self.env.observation(state)
        state = torch.tensor(state, dtype=torch.float32, device='cuda').unsqueeze(0)
        action = self.rl_model(state).max(1).indices.view(1, 1)
        if np.abs(val) > 10000:# or (self.k > 1000 and self.k<1200):
    #        sys.stdout = sys.__stdout__
            print(self.network.loop_id, self.k)
            print(self.network.loop_id, "val ", val)
    #    #    print(self.loop_id, "delay ", delay)
            print(self.network.loop_id, "rl state  ", acc_hist)
            print(self.network.loop_id, "bw  ", bw)
            print(self.network.loop_id, "action ", action[0][0].cpu().numpy())
    #        print(self.loop_id, "decision  ", action[0][0].cpu().numpy())
    #        sys.stdout = open(os.devnull, 'w')
    #    print(acc_hist, val, action)

        return action[0][0].cpu().numpy()

    def get_obs_hist(self, n=10):
        if self.k < n + 1:
            acc_list = self.network.rtt_list[:self.k]
            acc_list = [0] * (n - self.k) + list(acc_list)
        else:
            acc_list = self.network.rtt_list[self.k-n-1:self.k-1]
        m = 100
        if self.k < m + 1:
            acc_list_d = self.network.rtt_list[:self.k]
            acc_list_d = [0] * (m - self.k) + list(acc_list_d)
        else:
            acc_list_d = self.network.rtt_list[self.k-m-1:self.k-1]
        sum_data = np.sum(np.array(acc_list_d) > 0)
        sum_delay = np.sum(np.array(acc_list_d)[np.where(np.array(acc_list_d) > 0)])
        if sum_delay:
            bw = np.round(sum_data / sum_delay * 100)
        else:
            bw = 0
        acc_for_rl = []
        for i in range(len(acc_list)):
            k_cnt = self.k - n + i + 1
            if self.network.check_op(k_cnt) and k_cnt < 10:
                acc_for_rl.append(1)
            elif self.network.check_op(k_cnt):
                acc_for_rl.append(4)
            elif acc_list[i] == -1:
                acc_for_rl.append(4)
            elif acc_list[i] > 0:
                acc_for_rl.append(1)
            else:
                acc_for_rl.append(0)
        return acc_for_rl, bw




class Controller:
    def __init__(self, controller_type, N):
        self.x_est = 0
        self.x_last_obs = 0
        self.k_last_rec = 0
        self.u_list = np.zeros(N)
        self.k = 0
        self.age_list = np.zeros(N)
        self.A = 1.2
        self.controller_type = controller_type
        if self.controller_type == "LQG":
            self.K = -0.79352812
            self.B = 1
        if self.controller_type == "PID":
            self.K_p = 2.3
            self.K_i = 120
            self.K_d = 0.0015
            self.B = 0.5
            self.I_k = 0
            self.last_state = 0

            self.I_k_list = np.zeros(N+1)
            self.last_state_list = np.zeros(N+1)

    def proportional_term(self, state):
        return -self.K_p * state

    def integral_term(self, state):
        self.I_k -= self.K_i * state * 0.01
        self.I_k_list[self.k] = self.I_k
        return self.I_k

    def derivative_term(self, state):
        D_k = self.K_d * (state - self.last_state) / 0.01
        self.last_state = state
        self.last_state_list[self.k] = self.last_state
        return -D_k

    def step(self):

        self.x_est = self.x_last_obs
        age = self.k - self.k_last_rec - 1
        self.age_list[self.k] = age
        self.x_est *= self.A ** age
        for i in range(age):
            self.x_est += self.A ** (i) * self.B * self.u_list[self.k_last_rec + age - i]
        if self.controller_type == "LQG":
            control_input = self.K * self.x_est
        elif self.controller_type == "PID":
            control_input = self.proportional_term(self.x_est) + self.integral_term(self.x_est) + self.derivative_term(self.x_est)
        control_input = round(control_input, 5)
        self.u_list[self.k] = control_input
        self.k += 1
        self.x_est = self.A * self.x_est + self.B * control_input
        return control_input

    def update_rx(self, k_last_rec, x_last_obs):
        self.k_last_rec = k_last_rec
        self.x_last_obs = x_last_obs

    def update_on_ack(self, k_rx):
       # print('upd on ack')
        k_rx = int(k_rx)
        if self.controller_type == "PID":
            self.I_k = self.I_k_list[k_rx-1]
            self.last_state = self.last_state_list[k_rx-1]
        for k in range(k_rx, self.k):
            x_est = self.x_last_obs
            age = k - self.k_last_rec - 1
            self.age_list[k] = age
            self.x_est *= self.A ** age
            for i in range(age):
                x_est += self.A ** (i) * self.B * self.u_list[self.k_last_rec + age - i]
            if self.controller_type == "LQG":
                control_input = self.K * x_est
            elif self.controller_type == "PID":
                control_input = self.proportional_term(x_est) + self.integral_term(x_est) + self.derivative_term(x_est)
            control_input = round(control_input, 5)
            self.u_list[k] = control_input

    def get_x_est(self):
        return self.x_est
    
    def get_x_est_with_rx(self, generation_id, reception_id, state):
      #  print("sent at", generation_id, "received at", reception_id, "current", self.k, "sent state ", state)
        if reception_id > self.k:
            return self.x_est
        tmp_u_list = self.u_list.copy()
        if self.controller_type == "PID":
         #   print("last state before ", self.last_state)
            real_I_k = self.I_k
            real_I_k_list = self.I_k_list.copy()
            real_last_state = self.last_state
            real_last_state_list = self.last_state_list.copy()


            self.I_k = self.I_k_list[reception_id - 1]
            self.last_state = self.last_state_list[reception_id - 1]
        for t in range(reception_id, self.k):
            x_est = state
            age = t - generation_id
            x_est *= self.A ** age
            for r in range(age):
                x_est += self.A ** (age - r) * self.B * tmp_u_list[generation_id + r]
            if self.controller_type == "LQG":
                control_input = self.K * x_est
            elif self.controller_type == "PID":
                control_input = self.proportional_term(x_est) + self.integral_term(x_est) + self.derivative_term(x_est)
            tmp_u_list[t] = control_input

        x_est = state
        age = self.k - generation_id

        x_est *= self.A ** age
        for i in range(age):
            x_est += self.A ** (age - i) * self.B * tmp_u_list[generation_id + i]

        if self.controller_type == "PID":
            self.I_k = real_I_k
            self.I_k_list = real_I_k_list.copy()
            self.last_state = real_last_state
            self.last_state_list = real_last_state_list.copy()
           # print("last state after ", self.last_state)
      #  print("instantaneous estimation ", x_est)
        return x_est

class Network:
    def __init__(self, s_data, s_ack, loop_id, N, sensor):
        self.data_socket = s_data
        self.ack_socket = s_ack
        self.rtt_list = np.zeros(N)
        self.delay_list = np.zeros(N)
        self.loop_id = loop_id

        self.sensor = sensor

        self.in_network = []
        self.ack_estimator = ACKTimeoutEstimator(100)
        self.link_qual = LinkQualityEstimator()
        self.inter_ack_estimator = None
        #self.k = 0

    def get_num_ops(self):
        return len(self.in_network)

    def get_oldest_op(self):
        return self.in_network[0]

    def clear_op(self):
        self.in_network = self.in_network[1:]

    def check_op(self, packet_id):
        for pkt in self.in_network:
            if pkt.generation_id == packet_id:
                return True
        return False

    def register_inter_ack_time(self, time):
        if self.inter_ack_estimator is None:
            self.inter_ack_estimator = ExponentialMovingAverage(time, 0.25)
        else:
            self.inter_ack_estimator.add_value(time)



    def send_data(self, k, state, estimation):
        state = np.array([state], dtype=np.float64)
        state = state.tobytes()

        estimation = np.array([estimation], dtype=np.float64)
        estimation = estimation.tobytes()


        MESSAGE = self.loop_id.to_bytes(2, 'big') + (k).to_bytes(3, 'big') + state + estimation
        self.data_socket.sendto(MESSAGE, ("127.0.0.1", 6000 + self.loop_id))
        pkt = Packet(k, time.perf_counter())
        self.in_network.append(pkt)
        #self.channel_busy.append((self.k + 1, time.perf_counter(), len(self.op_cnt) + 1))
        #self.op_cnt.append(self.k + 1)
        self.rtt_list[k] = -1
        self.link_qual.register_tx()

    def receive_data(self, k):
        try:
            recv = self.data_socket.recvfrom(5)[0]
            k_last_rec = int.from_bytes(recv[2:], "big")
#            self.x_last_obs = self.state_list[self.k_last_rec]
            MESSAGE = b'!' + self.loop_id.to_bytes(2, 'big') + k_last_rec.to_bytes(3, 'big') + k.to_bytes(
                3, 'big')
            self.ack_socket.sendto(MESSAGE, ("127.0.0.1", 6030 + self.loop_id))
            self.delay_list[k_last_rec] = k - k_last_rec
            return k_last_rec
        except:
            pass

    def clear_buffer(self):
        if not self.get_num_ops():
            return
        while self.get_num_ops():
            if time.perf_counter() - self.get_oldest_op().generation_ts > self.ack_estimator.get_ack_timeout() / 1000:
                self.sensor.last_ack_time = None
                self.clear_op()
                #self.op_cnt = self.op_cnt[1:]

                #if self.method == "tcp" or self.method == "tcp_vegas":
                self.sensor.TCP_timeout()
            else:
                break

    def find_packet_in_buffer(self, msg_id):
        for i in range(len(self.in_network)):
            if self.in_network[i].generation_id == msg_id:
                return i
        return None


    def receive_ack(self, k):
        recv = self.ack_socket.recvfrom(9)[0]
        if str(recv)[2] == "!":
            msg_id = int.from_bytes(recv[3:6], "big")
            rx_timestep = int.from_bytes(recv[7:10], "big")
            i = self.find_packet_in_buffer(msg_id)
            if i is not None:
                pkt = self.in_network[i]
                if i > 0:
                    self.sensor.last_ack_time = None
                    self.sensor.TCP_timeout()
                if i >= self.get_num_ops():
                    self.in_network = []
                else:
                    self.in_network = self.in_network[i+1:]
                self.link_qual.register_rx()
                self.ack_estimator.register_new_ack(1000 * (time.perf_counter() - pkt.generation_ts))
                self.sensor.TCP_ack()

                
                if self.sensor.last_ack_time is not None:
                    self.register_inter_ack_time(1000 * (time.perf_counter() - self.sensor.last_ack_time))
                self.sensor.last_ack_time = time.perf_counter()
                self.sensor.age_estimate = k - msg_id
                self.rtt_list[msg_id] = 1000 * (time.perf_counter() - pkt.generation_ts)
                self.sensor.rtt = time.perf_counter() - pkt.generation_ts
              #  if self.sensor.augmentation_type == 'twin':
              #      self.sensor.controller_twin.update_rx(pkt.generation_id, self.sensor.get_state(pkt.generation_id))
              #      self.sensor.controller_twin.update_on_ack(pkt.generation_id+self.delay_list[pkt.generation_id])
                if self.sensor.tl_option == "vou_inst":
                    self.sensor.netw_registry.register_tx_delay(10*(rx_timestep - msg_id))
                    if i == 0:
                        self.sensor.netw_registry.register_inter_sending_time(100)
                    else:
                        self.sensor.netw_registry.register_inter_sending_time(
                            1000 * (pkt.generation_ts - self.in_network[i - 1].generation_ts))
            else:
                self.ack_estimator.register_new_ack(10 * (k - msg_id))
            if self.sensor.augmentation_type == 'twin':
                self.sensor.controller_twin.update_rx(msg_id, self.sensor.get_state(msg_id))
                self.sensor.controller_twin.update_on_ack(msg_id + self.delay_list[msg_id])


class Run:
    def __init__(self, id, N, tl_option, threshold, augmentation_type, controller_type, data_socket, ack_socket, env, rl_model):
        self.loop_id = id

        self.N = N
        self.k = 0
        self.sensor = Sensor(tl_option, N, augmentation_type, threshold=threshold, controller_type=controller_type, env=env, rl_model=rl_model)
        self.controller = Controller(controller_type, self.N)
        self.network = Network(data_socket, ack_socket, id, N, self.sensor)
        self.sensor.associate_network(self.network)
        self.last_sampling_ts = None


    def do(self):


        self.last_sampling_ts = time.perf_counter()
        self.sensor.start_ts = time.perf_counter()
        self.sensor.t_last_sent = time.perf_counter()

        while True:
            if self.k == self.N:
                break
            if (time.perf_counter() - self.last_sampling_ts) * 1000 >= 10:  # 328
                self.network.clear_buffer()
                self.last_sampling_ts = time.perf_counter()
                control_input = self.controller.step()
           #     print(self.k, 'estimation ', self.controller.x_est)
                self.sensor.observe(control_input)


                if self.sensor.trigger():
                    self.sensor.t_last_sent = time.perf_counter()
                    state = np.round(self.sensor.state, 4)
                  #  print('trigger')
                   #     print(self.loop_id, "send data ", self.k, state, self.sensor.A * self.controller.x_est + self.sensor.B * control_input)
                    self.network.send_data(self.k, state, self.sensor.A * self.controller.x_est + self.sensor.B * control_input)
                self.k += 1
            try:
                k_last_rec = self.network.receive_data(self.k)
                x_last_obs = self.sensor.get_state(k_last_rec)
                if k_last_rec is not None:
                    self.controller.update_rx(k_last_rec, x_last_obs)
               #     print(self.loop_id, "receive data ", k_last_rec, x_last_obs)
            except:
                pass

            try:
                self.network.receive_ack(self.k)
            except:
                pass

        return self.sensor.state_list, self.controller.u_list, self.network.rtt_list, self.network.delay_list#self.rtt_list, self.state_list, self.u_list, self.age_list


def sim():
    loop_id = int(sys.argv[1])
    exp_name = "rl"#sys.argv[2]
    threshold = 0#7    # 1 loop - 0 (2.26), 2 loops - 1(3.8), 3 loops - 3 (6.4), 4 loops - 4 (12) , 5 loops - 5 (21) ## pid0 1 - 8, 2 - 7, 3 - 8, 4 - 10, 5 - 13 ## pid2 1 - 5.2 5 - 22
    augm_type = "none" #sys.argv[3]
    print("run control loop ", loop_id)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 5000 + loop_id))
    s.setblocking(False)
    ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ack_socket.bind(("127.0.0.1", 5030 + loop_id))
    ack_socket.setblocking(False)
    N = 5000

    controller_type = "LQG"
    # time.sleep(0.5*loop_id)

    if exp_name == 'rl':
        env = NormalizeObservation(FlattenObservation(gym.make('gym_examples/TraceControlEnv-v0')))
        n_actions = env.action_space.n
        state, info = env.reset()
        n_observations = len(state)
        rl_model = DQN(n_observations, n_actions).to(device)
        rl_model.load_state_dict(torch.load('truncated_1.pth')) #pid15 trained_model_lqg2
    else:
        env = None
        rl_model = None

    for i in range(5):
        sim_run = Run(loop_id, N, exp_name, threshold, augm_type, controller_type, s, ack_socket, env, rl_model)
        state_list, u_list, rtt_list, delay_list = sim_run.do()
        print(loop_id, "mean sq state ", np.mean(state_list[1000:]**2), "sending rate ", np.sum(np.array(rtt_list) > 0)/5000, 'mean aoi ', np.mean(sim_run.controller.age_list))
        #print(rtt_list)
        
        with open('data_lqg_trun/5_loop_256/states' + str(loop_id) + '_' + str(i+1) + '.txt', 'w') as f:
            for line in state_list:
                f.write(f"{line}, ")
        with open('data_lqg_trun/5_loop_256/controls' + str(loop_id) + '_' + str(i+1) + '.txt', 'w') as f:
            for line in u_list:
                f.write(f"{line}, ")
        with open('data_lqg_trun/5_loop_256/rtts' + str(loop_id) + '_' + str(i+1) + '.txt', 'w') as f:
            for line in rtt_list:
                f.write(f"{line}, ")
        with open('data_lqg_trun/5_loop_256/delays' + str(loop_id) + '_' + str(i+1) + '.txt', 'w') as f:
            for line in delay_list:
                f.write(f"{line}, ")
        
        time.sleep(3)
    return


def sim_var_n():
    loop_id = int(sys.argv[1])
    if loop_id == 1 or loop_id == 2 or loop_id == 3:
        controller_type = "LQG"
        model_name = 'trained_model_lqg2.pth'
        threshold = 0#19
    else:
    #if loop_id == 1 or loop_id == 2 or loop_id == 3 or loop_id == 4 or loop_id == 5:
        controller_type = "PID"
        model_name = 'trained_model_pid15.pth'
        threshold = 0#34
    exp_name = "udp"#sys.argv[2]
   # threshold = 19 # 1 loop - 0 (2.26), 2 loops - 1(3.8), 3 loops - 3 (6.4), 4 loops - 4 (12) , 5 loops - 5 (21) ## pid0 1 - 8, 2 - 7, 3 - 8, 4 - 10, 5 - 13 ## pid2 1 - 5.2 5 - 22
    augm_type = "none" #sys.argv[3]
    print("run control loop ", loop_id)
   # sys.stdout = open(os.devnull, 'w')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 5000 + loop_id))
    s.setblocking(False)
    ack_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ack_socket.bind(("127.0.0.1", 5030 + loop_id))
    ack_socket.setblocking(False)
   # def __init__(self, tl_option, N, augmentation_type=None, threshold=None, network = None):
    N = 10000
    np.random.seed(loop_id)
    timestamps_on, timestamps_off = schedule_on_off(N, 1/500, 1/1000)
    #controller_type = "LQG"

    if exp_name == 'rl':
        env = NormalizeObservation(FlattenObservation(gym.make('gym_examples/TraceControlEnv-v0')))
        n_actions = env.action_space.n
        state, info = env.reset()
        n_observations = len(state)
        rl_model = DQN(n_observations, n_actions).to(device)
        rl_model.load_state_dict(torch.load(model_name))
    else:
        env = None
        rl_model = None
    for i in range(1):
        timestamps_on, timestamps_off = schedule_on_off(N, 1/500, 1/1000)
        print(loop_id, timestamps_on, timestamps_off)


        state_list = []
        u_list = []
        rtt_list = []
        delay_list = []
        t_start = time.perf_counter()
        if exp_name == 'acp':
            upd_period = 100
        while time.perf_counter() - t_start < N * 0.01:
            t_tmp = time.perf_counter()
            if t_tmp < t_start + timestamps_on[0]*0.01:
               # print('sleep for ', t_start + timestamps_on[0]*0.01 - t_tmp)
                time.sleep(t_start + timestamps_on[0]*0.01 - t_tmp)
            if t_tmp > t_start + timestamps_on[0]*0.01:
                N_tmp = int(timestamps_off[0] - timestamps_on[0])
                timestamps_on = timestamps_on[1:]
                timestamps_off = timestamps_off[1:]
               # print('run for ', N_tmp)
                sim_run = Run(loop_id, N_tmp, exp_name, threshold, augm_type, controller_type, s, ack_socket, env, rl_model)
                if exp_name == 'acp':
                    sim_run.sensor.upd_period = upd_period
                state_list_tmp, u_list_tmp, rtt_list_tmp, delay_list_tmp = sim_run.do()
                if exp_name == 'acp':
                    upd_period = sim_run.sensor.upd_period
                state_list.extend(state_list_tmp)
                u_list.extend(u_list_tmp)
                rtt_list.extend(rtt_list_tmp)
                delay_list.extend(delay_list_tmp)
        print(loop_id, "mean sq state ", np.mean(np.array(state_list)**2), "sending rate ", np.sum(np.array(rtt_list) != 0)/1000, 'mean aoi ', np.mean(sim_run.controller.age_list))
        
        with open('data_var_n/wiswarm_mix/states' + str(loop_id) + '_' + str(i+1) + '.txt', 'w') as f:
            for line in state_list:
                f.write(f"{line}, ")
        with open('data_var_n/wiswarm_mix/controls' + str(loop_id) + '_' + str(i+1) + '.txt', 'w') as f:
            for line in u_list:
                f.write(f"{line}, ")
        with open('data_var_n/wiswarm_mix/rtts' + str(loop_id) + '_' + str(i+1) + '.txt', 'w') as f:
            for line in rtt_list:
                f.write(f"{line}, ")
        with open('data_var_n/wiswarm_mix/delays' + str(loop_id) + '_' + str(i+1) + '.txt', 'w') as f:
            for line in delay_list:
                f.write(f"{line}, ")
        
        time.sleep(10)

if __name__ == "__main__":
    sim()
