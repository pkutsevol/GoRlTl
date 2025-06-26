import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import json
import ast
from os.path import exists
from tqdm import tqdm
from itertools import product, combinations


AoI_max = 40
buf_size = 10
val_max = 2000

acc_history_length = 10

def quantize_to_grid(number):
    # Define the grid for numbers up to 600
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10,
            12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
            120, 150, 200, 250, 300, 350, 400, 450, 500, 600]
    
    # Handle negative numbers
    if number < 0:
        return -quantize_to_grid(abs(number))
    
    # For numbers larger than 600, round to nearest hundred
    if number > 600:
        return np.round(round(number / 100) * 100, 1)
    
    # Find the closest value in the grid
    return min(grid, key=lambda x: abs(x - number))

def load_dict(n=10):
#    path = 'observations_txt1'+str(n)+'.json'
    if n == 10:
        path = '10bwlqg_observations_txt'+'.json'
    if n == 9:
        path = '9bwlqg_observations_txt'+'.json'

    """
     if n == 10:
        path = '10bwpid_observations_txt'+'.json'
    if n == 9:
        path = '9bwpid_observations_txt'+'.json'
    
    """

    if exists(path):
        with open(path) as json_file:
            data = json.load(json_file)
    #new_dict = {}
    #for key in tqdm(data.keys()):
    #    val = [tuple(ast.literal_eval(el)) for el in data[key]]
    #    new_dict[tuple(ast.literal_eval(key))] = val
    return data

def dict_val_to_obs(val):
    #print('dict to val', val)
    val = ast.literal_eval(val)
  ###  return np.array(val[:acc_history_length]), val[acc_history_length]#, val[acc_history_length + 1]
    return np.array(val[:acc_history_length]), val[acc_history_length], val[acc_history_length + 1]

### def obs_to_dict_val(acc_hist, val):#, delay):
def obs_to_dict_val(acc_hist, val, bw):

    #print('val to dict', acc_hist, val)
    #if np.abs(val) < 10:
    #    val = np.round(val, 0)
    #elif np.abs(val) < 100:
    #    val = np.round(val, -1)
    #else:
    #    val = np.round(val, -2)

    val = quantize_to_grid(val)
###    d = str(list(acc_hist) + [val])# + [delay])
    d = str(list(acc_hist) + [val] + [bw])

    return d#tuple(list(acc_hist) + [val])

#def revert_to_ops(acc_hist):
#    rt = []
#    for el in acc_hist:
#        if el == 1:
#            rt.append(4)
#        else:
#            rt.append(el)
#    return np.array(rt)

def acc_hist_list(acc_hist):
    acc_hist = np.array(acc_hist)
    mask_ones_fours = np.array((acc_hist == 1) | (acc_hist == 4), dtype=bool)
    num_ones_fours = np.sum(mask_ones_fours)
    numbers = [1, 4]
    combinations_list = list(product(numbers, repeat=num_ones_fours))
    res_lst = []
    for i in range(len(combinations_list)):
        tmp_lst = acc_hist.copy()
        tmp_lst[mask_ones_fours] = combinations_list[i]
        res_lst.append(tmp_lst)
    return res_lst


def acc_hist_list_losses(acc_hist_list):
    res_lst = []
    for acc_hist in acc_hist_list:
        acc_hist = np.array(acc_hist)
        mask_twos = np.array((acc_hist == 2), dtype=bool)
        num_twos = np.sum(mask_twos)
        numbers = [0, 2]
        combinations_list = list(product(numbers, repeat=num_twos))

        for i in range(len(combinations_list)):
            tmp_lst = acc_hist.copy()
            tmp_lst[mask_twos] = combinations_list[i]
            res_lst.append(tmp_lst)
    return res_lst

def acc_hist_list_tot(acc_hist):
    acc_hist = np.array(acc_hist)
    total_length = len(acc_hist)
    num_zeros = np.sum((acc_hist == 0) | (acc_hist == 2))
    num_non_zeros = total_length - num_zeros
    
    res_lst = []
    
    # Generate all possible positions for non-zero elements
    for non_zero_positions in combinations(range(total_length), num_non_zeros):
        # Generate all combinations of 1, 2, and 4 for these positions
        for values in product([1, 4], repeat=num_non_zeros):
            tmp_lst = np.zeros(total_length, dtype=int)
            tmp_lst[list(non_zero_positions)] = values
            res_lst.append(tmp_lst.tolist())
    
    return res_lst




def action_to_obs(key, action, obs_dicts):
    obs, val, bw = dict_val_to_obs(key)
  ###  obs, val = dict_val_to_obs(key)
    key_to_look = obs_to_dict_val(obs, val, bw)
 ###   key_to_look = obs_to_dict_val(obs, val)
    suc = 0
    try:
        if action == 0:
            possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
        else:
            possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
        if len(possible_observations):
        #    print("FOUND")
            suc = 1
    except:
        suc = 0
    if not suc:
        try:
            val1 = quantize_to_grid(np.round(val, -1))
###            key_to_look = obs_to_dict_val(obs, val1)
            key_to_look = obs_to_dict_val(obs, val1, bw)

            if action == 0:
                possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
            else:
                possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
            if len(possible_observations):
                print("FOUND 1")
                suc = 1
        except: 
            suc = 0

    if not suc:
        key_to_look = obs_to_dict_val(obs[-9:], val, bw)

###        key_to_look = obs_to_dict_val(obs[-9:], val)
        try:
            if action == 0:
                possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
            else:
                possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
            if len(possible_observations):
                suc = 1
                print("FOUND 2")

        except:
            suc = 0
        if not suc:
            try:
                val1 = quantize_to_grid(np.round(val, -1))
###                key_to_look = obs_to_dict_val(obs[-9:], val1)
                key_to_look = obs_to_dict_val(obs[-9:], val1, bw)

                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    suc = 1
                    print("FOUND 3")
            except: 
                suc = 0
    if not suc:
        obs_rev_list = acc_hist_list(obs)
        np.random.shuffle(obs_rev_list)
        for el in obs_rev_list:
            try:
###                key_to_look =  obs_to_dict_val(el, val)

                key_to_look =  obs_to_dict_val(el, val, bw)
                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    print("FOUND 4")
                    suc = 1
                    break
            except:
                suc = 0

    if not suc:
        obs_rev_list = acc_hist_list(obs)
        np.random.shuffle(obs_rev_list)
        val1 = quantize_to_grid(np.round(val, -1))
        for el in obs_rev_list:
            try:
###                key_to_look =  obs_to_dict_val(el, val1)

                key_to_look =  obs_to_dict_val(el, val1, bw)
                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    print("FOUND 5")
                    suc = 1
                    break
            except:
                suc = 0
    if not suc:
        obs_rev_list = acc_hist_list(obs)
        np.random.shuffle(obs_rev_list)
        for el in obs_rev_list:
            try:
                key_to_look =  obs_to_dict_val(el[-9:], val, bw)
###                key_to_look =  obs_to_dict_val(el[-9:], val)

                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    print("FOUND 6")
                    suc = 1
                    break
            except:
                suc = 0

    if not suc:
            obs_rev_list = acc_hist_list(obs)
            np.random.shuffle(obs_rev_list)
            val1 = quantize_to_grid(np.round(val, -1))
            for el in obs_rev_list:
                try:
                    key_to_look =  obs_to_dict_val(el[-9:], val1, bw)
###                    key_to_look =  obs_to_dict_val(el[-9:], val1)

                    if action == 0:
                        possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
                    else:
                        possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
                    if len(possible_observations):
                        print("FOUND 7")
                        suc = 1
                        break
                except:
                    suc = 0

    if not suc:
        obs_rev_list = acc_hist_list(obs)
        obs_rev_list = acc_hist_list_losses(obs_rev_list)
        np.random.shuffle(obs_rev_list)
        for el in obs_rev_list:
            try:
                key_to_look =  obs_to_dict_val(el, val, bw)
                ###key_to_look =  obs_to_dict_val(el, val)

                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    print("FOUND 8")
                    suc = 1
                    break
            except:
                suc = 0

    if not suc:
        obs_rev_list = acc_hist_list(obs)
        obs_rev_list = acc_hist_list_losses(obs_rev_list)
        np.random.shuffle(obs_rev_list)
        val1 = quantize_to_grid(np.round(val, -1))
        for el in obs_rev_list:
            try:
###                key_to_look =  obs_to_dict_val(el, val1)
                key_to_look =  obs_to_dict_val(el, val1, bw)

                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    print("FOUND 9")
                    suc = 1
                    break
            except:
                suc = 0

    if not suc:
        obs_rev_list = acc_hist_list(obs)
        obs_rev_list = acc_hist_list_losses(obs_rev_list)
        np.random.shuffle(obs_rev_list)
        for el in obs_rev_list:
            try:
###                key_to_look = obs_to_dict_val(el[-9:], val)

                key_to_look = obs_to_dict_val(el[-9:], val, bw)
                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if
                                             dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if
                                             dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    print("FOUND 10")
                    suc = 1
                    break
            except:
                suc = 0

    if not suc:
        obs_rev_list = acc_hist_list(obs)
        obs_rev_list = acc_hist_list_losses(obs_rev_list)
        np.random.shuffle(obs_rev_list)
        val1 = quantize_to_grid(np.round(val, -1))
        for el in obs_rev_list:
            try:
###                key_to_look = obs_to_dict_val(el[-9:], val1)

                key_to_look = obs_to_dict_val(el[-9:], val1, bw)
                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if
                                             dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[1][key_to_look] if
                                             dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    print("FOUND 11")
                    suc = 1
                    break
            except:
                suc = 0
            
    if not suc:
        obs_rev_list = acc_hist_list_tot(obs)
        np.random.shuffle(obs_rev_list)
        for el in obs_rev_list:
            try:
                ###key_to_look = obs_to_dict_val(el, val)

                key_to_look = obs_to_dict_val(el, val, bw)
                #print("try ", key_to_look)
                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if
                                             dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][key_to_look] if
                                             dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    suc = 1
                    break
            except:
                suc = 0
    if not suc:
        print("STATE NOT FOUND")
        return None, None, None

    if not suc:
        print("random")
        while True:
            t = np.random.randint(0, len(obs_dicts[0].keys()), size=1, dtype=int)[0]
            acceptance_history_r, val_r, bw_r = dict_val_to_obs(list(obs_dicts[0].keys())[t])

###            acceptance_history_r, val_r = dict_val_to_obs(list(obs_dicts[0].keys())[t])
            if val_r == val:
                if action == 0:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][list(obs_dicts[0].keys())[t]] if
                                             dict_val_to_obs(el)[0][-1] == action]
                else:
                    possible_observations = [dict_val_to_obs(el) for el in obs_dicts[0][list(obs_dicts[0].keys())[t]] if
                                             dict_val_to_obs(el)[0][-1] != 0]
                if len(possible_observations):
                    suc = 1
                    break


    if not suc:
        print("STATE NOT FOUND")
        return None, None
    else:
        idx = np.random.randint(0, len(possible_observations), size=1, dtype=int)[0]
      #  print("pos ", possible_observations)
        return possible_observations[idx]

#def find_close(obs_dicts, key, action):
#    return None, None
#  #  print("find close to ", key, action)
#    n_to_match = acc_history_length - 1
#    acc_hist, val = dict_val_to_obs(key)
#   # acc_hist_new = np.roll(acc_hist, 1)
#   # acc_hist_new[-1] = action
#    while True:
#        for n_to_match in range(acc_history_length - 1, 0, -1):
#            key_to_look = obs_to_dict_val(acc_hist[-n_to_match:], val)
#            # print("key to look ", key_to_look)
#            try:
#                for el in obs_dicts[acc_history_length - n_to_match][key_to_look]:
#                    # print("check el ", el)
#                    obs,v = dict_val_to_obs(el)
#                    if action == 0:
#                        if obs[-1] == action:
#                            #  print("found ", el)
#                          #  possible_observations = [(o,v) for o,v in dict_val_to_obs(obs_dicts[acc_history_length - n_to_match][key_to_look]) if o[-1] == action]
#                            possible_observations = [dict_val_to_obs(el) for el in obs_dicts[acc_history_length - n_to_match][key_to_look] if dict_val_to_obs(el)[0][-1] == action]
#                            # print("possible obs ", possible_observations)
#                            idx = np.random.randint(0, len(possible_observations), size=1, dtype=int)[0]
#                            return possible_observations[idx]
#                    else:
#                        if obs[-1] != 0:
#                            #  print("found ", el)
#                            possible_observations = [dict_val_to_obs(el) for el in obs_dicts[acc_history_length - n_to_match][key_to_look] if dict_val_to_obs(el)[0][-1] != 0]
#                            # print("possible obs ", possible_observations)
#                            idx = np.random.randint(0, len(possible_observations), size=1, dtype=int)[0]
#                            return possible_observations[idx]
#            except:
#                pass
#        print(":::::::::::::::not found ", key)
#        return None, None
#        t = np.random.randint(0, len(obs_dicts[0].keys()), size=1, dtype=int)[0]
#        return dict_val_to_obs(list(obs_dicts[0].keys())[t])
#       # for el in obs_dicts[0][key]:
#       #     return dict_val_to_obs(el)
#        '''
#    while True:
#        for el in list(obs_dict.keys()):
#            acc_hist_tmp, val_tmp = dict_val_to_obs(el)
#            acc_hist_new_tmp = np.roll(acc_hist_tmp, 1)
#            acc_hist_new_tmp[-1] = action
#            if np.all(acc_hist[i_to_match:] == acc_hist_tmp[i_to_match:]) and val == val_tmp:
#                next_states = obs_dict[el]
#                for next_st in next_states:
#                    if next_st[-2] == action:
#                        return el
#        i_to_match += 1
#    '''
#

class TraceControlEnv(gym.Env):
    #  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, delay=1):
        self.delay = delay
        self.cost = 50
        self.cost2 = 1
        self.cost3 = 10
        self.cost4 = 1
        self.obs_dicts = [load_dict(n) for n in tqdm(range(10,8,-1))]
        self.observation_space = spaces.Dict(
            {
                "val": spaces.Box(low=-val_max, high=val_max, dtype=np.float32),
                "bw": spaces.Box(low=0, high=10, dtype=np.float32),
                "acc_history": spaces.MultiDiscrete([5] * (10))
            }
        )

        self.action_space = spaces.Discrete(2)

    def _get_obs(self):
###        return {"val": self._val, "acc_history": self._acceptance_history}
        return {"val": self._val, "bw": self._bw, "acc_history": self._acceptance_history}

    
    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

       # while True:
#
 #           t = self.np_random.integers(0, len(self.obs_dicts[0].keys()), size=1, dtype=int)[0]
#
 #           self._acceptance_history, self._val = dict_val_to_obs(list(self.obs_dicts[0].keys())[t])
#
 #           if np.abs(self._val) < 10.0 and len(self.obs_dicts[0][list(self.obs_dicts[0].keys())[t]]) > 10:
  #              break

       # self._val = 1.0
       # self._acceptance_history = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        while True:
            t = self.np_random.integers(0, len(self.obs_dicts[0].keys()), size=1, dtype=int)[0]
###            self._acceptance_history, self._val = dict_val_to_obs(list(self.obs_dicts[0].keys())[t])
            self._acceptance_history, self._val, self._bw = dict_val_to_obs(list(self.obs_dicts[0].keys())[t])

            if np.abs(self._val) < 50.0:
                break
                
        observation = self._get_obs()
        info = {}

        print("obs after reset ", observation)

        return observation, info

    def step(self, action):

      #  print("prev obs ", self._acceptance_history, self._val, self._delay)

      ###  next_acceptance_hist, next_val  = action_to_obs(obs_to_dict_val(self._acceptance_history, self._val), action, self.obs_dicts)
        next_acceptance_hist, next_val, next_bw = action_to_obs(obs_to_dict_val(self._acceptance_history, self._val, self._bw), action,
                                                     self.obs_dicts)
        prev_val = np.abs(self._val).copy()
        if next_val is None:
            terminated = 1
            observation = self._get_obs()
            info = self._get_info()
            num_acc = np.sum(next_acceptance_hist == 4) + 1 * np.sum(next_acceptance_hist == 1)  #len(next_acceptance_hist != 0)
         ###   return observation, -self.cost2*np.abs(self._val)**2 - self.cost*action*num_acc, terminated, False, info
            return observation, -self.cost2*(np.abs(2000/2000)-1)**75    - self.cost*num_acc*action - min(self.cost4*((np.abs(2000) - np.abs(prev_val))/2000)**3, 0), terminated, False, info #(was power 35)
        #add_on =np.round(self.np_random.integers(-1, 2, size=1, dtype=int)[0]/10, 1)
        #print(add_on)
        #next_val += add_on
        next_val = np.round(next_val, 1)
        #print(next_val)
       # print("next obs ", next_acceptance_hist, next_val, next_bw)

    #    next_acceptance_hist, next_val = dict_val_to_obs(next_obs)


        #print("action ", action)
       # print("1", self.cost*action*num_acc, " 2 ", self._delay/10)

        self._acceptance_history = next_acceptance_hist
        self._val = next_val

        self._val = min(val_max, max(-val_max, self._val))
        self._bw = next_bw
        #num_acc = np.sum(next_acceptance_hist != 0)

        num_acc = np.sum(next_acceptance_hist == 4) + 1 * np.sum(next_acceptance_hist == 1)#np.sum(self._acceptance_history != 0)
        netw_reward = self.cost  * num_acc*action
       # print('reward ', -self.cost2 *np.log(np.abs(self._val/2000)+1))
       # print('full reward ', -self.cost2 *np.log(np.abs(self._val/2000)+1) - netw_reward / (next_bw + 1)**2 + self.cost3 * next_bw)
        ###reward = -self.cost2*np.abs(self._val)**2 -  netw_reward#self.cost*action*num_acc - (self._delay/10)
        #reward = -self.cost2 *(np.abs(self._val/2000)-1)**25 - netw_reward/(self._bw+1)**2  + self.cost3 * next_bw - min(self.cost4*((np.abs(self._val) - np.abs(prev_val))/4000 - 1), 0)# self.cost*action*num_acc - (self._delay/10)
        reward = -self.cost2 *(np.abs(self._val/2000)-1)**75 - netw_reward/(self._bw+1)**2  + self.cost3 * next_bw - min(self.cost4*((np.abs(self._val) - np.abs(prev_val))/2000)**3, 0)# self.cost*action*num_acc - (self._delay/10)

       # print('reward 4', prev_val, self._val, self.cost4*((np.abs(self._val) - np.abs(prev_val))/4000))
     #   print("reward, ", reward)
        terminated = 0
        if (np.abs(self._val) >= val_max - 2) and False:
            terminated = 1
           ### reward = -self.cost2*np.abs(self._val)**2 - self.cost*action*num_acc #- self.cost2*(self._delay/10*action)
            reward = -self.cost2*np.abs(self._val)**2 - self.cost*num_acc*action + self.cost3 * next_bw#- self.cost2*(self._delay/10*action)

            # reward = 0
            print('terminate')
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def set_cost(self, cost):
        self.cost = cost

    def set_cost2(self, cost2):
        self.cost2 = cost2

    def set_cost3(self, cost3):
        self.cost3 = cost3
    
    def set_cost4(self, cost4):
        self.cost4 = cost4
