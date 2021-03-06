from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features
from math import fabs as abs
from agents.network import build_net
import utils as U

_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

up_left = 0
up = 1
up_right = 2
left = 3
right = 4
left_down = 5
down = 6
right_down = 7
NO_OP = 8
DEBUG = False


def filter_enemy_hp(hp_map, normal_map, enemy_id):
    filtered_map = np.zeros(hp_map.shape)
    for i in range(hp_map.shape[0]):
        for j in range(hp_map.shape[-1]):
            if normal_map[i][j] == enemy_id:
                filtered_map[i][j] = hp_map[i][j]
    return filtered_map


def turn_hp_into_vector(map, self_x, self_y):
    row = map.shape[0]
    col = map.shape[1]
    self_x = int(self_x)
    self_y = int(self_y)
    vec1 = map[0:self_x, 0:self_y]
    vec2 = map[0:self_x, self_y:]
    vec3 = map[self_x:, 0:self_y]
    vec4 = map[self_x:, self_y:]
    hp1 = np.sum(vec1)
    hp2 = np.sum(vec2)
    hp3 = np.sum(vec3)
    hp4 = np.sum(vec4)
    res = np.zeros(4)
    res[0] = hp1
    res[1] = hp2
    res[2] = hp3
    res[3] = hp4
    return res


def calulate_phase_simple(self_x, self_y, enemy_x, enemy_y):
    vector = np.zeros(8)
    phase_dim = -1
    delta_x = float(enemy_x - self_x)
    delta_y = float(enemy_y - self_y)

    if math.fabs(delta_x) < 0.01 and delta_y <= 0:
        phase_dim = 6
    elif math.fabs(delta_x) < 0.01 and delta_y > 0:
        phase_dim = 2
    else:
        if delta_x >= 0 and delta_y >= 0 and abs(delta_x) > abs(delta_y):
            phase_dim = 0
        elif delta_x >= 0 and delta_y >= 0 and abs(delta_x) <= abs(delta_y):
            phase_dim = 1
        elif delta_x <= 0 and delta_y >= 0 and abs(delta_x) > abs(delta_y):
            phase_dim = 2
        elif delta_x <= 0 and delta_y >= 0 and abs(delta_x) <= abs(delta_y):
            phase_dim = 3
        elif delta_x <= 0 and delta_y <= 0 and abs(delta_x) > abs(delta_y):
            phase_dim = 4
        elif delta_x <= 0 and delta_y <= 0 and abs(delta_x) <= abs(delta_y):
            phase_dim = 5
        elif delta_x >= 0 and delta_y <= 0 and abs(delta_x) > abs(delta_y):
            phase_dim = 6
        elif delta_x >= 0 and delta_y <= 0 and abs(delta_x) <= abs(delta_y):
            phase_dim = 7

    if phase_dim == -1:
        print('calulate_phase_simple maybe wrong')
        return vector
    else:
        vector[phase_dim] = 1
        return vector


def calulate_distance_in_phase_simple(self_x, self_y, enemy_x, enemy_y):
    vector = np.zeros(8)
    phase_dim = -1
    delta_x = float(enemy_x - self_x)
    delta_y = float(enemy_y - self_y)
    distance = math.sqrt(delta_y ** 2 + delta_x ** 2)
    if delta_x >= 0 and delta_y >= 0 and delta_x > delta_y:
        phase_dim = 0
    elif delta_x >= 0 and delta_y >= 0 and delta_x <= delta_y:
        phase_dim = 1
    elif delta_x <= 0 and delta_y >= 0 and abs(delta_x) > abs(delta_y):
        phase_dim = 2
    elif delta_x <= 0 and delta_y >= 0 and abs(delta_x) <= abs(delta_y):
        phase_dim = 3
    elif delta_x <= 0 and delta_y <= 0 and abs(delta_x) > abs(delta_y):
        phase_dim = 4
    elif delta_x <= 0 and delta_y <= 0 and abs(delta_x) <= abs(delta_y):
        phase_dim = 5
    elif delta_x >= 0 and delta_y <= 0 and abs(delta_x) > abs(delta_y):
        phase_dim = 6
    elif delta_x >= 0 and delta_y <= 0 and abs(delta_x) <= abs(delta_y):
        phase_dim = 7

    if phase_dim == -1:
        print('maybe wrong in  calulate_distance_in_phase_simple')
        return vector
    else:
        vector[phase_dim] = distance
        return vector


def calulate_phase(self_x, self_y, enemy_x, enemy_y):
    vector = np.zeros(8)
    phase_dim = -1
    delta_x = float(enemy_x - self_x)
    delta_y = float(enemy_y - self_y)

    if math.fabs(delta_x) < 0.01 and delta_y <= 0:
        phase_dim = 6
    elif math.fabs(delta_x) < 0.01 and delta_y > 0:
        phase_dim = 2
    else:
        tan = delta_y / delta_x
        theta = math.atan(tan)
        if theta < 0:
            theta += math.pi
        if 0 <= theta < (math.pi / 6) or (math.pi * 11 / 6) <= theta < math.pi * 2:
            phase_dim = 0
        elif (math.pi / 6) <= theta < (math.pi / 3):
            phase_dim = 1
        elif (math.pi / 3) <= theta < (math.pi / 3 + math.pi / 6):
            phase_dim = 2
        elif (math.pi / 3 + math.pi / 6) <= theta < (math.pi / 3 + math.pi / 3):
            phase_dim = 3
        elif (math.pi * 2 / 3) <= theta < (math.pi * 2 / 3 + math.pi / 6):
            phase_dim = 4
        elif (math.pi * 2 / 3 + math.pi / 6) <= theta < (math.pi * 2 / 3 + math.pi / 3):
            phase_dim = 5
        elif (math.pi * 2 / 3 + math.pi / 3) <= theta < (math.pi * 2 / 3 + math.pi / 3 + math.pi / 6):
            phase_dim = 6
        elif (math.pi * 4 / 3) <= theta <= (math.pi * 4 / 3 + math.pi / 6):
            phase_dim = 7
    if phase_dim == -1:
        return vector
    else:
        vector[phase_dim] = 1
        return vector


def calulate_distance_in_phase(self_x, self_y, enemy_x, enemy_y):
    vector = np.zeros(8)
    phase_dim = -1
    delta_x = float(enemy_x - self_x)
    delta_y = float(enemy_y - self_y)
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    if math.fabs(delta_x) < 0.01 and delta_y < 0:
        phase_dim = 6
    elif math.fabs(delta_x) < 0.01 and delta_y > 0:
        phase_dim = 2
    else:
        tan = delta_y / delta_x
        theta = math.atan(tan)
        if theta < 0:
            theta += math.pi
        if 0 <= theta < (math.pi / 6) or (math.pi * 11 / 6) <= theta < math.pi * 2:
            phase_dim = 0
        elif (math.pi / 6) <= theta < (math.pi / 3):
            phase_dim = 1
        elif (math.pi / 3) <= theta < (math.pi / 3 + math.pi / 6):
            phase_dim = 2
        elif (math.pi / 3 + math.pi / 6) <= theta < (math.pi / 3 + math.pi / 3):
            phase_dim = 3
        elif (math.pi * 2 / 3) <= theta < (math.pi * 2 / 3 + math.pi / 6):
            phase_dim = 4
        elif (math.pi * 2 / 3 + math.pi / 6) <= theta < (math.pi * 2 / 3 + math.pi / 3):
            phase_dim = 5
        elif (math.pi * 2 / 3 + math.pi / 3) <= theta < (math.pi * 2 / 3 + math.pi / 3 + math.pi / 6):
            phase_dim = 6
        elif (math.pi * 4 / 3) <= theta <= (math.pi * 4 / 3 + math.pi / 6):
            phase_dim = 7
    if phase_dim == -1:
        return vector
    else:
        vector[phase_dim] = distance
        return vector


def calulate_enemy_num_distribution(self_x, self_y, enemy_x_list, enemy_y_list):
    total_vector = np.zeros(8)
    for index, one in enumerate(enemy_x_list):
        vec_tmp = calulate_phase_simple(self_x, self_y, one, enemy_y_list[index])
        total_vector += vec_tmp
    return total_vector


def calulate_enemy_distance_distribution(self_x, self_y, enemy_x_list, enemy_y_list):
    total_vector = list()
    for index, one in enumerate(enemy_x_list):
        vec_tmp = calulate_distance_in_phase_simple(self_x, self_y, one, enemy_y_list[index])
        total_vector.append((vec_tmp, np.linalg.norm(vec_tmp)))
    total_vector = sorted(total_vector, key=lambda x: x[1])
    return total_vector[0][0]


def get_group_x_y(player_selected):
    selected_y, selected_x = (player_selected == _PLAYER_FRIENDLY).nonzero()
    if len(selected_x) == 0 or len(selected_y) == 0:
        return -1, -1
    mean_x = np.mean(selected_x)
    mean_y = np.mean(selected_y)
    return int(mean_x), int(mean_y)


def merge_target_with_position(target, x, y, ssize):
    mov_step = 5
    # 0 up left ；1 up； 2 up right； 3 left； 4 right；5 left down 6 down 7 right down
    target_x = x
    target_y = y
    if target == up_left:
        target_x = x - mov_step
        target_y = y - mov_step
        if DEBUG: print("up left")
    elif target == up:
        target_x = x
        target_y = y - mov_step
        if DEBUG: print("up ")
    elif target == up_right:
        target_x = x + mov_step
        target_y = y - mov_step
        if DEBUG: print("up right")
    elif target == left:
        target_x = x - mov_step
        target_y = y
        if DEBUG: print("left")
    elif target == right:
        target_x = x + mov_step
        target_y = y
        if DEBUG: print("right")
    elif target == left_down:
        target_x = x - mov_step
        target_y = y + mov_step
        if DEBUG: print("left down")
    elif target == down:
        target_x = x
        target_y = y + mov_step
        if DEBUG: print("down")
    elif target == right_down:
        target_x = x + mov_step
        target_y = y + mov_step
        if DEBUG: print("down right")
    if target_y >= ssize:
        target_y = ssize - 1
    if target_x >= ssize:
        target_x = ssize - 1
    if target_x < 0:
        target_x = 0
    if target_y < 0:
        target_y = 0
    return target_x, target_y


def convert_xy_2_eight_direction(x, y, new_x, new_y):
    if new_x < x and new_y < y:
        return up_left
    elif new_x == x and new_y < y:
        return up
    elif new_x > x and new_y < y:
        return up_right
    elif new_x < x and new_y == y:
        return left
    elif new_x > x and new_y == y:
        return right
    elif new_x < x and new_y > y:
        return left_down
    elif new_x == x and new_y > y:
        return down
    elif new_x > x and new_x > y:
        return right_down
    else:
        return NO_OP


def compute_distance_reward(dist):
    return (1 * 100) / (dist + 0.1)


def mean_fun(num_list):
    if num_list is None or len(num_list) == 0:
        return 0
    else:
        return sum(num_list) / len(num_list)


def compute_curiosity_reward(minimap, next_minimap):
    flat_minimap = np.ndarray.flatten(minimap)
    flat_next_minimap = np.ndarray.flatten(next_minimap)
    # flat_minimap = flat_minimap[3:]
    # flat_next_minimap = flat_next_minimap[3:]
    reward_curiosity = np.linalg.norm(flat_minimap - flat_next_minimap)
    # reward_curiosity = reward_curiosity if reward_curiosity <= 50 else 50
    return reward_curiosity


class A3CAgent(object):
    """An agent specifically for solving the mini-game maps."""
    EIGHT_SPATIAL_ACTION = 9

    def __init__(self, training, msize, ssize, name='A3C/A3CAgent'):
        self.name = name
        self.training = training
        self.summary = []
        # Minimap size, screen size and info size
        assert msize == ssize
        self.msize = msize
        self.ssize = ssize
        self.isize = len(actions.FUNCTIONS)
        self.agent_num = 0

    def setup(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reset(self):
        # Epsilon schedule
        self.epsilon = [0.05, 0.]

    def build_model(self, reuse, dev, ntype):
        with tf.variable_scope(self.name) and tf.device(dev):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse

            # Set inputs of networks
            # self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
            self.minimap = tf.placeholder(tf.float32, [None, U.hand_crafted_feature_num()], name='minimap')
            self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
            self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

            # Build networks
            net = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS), ntype)
            self.spatial_action, self.non_spatial_action, self.value = net

            # Set targets and masks
            self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
            self.spatial_action_selected = tf.placeholder(tf.float32, [None, A3CAgent.EIGHT_SPATIAL_ACTION],
                                                          name='spatial_action_selected')
            self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                           name='valid_non_spatial_action')
            self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                              name='non_spatial_action_selected')
            self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

            # Compute log probability
            spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
            non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
            valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action,
                                                          axis=1)
            valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
            non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
            self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

            # Compute losses, more details in https://arxiv.org/abs/1602.01783
            # Policy loss and value loss
            action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
            advantage = tf.stop_gradient(self.value_target - self.value)
            policy_loss = - tf.reduce_mean(action_log_prob * advantage)
            value_loss = - tf.reduce_mean(self.value * advantage)
            # action_ent = tf.reduce_mean(action_log_prob)
            self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
            self.summary.append(tf.summary.scalar('value_loss', value_loss))

            # TODO: policy penalty
            loss = policy_loss + value_loss

            # Build the optimizer
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
            grads = opt.compute_gradients(loss)
            cliped_grad = []
            for grad, var in grads:
                self.summary.append(tf.summary.histogram(var.op.name, var))
                self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))
                grad = tf.clip_by_norm(grad, 10.0)
                cliped_grad.append([grad, var])
            self.train_op = opt.apply_gradients(cliped_grad)
            self.summary_op = tf.summary.merge(self.summary)

            self.saver = tf.train.Saver(max_to_keep=100)


    def get_enemy_friend_distance(self, obs):
        _PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        _PLAYER_SELECTED = features.SCREEN_FEATURES.selected.index
        # player_selected = obs.observation["screen"][_PLAYER_SELECTED]
        # selected_x, selected_y = get_group_x_y(player_selected)
        player_selected = obs.observation["screen"][_PLAYER_SELECTED]
        selected_x, selected_y = get_group_x_y(player_selected)
        # enemy_hp_map = filter_enemy_hp(player_hit_points, player_relative, _PLAYER_HOSTILE)
        enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()
        # enemy_y, enemy_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        if enemy_y is None or len(enemy_y) == 0:
            enemy_mean_y = -1
            enemy_mean_x = -1
            return 0
        else:
            enemy_mean_y = mean_fun(enemy_y)
            enemy_mean_x = mean_fun(enemy_x)
        friend_y, friend_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        if friend_y is None or len(friend_y) == 0 or friend_x is None or len(friend_x) == 0:
            friend_mean_y = 1000
            friend_mean_x = 1000
        else:
            friend_mean_y = mean_fun(friend_y)
            friend_mean_x = mean_fun(friend_x)
        delta = (friend_mean_y - enemy_mean_y) ** 2 + (enemy_mean_x - friend_mean_x) ** 2
        distance = math.sqrt(delta)
        return distance

    def get_hand_crafted_feature(self, obs):
        _PLAYER_HIT = features.SCREEN_FEATURES.unit_hit_points.index
        player_hit_points = obs.observation["screen"][_PLAYER_HIT]
        _PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        _PLAYER_SELECTED = features.SCREEN_FEATURES.selected.index
        player_selected = obs.observation["screen"][_PLAYER_SELECTED]
        selected_x, selected_y = get_group_x_y(player_selected)
        enemy_hp_map = filter_enemy_hp(player_hit_points, player_relative, _PLAYER_HOSTILE)
        enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()
        # enemy_y, enemy_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        friend_y, friend_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        if len(friend_x) != len(friend_y):
            print("friend_x num != friend_y num")

        if self.agent_num >= len(friend_y) - 1:
            self.agent_num = 0
        else:
            self.agent_num = self.agent_num + 1

        if len(enemy_y) > 0 and len(enemy_x) > 0 and len(friend_x) > 0 and len(friend_y) > 0:
            hp_vec_enemy = turn_hp_into_vector(enemy_hp_map, selected_x, selected_y)
            vector_enemy_num_8_dim = calulate_enemy_num_distribution(selected_x, selected_y, enemy_x,
                                                                     enemy_y)
            vector_enemy_dist_8_dim = calulate_enemy_distance_distribution(selected_x, selected_y,
                                                                           enemy_x, enemy_y)
            vector_friend_num_8_dim = calulate_enemy_num_distribution(selected_x, selected_y,
                                                                      friend_x,
                                                                      friend_y)
            vector_friend_dist_8_dim = calulate_enemy_distance_distribution(selected_x, selected_y,
                                                                            friend_x, friend_y)
        else:
            vector_enemy_num_8_dim = np.zeros(8)
            vector_enemy_dist_8_dim = np.zeros(8)
            vector_friend_num_8_dim = np.zeros(8)
            vector_friend_dist_8_dim = np.zeros(8)
            hp_vec_enemy = np.zeros(4)
        if len(friend_x) > 0 and len(friend_y) > 0:
            hit_points = player_hit_points[selected_y][selected_x]
        else:
            hit_points = 0
        hp = np.zeros(1)
        hp[0] = hit_points
        self_pos = np.zeros(2)
        self_pos[0] = selected_x
        self_pos[1] = selected_y
        minimap = np.hstack((self_pos, hit_points, vector_enemy_num_8_dim, vector_enemy_dist_8_dim,
                             vector_friend_num_8_dim, vector_friend_dist_8_dim, hp_vec_enemy))
        minimap = np.expand_dims(minimap, axis=0)
        return minimap

    def step(self, obs):
        minimap = self.get_hand_crafted_feature(obs)
        # minimap = np.expand_dims(minimap, axis=0)
        screen = np.array(obs.observation['screen'], dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        # TODO: only use available actions
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        feed = {self.minimap: minimap,
                self.screen: screen,
                self.info: info}
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed)

        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel()
        spatial_action = spatial_action.ravel()
        valid_actions = obs.observation['available_actions']
        valid_actions = [ind for ind in valid_actions if ind > 0 or ind >= 331]
        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)
        x = minimap[0][0]
        y = minimap[0][1]
        if DEBUG: print('y, x ', y, x)
        next_y, next_x = merge_target_with_position(target, x, y, self.ssize)
        # target = [int(target // self.ssize), int(target % self.ssize)]
        target = [next_x, next_y]

        if DEBUG:
            print(actions.FUNCTIONS[act_id].name, target)

        # Epsilon greedy exploration
        if self.training and np.random.rand() < self.epsilon[0]:
            act_id = np.random.choice(valid_actions)
        if self.training and np.random.rand() < self.epsilon[1]:
            dy = np.random.randint(-2, 2)
            target[0] = int(max(0, min(self.ssize - 1, target[0] + dy)))
            dx = np.random.randint(-2, 2)
            target[1] = int(max(0, min(self.ssize - 1, target[1] + dx)))

        # Set act_id and act_args
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])  # TODO: Be careful
        return actions.FunctionCall(act_id, act_args)

    def update(self, rbs, disc, lr, cter):
        # Compute R, which is value of the last observation
        obs = rbs[-1][-1]
        if obs.last():
            R = 0
            # R_distance_reward = 0
            R_distance_reward = compute_distance_reward(self.get_enemy_friend_distance(obs))
        else:
            # minimap = np.array(obs.observation['minimap'], dtype=np.float32)
            # minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            minimap = self.get_hand_crafted_feature(obs)
            screen = np.array(obs.observation['screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info: info}
            R = self.sess.run(self.value, feed_dict=feed)[0]
            R_distance_reward = compute_distance_reward(self.get_enemy_friend_distance(obs))

        Tot_reward = R
        # Compute targets and masks
        minimaps = []
        screens = []
        infos = []

        value_target = np.zeros([len(rbs)], dtype=np.float32)
        value_target[-1] = Tot_reward

        valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
        spatial_action_selected = np.zeros([len(rbs), A3CAgent.EIGHT_SPATIAL_ACTION], dtype=np.float32)
        valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
        non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)

        rbs.reverse()
        for i, [obs, action, next_obs] in enumerate(rbs):
            # minimap = np.array(obs.observation['minimap'], dtype=np.float32)
            # minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
            minimap = self.get_hand_crafted_feature(obs)
            player_relative_mat = obs.observation["screen"][_PLAYER_RELATIVE]
            player_relative_mat_next = next_obs.observation["screen"][_PLAYER_RELATIVE]
            # next_minimap = self.get_hand_crafted_feature(next_obs)
            reward_curiosity = compute_curiosity_reward(player_relative_mat, player_relative_mat_next)
            # reward_curiosity = np.linalg.norm(next_minimap - minimap)
            # reward_curiosity = reward_curiosity if reward_curiosity <= 50 else 50
            current_x = minimap[0][0]
            current_y = minimap[0][1]
            screen = np.array(obs.observation['screen'], dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)

            reward = obs.reward
            # tmp_distance_reward = compute_distance_reward(self.get_enemy_friend_distance(obs))
            tmp_tot_reward = reward + reward_curiosity
            if DEBUG and reward_curiosity > 0:
                print('reward_curiosity ', i, ' ', reward_curiosity, ' ', reward)
            act_id = action.function
            act_args = action.arguments

            value_target[i] = tmp_tot_reward * (disc ** i)
            # value_target[i] = tmp_tot_reward + disc * value_target[i-1]

            valid_actions = obs.observation["available_actions"]
            valid_non_spatial_action[i, valid_actions] = 1
            non_spatial_action_selected[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    # act_arg[0] -->y act_arg[1] -->x
                    eight_digit = convert_xy_2_eight_direction(current_x, current_y, act_arg[1], act_arg[0])
                    ind = eight_digit
                    # ind = act_arg[1] * self.ssize + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)

        # Train
        feed = {self.minimap: minimaps,
                self.screen: screens,
                self.info: infos,
                self.value_target: value_target,
                self.valid_spatial_action: valid_spatial_action,
                self.spatial_action_selected: spatial_action_selected,
                self.valid_non_spatial_action: valid_non_spatial_action,
                self.non_spatial_action_selected: non_spatial_action_selected,
                self.learning_rate: lr}
        _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
        self.summary_writer.add_summary(summary, cter)

    def save_model(self, path, count):
        self.saver.save(self.sess, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])
