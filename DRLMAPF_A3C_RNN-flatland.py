#!/usr/bin/env python
# coding: utf-8

# # Pathfinding via Reinforcement and Imitation Multi-Agent Learning (PRIMAL)
# 
# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
# `tensorboard --logdir train_primal`

# In[ ]:


#this should be the thing, right?
from __future__ import division
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import threading
import copy
import time
import scipy.signal as signal
import os
import sys
from GroupLock import GroupLock
from StateMaskingObs import StateMaskingObs
from NewAgentInitObs import StateMaskingObs as TrafficLightObs
from expert2 import Solver, Global_H, my_controller
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.core.grid.grid_utils import distance_on_rail as manhattan_distance

import pickle
from ACNet4 import ACNet
import imageio

from tensorflow.python.client import device_lib
dev_list = device_lib.list_local_devices()
print(dev_list)

# assert len(dev_list) > 1


def Complex_params():
    grid_width = np.random.randint(12, 30)  # min(int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )),

    # int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )))
    grid_height = grid_width  # min(int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1])),

    # nt(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )))
    rnd_start_goal = 8 + np.random.randint(0,
                                           3)  # int(np.random.uniform(num_workers, num_workers+1+episode_difficulty ))
    # int(np.random.uniform( num_workers , min(grid_width,grid_height))),

    rnd_extra = np.random.randint(3, 7)  # int(np.random.uniform(0 , 1+2*episode_difficulty ))
    # int(np.random.uniform( 0 , min(grid_width,grid_height))))
    rnd_min_dist = np.random.randint(int(0.2 * min(grid_height, grid_width)), int(
        0.75 * min(grid_height, grid_width)))  # int(np.random.uniform( episode_difficulty , 4+2*episode_difficulty ))
    rnd_max_dist = rnd_min_dist + np.random.randint(5,
                                                    15)  # int(np.random.uniform(3+episode_difficulty, 6+2*episode_difficulty))
    rnd_seed = int(np.random.rand() * 2 * 200)

    return grid_width, grid_height, rnd_start_goal, rnd_extra, rnd_min_dist, rnd_max_dist, rnd_seed


def Sparse_params():
    tid = np.random.randint(0, 50)
    seed = tid * 19997 + 997
    random.seed(seed)

    nSize = random.randint(0, 5)

    width = 20 + nSize * 5
    height = 20 + nSize * 5
    nr_cities = 2 + nSize // 2 + random.randint(0, 2)
    nr_trains = min(nr_cities * 5, 5 + random.randint(0, 5))  # , 10 + random.randint(0, 10))
    max_rails_between_cities = 2
    max_rails_in_cities = 3 + random.randint(0, nSize)
    malfunction_rate = 30 + random.randint(0, 100)
    malfunction_min_duration = 3 + random.randint(0, 7)
    malfunction_max_duration = 20 + random.randint(0, 80)
    return (
        seed, width, height,
        nr_trains, nr_cities,
        max_rails_between_cities, max_rails_in_cities,
        malfunction_rate, malfunction_min_duration, malfunction_max_duration
    )

def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("\nwrote gif")

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.


def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def good_discount(x, gamma):
    return discount(x, gamma)
#     positive = np.clip(x,0,None)
#     negative = np.clip(x,None,0)
#     return signal.lfilter([1], [1, -gamma], positive[::-1], axis=0)[::-1]+negative


# ## Worker Agent

# In[ ]:


class Worker:
    def __init__(self, gameEnv, metaAgentID, workerID, a_size, groupLock):
        self.workerID     = workerID
        self.env          = gameEnv
        self.metaAgentID  = metaAgentID
        self.name         = "worker_"+str(workerID)
        self.agentID      = workerID % num_workers
        self.groupLock    = groupLock
        self.nextGIF      = episode_count  # For GIFs output
        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC     = ACNet(self.name, a_size, trainer, True, GLOBAL_NET_SCOPE, OBS_SIZE)
        self.pull_global  = update_target_graph(GLOBAL_NET_SCOPE, self.name)
        self.prune_rate   = PRUNE_ACTION
        if self.workerID == 0:
            self.env_renderer = None

        self.path_finder = None

    def synchronize(self):
        # handy thing for keeping track of which to release and acquire
        if not hasattr(self,"lock_bool"):
            self.lock_bool=False
        self.groupLock.release(int(self.lock_bool),self.name)
        self.groupLock.acquire(int(not self.lock_bool),self.name)
        self.lock_bool = not self.lock_bool

    def train(self, rollout, sess, gamma, bootstrap_value, imitation=False):
        global episode_count

        if imitation:
            # we calculate the loss differently for imitation
            # if imitation=True the rollout is assumed to have different dimensions:
            # [o[0],o[1],optimal_actions]
            # rnn_state = self.local_AC.state_init

            rollout_obs = np.array(rollout[0])
            rollout_obs = np.reshape(rollout_obs, (-1, OBS_SIZE))

            rollout_action = np.array([rollout[1]])
            rollout_action = np.reshape(rollout_action, (-1, 1))
            rollout_action = rollout_action.squeeze()
            # rnn_state = self.local_AC.state_init
            feed_dict = {global_step: episode_count,
                         self.local_AC.inputs: rollout_obs,
                         self.local_AC.optimal_actions: rollout_action,
                         # self.local_AC.state_in[0]: rnn_state[0],
                         # self.local_AC.state_in[1]: rnn_state[1]
                         }
            _, i_l, _ = sess.run([self.local_AC.policy, self.local_AC.imitation_loss,
                                 self.local_AC.apply_imitation_grads],
                                 feed_dict=feed_dict)
            return i_l

        rollout      = np.array(rollout)

        observations = rollout[:, 0]
        observations = np.stack(observations)
        observations = np.reshape(observations, (-1, OBS_SIZE))

        actions      = rollout[:, 1]
        rewards      = rollout[:, 2]
        values       = rollout[:, 3]
        valids       = rollout[:, 4]

        train_value  = rollout[:, -1]
        # train_astar  = rollout[:, -1]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = good_discount(advantages, gamma)

        num_samples = min(EPISODE_SAMPLES, len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # rnn_state = self.local_AC.state_init
        feed_dict = {
            global_step: episode_count,
            self.local_AC.target_v: np.stack(discounted_rewards),
            self.local_AC.inputs: np.stack(observations),
            self.local_AC.actions: actions,
            self.local_AC.valid_actions: np.stack(valids),
            self.local_AC.advantages: advantages,
            self.local_AC.train_value: train_value,
            # self.local_AC.state_in[0]: rnn_state[0],
            # self.local_AC.state_in[1]: rnn_state[1]
        }

        v_l, p_l, valid_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                                        self.local_AC.policy_loss,
                                                        self.local_AC.valid_loss,
                                                        self.local_AC.entropy,
                                                        self.local_AC.grad_norms,
                                                        self.local_AC.var_norms,
                                                        self.local_AC.apply_grads], feed_dict=feed_dict)
        return v_l/len(rollout), p_l/len(rollout), valid_l/len(rollout), e_l/len(rollout), g_n, v_n, np.sum(rewards)

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    def resetEnv(self):
        assert self.agentID == 0

        if int(joint_episode_count[self.metaAgentID]) % ENV_CHANGE_FREQUENCY == 0:
            random_number = np.random.rand()
            if random_number <= SPARSE_POSSIBILITY:
                IS_SPARSE[self.metaAgentID] = True
                while True:
                    try:
                        seed, width, height, nr_trains, nr_cities,\
                        max_rails_between_cities, max_rails_in_cities, _, _, _ = Sparse_params()

                        gameEnv = RailEnv(width=width,
                                          height=height,
                                          rail_generator=sparse_rail_generator(
                                                max_num_cities=nr_cities,
                                                max_rails_between_cities=max_rails_between_cities,
                                                max_rails_in_city=max_rails_in_cities,
                                                seed=seed,  # Random seed
                                                grid_mode=False  # Ordered distribution of nodes
                                                ),
                                          schedule_generator=sparse_schedule_generator(),
                                          obs_builder_object=TrafficLightObs(),
                                          number_of_agents=num_agents)
                        gameEnv.global_reward = 20
                        gameEnv.step_penalty = -0.3
                        self.env = gameEnv
                        obs = self.env.reset(True, True)
                        joint_env[self.metaAgentID] = copy.deepcopy(self.env)

                        break
                    except Exception:
                        print('bad init')
                        pass
            else:
                IS_SPARSE[self.metaAgentID] = False
                grid_width , grid_height , rnd_start_goal , rnd_extra , rnd_min_dist , rnd_max_dist , rnd_seed = Complex_params()
                gameEnv = RailEnv(width=grid_width, height=grid_height,
                                      rail_generator=complex_rail_generator(
                                         nr_start_goal=rnd_start_goal,nr_extra=rnd_extra,min_dist=rnd_min_dist,max_dist=rnd_max_dist,seed=rnd_seed) ,
                                      schedule_generator=complex_schedule_generator(),
                                      obs_builder_object=TrafficLightObs(),
                                      number_of_agents=num_agents)
                gameEnv.global_reward = 20
                gameEnv.step_penalty = -0.3
                self.env = gameEnv
                obs = self.env.reset(True, True)
                joint_env[self.metaAgentID] = copy.deepcopy(self.env)
        else:
            obs = self.env.reset(True, True)
        print('init succeed')
        return obs

    def parse_path(self, obss, actions):
        """
        input:
        output: rollout:[[obs, action],[...],...]
        """

        if not obss or not actions:
            return None
        agents_rollout = [
                            [[], []] for i in range(NUM_THREADS)
                         ]

        for agentID in range(NUM_THREADS):
            for step in range(len(obss)):
                if agentID not in actions[step].keys():
                    # End of episode for that agent, go to next agent
                    continue

                if agentID in obss[step].keys():
                    agents_rollout[agentID][0].append(obss[step][agentID])

                    if a_size == 4:
                        agents_rollout[agentID][1].append(actions[step][agentID] - 1)
                    else:
                        agents_rollout[agentID][1].append(actions[step][agentID])

        return agents_rollout

    def StateClassifier(self, agent_pos, agent_dir):
        """
        returns 0 : No decision point
        returns 1 : Stopping point (Decision at next cell)
        returns 2 : At decision point currently (More than 1 available transition)
        returns 3 : MUST STOP point - Agent Ahead
        returns 4 : MUST STOP point + Stopping Point 
        returns None: invalid cell
        """
        avb_moves = self.env.rail.get_transitions(*agent_pos, agent_dir)
        move2grid = np.array([[[0, -1], [-1, 0], [0, +1]], [[-1, 0], [0, +1], [+1, 0]], [[0, +1], [+1, 0], [0, -1]],
                              [[+1, 0], [0, -1], [-1, 0]]])  # Obtained from collidingagent code
        trans2act = np.array([[2, 3, 0, 1], [1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2]])  # Maps transition to an action
        # next_dir_grid = np.array([-1,0,1])  # Maps action to a change in agent direction
        if sum(avb_moves) > 1:  # This is definitely a decision junction since more than 1 move possible
            return 2
        elif sum(avb_moves) == 1:
            avbmove = avb_moves.index(1)  # Get the available transition to next cell
            action = trans2act[agent_dir][avbmove]  # Get the corresponding action for that transition
            if action == 0:
                next_pos = agent_pos + move2grid[(agent_dir + 2) % 4][
                    1]  # This is a dead end, so turn around and move forward
            else:
                next_pos = agent_pos + move2grid[agent_dir][action - 1]
            # next_dir = (agent_dir + (next_dir_grid[action-1]) )%4
            sumnextcell = 0  # How many possible transitions at next cell
            for i in range(0, 4):
                new_avb_moves = self.env.rail.get_transitions(*next_pos, i)
                sumnextcell += sum(new_avb_moves)
                # Also have to check whether the junction is occupied
            Occupied = False
            for k in range(len(self.env.agents)):
                if self.env.agents[k].position is None:
                    if self.env.dones[k]:
                     my_pos = (-3,-3)
                    else:
                        my_pos = (-3,-3)
                else:
                    my_pos = self.env.agents[k].position
                if my_pos[0] == next_pos[0] and my_pos[1] == next_pos[1]:
                    Occupied = True
                    break   
            if (sumnextcell > 2) and Occupied:
                return 4  # The agent is currently at a MUST STOP point
            elif (sumnextcell > 2) and (not Occupied):  # The agent is at a stopping point
                return 1
            elif (sumnextcell <= 2) and Occupied:
                return 3  # The agent is at a MUST STOP point
            else:
                return 0  # The agent is at a no decision point

        else:
            # print("The agent is at an impossible cell")  # This happen when checking stopping agents
            # print("agent_dir:", agent_dir, " agent_pos:", agent_pos)
            return None

    def get_collided_agent(self, action, episode_step):
        # This whole function is in 5-action style, regardless of a_size

        if action == 4:
            # If we are stuck because we stopped too long, it's 100% on us
            return False, [], False

        # (N, E, S, W)
        agent_dir = self.env.agents[self.agentID].direction if self.env.agents[self.agentID].direction \
                            else self.env.agents[self.agentID].initial_direction
        agent_pos = self.env.agents[self.agentID].position if self.env.agents[self.agentID].position \
                            else self.env.agents[self.agentID].initial_position

        #         move2grid = [[-1, 0], [0, +1], [+1, 0], [0, -1]] # N, E, S, W
        move2grid = np.array([[[0, -1], [-1, 0], [0, +1]],
                              [[-1, 0], [0, +1], [+1, 0]],
                              [[0, +1], [+1, 0], [0, -1]],
                              [[+1, 0], [0, -1], [-1, 0]]])  # West:  SWN

        collision_position = np.asarray(agent_pos) + move2grid[agent_dir][action - 1]
        collided_agent = None
        for i in range(len(self.env.agents)):
            if (self.env.agents[i].position == collision_position).all():
                collided_agent = i
                if len(shared_nb_va[self.metaAgentID][i]) < len(shared_nb_va[self.metaAgentID][self.agentID]):
                    # collided_agent was already stuck/done, it's our fault
                    ep_i, steps_cc = episode_step, []
                    while ep_i >= 0:
                        steps_cc.append(ep_i)
                        if shared_nb_va[self.metaAgentID][self.agentID][ep_i] > 2:  # I had a choice then!
                            break
                        ep_i -= 1
                    steps_cc.remove(episode_step)
                    return True, steps_cc, True
                break

        if collided_agent is None:
            #             print('\n({:d}) Hit a ghost (should be rare)!!'.format(int(episode_step)))
            #             print(agent_pos, agent_dir, list(collision_position))
            #             for agent in self.env.agents:
            #                 print(agent.position, agent.direction)
            # I hit a ghost: I hit an agent by moving into its cell while it was moving out. It's clearly my fault
            return True, [], False

        ep_i, steps_cc = episode_step, []
        while ep_i >= 0:
            if shared_nb_va[self.metaAgentID][collided_agent][ep_i] >= 2:  # he had a choice then!
                if episode_step in steps_cc:
                    steps_cc.remove(episode_step)
                if shared_nb_va[self.metaAgentID][self.agentID][ep_i] >= 2:  # I also had a choice at the same time!
                    return True, steps_cc, True
                return False, steps_cc, False  # It's all his fault
            elif shared_nb_va[self.metaAgentID][self.agentID][ep_i] >= 2:  # I had a choice then!
                if shared_nb_va[self.metaAgentID][collided_agent][ep_i] >= 2:  # He also had a choice at the same time!
                    if episode_step in steps_cc:
                        steps_cc.remove(episode_step)
                    return True, steps_cc, True
                if episode_step in steps_cc:
                    steps_cc.remove(episode_step)
                return True, steps_cc, True  # It's all my fault
            steps_cc.append(ep_i)
            ep_i -= 1

        # Commented out to check for bugs, should never happen anyway
        #         return responsible, steps_cc, collisioncourse
        print("\nWeird, exiting get_collided_agent() via default exit")
        return False, [], False

    def _NextValidActions(self):
        """
        returns list of valid actions
        List[0]= LEFT , List[1] = Straight , List[2] = Right , List[3] = Stop

        If at NO decision point, just go forward [0]
        If at stopping point, look 1 timestep into future and conclude : returns [0,3](stop,go) or [3](stop)
        If at junction, get valid directions to go in. No stopping allowed here
        If no available direction at junction : return [3](stop) , This means we're screwed
        """

        currentobs = joint_observations[self.metaAgentID][self.agentID][0:OBS_SIZE-ADDITIONAL_INPUT]
        traffic_signal = joint_observations[self.metaAgentID][self.agentID][OBS_SIZE-ADDITIONAL_INPUT]
        homo_junctions = joint_observations[self.metaAgentID][self.agentID][(OBS_SIZE-3) :OBS_SIZE]
        if traffic_signal == -1:
            validactions = [3]
            return validactions 
        currentobs = np.reshape(currentobs, (3, -1))
        if self.env.agents[self.agentID].position is None:
            if self.env.dones[self.agentID]:
                actual_dir = self.env.agents[self.agentID].old_direction
                actual_pos = self.env.agents[self.agentID].target
            else:
                actual_dir = self.env.agents[self.agentID].initial_direction
                actual_pos = self.env.agents[self.agentID].initial_position
        else:
            actual_dir = self.env.agents[self.agentID].direction
            actual_pos = self.env.agents[self.agentID].position

        state = self.StateClassifier(actual_pos, actual_dir)
        # currentobs = joint_observations[self.metaAgentID][self.agentID]
        if state in [3, 4]:  # Must Stop Point
            validactions = [3]
            return validactions
        elif state == 0:  # Currently at NO decision point
            validactions = [1]
            return validactions
        elif state == 1:  # Currently at stopping point
            SolExist = [currentobs[0][0], currentobs[1][0], currentobs[2][0]]  # Imagine we are at decision junction
            agentsblocking = [currentobs[0][2], currentobs[1][2], currentobs[2][2]]
            agentsblockingjunction = [currentobs[0][3], currentobs[1][3], currentobs[2][3]]
            agentsdiff = [currentobs[0][4], currentobs[1][4], currentobs[2][4]]
            for i in range(0, 3):
                # Check if there is any available non-blocked path which leads to a solution
                if (SolExist[i] == 1) and (agentsblocking[i] == 0):
                    validactions = [1, 3]  # If there is such a path, then going forward allowed
                    return validactions
            if homo_junctions.count(1) >=2 :
                for i in range(0,3) :
                    if (SolExist[i] == 1) and (agentsblockingjunction[i] == 0) and homo_junctions[i]== 1 :
                        validactions = [1, 3]
                        return validactions 
            validactions = [3]  # If there is no such path, only stopping allowed
            return validactions
        else:  # Currently at junction
            SolExist = [currentobs[0][0], currentobs[1][0], currentobs[2][0]]
            agentsblocking = [currentobs[0][2], currentobs[1][2], currentobs[2][2]]
            agentsblockingjunction = [currentobs[0][3], currentobs[1][3], currentobs[2][3]]
            agentsdiff = [currentobs[0][4], currentobs[1][4], currentobs[2][4]]
            # stoppingoccupied = [currentobs[0][ENTRY_PER_COLUMN - 1],
            #                     currentobs[1][ENTRY_PER_COLUMN - 1],
            #                     currentobs[2][ENTRY_PER_COLUMN - 1]]
            validactions = []
            for i in range(0, 3):
                if (SolExist[i] == 1) and (agentsblocking[i] == 0):  # and stoppingoccupied[i]==False:
                    validactions.append(i)
            if validactions:
                return validactions
            else:
                if homo_junctions.count(1) >=2 :
                    for i in range(0,3) :
                        if (SolExist[i] == 1) and (agentsblockingjunction[i] == 0) and homo_junctions[i]== 1  :
                            validactions.append(i) 
                            break 
                    if validactions:    
                        return validactions 
                # print("Oops we screwed up , we should have stopped at the stopping point")
                # validactions = []
                # for j in range(0, 3):
                #    if SolExist[j] == 0 and stoppingoccupied[j] == False:
                #        validactions.append(j)
                # if validactions:
                #    return validactions
                # else:
                return [3]

    def work(self, max_episode_length, gamma, sess, coord, saver):
        global episode_count, swarm_reward, episode_rewards, episode_lengths, \
            episode_mean_values, episode_invalid_ops
#        global joint_success, MAX_DIFFICULTY
        total_steps = 0
        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                sess.run(self.pull_global)
#               sess.run(self.copy_weights)

                episode_buffer, episode_values = [], []
                episode_step_count = 0

                # Initial state from the environment
                if self.agentID == 0:
                    # print('Meta-agent {}: resetting environment...'.format(self.metaAgentID), end='')
                    all_obs = self.resetEnv()
                    # all_obs = self.env.reset(True,True)
                    if len(all_obs[0]) != NUM_THREADS:
                        continue
                    for i in range(num_workers):
                        joint_observations[self.metaAgentID][i] = all_obs[0][i]
                self.synchronize()  # synchronize starting time of the threads
                if self.env is not joint_env[self.metaAgentID]:
                    self.env = joint_env[self.metaAgentID]
                validActions              = self._NextValidActions()
                s                         = joint_observations[self.metaAgentID][self.agentID]
                assert len(s) == OBS_SIZE
                # rnn_state                 = self.local_AC.state_init
                stopped_counter           = 0
                valid_stopped_counter     = 0
                done_tag                  = False 
                

                end_episode               = max_episode_length+1
                successful_ep             = False  # only set to true if everyone got to their goal
                joint_stuck[self.metaAgentID][self.agentID] = False
                joint_done[self.metaAgentID][self.agentID]  = False
                shared_nb_va[self.metaAgentID][self.agentID] = [len(validActions)]
                # number of valid actions at each timestep,
                # used for collision course estimation at the end of an episode

                if self.agentID == 0:
                    global demon_probs
                    demon_probs[self.metaAgentID] = np.random.rand()

                self.synchronize()  # synchronize starting time of the threads

                # reset swarm_reward (for tensorboard)
                swarm_reward[self.metaAgentID] = 0

#                 # Imitation Learning from the expert #
                if episode_count < PRIMING_LENGTH or demon_probs[self.metaAgentID] < DEMONSTRATION_PROB:
                    #  for the first PRIMING_LENGTH episodes, or with a certain probability
                    #  don't train on the episode and instead observe a demonstration from M*
                    if self.workerID == 0 and int(episode_count) % 100 == 0:
                        saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                    global rollouts
                    rollouts[self.metaAgentID] = None
                    if self.agentID == 0:
                        heuristic = Global_H(self.env)
                        self.path_finder = Solver(self.env, heuristic)
                        masked_obs = []
                        all_actions = []
                        masked_actions = []
                        while self.env.dones["__all__"] is not True:

                            joint_actions_single_step = my_controller(self.env, self.path_finder)
                            obs_single_step, _, _, _ = self.env.step(joint_actions_single_step)

                            all_actions.append(joint_actions_single_step)

                            for a_id in range(NUM_THREADS):
                                pos = self.env.agents[a_id].position if self.env.agents[a_id].position \
                                    else self.env.agents[a_id].initial_position
                                direction = self.env.agents[a_id].direction if \
                                    self.env.agents[a_id].direction is not None \
                                    else self.env.agents[a_id].initial_position

                                if a_id not in joint_actions_single_step.keys() or joint_actions_single_step[a_id] == 0:
                                    recursive_counter = -1
                                    while recursive_counter > - len(all_actions) - 1:
                                        # assume that agent cannot 'do nothing' for all steps
                                        if a_id in all_actions[recursive_counter].keys():
                                            if all_actions[recursive_counter][a_id] == 4:  # check last action
                                                joint_actions_single_step[a_id] = 3 if a_size == 4 else 4
                                            else:
                                                joint_actions_single_step[a_id] = 1 if a_size == 4 else 2
                                            break
                                        else:  # recurse for previous action
                                            recursive_counter -= 1

                                if a_id not in joint_actions_single_step.keys():
                                    # if all steps are 'doing nothing', gives a 'going forward'
                                    joint_actions_single_step[a_id] = 1 if a_size == 4 else 2

                                if self.StateClassifier(pos, direction) not in [1, 2]:
                                    obs_single_step.pop(a_id)
                                    joint_actions_single_step.pop(a_id)
                            masked_obs.append(obs_single_step)
                            masked_actions.append(joint_actions_single_step)
                            #   all_obs and all_actions are dicts!

                        rollouts[self.metaAgentID] = self.parse_path(masked_obs, masked_actions)
                        print('env:', self.metaAgentID, 'episode ', episode_count, ' finish IL')
                    self.synchronize()
                    if rollouts[self.metaAgentID] is not None and len(rollouts[self.metaAgentID][0]) > 0:
                        i_l = self.train(rollouts[self.metaAgentID][self.agentID], sess, gamma, None, imitation=True)
                        if self.agentID == 0:
                            episode_count += 1
                            summary = tf.Summary()
                            summary.value.add(tag='Losses/Imitation loss', simple_value=i_l)
                            global_summary.add_summary(summary, int(episode_count))
                            global_summary.flush()

                        continue

                    continue

                saveGIF = False
                if OUTPUT_GIFS and self.workerID == 0 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF = int(episode_count) + 128
                    GIF_episode = int(episode_count)

                    self.env_renderer = RenderTool(self.env)
                    self.env_renderer.render_env(show=ON_SCREEN_RENDERING, frames=False, show_observations=False)
                    episode_frames = [self.env_renderer.get_image()]
                    print('\nGoing for a GIF episode (next one should be around {:d})'.format(self.nextGIF))

                k = 0
                t = 0
                if self.agentID == 0:
                    if not IS_SPARSE[self.metaAgentID]:
                        all_obs, _, _, _ = self.env.step({i: 2 for i in range(NUM_THREADS)})
                        for i in range(num_workers):
                            joint_observations[self.metaAgentID][i] = all_obs[i]
                        s = joint_observations[self.metaAgentID][self.agentID]
                        validActions = self._NextValidActions()
                        shared_nb_va[self.metaAgentID][self.agentID] = [len(validActions)]
                    else:
                        s = joint_observations[self.metaAgentID][self.agentID]
                        validActions = self._NextValidActions()
                        shared_nb_va[self.metaAgentID][self.agentID] = [len(validActions)]
                self.synchronize()


                episode_inv_count, decision_count, decision_go_straight_count, stopping_count, \
                stopping_inv_count, must_stop_inv_count, decision_count_inv_count,initialized,episode_reward = 0, 0, 0, 0, 0, 0, 0, 0,0

                while not self.env.dones["__all__"]:  # Give me something!

                    IS_NON_DECISION = False
                    previous_pos = self.env.agents[self.agentID].position if self.env.agents[self.agentID].position \
                        else self.env.agents[self.agentID].initial_position
                    previous_dir = self.env.agents[self.agentID].direction \
                        if self.env.agents[self.agentID].direction is not None else \
                        self.env.agents[self.agentID].initial_direction

                    if episode_step_count <= end_episode:
                        # state masking
                        initialization = joint_observations[self.metaAgentID][self.agentID][OBS_SIZE-ADDITIONAL_INPUT+1]
                        # print(initialization)
                        state = self.StateClassifier(previous_pos, previous_dir)
                        if initialization == 1:
                            joint_actions[self.metaAgentID][self.agentID] = 0
                            a = 0 
                            # print('initialized' , initialized)
                        elif initialization == 0 and initialized == 0:
                            joint_actions[self.metaAgentID][self.agentID] = 2
                            initialized = 1 
                            a = 2
                        elif state == 0:  # no decision point
                            a = 1
                            initialized += 1
                            joint_actions[self.metaAgentID][self.agentID] = 2  # just go forward
                            IS_NON_DECISION = True
                        elif state in [3,4] :
                            a=3 
                            joint_actions[self.metaAgentID][self.agentID] = 4
                            IS_NON_DECISION = True   
                            initialized +=1
                        else:  # state == 1 or state == 2
                        # Take an action using probabilities from policy network output.
                            if state == 1:
                                stopping_count += 1
                            initialized +=1
                            s_feed = np.reshape(s, (1, OBS_SIZE))
                            a_dist, v = sess.run([self.local_AC.policy,
                                                             self.local_AC.value,
                                                             # self.local_AC.state_out
                                                             ],
                                                 feed_dict={self.local_AC.inputs: s_feed,
                                                            # self.local_AC.state_in[0]: rnn_state[0],
                                                            # self.local_AC.state_in[1]: rnn_state[1]
                                                            })

                            valid_actions = np.zeros(a_size)
                            valid_actions[validActions] = 1
                            valid_dist = np.array([a_dist[0, validActions]])
                            # valid_dist /= np.sum(valid_dist)
                            if np.sum(valid_dist) > 0:
                                valid_dist /= np.sum(valid_dist)
                            else:
                                valid_dist = np.array([1. / len(valid_dist.ravel())
                                                       for _ in range(len(valid_dist.ravel()))])
                                valid_dist = np.reshape(valid_dist, (1, -1))
                            train_value = 1.
                            if TRAINING:
                                if not self.env.dones[self.agentID]:

                                    if not (np.argmax(a_dist.flatten()) in validActions):
                                        prune_possibility = np.random.rand()
                                        if PRUNE_ACTION and prune_possibility < self.prune_rate:
                                            a = validActions[np.random.choice(range(valid_dist.shape[1]))]
                                        else:
                                            a = np.random.choice(range(a_dist.shape[1]), p=a_dist.ravel())

                                        episode_inv_count += int(not (joint_stuck[self.metaAgentID][self.agentID]
                                                                      or joint_done[self.metaAgentID][self.agentID]))
                                        if state == 1:
                                            stopping_inv_count += 1
                                        if state in [3, 4]:
                                            must_stop_inv_count += 1
                                        if state == 2:
                                            decision_count_inv_count += 1
                                        train_value = 0.
                                    else:
                                        #a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                                        a= np.argmax(a_dist.flatten())


                                    if joint_stuck[self.metaAgentID][self.agentID] or joint_done[self.metaAgentID][self.agentID]:
                                        a                  = a_size - 1  # stay there and get punished if stuck...
                                    if state == 1:
                                        if np.random.rand() < np.exp(episode_count*-0.00085):
                                            a = np.random.choice(validActions)

                                    if state == 2:
                                        decision_count += 1
                                        if a == 1:
                                            decision_go_straight_count += 1
                                        if np.random.rand() < np.exp(episode_count*-0.00085):
                                            if 3 not in validActions:
                                                currentobs = joint_observations[self.metaAgentID][self.agentID][0:OBS_SIZE-ADDITIONAL_INPUT]
                                                currentobs = np.reshape(currentobs, (3, -1))
                                                shortest_paths = []
                                                for action in validActions:
                                                    shortest_paths.append(float(currentobs[action][1]))
                                                greedyaction = validActions[np.argmin(shortest_paths)]
                                                if np.random.rand() < 0.75:
                                                    a = greedyaction
                                                else :
                                                    a = np.random.choice(validActions)
                                                train_value        = 0.

                                            #if self.agentID ==0:
                                             #   print("Random Action taken , Valid Actions =",validActions , 'Action=', a+1)

                                else:
                                    a = 3  # just stop bc we already achieve the goal,
                                    # note that a is of 4-action style, so a=3
                            else:
                                if GREEDY:
                                    a     = np.argmax(a_dist.flatten())
                                else:
                                    a     = np.random.choice(range(a_dist.shape[1]), p=a_dist.ravel())
#                                if a not in validActions or not GREEDY:
#                                    a     = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]

                            # Choose action
                            if a_size == 4:
                                joint_actions[self.metaAgentID][self.agentID] = a + 1
                            else:
                                joint_actions[self.metaAgentID][self.agentID] = a

                    self.synchronize()  # synchronize threads

                    # Take single joint step and share new info

                    if self.agentID == 0:
                        all_obs, all_rewards, all_done, _ = self.env.step(joint_actions[self.metaAgentID])

                        for i in range(num_workers):
                            joint_done[self.metaAgentID][i] = all_done[i]

                        # If all agents are done (on goal) or stuck (collisions), stop episode early
                        if any(joint_stuck[self.metaAgentID]):
                            # done_mask = False  # fake true to finish episode early
                            # base_reward = - self.env.global_reward  # and give massive penalty
                            # todo: we ignore global punishment for stuck env
                            base_reward = 0

                        else:
                            base_reward = 0
                            if all_done['__all__']:
                                successful_ep = True

                        for i in range(num_workers):
                            joint_observations[self.metaAgentID][i] = all_obs[i]
                            joint_rewards[self.metaAgentID][i]      = base_reward * int(not all_done[i]) + all_rewards[i]
                            # if someone stucks, all agents are penalized by base_reward

                    self.synchronize()  # synchronize threads
                    
                    if sum(joint_stuck[self.metaAgentID]) >= STOP_PARAMETER :
                              done_tag = True 
                    self.synchronize()
                    if joint_done[self.metaAgentID][self.agentID] and k ==0:
                        # print('At Target', self.agentID)
                        k += 1
                    if episode_step_count <= end_episode:
                        # Get common observation for all agents after all individual actions have been performed
                        s1               = joint_observations[self.metaAgentID][self.agentID]
                        new_pos          = self.env.agents[self.agentID].position if self.env.agents[self.agentID].position \
                            else self.env.agents[self.agentID].initial_position
                        new_dir          = self.env.agents[self.agentID].direction if self.env.agents[self.agentID].direction is not None else \
                        self.env.agents[self.agentID].initial_direction
                        r                = joint_rewards[self.metaAgentID][self.agentID]

                        if new_pos == previous_pos and new_dir == previous_dir and \
                                not joint_stuck[self.metaAgentID][self.agentID]\
                                and not joint_done[self.metaAgentID][self.agentID] and initialized >1:
                            if a == a_size - 1:
                                stopped_counter += 1
                                if len(validActions) ==1:
                                    valid_stopped_counter += 1
                            if a != a_size - 1 or stopped_counter >= FUSE_LAYERS :
                                joint_stuck[self.metaAgentID][self.agentID] = True
                                if joint_stuck[self.metaAgentID][self.agentID] and t == 0:
                                    # print('Stuck', self.agentID)
                                    t += 1
                                responsible, steps_cc, _ = self.get_collided_agent(
                                                joint_actions[self.metaAgentID][self.agentID], episode_step_count)
                                if (stopped_counter >= FUSE_LAYERS and valid_stopped_counter< FUSE_LAYERS-1 and responsible==False ) :
                                    responsible = True               
                                if responsible:
                                    r = r - self.env.global_reward//2 \
                                        + (max_episode_length - episode_step_count) * self.env.step_penalty
                                else:
                                    if max(joint_observations[self.metaAgentID][self.agentID][0],
                                                joint_observations[self.metaAgentID][self.agentID][0+ENTRY_PER_COLUMN],
                                           joint_observations[self.metaAgentID][self.agentID][0 + ENTRY_PER_COLUMN * 2]) == 0:
                                        # no optimal path to the goal
                                        r = r + (max_episode_length - episode_step_count) * self.env.step_penalty
                                    else:
                                        # possible to reach goal
                                        r = r +(1)*(max_episode_length - episode_step_count) * self.env.step_penalty
                                # for ep_i in steps_cc:
                                #     episode_buffer[ep_i][5] = 1
                                 #- max(joint_observations[self.metaAgentID][self.agentID][1],
                                 #               joint_observations[self.metaAgentID][self.agentID][1+9],
                                 #               joint_observations[self.metaAgentID][self.agentID][1 + 9 * 2]))
                        else:
                            stopped_counter = 0
                            valid_stopped_counter = 0 

                    if saveGIF and self.workerID == 0:
                        self.env_renderer.render_env(show=False, frames=False, show_observations=False)
                        episode_frames.append(self.env_renderer.get_image())
                    episode_step_count += 1

                    if episode_step_count <= end_episode:
                        if joint_stuck[self.metaAgentID][self.agentID] or joint_done[self.metaAgentID][self.agentID]:
                            end_episode = min(end_episode, episode_step_count)
                            if joint_done[self.metaAgentID][self.agentID] and not self.env.dones["__all__"]:
                                r = r + self.env.global_reward
                        if episode_step_count <= end_episode and (not IS_NON_DECISION or episode_step_count == end_episode) and initialized>1:
                            episode_buffer.append([s, a, r, v[0, 0], valid_actions, train_value])
                            episode_values.append(v[0, 0])

                        if episode_step_count != end_episode:
                            validActions              = self._NextValidActions()
                            # only keep track of number of valid actions until we get stuck/done
                            shared_nb_va[self.metaAgentID][self.agentID].append(len(validActions))

                        s = s1
                        assert len(s) == OBS_SIZE
                    total_steps += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.

                    if TRAINING and ((int(episode_step_count) % EXPERIENCE_BUFFER_SIZE == 0 and end_episode== max_episode_length+1) or self.env.dones["__all__"]
                                     or episode_step_count == end_episode or done_tag==True ) and len(episode_buffer) > 0:
                        # todo: nuke assertion
                        # assert(len(episode_buffer) == end_episode or end_episode == max_episode_length+1)
                        #  Since we don't know what the true final return is, we "bootstrap"
                        #  from our current value estimation.

                        if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                            training_buffer = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                        else:
                            training_buffer = episode_buffer[:]

                        if self.env.dones["__all__"]:
                            s1Value = 0
                        else:
                            s_feed = np.array(s)
                            s_feed = np.reshape(s_feed, (1, OBS_SIZE))
                            s1Value = sess.run(self.local_AC.value,
                                               feed_dict={self.local_AC.inputs: s_feed,
                                                          # self.local_AC.state_in[0]: rnn_state[0],
                                                          # self.local_AC.state_in[1]: rnn_state[1]
                                                          })[0, 0]

                        v_l, p_l, valid_l, e_l, g_n, v_n, episode_reward = self.train(
                                                            training_buffer, sess, gamma, s1Value, imitation=False)

                    self.synchronize()  # synchronize threads

                    if episode_step_count >= max_episode_length or \
                            (not self.env.dones["__all__"] and
                             sum(joint_stuck[self.metaAgentID])+sum(joint_done[self.metaAgentID]) == NUM_THREADS)\
                            or done_tag is True:
                        if self.agentID == 0:
                            print('env:', self.metaAgentID, self.env.height, self.env.width, ' episode:', episode_count,
                                'done:',
                                  sum(joint_done[self.metaAgentID]), ' stuck:',
                                  sum(joint_stuck[self.metaAgentID]), '/', NUM_THREADS,
                                  'at ', episode_step_count, 'th step')
                        break

                    if self.env.dones["__all__"]:
                        if self.agentID == 0:
                            joint_all_success_count[self.metaAgentID] += 1
                            print('env:', self.metaAgentID, 'episode ', episode_count, ' finish at ', episode_step_count, 'th step')
                        break

#                # Curriculum update
#                if self.agentID == 0:
#                    global_mutex.acquire()
#                    if successful_ep and episode_difficulty == MAX_DIFFICULTY:  # we successfully finished the latest episode
#                        joint_success += 1
#                        if joint_success >= SUCCESS_NEEDED:
#                            joint_success = 0
#                            MAX_DIFFICULTY += 1
#                            print("\n\n\t\tIncreasing Difficult Level to: {:d}\n".format(int(MAX_DIFFICULTY)))
#                    else:
#                        joint_success = 0
#                    global_mutex.release()
                if self.env.dones[self.agentID]:
                    joint_success_count[self.metaAgentID] += 1
                actual_episode_lengths[self.metaAgentID].append(episode_step_count)
                episode_lengths[self.metaAgentID].append(episode_step_count if self.env.dones["__all__"] else max_episode_length)
                episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))

                if PRUNE_ACTION:
                    effective_count = decision_count + stopping_count
                else:
                    effective_count = episode_step_count
                if effective_count != 0:
                    episode_invalid_ops[self.metaAgentID].append(float(effective_count - episode_inv_count) / effective_count)
                    episode_invalid_ops_on_decision[self.metaAgentID].append(float(effective_count
                                                                                   - decision_count_inv_count)/effective_count)
                    episode_invalid_ops_on_stopping[self.metaAgentID].append(float(effective_count
                                                                                   - stopping_inv_count) / effective_count)
                    episode_invalid_ops_on_muststop[self.metaAgentID].append(float(effective_count
                                                                                   - must_stop_inv_count) / effective_count)
                else:
                    episode_invalid_ops_on_decision[self.metaAgentID].append(1)
                    episode_invalid_ops_on_stopping[self.metaAgentID].append(1)
                    episode_invalid_ops_on_muststop[self.metaAgentID].append(1)
                if decision_count != 0:
                    episode_steer_rate[self.metaAgentID].append(float(decision_count
                                                                      - decision_go_straight_count) / decision_count)
                else:
                    episode_steer_rate[self.metaAgentID].append(0)
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if int(episode_count) % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print('                                                                                   ', end='\r')
                    print('({}) Episode terminated ({},{})'.format(int(episode_count), self.agentID, episode_reward), end='\r')

                swarm_reward[self.metaAgentID] += episode_reward

                self.synchronize()  # synchronize threads

                episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])

                if not TRAINING:
                    mutex.acquire()
                    if self.agentID == 0:
                        _, _, dones, _ = self.env.step({i: 0 for i in range(num_workers)})
                        agents_on_goal[episode_count] = sum([done for [train, done] in dones.items() ]) - dones['__all__']
                        episode_count += 1
                        joint_episode_count[self.metaAgentID] += 1
#                       print('({}) Thread {}: {} steps, {:.2f} reward ({} invalids).'.format(int(episode_count), self.workerID, episode_step_count, episode_reward, episode_inv_count))
                        print([np.nanmean( (agents_on_goal == num_workers) * (agents_on_goal/np.maximum(1,agents_on_goal)) ), np.sqrt(np.nanvar( (agents_on_goal == num_workers) * (agents_on_goal/np.maximum(1,agents_on_goal)) )), np.nanmean(agents_on_goal / num_workers), np.sqrt(np.nanvar(agents_on_goal / num_workers))])
                    GIF_episode = int(episode_count)
                    mutex.release()
                elif self.agentID == 0:
                    episode_count += 1
                    joint_episode_count[self.metaAgentID] += 1

                    if int(joint_episode_count[self.metaAgentID]) % SUMMARY_WINDOW == 0:

                        successful_rate = joint_success_count[self.metaAgentID] / (SUMMARY_WINDOW * NUM_THREADS)
                        all_done_rate = joint_all_success_count[self.metaAgentID] / SUMMARY_WINDOW
                        joint_success_count[self.metaAgentID] = 0
                        joint_all_success_count[self.metaAgentID] = 0
                        summary = tf.Summary()
                        summary.value.add(tag='Losses/Individual Successful Rate', simple_value=successful_rate)
                        summary.value.add(tag='Losses/All-Done Rate', simple_value=all_done_rate)
                        global_summary.add_summary(summary, int(episode_count))
                        global_summary.flush()
                    if int(episode_count % SUMMARY_WINDOW) == 0:
                        if int(episode_count) % 100 == 0:
                            print('\nSaving Model for episode ', episode_count)
                            saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                            print('\nSaved Model')
                        SL = SUMMARY_WINDOW * num_workers
                        mean_reward = np.nanmean(episode_rewards[self.metaAgentID][-SL:])
                        mean_length = np.nanmean(episode_lengths[self.metaAgentID][-SL:])
                        mean_actual_length = np.nanmean(actual_episode_lengths[self.metaAgentID][-SL:])
                        # mean_value = np.nanmean(episode_mean_values[self.metaAgentID][-SL:])
                        mean_invalid = np.nanmean(episode_invalid_ops[self.metaAgentID][-SL:])
                        mean_invalid_on_desicion = np.nanmean(episode_invalid_ops_on_decision[self.metaAgentID][-SL:])
                        mean_invalid_on_stopping = np.nanmean(episode_invalid_ops_on_stopping[self.metaAgentID][-SL:])
                        mean_invalid_on_muststop = np.nanmean(episode_invalid_ops_on_muststop[self.metaAgentID][-SL:])
                        mean_steer_rate = np.nanmean(episode_steer_rate[self.metaAgentID][-SL:])

                        # current_learning_rate = sess.run(lr, feed_dict={global_step: episode_count})

                        summary = tf.Summary()
#                         summary.value.add(tag='Perf/Learning Rate',simple_value=current_learning_rate)
                        summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                        # summary.value.add(tag='Perf/Difficulty Level', simple_value=MAX_DIFFICULTY)
                        summary.value.add(tag='Perf/Length', simple_value=mean_length)
                        summary.value.add(tag='Perf/Actual Length', simple_value=mean_actual_length)
                        summary.value.add(tag='Perf/Valid Rate', simple_value=mean_invalid)

                        summary.value.add(tag='Perf/Action Pruning Rate', simple_value=self.prune_rate)
                        if mean_invalid > .9:
                            self.prune_rate -= 0.1

                        summary.value.add(tag='Perf/Valid Rate On Decision', simple_value=mean_invalid_on_desicion)
                        summary.value.add(tag='Perf/Valid Rate On Stopping Point', simple_value=mean_invalid_on_stopping)
                        if not PRUNE_ACTION:
                            summary.value.add(tag='Perf/Valid Rate On Must Stop Point', simple_value=mean_invalid_on_muststop)
                        summary.value.add(tag='Perf/Steering Rate On Decision', simple_value=mean_steer_rate)
                        try:
                            summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                        except Exception:
                            summary.value.add(tag='Losses/Value Loss', simple_value=0)
                        summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                        summary.value.add(tag='Losses/Valid Loss', simple_value=valid_l)
                        summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                        summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                        global_summary.add_summary(summary, int(episode_count))

                        global_summary.flush()

                        if printQ:
                            print('({}) Tensorboard updated ({})'.format(int(episode_count), self.workerID), end='\r')

                if saveGIF:
                    # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                    time_per_step = 0.1
                    images = np.array(episode_frames)
                    if TRAINING:
                        make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}{:s}.gif'.format(gifs_path,GIF_episode,episode_step_count,swarm_reward[self.metaAgentID],("_success" if successful_ep else "")))
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)
                if SAVE_EPISODE_BUFFER:
                    with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
                        pickle.dump(episode_buffer, file)


# ## ----------------------------parameters----------------------------------------------

# Learning parameters
max_episode_length      = 384
episode_count           = 0

EPISODE_START           = episode_count
gamma                   = .95  # discount rate for advantage estimation and reward discounting
# moved network parameters to ACNet2.py
EXPERIENCE_BUFFER_SIZE  = 64
# GRID_SIZE               = 11 # the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE        = (15, 51)  # the total size of the environment (length of one side)
a_size                  = 4  # removed DO_NOTHING. Otherwise should equal 5
SUMMARY_WINDOW          = 10
NUM_META_AGENTS         = 3
NUM_THREADS             = 8  # int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
NUM_BUFFERS             = 1  # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)
EPISODE_SAMPLES         = EXPERIENCE_BUFFER_SIZE # 64
ENV_CHANGE_FREQUENCY    = 10
SPARSE_POSSIBILITY      = 1
STOP_PARAMETER          = 4

FUSE_LAYERS             = 80
LR_Q                    = 2.e-5  # 8.e-5 / NUM_THREADS # default: 1e-5
ADAPT_LR                = False
ADAPT_COEFF             = 5.e-6  # the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
PRUNE_ACTION            = 1
ON_SCREEN_RENDERING     = False
load_model              = False 
RESET_TRAINER           = False
version                 = 'dynamic_prune'
model_path              = './model_' + version
gifs_path               = './gifs_' + version
train_path              = 'train_' + version
GLOBAL_NET_SCOPE        = 'global'

# Curriculum
#MAX_DIFFICULTY          = 5
#SUCCESS_NEEDED          = 8
#global_mutex            = threading.Lock()
#joint_success           = 0

# Imitation options
PRIMING_LENGTH          = 0  # number of episodes at the beginning to train only on demonstrations
DEMONSTRATION_PROB      = 0

# Simulation options
FULL_HELP               = False
OUTPUT_GIFS             = True
SAVE_EPISODE_BUFFER     = False

# Testing
TRAINING                = True
GREEDY                  = False
NUM_EXPS                = 100
MODEL_NUMBER            = 40000

# Shared variables for actions / observations / done
joint_episode_count     = [0 for _ in range(NUM_META_AGENTS)]
joint_success_count     = [0 for _ in range(NUM_META_AGENTS)]
joint_all_success_count = [0 for _ in range(NUM_META_AGENTS)]
joint_end_episode       = [[max_episode_length + FUSE_LAYERS for _ in range(NUM_THREADS)] for _ in range(NUM_META_AGENTS)]
joint_actions           = [{} for _ in range(NUM_META_AGENTS)]
# joint actions consider 5-action !!!
joint_env               = [None for _ in range(NUM_META_AGENTS)]
joint_observations      = [[[] for _ in range(NUM_THREADS)] for _ in range(NUM_META_AGENTS)]
joint_rewards           = [[0 for _ in range(NUM_THREADS)] for _ in range(NUM_META_AGENTS)]
joint_done              = [[False for _ in range(NUM_THREADS)] for _ in range(NUM_META_AGENTS)]
joint_stuck             = [[False for _ in range(NUM_THREADS)] for _ in range(NUM_META_AGENTS)]
shared_nb_va            = [[[] for _ in range(NUM_THREADS)] for _ in range(NUM_META_AGENTS)]

# Shared arrays for tensorboard
episode_rewards         = [[] for _ in range(NUM_META_AGENTS)]
episode_lengths         = [[] for _ in range(NUM_META_AGENTS)]
actual_episode_lengths  = [[] for _ in range(NUM_META_AGENTS)]
episode_mean_values     = [[] for _ in range(NUM_META_AGENTS)]
episode_invalid_ops     = [[] for _ in range(NUM_META_AGENTS)]
# episode_stopping_ratios = [[] for _ in range(NUM_META_AGENTS)]
rollouts                = [None for _ in range(NUM_META_AGENTS)]
demon_probs             = [np.random.rand() for _ in range(NUM_META_AGENTS)]
printQ                  = False  # (for headless)
swarm_reward            = [0]*NUM_META_AGENTS
episode_invalid_ops_on_decision = [[] for _ in range(NUM_META_AGENTS)]
episode_invalid_ops_on_stopping = [[] for _ in range(NUM_META_AGENTS)]
episode_invalid_ops_on_muststop = [[] for _ in range(NUM_META_AGENTS)]
episode_steer_rate = [[] for _ in range(NUM_META_AGENTS)]
IS_SPARSE               = [False for _ in range(NUM_META_AGENTS)]

OBS_SIZE                = TrafficLightObs.OBS_SIZE
ENTRY_PER_COLUMN        = TrafficLightObs.ENTRYS_PER_COLUMN
TRAFFIC_LIGHT_SIZE      = TrafficLightObs.TRAFFIC_LIGHT_SIZE
HOMO_SIZE               = TrafficLightObs.HOMO_SIZE
ADDITIONAL_INPUT        = TRAFFIC_LIGHT_SIZE + HOMO_SIZE

# ------------------------------------------training------------------------------
print('OBS_SIZE=', OBS_SIZE)
tf.reset_default_graph()
print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
if not TRAINING:
    agents_on_goal = np.array([np.nan for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
#    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
#        os.makedirs('gifs3D')

#  Create a directory to save episode playback gifs to
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

with tf.device("/gpu:0"):
    master_network = ACNet(GLOBAL_NET_SCOPE, a_size, None, False, GLOBAL_NET_SCOPE, OBS_SIZE)  # Generate global network

    global_step = tf.placeholder(tf.float32)
    if ADAPT_LR:
        #  computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        #  we need the +1 so that lr at step 0 is defined
        lr = tf.divide(tf.constant(LR_Q), tf.sqrt(tf.add(1., tf.multiply(tf.constant(ADAPT_COEFF), global_step))))
    else:
        lr = tf.constant(LR_Q)
    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

    if TRAINING:
        num_workers = NUM_THREADS  # Set workers # = # of available CPU threads
    else:
        num_workers = NUM_THREADS
        #if OUTPUT_GIFS:
        #    NUM_META_AGENTS = 1

    gameEnvs            = [None for _ in range(NUM_META_AGENTS)]
    workers, groupLocks = [], []
    n = 0  # counter of total number of agents (for naming)
    for ma in range(NUM_META_AGENTS):
        num_agents = NUM_THREADS

        grid_width, grid_height, rnd_start_goal, rnd_extra, rnd_min_dist, rnd_max_dist, rnd_seed = Complex_params()
        gameEnv = RailEnv(width=grid_width, height=grid_height,
                          rail_generator=complex_rail_generator(
                              nr_start_goal=rnd_start_goal,
                              nr_extra=rnd_extra,
                              min_dist=rnd_min_dist,
                              max_dist=rnd_max_dist,
                              seed=rnd_seed)
                          ,
                          schedule_generator=complex_schedule_generator(),
                          obs_builder_object=TrafficLightObs(),
                          number_of_agents=num_agents)
        gameEnv.global_reward = 20
        gameEnv.step_penalty = -0.3
        gameEnv.stop_penalty = 0

        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = ["worker_"+str(i) for i in range(n, n+num_workers)]
        groupLock = GroupLock([workerNames, workerNames])
        groupLocks.append(groupLock)

        # Create worker classes
        workersTmp = []
        for _ in range(num_workers):
            workersTmp.append(Worker(gameEnv, ma, n, a_size, groupLock))
            n += 1
        workers.append(workersTmp)

    global_summary = tf.summary.FileWriter(train_path)
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model...')
            if not TRAINING and MODEL_NUMBER is not None:
                with open(model_path+'/checkpoint', 'w') as file:
                    file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                    file.close()
                    print('Loading model-{}'.format(MODEL_NUMBER))
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            if TRAINING:
                p = ckpt.model_checkpoint_path
                p = p[p.find('-')+1:]
                p = p[:p.find('.')]
                episode_count = int(p)
                print("episode_count set to ", episode_count)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []

        # first meta-agent is dealt with differently so that agentID=workerID=0 can be the main thred (for GIF purposes)
        ma = 0
        groupLocks[ma].acquire(0,workers[ma][0].name)
        for worker in workers[ma][1:]:
            groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
            worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
            print("Starting worker " + str(worker.workerID))
            t = threading.Thread(target=(worker_work))
            t.start()
            worker_threads.append(t)
        for ma in range(1,NUM_META_AGENTS):
            for worker in workers[ma]:
                groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
                worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
                print("Starting worker " + str(worker.workerID))
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
        print("Starting worker " + str(workers[0][0].workerID))
        workers[0][0].work(max_episode_length, gamma, sess, coord, saver)
        coord.join(worker_threads)

if not TRAINING:
    print([np.mean(agents_on_goal), np.sqrt(np.var(agents_on_goal)), np.mean(np.asarray(agents_on_goal < max_episode_length, dtype=float))])

