from __future__ import division
import tensorflow as tf
from ACNET4_test import ACNet
import numpy as np
import json
import os
import time
import pickle
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import threading
from datetime import datetime 
import copy
import scipy.signal as signal
import sys
from NewAgentInitObs import StateMaskingObs as TrafficLightObs
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.core.grid.grid_utils import distance_on_rail as manhattan_distance
import imageio
import random
environment_path = "saved_environments"

class FLATLAND(object):
    '''
    This class provides functionality for running multiple instances of the
    trained network in a single environment
    '''

    def __init__(self, model_path, obs_size, TEST_FLATLAND_ENVIRONMENTS,saveGIF,gifs_path):
        self.obs_size = obs_size
        self.ADDITIONAL_INPUT = 6 
        self.TEST_FLATLAND_ENVIRONMENTS = TEST_FLATLAND_ENVIRONMENTS
        self.PRUNE_ACTIONS = True  
        self.saveGIF = saveGIF
        self.SAVEGIFFREQUENCY = 5 
        self.SKIPLARGE = True 
        self.gifs_path = gifs_path 
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.network = ACNet("global", 4, None, False, "global", obs_size)
        self.episode_count =0 
        # load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt.model_checkpoint_path)

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
        trans2act = np.array([[2, 3, 0, 1], [1, 2, 3, 0], [0, 1, 2, 3], [
                             3, 0, 1, 2]])  # Maps transition to an action
        # next_dir_grid = np.array([-1,0,1])  # Maps action to a change in agent direction
        if sum(avb_moves) > 1:  # This is definitely a decision junction since more than 1 move possible
            return 2
        elif sum(avb_moves) == 1:
            # Get the available transition to next cell
            avbmove = avb_moves.index(1)
            # Get the corresponding action for that transition
            action = trans2act[agent_dir][avbmove]
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
                        my_pos = (-3, -3)
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

    def _NextValidActions(self,obs,agentID):
        """
        returns list of valid actions
        List[0]= LEFT , List[1] = Straight , List[2] = Right , List[3] = Stop

        If at NO decision point, just go forward [0]
        If at stopping point, look 1 timestep into future and conclude : returns [0,3](stop,go) or [3](stop)
        If at junction, get valid directions to go in. No stopping allowed here
        If no available direction at junction : return [3](stop) , This means we're screwed
        """

        currentobs = obs[0:self.obs_size-self.ADDITIONAL_INPUT]
        traffic_signal = obs[self.obs_size-self.ADDITIONAL_INPUT]
        homo_junctions = obs[(self.obs_size-3) :self.obs_size]
        if traffic_signal == -1:
            validactions = [3]
            return validactions 
        currentobs = np.reshape(currentobs, (3, -1))
        if self.env.agents[agentID].position is None:
            if self.env.dones[agentID]:
                actual_dir = self.env.agents[agentID].old_direction
                actual_pos = self.env.agents[agentID].target
            else:
                actual_dir = self.env.agents[agentID].initial_direction
                actual_pos = self.env.agents[agentID].initial_position
        else:
            actual_dir = self.env.agents[agentID].direction
            actual_pos = self.env.agents[agentID].position

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
                return [3]

    def getparams(self, size):
        tid = np.random.randint(0, 50)
        seed = tid * 19997 + 997
        random.seed(seed)
        nSize = int((size-20)/5)
        nr_cities = 2 + nSize // 2 + random.randint(0, 2)
        # , 10 + random.randint(0, 10))
        nr_trains = min(nr_cities * 5, 5 + random.randint(0, 5))
        max_rails_between_cities = 2
        max_rails_in_cities = 3 + random.randint(0, nSize)
        malfunction_rate = 30 + random.randint(0, 100)
        malfunction_min_duration = 3 + random.randint(0, 7)
        malfunction_max_duration = 20 + random.randint(0, 80)
        return (
            seed, nr_trains, nr_cities,
            max_rails_between_cities, max_rails_in_cities,
            malfunction_rate, malfunction_min_duration, malfunction_max_duration
        )

    def make_gif(self,images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
        imageio.mimwrite(fname,images,subrectangles=True)
        print("\nwrote gif")

    def set_env(self, num_agents, id, width, height, max_cities=None, max_rails=None):
        if not TEST_FLATLAND_ENVIRONMENTS:
            if id % 10 == 0:
                while True:
                    try:
                        seed, nr_trains, nr_cities,\
                            max_rails_between_city, max_rails_in_cities, _, _, _ = self.getparams(
                                width)
                        print('size:', width)
                        print('cities:', nr_cities)
                        print('agents', num_agents)
                        print('seed', seed)
                        gameEnv = RailEnv(width=width, height=width, rail_generator=sparse_rail_generator(
                            max_num_cities=nr_cities, max_rails_between_cities=max_rails_between_city,
                            max_rails_in_city=max_rails_in_cities, seed=seed, grid_mode=False),
                            schedule_generator=sparse_schedule_generator(), obs_builder_object=TrafficLightObs(),
                            number_of_agents=num_agents)
                        gameEnv.global_reward = 20
                        gameEnv.step_penalty = -0.3
                        self.env = gameEnv
                        obs = self.env.reset(True, True)
                        break
                    except Exception:
                        pass
            else:
                obs = self.env.reset(True, True)
                print('Reset Successfully')
                self.episode_count +=1  

            if self.saveGIF and self.episode_count % self.SAVEGIFFREQUENCY ==0 :
                self.env_renderer = RenderTool(self.env)
                self.env_renderer.render_env(show=False, frames=False, show_observations=False)
                self.episode_frames = [self.env_renderer.get_image()]         
            return obs
        else:
            if id == 0:
                while True:
                    try:
                        tid = np.random.randint(0, 50)
                        seed = tid * 19997 + 997
                        gameEnv = RailEnv(width=width, height=height, rail_generator=sparse_rail_generator(
                            max_num_cities=max_cities, max_rails_between_cities=2,
                            max_rails_in_city=max_rails, seed=seed, grid_mode=False),
                            schedule_generator=sparse_schedule_generator(), obs_builder_object=TrafficLightObs(),
                            number_of_agents=num_agents)
                        gameEnv.global_reward = 20
                        gameEnv.step_penalty = -0.3
                        self.env = gameEnv
                        obs = self.env.reset(True, True)
                        break
                    except Exception:
                        pass
            else:
                obs = self.env.reset(True, True)
                print('Reset Successfully')
                self.episode_count +=1
                 
            if self.saveGIF and self.episode_count % self.SAVEGIFFREQUENCY ==0 :
                self.env_renderer = RenderTool(self.env)
                self.env_renderer.render_env(show=False, frames=False, show_observations=False)
                self.episode_frames = [self.env_renderer.get_image()]         
            return obs

    def check_action(self, obs, id, done_tag):
        if done_tag == 1:
            return 0

        initialization = obs[self.obs_size - 5]
        previous_pos = self.env.agents[id].position if self.env.agents[id].position \
            else self.env.agents[id].initial_position
        previous_dir = self.env.agents[id].direction if self.env.agents[id].direction is not None else \
            self.env.agents[id].initial_direction
        state = self.StateClassifier(previous_pos, previous_dir)

        if initialization == 1:
            return 0
        elif initialization == 0 and self.initialized[id] == 0:
            self.initialized[id] = 1
            return 2
        elif state == 0:  # no decision point
            return 2
        elif state in [3, 4]:
            return 4
        else:
            return -1

    def step_all_parallel(self, step, all_obs):
        '''advances the state of the environment by a single step across all agents'''
        joint_actions = {}
        if step == 0:
            for agent in range(0, len(self.env.agents)):
                o = all_obs[0][agent]
                s_feed = np.reshape(o, (1, self.obs_size))
                a = self.check_action(o, agent, self.agent_done[agent])
                if a == -1:
                    a_dist = self.sess.run([self.network.policy], feed_dict={
                                           self.network.inputs: s_feed})
                    a = np.random.choice(
                        range(a_dist.shape[1]), p=a_dist.ravel()) + 1
                joint_actions[agent] = a
        else:
             if len(self.env.agents) < 81 or ((len(self.env.agents)==100) and ((self.env.height+self.env.width)==200)) or self.SKIP_LARGE == False:
                observations =[] 
                for i in range(0,len(self.env.agents)) :
                    observations.append(all_obs[i]) 
                s_feed = np.reshape(observations, (len(self.env.agents), self.obs_size))
                action_set = self.sess.run([self.network.policy], feed_dict={
                                            self.network.inputs: s_feed})
                for agent in range(0, len(self.env.agents)):
                    o = all_obs[agent]
                    #s_feed = np.reshape(o, (1, self.obs_size))
                    a = self.check_action(o, agent, self.agent_done[agent])
                    if a == -1:
                    #   a_dist = self.sess.run([self.network.policy], feed_dict={
                                            #   self.network.inputs: s_feed})
                        a_dist = action_set[0][agent]                        
                        a_dist = np.array(a_dist)
                        #a_dist = a_dist[0]
                        if self.PRUNE_ACTIONS : 
                            validactions = self._NextValidActions(o,agent)
                            if not (np.argmax(a_dist.flatten()) in validactions):
                                a = np.random.choice(validactions) + 1 
                            else :
                                a = np.argmax(a_dist.flatten()) + 1     
                        else :
                            a = np.argmax(a_dist.flatten()) + 1  # a = np.random.choice(range(a_dist.shape[1]), p=a_dist.ravel()) + 1
                    joint_actions[agent] = a
        starttime = time.time()           
        all_obs, _, all_done, _ = self.env.step(joint_actions)
        self.timeobs += round((time.time()-starttime), 2)
        return all_obs, all_done

    def find_path(self, all_obs, max_step=384):
        '''run a full environment to completion, or until max_step steps'''
        solution = []
        step = 0
        self.initialized = [0 for i in range(len(self.env.agents))]
        self.agent_done = [0 for i in range(len(self.env.agents))]
        self.timeobs =0 
        while(not self.env.dones["__all__"] and step < max_step):
            timestep = []
            for agent in range(0, len(self.env.agents)):
                position = self.env.agents[agent].position if self.env.agents[agent].position is not None else \
                    self.env.agents[agent].initial_position
                timestep.append(position)
            solution.append(np.array(timestep))
            all_obs, all_done = self.step_all_parallel(step, all_obs)
            for agent in range(0, len(self.env.agents)):
                self.agent_done[agent] = all_done[agent]
            step += 1
            if self.saveGIF and self.episode_count% self.SAVEGIFFREQUENCY ==0 :
                self.env_renderer.render_env(show=False, frames=False, show_observations=False)
                self.episode_frames.append(self.env_renderer.get_image())

        if self.saveGIF and self.episode_count% self.SAVEGIFFREQUENCY ==0  :
            time_per_step = 0.1
            images = np.array(self.episode_frames)  
            self.make_gif(images, '{}/test_episode_{:d}_{:d}_{:s}.gif'.format(self.gifs_path,self.episode_count,step,("_success" if self.env.dones["__all__"] else "")))       
        
        print('step', step)
        print('Done', self.agent_done.count(1), '/', len(self.env.agents))
        for agent in range(0, len(self.env.agents)):
            position = self.env.agents[agent].position if self.env.agents[
                agent].position is not None else self.env.agents[agent].initial_position
            timestep.append(position)
        all_done = self.agent_done.count(1) == len(self.env.agents)
        return np.array(solution), all_done, self.agent_done.count(1), self.timeobs


def make_name(num_agents, size, id, extension, dirname, extra=""):
    if extra == "":
        return dirname+'/'+"{}_agents_{}_size_{}_id_{}".format(num_agents, size, id, extension)
    else:
        return dirname+'/'+"{}_agents_{}_size_{}_id_{}{}".format(num_agents, size, id, extra, extension)


def run_simulations(next, flatland_test):
    (num_agents, id, width, height, max_cities, max_rails) = next
    all_obs = flatland_test.set_env(
        num_agents, id, width, height, max_cities, max_rails)
    results = dict()
    start_time = time.time()
    print('Starting test ({},{},{},{})'.format(num_agents, width, height, id))
    max_time = int(8*(height + width + (num_agents/max_cities))) -2 
    path, all_done, num_done , obs_time = flatland_test.find_path(all_obs, max_time)
    results['Successful_Agents'] = num_done
    results['Observetime'] = obs_time 
    results['finished'] = True if all_done else False
    results['time'] = round((time.time()-start_time), 2)
    results['length'] = len(path)

    return results


if __name__ == "__main__":
    def getfilename() :
        today = datetime.today()
        d1 = today.strftime("%d-%m") 
        now = datetime.now()
        current_time = now.strftime("%H")
        filename =  "Flatland_Test" + "_"+ d1 + "_" + current_time + ".txt" 
        return filename 

    obs_size = TrafficLightObs.OBS_SIZE
    num_agents = 4
    num_iterations = 10
    min_grid_size = 30
    max_grid_size = 80
    max_agents = 128
    TEST_FLATLAND_ENVIRONMENTS = True
    saveGIF = False  
    filename = str(getfilename())  
    flatland_environments = [[50, 5, 25, 25, 2, 3, 50], [50, 10, 30, 30, 2, 3, 100], [50, 20, 30, 30, 3, 3, 200], [40, 50, 20, 35, 3, 3, 500],
                             [30, 80, 35, 20, 5, 3, 800], [30, 80, 35, 35, 5, 4, 800], [
                                 30, 80, 40, 60, 9, 4, 800], [30, 80, 60, 40, 13, 4, 800],
                             [20, 80, 60, 60, 17, 4, 800], [20, 100, 80, 120, 21, 4, 1000], [
                                 20, 100, 100, 80, 25, 4, 1000], [10, 200, 100, 100, 29, 4, 2000],
                             [10, 200, 150, 150, 33, 4, 2000], [10, 400, 150, 150, 37, 4, 4000]]

    flatland_test = FLATLAND('newmod', obs_size,
                             TEST_FLATLAND_ENVIRONMENTS,saveGIF,'./gifs_SMObs')
    summary_file = open(filename, "w+")
    summary_file.write("Summary of Flatland Testing")
    summary_file.write("\n")
    summary_file.close()

    if not TEST_FLATLAND_ENVIRONMENTS:
        while num_agents <= max_agents:
            num_agents *= 2

            print("Starting tests for %d agents" % num_agents)
            for size in range(min_grid_size, max_grid_size, 5):
                summary_file = open(filename, "a+")
                if size != 30:
                    successful_rate = round(
                        (100*total_completed/(num_iterations*num_agents)), 2)
                    episode_success_rate = round(
                        (100*success_count/num_iterations), 2)
                    summary_file.write(
                        "Agent Success Rate: {}".format(successful_rate))
                    summary_file.write("\n")
                    summary_file.write(
                        "Episode Success Rate: {}".format(episode_success_rate))
                    summary_file.write("\n")
                    summary_file.write("\n")

                summary_file.write(
                    "Size: {} Agents: {}".format(size, num_agents))
                summary_file.write("\n")
                summary_file.write("\n")
                summary_file.close()
                total_completed = 0
                success_count = 0
                total_time = 0 

                for iter in range(num_iterations):
                    results = run_simulations(
                        (num_agents, iter, size, size, 3, None), flatland_test)

                    total_completed += results['Successful_Agents']
                    if results['finished'] == True:
                        success_count += 1

                    summary_file = open(filename, "a+")
                    summary_file.write(" Finished: {} CompletedAgents: {}  TimeTaken: {} Length: {} ".format(results['finished'],
                                                                                                             results['Successful_Agents'], results['time'], results['length']))
                    summary_file.write("\n")
                    summary_file.write("\n")
                    summary_file.close()

    else:
        total_done = 0 
        total_agents = 0 
        TOTAL_TIME = 0 
        for index in range(len(flatland_environments)):
            summary_file = open(filename, "a+")
            if index != 0:
                num_iterations = flatland_environments[index-1][0]
                successful_rate = round(
                    (100*total_completed/(num_iterations*num_agents)), 2)
                episode_success_rate = round(
                    (100*success_count/num_iterations), 2)
                total_done += total_completed 
                total_agents +=  num_iterations*num_agents    
                summary_file.write(
                    "Agent Success Rate: {}".format(successful_rate))
                summary_file.write("\n")
                summary_file.write(
                    "Episode Success Rate: {}".format(episode_success_rate))
                summary_file.write("\n")
                summary_file.write("\n")
                summary_file.write(
                    "Time Taken : {} Minutes".format(round((total_time/60),2)))
                summary_file.write("\n")
                summary_file.write("\n")
                summary_file.write(
                    "Average Time Taken : {} Seconds".format(round((total_time/num_iterations),2)))
                summary_file.write("\n")
                summary_file.write("\n")
                summary_file.write(
                    "Average Observation Time : {} Seconds".format(round((obs_time/num_iterations),2)))
                summary_file.write("\n")
                summary_file.write("\n")
                summary_file.write(
                    "Time Elapsed so Far: {} Minutes".format(round((TOTAL_TIME/60),2)))
                summary_file.write("\n")
                summary_file.write("\n")
                
            num_agents = flatland_environments[index][1]
            width = flatland_environments[index][2]
            height = flatland_environments[index][3]
            max_cities = flatland_environments[index][4]
            max_rails = flatland_environments[index][5]

            summary_file.write("Environment: {} Agents: {} Width: {} Height: {}".format(index,num_agents,width,height))

            summary_file.write("\n")
            summary_file.write("\n")
            summary_file.close()
            total_completed = 0
            total_time = 0 
            success_count = 0
            obs_time = 0 
            

            for iter in range(flatland_environments[index][0]):
                results = run_simulations(
                    (num_agents, iter, width, height, max_cities, max_rails), flatland_test)
                total_completed += results['Successful_Agents']
                total_time+= results['time']
                TOTAL_TIME += results['time']
                obs_time += results['Observetime']
                if results['finished'] == True:
                    success_count += 1
                summary_file = open(filename, "a+")
                summary_file.write(" Finished: {} CompletedAgents: {}  TimeTaken: {} Length: {} ".format(results['finished'],
                                                                                                         results['Successful_Agents'], results['time'], results['length']))
                summary_file.write("\n")
                summary_file.write("\n")
                summary_file.close()


        summary_file = open(filename, "a+")
        num_iterations = flatland_environments[index-1][0]
        successful_rate = round(
                    (100*total_completed/(num_iterations*num_agents)), 2)
        episode_success_rate = round(
                    (100*success_count/num_iterations), 2)
        total_done += total_completed 
        total_agents +=  num_iterations*num_agents    
        summary_file.write(
                    "Agent Success Rate: {}".format(successful_rate))
        summary_file.write("\n")
        summary_file.write(
                    "Episode Success Rate: {}".format(episode_success_rate))
        summary_file.write("\n")
        summary_file.write("\n")
        summary_file.write(
                    "Time Taken : {} Minutes".format(round((total_time/60),2)))
        summary_file.write("\n")
        summary_file.write("\n")
        summary_file.write(
                    "Average Time Taken : {} Seconds".format(round((total_time/num_iterations),2)))
        summary_file.write("\n")
        summary_file.write("\n")
        summary_file.write(
                    "Average Observation Time : {} Seconds".format(round((obs_time/num_iterations),2)))
        summary_file.write("\n")
        summary_file.write("\n")
        summary_file.write(
                    "Time Elapsed so Far: {} Minutes".format(round((TOTAL_TIME/60),2)))
        summary_file.write("\n")
        summary_file.write("\n")
        summary_file.write(
                    "AVERAGE SUCCESS RATE: {}".format((100*total_done)/total_agents))
        summary_file.close()



print("finished all tests!")
