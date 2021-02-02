from __future__ import division

import copy
from operator import itemgetter

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.distance_map import DistanceMap
#from flatland.envs.rail_env_shortest_paths import get_shortest_paths
from shortestpath import get_shortest_paths 

class StateMaskingObs(ObservationBuilder):
    ENTRYS_PER_COLUMN = 8
    TRAFFIC_LIGHT_SIZE = 3
    HOMO_SIZE = 3
    OBS_SIZE = 3 * ENTRYS_PER_COLUMN + TRAFFIC_LIGHT_SIZE + HOMO_SIZE
    ADDITIONAL_INPUT = OBS_SIZE - 3 * ENTRYS_PER_COLUMN

    def __init__(self):
        super(StateMaskingObs, self).__init__()
        self.fake_envs = []
        self.single_solver = []
        self.SKIPLARGE = True 

    def set_env(self, env):
        super().set_env(env)
        
    def reset(self):
        self.time = 0
        self.junctions = []
        self.visited = []
        self.actual_stopping_positions = []
        self.stopping_positions_only = []
        self.actual_junction_cluster = []
        self.permanent_pointer_position = []
        self.temporary_pointer_position = []
        self.initialize_list = [0 for i in range(len(self.env.agents))]
        self.num_agent = len(self.env.agents)
        self.agent_in_clusters = [[-1, -1] for i in range(len(self.env.agents))]
        self.num_active_agents = [0 for i in range(len(self.env.agents))]
        self.initialization_timestep = 0
        self.max_timestep = int((8 * (self.env.height + self.env.width)) / len(self.env.agents))
        self.upper_bound = int((self.env.height + self.env.width) / 12)
        self.observations = [0 for i in range(len(self.env.agents))]
        self.queues = {}
        self.agents_activated = []
        self.clusters_activated = []
        self.path_dict= {}
        self.State ={}  
        self.Next_Positions = {}
        self.agents_stuck = [[0,0] for i in range(len(self.env.agents))]  
        self.old_info = [ [0,0 ] for i in range(len(self.env.agents))] 
        return

    def _cheat_expert(self, start_pos, orientation, agentID):
        """
        return the next position when expert is standing on a junction
        """
        target = self.env.agents[agentID].target
        agent_inform = (start_pos,orientation,target)
        if agent_inform in self.path_dict.keys() :
            return self.path_dict[agent_inform] 
        else :
            path = get_shortest_paths(self.distance_map,start_pos,orientation, agent_handle = agentID)
            self.path_dict[agent_inform] = path[agentID] 
            return path[agentID]

    def get_distance_map(self) :
        self.distance_map = DistanceMap(env_width=self.env.rail.width, env_height=self.env.rail.height,
                                   agents=self.env.agents)
        self.distance_map.reset(self.env.agents, self.env.rail)                           
        return   

    def get(self, handle=0):
        """
        param-handle: agent id
        if agent_id==0, add Obs of all agents to self.observations,
        return respective Obs of agent_id
        New obs is a 3*8+3+3 tuple observation for RL
        """

        def is_junction_homo(cell_list):
            assert len(cell_list) == 3
            homo_output = [0, 0, 0]
            count_dict = {}
            junction_pos = None
            # index = []
            for i in cell_list:
                index = [x for x in range(len(cell_list)) if cell_list[x] == i]
                # for x in range(len(cell_list)):
                #     if cell_list[x] is not None:
                #         print(cell_list[x], i)
                #         if cell_list[x] == i:
                #             index.append(x)
                count_dict.update({i: index})
            if 1 <= len(count_dict) <= 3:
                if len(count_dict) == 1:
                    k = list(count_dict.keys())
                    if k[0] is not None:
                        homo_output = [1, 1, 1]
                        junction_pos = k[0]
                elif len(count_dict) == 2:
                    for key, value in count_dict.items():
                        if len(value) == 2 and key is not None:
                            junction_pos = key
                            for index in value:
                                homo_output[index] = 1
            else:
                raise RuntimeError('bug somewhere')
            return homo_output

        def initialize_stopping_points():
            """
            Compute all junction clusters and the stopping points associated with them
            Only needs to be called once for an episode
            """
            self.compute_all_junctions()
            self.compute_stopping_points()
            self.set_stopping_pointers()

        #    if self.initialize_list[handle] == 1:
        #        self.agent_initial_positions[handle] = [0, (-3, -3)]
        if len(self.env.agents) >81 and self.SKIPLARGE == True :
            if ((len(self.env.agents)==100) and ((self.env.height+self.env.width)!=200)) or len(self.env.agents) >100   :
                if self.time == 0 :
                    for agent in range(len(self.env.agents)) :
                        self.observations[agent] = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                        0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                        0.0, 0.0, 0.0,  
                                                        1.0, 0.0, -1.0]
                self.time+=1                                         
                return                                     
        if handle == 0:
            self.get_others_complete() 
            if self.time == 0:
                self.GetAllStates() 
                initialize_stopping_points()
                #self.get_initial_positions()
                self.get_initialization_queue()
                self.get_distance_map() 
            self.update_pointers()
            # Get clusters with stuck agents
            self.get_stuck_clusters()
            self.get_timed_out_clusters()
            # Increment Time
            self.time += 1
            self.initialization_timestep +=1 
            for agent in range(len(self.env.agents)) :
                
                my_pos = self.env.agents[agent].position if self.env.agents[agent].position is not None else self.env.agents[
                    agent].initial_position
                my_direction = self.env.agents[agent].direction if self.env.agents[agent].direction is not None else \
                    self.env.agents[
                        agent].initial_direction
                
                if self.time >2 and not self.env.dones[agent] and self.agents_stuck[agent][0] ==0 and agent in self.agents_activated:
                    if self.old_info[agent][0] == my_pos and self.old_info[agent][1] == my_direction:
                        self.agents_stuck[agent][1] +=1 
                    else :
                        self.agents_stuck[agent][1] = 0     

                self.old_info[agent][0] = copy.deepcopy(my_pos)
                self.old_info[agent][1] = copy.deepcopy(my_direction)         
                
                if self.agents_stuck[agent][1] >100 or self.agents_stuck[agent][0] == 1 and not self.env.dones[agent] :
                    self.agents_stuck[agent][0] = 1  
                    self.observations[agent] = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0,  # homogenous junction cell
                                                1.0, 0.0, -1.0]
                    continue                             
                

                if self.env.dones[agent]:
                    self.num_active_agents[agent] = 2
                    self.observations[agent] = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0,  # homogenous junction cell
                                                1.0, 0.0, -1.0]
                    continue  # last element: traffic light            
                    
                state_of_agent = self.StateClassifier(my_pos,my_direction) 
                if state_of_agent in [0,3,4] and agent in self.agents_activated :
                    self.observations[agent] = [0.0, -1.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0,
                            0.0, -1.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0,
                            0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0  , -1 ,  # homagogenous junction cell
                            1.0, 0.0, -1.0]
                    continue         

                isStoppingPoint = False        

                actual_pos = self.env.agents[agent].position if self.env.agents[agent].position is not None else \
                    self.env.agents[
                        agent].initial_position
                actual_dir = self.env.agents[agent].direction if self.env.agents[agent].direction is not None else \
                    self.env.agents[
                        agent].initial_direction

                # Get Traffic Signal
                if agent in self.agents_activated: 
                    others_traffic = self.get_others(agent,1)
                    traffic_signal = self.get_traffic_signal(actual_pos, actual_dir, others_traffic, agent)
                else :
                    traffic_signal = 0     

                obs = []
                others = []
                next_junction_list = []
                all_handles = [i for i in range(len(self.env.agents))]


                # Adjust Position if at Stopping Point 
                if state_of_agent in [1, 4]:
                    valid_act_pos_pair = self._get_next_valid_position(my_pos, my_direction)
                    for action, next_pos in valid_act_pos_pair.items():
                        if next_pos[1] is not None:
                            my_pos = next_pos[1]
                            my_direction = next_pos[0]
                            isStoppingPoint = True
                            break

                # Check whether agent is at decision state 
                if state_of_agent ==2 :
                    state_of_agent = 1
                else:
                    state_of_agent = -1
                
                valid_act_pos_pair = self._get_next_valid_position(my_pos, my_direction)
                for action, next_pos in valid_act_pos_pair.items():
                    # stopping = self.stopping_point_occupied(actual_pos, actual_dir, my_pos, my_direction, action, others)
                    if next_pos[1] is not None:  # valid action
                        # print(next_pos)
                        has_solution, obs_1_direction, junction_cell = self._count_path_block(next_pos[1], next_pos[0],
                                                                                              agent,
                                                                                              my_pos,
                                                                                              my_direction)
                        next_junction_list.append(junction_cell)
                        if has_solution:

                            obs_1_direction[0] += 1  # compensate 2 cell for current pos

                            if isStoppingPoint:
                                #    obs_1_direction[6] = 1
                                obs_1_direction[
                                    0] += 1  # compensate 1 cell for stopping point bc it receives future obs
                            obs_1_direction.insert(0, True)
                            obs.append(obs_1_direction)
                        else:
                            # print(my_pos, my_direction, self.StateClassifier(my_pos, my_direction), "still fail!")
                            obs_1_direction.insert(0, False)
                            obs.append(obs_1_direction)
                    else:
                        next_junction_list.append(None)
                        obs.append([False, 0, 0, 0, 0, 0, 0, 0])
                obs = np.array(obs, dtype=float)
                max_length = max(obs[:, 1])
                for i in range(3):
                    obs[i, 1] = obs[i, 1] / max_length if obs[i, 1] > 0 else -1
                obs = np.reshape(obs, (1, -1))
                obs = obs.squeeze()
                obs = obs.tolist()
                assert len(obs) == 3 * self.ENTRYS_PER_COLUMN

                obs.append(traffic_signal)

                if agent in self.agents_activated:
                    obs.append(0)
                else:
                    obs.append(1)

                obs.append(state_of_agent)
                if agent in self.agents_activated: 
                    junction_homo = is_junction_homo(next_junction_list)
                    obs.extend(junction_homo)
                else :
                    junction_homo = [0,0,0] 
                    obs.extend(junction_homo)  
                      
                assert len(junction_homo) == self.HOMO_SIZE
                assert len(obs) == self.OBS_SIZE
                self.observations[agent] = obs

                # initialization_code()  
            self.get_initialization()
            return self.observations[0]

        else:
            return self.observations[handle]

    def get_many(self, handles=None):
        observations = {}
        self.get(0)  # store all obs in self.observation
        if handles is None:
            handles = []
        for h in handles:
            observations[h] = self.observations[h]
        return observations

    def StateComputation(self,agent_pos,agent_dir) :
        avb_moves = self.env.rail.get_transitions(*agent_pos, agent_dir)
        move2grid = np.array([[[0, -1], [-1, 0], [0, +1]], [[-1, 0], [0, +1], [+1, 0]], [[0, +1], [+1, 0], [0, -1]],
                              [[+1, 0], [0, -1], [-1, 0]]])  # Obtained from colliding agent code
        trans2act = np.array([[2, 3, 0, 1], [1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2]])  # Maps transition to an action
        # next_dir_grid = np.array([-1,0,1])  # Maps action to a change in agent direction
        if sum(avb_moves) > 1:  # This is definitely a decision junction since more than 1 move possible
            self.State[agent_pos,agent_dir] = [2,None] 
            self.Get_Valid_Positions(agent_pos,agent_dir,2)  
            return 
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
            if (sumnextcell > 2) :
                self.State[agent_pos,agent_dir] = [1,tuple(next_pos)] 
                self.Get_Valid_Positions(agent_pos,agent_dir,1) 
                return
            elif (sumnextcell <= 2) :
                self.State[agent_pos,agent_dir] = [0,tuple(next_pos)] 
                self.Get_Valid_Positions(agent_pos,agent_dir,0)  
                return 
        else:
            self.State[agent_pos,agent_dir] = [-1,None]
            next_positions = {}
            for action in [1, 2, 3]:
                next_positions.update({action: [None, None]}) 
            self.Next_Positions[agent_pos,agent_dir] = next_positions     
            return

    def GetAllStates(self):
        for row in range(self.env.height):
            for column in range(self.env.width):
                position = (row, column)
                for direction in range(0,4) :
                    self.StateComputation(position,direction) 
        return             
                
    def StateClassifier(self, agent_pos, agent_dir):
        """
        returns 0 : No decision point
        returns 1 : Stopping point (Decision at next cell)
        returns 2 : At decision point currently (More than 1 available transition)
        returns 3,4 : MUST STOP point - Agent Ahead
        returns None: invalid cell
        """
        output = self.State[agent_pos,agent_dir]
        state = output[0]
        next_position = output[1] 
        if state ==2 :
            return 2 
        elif state ==0 :
            others = self.get_others(0,2) 
            if next_position in others:
                return 3
            else :
                return 0 
        elif state ==1 :
            others = self.get_others(0,2) 
            if next_position in others:
                return 4
            else :
                return 1
        else : 
            return None                  

    def Get_Valid_Positions(self, my_pos, my_direction,state):
        """
        action: 0 ---> stop
                1 ---> left
                2 ---> forward
                3 ---> right
        """
        avb_moves = self.env.rail.get_transitions(*my_pos, my_direction)
        action2direction = [[3, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 0]]
        dir2grid = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        next_dir_grid = np.array([-1, 0, 1])
        move2grid = np.array([[[0, -1], [-1, 0], [0, +1]],
                              [[-1, 0], [0, +1], [+1, 0]],
                              [[0, +1], [+1, 0], [0, -1]],
                              [[+1, 0], [0, -1], [-1, 0]]])
        avbmove = [i for i, x in enumerate(avb_moves) if x == 1]
        trans2act = np.array([[2, 3, 0, 1], [1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2]])  # Maps transition to an action
        next_positions = {}
        for action in [1, 2, 3]:
            next_positions.update({action: [None, None]})

        if state in [0, 1, 2, 3, 4]:
            if state == 2:  # decision point
                for action in [1, 2, 3]:
                    i = action2direction[my_direction][action - 1]
                    if i in avbmove:  # available NSWE direction
                        next_pos = my_pos + dir2grid[i]
                        next_positions[action] = [i, tuple(next_pos)]

            else:
                avbmove = avb_moves.index(1)  # Get the available transition to next cell
                action = trans2act[my_direction][avbmove]  # Get the corresponding action for that transition
                if action == 0:
                    next_dir = (my_direction + 2) % 4
                    next_pos = my_pos + move2grid[next_dir][1]
                    # This is a dead end, so turn around and move forward
                else:
                    next_pos = my_pos + move2grid[my_direction][action - 1]
                    next_dir = (my_direction + (next_dir_grid[action - 1])) % 4
                if action == 2 or action == 0 or sum(avb_moves) == 1:
                    next_positions[2] = [next_dir, tuple(next_pos)]
                else:
                    next_positions[action] = [next_dir, tuple(next_pos)]
        self.Next_Positions[my_pos,my_direction] = next_positions
        return next_positions

    def _get_next_valid_position(self,my_pos,my_direction):
        return self.Next_Positions[my_pos,my_direction]   

    def DistToNextJunction(self, agentID, full_path, old_pos, old_heading):
        """
        Returns 1 : If at Stopping Point 
        Returns Distance to Junction (Greater than 1) : If at No Decision Point
        Returns 0 : If at Junction currently 
        """
        full_path_cp = copy.deepcopy(full_path)
        state = self.StateClassifier(old_pos, old_heading)
        sumcell = 0  # How many possible transitions at next cell
        # for j in range(0, 4):
        #     new_avb_moves = self.env.rail.get_transitions(*old_pos, j)
        #     sumcell += sum(new_avb_moves)
        if state in [1, 4]:
            return 1
        elif state in [0, 2, 3]:
            distance = 0
            if state == 2:
                full_path_cp.pop(0)  # remove current junction pos, and add 1 more step
                distance += 1
            for i in range(1, len(full_path_cp) - 1):  # full_path_cp[0] is current pos, not moving yet
                distance += 1
                statecell = self.StateClassifier(full_path_cp[i].position, full_path_cp[i].direction)
                if statecell in [1, 4]:
                    return distance + 1
                elif statecell == 2:
                    return distance
            return distance
        else:
            print("Some error in DistToNextJunction")
            return 0

    def check_transition_validity(self, next_pos, current_pos):
        directions = [-1, -1, -1, -1]
        current_pos = tuple(current_pos)
        if (current_pos[0] >= 0 and 0 <= current_pos[1] < self.env.width and current_pos[
            0] < self.env.height):
            for i in range(0, 4):
                next_positions = self._get_next_valid_position(tuple(current_pos), i)
                for j in range(1, 4):
                    if next_positions[j][1] == next_pos:
                        directions[i] = i
                        break
            if sum(directions) >= -3:
                return True, directions
            else:
                return False, directions
        else:
            return False, directions

    def stopping_point_occupied(self, actual_pos, actual_dir, current_pos, current_dir, action, others):
        action = action - 1
        # print(others)
        count = 0

        movegrid = np.array([[[0, -1], [-1, 0],
                              [0, +1], [+1, 0]],
                             [[-1, 0], [0, +1],
                              [+1, 0], [0, -1]],
                             [[0, +1], [+1, 0],
                              [0, -1], [-1, 0]]])
        if self.StateClassifier(actual_pos, actual_dir) == 2:
            next_pos = actual_pos + movegrid[action][actual_dir]
            validity, directions = self.check_transition_validity(actual_pos, next_pos)
            if validity:
                for i in range(0, 4):
                    if [tuple(next_pos.reshape(1, -1)[0]), directions[i]] in others:
                        return 1
            return 0

        elif self.StateClassifier(actual_pos, actual_dir) in [1, 4]:
            next_pos = current_pos + movegrid[action][current_dir]
            validity, directions = self.check_transition_validity(current_pos, next_pos)
            if validity:
                for i in range(0, 4):
                    if [tuple(next_pos.reshape(1, -1)[0]), directions[i]] in others:
                        return 1
            return 0

        elif self.StateClassifier(actual_pos, actual_dir) in [0, 3] and action == 1:
            new_pos = actual_pos
            new_dir = actual_dir
            while self.StateClassifier(new_pos, new_dir) not in [1, 2, 4]:
                next_positions = self._get_next_valid_position(new_pos, new_dir)
                new_dir = next_positions[2][0]
                new_pos = next_positions[2][1]
                count += 1
                if count > 30:
                    return 0
            for i in range(0, 4):
                if [new_pos, i] in others:
                    return 1
            return 0
        elif self.StateClassifier(actual_pos, actual_dir) in [0, 3] and action in [0, 2]:
            return 0
        else:
            return 0

    def isJuntion(self, agent_pos):
        trans = []
        for direction in range(0, 4):
            trans.append(sum(self.env.rail.get_transitions(*agent_pos, direction)))
        return sum(trans) > 2

    def _count_path_block(self, start_pos, heading, agentID, old_pos, old_heading):
        """
        input: the start position (x,y).
        make sure the start position is a rail cell not a obstacle!
        Convert a list of directions to the opposite direction list.
        return: a bool and a 6-element list
                int: does_this_direction has expert solution?,
                [int: astar_path length,
                 int:num_blocking within the first junction,
                 int:num_all_blocking,
                 int:num_blocking on junctions,
                 int: distance to the next junction]
        """

        def next_junction_cell(full_path):
            for i in range(len(full_path)):
                check_pos = full_path[i]
                if self.isJuntion(check_pos.position):
                    return full_path[i][0]
            return None

        assert self.env.rail.grid[(start_pos[0], start_pos[1])] != 0, "start position " + str(start_pos) \
                                                                      + " is not valid in the map!"

        count_first_decision_block = 0
        count_first_junction_block = 0
        count_all_block = 0
        count_all_decision_block = 0
        count_decision_point = 0
        count_junction = 0
        crash_buffer = []
        visit_first_decision_point = False
        visit_first_stopping = False

        actual_pos = self.env.agents[agentID].position if self.env.agents[agentID].position is not None else \
            self.env.agents[
                agentID].initial_position
        actual_direction = self.env.agents[agentID].direction if self.env.agents[agentID].direction is not None else \
            self.env.agents[
                agentID].initial_direction

        all_handle = [i for i in range(len(self.env.agents))]

        others_pos = self.get_others(agentID,3) 
        others_directions = self.get_others(agentID,4) 
        #others_moving = []

        # start to compute quantities ---------------------------------------
        full_path = self._cheat_expert(start_pos, heading, agentID)

        if full_path is None:
            return False, [-1,  # single agent path length
                           min(1, count_first_decision_block),  # num_block within the first junction
                           min(1, count_first_junction_block),
                           count_first_decision_block,
                           count_all_block / self.num_agent,  # num_block all along the path
                           count_all_decision_block,  # num_block standing on junctions
                           -1], None

        distance_to_next_junction = self.DistToNextJunction(agentID, full_path, actual_pos,
                                                            actual_direction)
        junction_cell = next_junction_cell(full_path)

        for num_step in range(len(full_path) - 1):
            checking_cell = full_path[num_step].position  # checking cell is the cell we want to check blocking
            checking_cell_dir = full_path[num_step].direction

            if self.isJuntion(checking_cell):
                count_junction += 1
            if self.StateClassifier(checking_cell, checking_cell_dir) == 2:
                count_decision_point += 1

            for direction in range(4):
                if self.StateClassifier(checking_cell, direction) in [1, 4] and not visit_first_stopping:
                    visit_first_stopping = True

            if (checking_cell in others_pos.keys()) is True:
                idx = others_pos[checking_cell]
                crash_buffer.append(checking_cell)
                # test if there is other agents stepping on the stopping point

                if others_directions[idx] == checking_cell_dir:
                    # same heading, not moving, so waiting
                    if num_step == 0:
                        count_first_decision_block += 1

                elif (others_directions[idx] + checking_cell_dir) % 2 == 0:
                    # opposite heading, so blocking
                    count_all_block += 1
                    if self.StateClassifier(checking_cell, checking_cell_dir) == 2:
                        count_all_decision_block += 1

                    if not visit_first_decision_point:
                        count_first_decision_block += 1
                    if not visit_first_stopping:  # stopping point must lead to a junction
                        count_first_junction_block += 1
                else:  # neither same direction or opposite direction, meaning that an agent staying at a junction
                    # but that is not a junction for the current direction (non-decision point)
                    count_all_block += 1
                    count_all_decision_block += 1
                    if not visit_first_decision_point:
                        count_first_decision_block += 1

            if self.StateClassifier(checking_cell, checking_cell_dir) == 2:
                visit_first_decision_point = True

        return True, [len(full_path),  # single agent path length
                      min(1, count_first_decision_block),  # num_block within the first decision point
                      min(1, count_first_junction_block),  # num_block within the first junction
                      ((count_first_decision_block - count_first_junction_block) / max(1, count_first_decision_block)),
                      count_all_block / self.num_agent,  # num_block all along the path
                      count_all_decision_block / count_decision_point if count_decision_point else 0,
                      # num_block standing on junctions
                      distance_to_next_junction / len(full_path) if len(full_path) > 0 else -1,
                      ], junction_cell

    def compute_all_junctions(self):
        """
        Finds all junctions : Any cell with greater than 2 transitions available
        """
        for row in range(self.env.height):
            for column in range(self.env.width):
                position = (row, column)
                if self.total_transitions(position) > 2:
                    self.junctions.append(position)
        return

    def total_transitions(self, position):
        """
        Input - Position
        Return- Total transitions available at a particular cell location
        Called by compute_all_junctions
        """
        sum_transitions = 0
        for i in range(0, 4):
            sum_transitions += sum(self.env.rail.get_transitions(*position, i))
        return sum_transitions

    def compute_stopping_points(self):
        """
        Input - Environment
        Return - None
        Computes all stopping points and junction clumps and stores them
        """
        for position in self.junctions:  # Iterate over all junctions
            if position not in self.visited:  # Only visit junctions which have not been visited before
                self.visited.append(position)  # Keep track of junctions visited
                self.stopping_points = []
                self.stopping_points_only = []
                self.junction_cluster = []
                self.junction_cluster.append(position)  # Computing Junction clusters
                self.visit(position)  # VISIT the current junction
                self.actual_stopping_positions.append(self.stopping_points)
                self.actual_junction_cluster.append(self.junction_cluster)
                self.stopping_positions_only.append(self.stopping_points_only)
        return

    def get_possible_positions(self, position):
        """
        Input - Position
        Returns - 4 Positions which can be reached by this cell , doesn't check for validity of the transition
        """

        movements = np.array([[[-1, 0], [0, +1],
                               [1, 0], [0, -1]]])
        possible_positions = []
        for i in range(0, 4):
            next_pos = position + movements[0][i]
            next_pos = tuple(next_pos.reshape(1, -1)[0])
            possible_positions.append(next_pos)
        return possible_positions

    def visit(self, position):
        """
        Recursive code to visit a junction position and compute the junction clump
        """
        possible_stopping_points = self.get_possible_positions(
            position)  # Get possible positions from the current position
        for stopping_position in possible_stopping_points:
            if (stopping_position is not None):
                valid, directions = self.check_transition_validity(position,
                                                                   stopping_position)  # Check if transition to that cell is possible
                if not valid:
                    continue
                else:
                    for j in range(0, 4):
                        if (0 < self.total_transitions(stopping_position) <= 2) and (
                                (stopping_position, directions) not in self.stopping_points) and (
                                directions[j] != -1):  # Check whether that cell is a stopping point
                            self.stopping_points.append(
                                (stopping_position, directions[j]))  # which has not been visited yet
                            self.stopping_points_only.append(stopping_position)
                        elif self.total_transitions(stopping_position) > 2 and (
                                stopping_position not in self.visited):  # Check whether that cell is a junction
                            self.junction_cluster.append(stopping_position)  # which has not been visited yet
                            self.visited.append(stopping_position)  # Mark this junction visited
                            self.visit(
                                stopping_position)  # Make a recursive call to the function if that cell hasn't been visited yet
                        else:
                            pass
        return

    def set_stopping_pointers(self):
        """ 
        Initializes values for all traffic lights at the computed stopping points
        """
        for _ in self.actual_stopping_positions:
            self.permanent_pointer_position.append(0)  # Permanent pointer position moves by one at each time step
            self.temporary_pointer_position.append(
                0)  # Temporary pointer position makes the traffic light smart by checking for incoming traffic
        return

    def stopping_point_find(self, agent_pos, agent_dir):
        """
        Input - Agent Position and Direction
        Returns - Is Agent At Stopping Point, Stopping Cluster, Index within Cluster
        """
        agent_info = (agent_pos, agent_dir)
        for stopping_cluster in range(len(self.actual_stopping_positions)):
            for stopping_point in self.actual_stopping_positions[stopping_cluster]:
                if stopping_point == agent_info:
                    return True, stopping_cluster, self.actual_stopping_positions[stopping_cluster].index(
                        stopping_point)
        return False, None, None

    def junction_find(self, agent_pos):
        for cluster in range(len(self.actual_junction_cluster)):
            for point in self.actual_junction_cluster[cluster]:
                if point == agent_pos:
                    return True, cluster, self.actual_junction_cluster[cluster].index(point)
        return False, None, None

    def get_traffic_signal(self, agent_pos, agent_dir, others, handle):
        """
        Input - Agent Position, Direction, handle and others 
        Get the traffic signal for an agent
        Returns 1 : If Traffic Signal is Green
        Returns -1 : If Traffic Signal is Red 
        Two clearances are required to be obtained except if an agent is stuck within the cluster
        """

        validity, cluster, index = self.stopping_point_find(agent_pos,
                                                            agent_dir)  # Check whether agent is at Stopping Point
        if validity == False:  # If agent not at stopping point, traffic signal always green
            return 1
        if validity == True:  # If agent is at stopping point
            if cluster in self.stuck_clusters:  # First check if another agent is stuck inside the cluster
                current_pointer_pos = self.permanent_pointer_position[cluster]
                if current_pointer_pos == index:
                    return 1
                clearance = self.get_clearence(cluster, index, others)
                if clearance == True:  # If another agent is stuck, then directly get clearance for the traffic light
                    return 1
                else:
                    return -1

            if cluster in self.timed_out_clusters:
                current_pointer_pos = self.permanent_pointer_position[cluster]
                if current_pointer_pos == index:
                    return 1
                clearance = self.get_clearence(cluster, index, others)
                if clearance == True:  # If another agent is stuck, then directly get clearance for the traffic light
                    return 1
                else:
                    return -1

            # If no agent is stuck inside the cluster, then 2 clearances need to be obtained        
            cleared = self.get_first_clearance(cluster,
                                               handle)  # First clearance checks whether any agent is already occupying the cluster
            if cleared == False:
                return -1

            current_pointer_pos = self.permanent_pointer_position[cluster]
            if current_pointer_pos == index:
                return 1
            else:
                clearance = self.get_clearence(cluster, index,
                                               others)  # Second clearance checks whether other agents are waiting to go inside the cluster,
                if clearance == True:  # If other agents are waiting to go inside cluster, only 1 of them gets a green traffic light
                    return 1
                else:
                    return -1

    def get_agent_stuck(self, cluster, others, position, direction, handle):
        """ 
        Input - Current cluster, others, agent pos, agent dir , agent handle
        Recursively checks whether agent inside a cluster has any possible exit
        """
        # Get the next valid positions
        valid_act_pos_pair = self._get_next_valid_position(position, direction)
        for action, next_pos in valid_act_pos_pair.items():
            if next_pos[1] is not None:
                # If that position,direction has not been checked before
                if (next_pos[1], next_pos[0]) not in self.possible_positions:
                    self.possible_positions.append((next_pos[1], next_pos[0]))
                    # If that position is a stopping point and it is unoccupied , then agent has a free exit
                    if next_pos[1] in self.stopping_positions_only[cluster]:
                        if (next_pos[1], next_pos[0]) not in others.keys():
                            return 0
                    # If that position is another junction whithin the cluster, then recursively call this function        
                    elif next_pos[1] in self.actual_junction_cluster[cluster]:
                        output = self.get_agent_stuck(cluster, others, next_pos[1], next_pos[0], handle)
                        if output == 0:
                            return 0
                    else:
                        pass
        # If all exits are occupied by other agents, then the agent inside the cluster is stuck
        # In such a case, allowing 1 more agent entry into the cluster might be able to solve the problem            
        return 1

    def get_stuck_clusters(self):
        """
        Computes the clusters inside which agents are stuck (no unoccupied agents from that cluster) 
        Returns - Cluster indexes in which agents are stuck
        """
        self.stuck_clusters = []
        others = self.get_others(0,0)  # Get information of all agents
        for cluster in range(len(self.actual_junction_cluster)):  # Iterate over all clusters
            counter = 0
            total = 0
            for handle in range(len(self.env.agents)):  # Iterate over all agents
                position = self.env.agents[handle].position
                direction = self.env.agents[handle].direction
                if position in self.actual_junction_cluster[
                    cluster]:  # If an agent is inside a particular cluster, call function
                    self.possible_positions = []  # to determine whether the agent is stuck
                    self.possible_positions.append(
                        (self.env.agents[handle].position, self.env.agents[handle].direction))
                    stuck = self.get_agent_stuck(cluster, others, position, direction, handle)
                    if stuck:  # If the agent is stuck inside cluster, increment the counter
                        counter += 1
                    total += 1
            if counter == 1 and total < 2:  # Make sure that no more than 2 agents are allowed entry into the cluster
                self.stuck_clusters.append(cluster)
        return self.stuck_clusters

    def get_timed_out_clusters(self):
        self.timed_out_clusters = []
        self.num_agents_in_clusters = [0 for i in range(len(self.actual_junction_cluster))]
        self.timed_out = [0 for i in range(len(self.actual_junction_cluster))]
        for handle in range(len(self.env.agents)):
            if self.env.agents[handle].position != None:
                validity, cluster, index = self.junction_find(self.env.agents[handle].position)
                if validity == True:
                    if self.agent_in_clusters[handle][0] == cluster:
                        self.agent_in_clusters[handle][1] += 1
                    else:
                        self.agent_in_clusters[handle][0] = cluster
                        self.agent_in_clusters[handle][1] = 1

                    self.num_agents_in_clusters[cluster] += 1
                    if self.agent_in_clusters[handle][1] > 10:
                        self.timed_out[cluster] = 1
        for cluster in range(len(self.actual_junction_cluster)):
            if self.num_agents_in_clusters[cluster] < 2 and self.timed_out[cluster] == 1:
                self.timed_out_clusters.append(cluster)
        return self.timed_out_clusters

    def get_others_complete(self):
        """
        Returns tuple of (position,opposite direction) of each agent.
        Unborn/Completed agents are set to have : (-3,-3,0) 
        """
        self.others0 = {} 
        self.others1 ={}
        self.others2 ={}  
        self.others3 ={}  
        self.others4 ={} 
        for id in range(len(self.env.agents)):
            if self.env.agents[id].position is None:
                if self.env.dones[id] is True:
                    otherspos = (-id, -id)
                    othersdir = 0
                    othersdirections =0 
                else:
                    otherspos = (-id, -id)
                    othersdir = 0
                    othersdirections =0 
            else:  # position not None
                otherspos = self.env.agents[id].position
                othersdir = (self.env.agents[id].direction + 2) % 4
                othersdirections = self.env.agents[id].direction 
            self.others0[otherspos, othersdir] = id 
            self.others1[otherspos, othersdirections] = id 
            self.others2[otherspos] = id 
            self.others3[otherspos] = id  
            self.others4[id] = othersdirections 
        return 

    def get_others(self,handle,value) :
        if value ==0 :
            others = copy.copy(self.others0) 
            return others   
        elif value==1 :
            others = copy.copy(self.others1)
            if self.env.agents[handle].position is not None :
                agentpos = self.env.agents[handle].position
                agentdir = self.env.agents[handle].direction 
                del others[agentpos,agentdir] 
            return others 
        elif value ==2 :
            others = copy.copy(self.others2) 
            return others  
        elif value ==3 :
            others = copy.copy(self.others3) 
            if self.env.agents[handle].position is not None :
                agentpos = self.env.agents[handle].position
                del others[agentpos]
            return others 
        else : 
            others = copy.copy(self.others4) 
            del others[handle] 
            return others             

    def get_first_clearance(self, cluster, handle):
        """
        Checks whether a particular cluster is occupied 
        Returns 0 - If Occupied
        Returns 1 - If Free 
        """
        all_handles = [i for i in range(len(self.env.agents))]
        others = self.get_others(handle,3)   
        for value in range(len(self.actual_junction_cluster[cluster])):  # Checks whether the cluster is occupied
            if self.actual_junction_cluster[cluster][value] in others.keys():
                return 0
        return 1

    def get_clearence(self, cluster, index, others):
        """
        Executes smart time-dependent traffic light 
        Returns 1 - If traffic light is green
        Returns 0 - If traffic light is red
        If more than 1 agent is waiting to enter a cluster, only a single agent gets a clearance
        If only 1 agent waiting to enter a cluster, clearance always green for that agent 
        Pointers change values at each timestep
        """

        for positions in range(
                len(self.actual_stopping_positions[cluster])):  # Iterate over all stopping positions of a cluster
            if self.actual_stopping_positions[cluster][self.permanent_pointer_position[cluster]] in others.keys():
                return 0,  # Check if the position pointed by permanent pointer is occupied by another agent
            if self.temporary_pointer_position[cluster] == index:
                self.temporary_pointer_position[cluster] = self.permanent_pointer_position[cluster]
                return 1  # Check if temporary pointer index is the same as our agent's index, if yes return 1 (green)
            else:
                self.temporary_pointer_position[cluster] = (self.temporary_pointer_position[cluster] + 1) % len(
                    self.actual_stopping_positions[cluster])  # Increment temporary pointer to check at next position
                if self.actual_stopping_positions[cluster][self.temporary_pointer_position[cluster]] in others.keys():
                    self.temporary_pointer_position[cluster] = self.permanent_pointer_position[cluster]
                    return 0  # If temporary pointer index is same as another agent's index , then traffic signal for our agent is red

    def update_pointers(self):
        """
        Updates pointer by 1  at each timestep 
        """
        for i in range(len(self.actual_stopping_positions)):
            self.permanent_pointer_position[i] = (self.time) % len(self.actual_stopping_positions[i])
            self.temporary_pointer_position[i] = (self.time) % len(self.actual_stopping_positions[i])
        return

#    def get_others_traffic_light(self, handle):
#        """
#        Compute positions of all agents except our agent 
#        Returns - Others
#        """
#        all_handles = [i for i in range(len(self.env.agents))]
#        others = []
#        others2 = []
#        for id in all_handles:
#            if id != handle:
#                if self.env.agents[id].position is None:
#                    if self.env.dones[id] is True:
##                        otherspos = (-3, -3)
 #                       othersdirections = 0
 #                   else:
 #                       otherspos = (-3, -3)
 #                       othersdirections = self.env.agents[id].initial_direction
 #               else:  # position not None
 #                   otherspos = self.env.agents[id].position
 #                   othersdirections = self.env.agents[id].direction
 #               others.append((otherspos, othersdirections))
 #               others2.append([otherspos, othersdirections])
 #       return others, others2

#    def get_initial_positions(self):
#        """
#        Get initial position of all agents 
#        """
#        all_handles = [i for i in range(len(self.env.agents))]
#        for id in all_handles:
#            self.agent_initial_positions.append(
#                [id, self.env.agents[id].initial_position, self.env.agents[id].initial_direction,
#                 self.env.agents[id].target])
#        return

    def get_initialization_queue(self):
        for id in range(len(self.env.agents)):
            info = (self.env.agents[id].initial_position, self.env.agents[id].initial_direction,
                    self.env.agents[id].target)
            if info in self.queues.keys():
                self.queues[info][0] += 1
                self.queues[info].append(id)
            else:
                self.queues[info] = [1]
                self.queues[info].append(id)
        queue = sorted(self.queues.values(), key=itemgetter(0), reverse=True)
        self.queue = []
        for element in queue:
            self.queue.append([0, element[1:]])
        return

    def get_initialization(self):
        clusters_to_check = []
        queue_size = 0
        if (self.num_active_agents.count(1) > self.upper_bound) and (self.initialization_timestep < self.max_timestep):
            return
        if len(self.agents_activated) == len(self.env.agents):
            return
        for index_cluster in range(len(self.queue)):
            if index_cluster not in self.clusters_activated:
                queue_size += len(self.queue[index_cluster][1])
                clusters_to_check.append(index_cluster)
            if self.upper_bound < queue_size:
                break
        for index in clusters_to_check:
            if self.num_active_agents.count(1) > self.upper_bound:
                break
            for agent in self.queue[index][1]:
                if agent not in self.agents_activated:
                    others = self.get_others(agent,3 )
                    allowed = self.initialize(agent, self.observations[agent], others)
                    if allowed:
                        self.agents_activated.append(agent)
                        self.queue[index][0] += 1
                        self.num_active_agents[agent] = 1
                        self.observations[agent][self.OBS_SIZE - self.ADDITIONAL_INPUT + 1] = 0
                        if self.queue[index][0] == len(self.queue[index][1]):
                            self.clusters_activated.append(index)
                        self.initialization_timestep = 0
                        break
        if self.initialization_timestep > self.max_timestep:
            for index_cluster in range(len(self.queue)):
                if index_cluster not in self.clusters_activated :
                    for agent in self.queue[index_cluster][1]:
                        if agent not in self.agents_activated:
                            others = self.get_others(agent,3 )
                            allowed = self.initialize(agent, self.observations[agent], others)
                            if allowed:
                                self.num_active_agents[agent] = 1
                                self.agents_activated.append(agent)
                                self.queue[index_cluster][0] += 1
                                self.observations[agent][self.OBS_SIZE - self.ADDITIONAL_INPUT + 1] = 0
                                if self.queue[index_cluster][0] == len(self.queue[index_cluster][1]):
                                    self.clusters_activated.append(index_cluster)
                                self.initialization_timestep = 0
                                break
                    if allowed:
                        break

        return

    def initialize(self, handle, obs, others):
        """
        Input - Observation, agent handle, others
        Returns 1 if an agent is not allowed to initialize
        Returns 0 if an agent is allowed to initialize 
        """
        # An agent is only allowed to initialize if - 
        # 1) No other unborn agent with a smaller handle is currently at the same position waiting to be born
        # 2) No currently alive agent is at the agent's position
        # 3) No currently alive agent is currently behind the agent , this would lead to a crash
        # 4) The agent won't be blocked if it initializes 

        for agent in range(len(self.env.agents)):
            if (agent != handle) and (agent in self.agents_activated) and (self.env.agents[agent].position == None and \
                                                                           self.num_active_agents[agent] != 2) and (
                    self.env.agents[agent].initial_position == self.env.agents[handle].initial_position):
                return 0

        
        if (self.env.agents[handle].initial_position in others.keys()):
                return 0
        valid_act_pos_pair = self._get_next_valid_position(self.env.agents[handle].initial_position,
                                                           ((self.env.agents[handle].direction + 2) % 4))
        for action, pos in valid_act_pos_pair.items():
            if pos[1] is not None:
                    if (pos[1] in others.keys()):
                        return 0
        agentblocking = obs[self.ENTRYS_PER_COLUMN + 2]
        if agentblocking != 0:
            return 0
        # if (self.num_active_agents.count(1) < self.upper_bound) or (self.initialization_timestep>self.max_timestep) : 
        #     if (self.initialization_timestep>self.max_timestep) :
        #         self.initialization_timestep = 0 
        #     self.num_active_agents[handle] = 1     
        #     return 0 
        return 1


if __name__ == '__main__':
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.schedule_generators import sparse_schedule_generator
    from flatland.envs.rail_generators import sparse_rail_generator

    from flatland.utils.rendertools import RenderTool

    num_agent = 20


    def Complex_params():
        grid_width = 25  # min(int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )),

        # int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )))
        grid_height = 25  # min(int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1])),

        # nt(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )))
        rnd_start_goal = 8 + np.random.randint(0,
                                               3)  # int(np.random.uniform(num_workers, num_workers+1+episode_difficulty ))
        # int(np.random.uniform( num_workers , min(grid_width,grid_height))),

        rnd_extra = 1  # int(np.random.uniform(0 , 1+2*episode_difficulty ))
        # int(np.random.uniform( 0 , min(grid_width,grid_height))))
        rnd_min_dist = int(
            0.6 * min(grid_height, grid_width))  # int(np.random.uniform( episode_difficulty , 4+2*episode_difficulty ))
        rnd_max_dist = 99999  # int(np.random.uniform(3+episode_difficulty, 6+2*episode_difficulty))
        rnd_seed = 3

        return grid_width, grid_height, rnd_start_goal, rnd_extra, rnd_min_dist, rnd_max_dist, rnd_seed


    grid_width, grid_height, rnd_start_goal, rnd_extra, rnd_min_dist, rnd_max_dist, rnd_seed = Complex_params()
    gameEnv = RailEnv(width=35, height=20,
                      rail_generator=sparse_rail_generator(
                          max_num_cities=5,
                          max_rails_between_cities=2,
                          max_rails_in_city=3,
                          grid_mode=False,
                          seed=rnd_seed)
                      ,
                      schedule_generator=sparse_schedule_generator(),
                      obs_builder_object=StateMaskingObs(),
                      number_of_agents=num_agent)
    gameEnv.reset(regenerate_rail=True, regenerate_schedule=True)
    env_renderer = RenderTool(gameEnv)

    for t in range(500):
        gameEnv.reset(regenerate_rail=True, regenerate_schedule=True)
        print(t) 
        for i in range(30):
            obs, _, _, _ = gameEnv.step({0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2})
            #print(obs)
            env_renderer.render_env(show=True, frames=True, show_observations=True)
            # print(obs)
            # print(len(obs[0]))
            #input()
