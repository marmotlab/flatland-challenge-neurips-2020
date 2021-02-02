from __future__ import division
import numpy as np
import copy
from flatland.core.grid.grid4_utils import get_direction
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_astar import a_star
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env_shortest_paths import get_shortest_paths
from flatland.envs.distance_map import DistanceMap
from gym import spaces
import networkx as nx
from expert2 import Solver, Global_H
from expert2 import my_controller as expert_planner
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_generators import complex_rail_generator
import time


def countSetBits(num):
    binary = bin(num)
    setBits = [ones for ones in binary[2:] if ones == '1']
    return len(setBits)


def create_nodes(pos, transitions):
    nodes = {}
    directions = np.array_split([int(d) for d in format(transitions, '016b')], 4)
    for direction, next_directions in enumerate(directions):
        if 1 in next_directions:
            node = RailNode(pos, direction, next_directions)
            nodes[node] = node

    for node in nodes:
        node.blocks = [value for key, value in nodes.items() if key is not node]
    return nodes


def get_next_pos(pos, direction):
    if direction == 0:
        next_pos = (pos[0] - 1, pos[1])
    if direction == 1:
        next_pos = (pos[0], pos[1] + 1)
    if direction == 2:
        next_pos = (pos[0] + 1, pos[1])
    if direction == 3:
        next_pos = (pos[0], pos[1] - 1)
    else:
        next_pos = None
    return next_pos


class RailNode:
    def __init__(self, pos, direction, next_directions=None):
        self.direction = direction
        self.next_directions = next_directions
        self.pos = pos

    def __str__(self):
        return str(self.pos) + " " + str(self.direction)

    def get_next_pos(self):
        return [RailNode(get_next_pos(self.pos, next_direction), next_direction) for next_direction in
                np.where(self.next_directions == 1)[0]]

    def __eq__(self, other):
        try:
            return self.pos == other.pos and self.direction == other.direction
        except:
            return False

    def __hash__(self):
        return hash((self.pos, self.direction))


class StateMaskingObs(ObservationBuilder):
    def __init__(self):
        super(StateMaskingObs, self).__init__()
        self.obs_size = 18
        self.fake_envs = []
        self.single_solver = []

    def set_env(self, env):
        super().set_env(env)

    def reset(self):
        return

    def _cheat_expert(self, start_pos, orientation, agentID):
        """
        return the next position when expert is standing on a junction
        """
        fake_env = copy.deepcopy(self.env)
        fake_env.agents = [fake_env.agents[agentID]]
        # if fake_env.agents[0].position is not None:
        fake_env.agents[0].position = start_pos
        fake_env.agents[0].direction = orientation
        fake_env.agents[0].handle = 0
        # else:
        fake_env.agents[0].initial_position = start_pos
        fake_env.agents[0].initial_direction = orientation

        distance_map = DistanceMap(env_width=self.env.rail.width, env_height=self.env.rail.height,
                                   agents=fake_env.agents)
        distance_map.reset(fake_env.agents, self.env.rail)
        path = get_shortest_paths(distance_map, agent_handle=0)
        return path[0]

    def get(self, handle=0):
        """
        param-handle: agent id
        return NewObs and GlobalObs
        New obs is a 3*6 tuple observation for RL
        """
        isStoppingPoint = False
        my_pos = self.env.agents[handle].position if self.env.agents[handle].position is not None else self.env.agents[
            handle].initial_position

        if self.env.dones[handle]:
            return [0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, -1.0, 0.0, 0.0, 0.0, 0.0]

        my_direction = self.env.agents[handle].direction if self.env.agents[handle].direction is not None else \
            self.env.agents[
                handle].initial_direction
        valid_act_pos_pair = self._get_next_valid_position(my_pos, my_direction)
        obs = []
        all_handles = [i for i in range(len(self.env.agents))]

        others = []
        for id in all_handles:
            if id != handle:
                if self.env.agents[id].position is None:
                    if self.env.dones[id] is True:
                        otherspos = self.env.agents[id].target
                        othersdirections = self.env.agents[id].old_direction
                    else:
                        otherspos = self.env.agents[id].initial_position
                        othersdirections = self.env.agents[id].initial_direction
                else:  # position not None
                    otherspos = self.env.agents[id].position
                    othersdirections = self.env.agents[id].direction
                others.append([otherspos, othersdirections])

        if self.StateClassifier(my_pos, my_direction) in [1, 4]:
            for action, next_pos in valid_act_pos_pair.items():
                if next_pos[1] is not None:
                    my_pos = next_pos[1]
                    my_direction = next_pos[0]
                    isStoppingPoint = True
                    break

        valid_act_pos_pair = self._get_next_valid_position(my_pos, my_direction)
        for action, next_pos in valid_act_pos_pair.items():
            if next_pos[1] is not None:  # valid action
                # print(next_pos)
                has_solution, obs_1_direction = self._count_path_block(next_pos[1], next_pos[0], handle, my_pos,
                                                                       my_direction)
                if has_solution:

                    obs_1_direction[0] += 1  # compensate 1 cell for current pos

                    if isStoppingPoint:
                        obs_1_direction[0] += 1  # compensate 1 cell for stopping point bc it receives future obs
                    obs_1_direction.insert(0, True)
                    obs.append(obs_1_direction)
                else:
                    print(my_pos, my_direction, self.StateClassifier(my_pos, my_direction), "still fail!")
                    obs_1_direction.insert(0, True)

                    obs.append(obs_1_direction)
            else:

                obs.append([False, -1, 0, 0, 0, 0])
        obs = np.array(obs, dtype=float)
        for i in range(3):
            obs[i, 1] = obs[i, 1]/max(obs[:, 1]) if obs[i, 1] > 0 else -1

        obs = np.reshape(obs, (1, -1))
        obs = obs.squeeze()
        obs = obs.tolist()
        return obs

    def StateClassifier(self, agent_pos, agent_dir):
        """
        returns 0 : No decision point
        returns 1 : Stopping point (Decision at next cell)
        returns 2 : At decision point currently (More than 1 available transition)
        returns 3 : MUST STOP point - Agent Ahead
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
                        my_pos = (-3, -3)
                    else:
                        my_pos = self.env.agents[k].initial_position
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

    def _get_next_valid_position(self, my_pos, my_direction):
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

        state = self.StateClassifier(my_pos, my_direction)
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
        # else:
        #     print(avb_moves)
        #     print(state)
        #     print('my_pos, my_direction:', my_pos, my_direction)
        #     raise RuntimeError('Messed up, needs more debugging')
        return next_positions

    def disttonextjunction(self, full_path, old_pos, old_heading):
        """
        Returns 1 : If at Stopping Point 
        Returns Distance to Junction (Greater than 1) : If at No Decision Point
        Returns 0 : If at Junction currently 
        """
        state = self.StateClassifier(old_pos, old_heading)
        sumcell = 0  # How many possible transitions at next cell
        for j in range(0, 4):
            new_avb_moves = self.env.rail.get_transitions(*old_pos, j)
            sumcell += sum(new_avb_moves)
        if sumcell > 2:
            return 0
        elif state in [1, 4]:
            return 1
        elif state in [0, 3]:
            distance = 0
            for i in range(len(full_path) - 1):
                distance += 1
                statecell = self.StateClassifier(full_path[i].position, full_path[i].direction)
                if statecell in [1, 4]:
                    return distance + 1
                elif statecell == 2:
                    return distance
            return distance  # No decision junction found , means straight path to goal
        #        elif state== 2 :
        #            distance =0
        #            for i in range (len(full_path) - 1):
        #                distance +=1
        #                statecell = self.StateClassifier(full_path[i].position , full_path[i].direction)
        #                sumcell = 0  # How many possible transitions at next cell
        #                for j in range(0, 4):
        #                    new_avb_moves = self.env.rail.get_transitions(*full_path[i].position , (full_path[i].direction+j)%4)
        #                    sumcell += sum(new_avb_moves)
        #                if (sumcell > 2) :
        #                    return distance
        #            return distance # No decision junction found , means straight path to goal
        else:
            print("Some error in disttonextjunction")
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

        elif (self.StateClassifier(actual_pos, actual_dir) in [0, 3] and action == 1):
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
        elif (self.StateClassifier(actual_pos, actual_dir) in [0, 3] and action in [0, 2]):
            return 0
        else:
            return 0

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

        assert self.env.rail.grid[(start_pos[0], start_pos[1])] != 0, "start position " + str(start_pos) \
                                                                      + " is not valid in the map!"

        def isJuntion(agent_pos):
            trans = []
            for direction in range(0, 4):
                trans.append(sum(self.env.rail.get_transitions(*agent_pos, direction)))
            return sum(trans) > 2

        count_first_block = 0
        count_all_block = 0
        count_junction_block = 0
        count_junction = 0
        count_waiting = 0
        crash_buffer = []

        actual_pos = self.env.agents[agentID].position if self.env.agents[agentID].position is not None else \
            self.env.agents[
                agentID].initial_position
        actual_direction = self.env.agents[agentID].direction if self.env.agents[agentID].direction is not None else \
            self.env.agents[
                agentID].initial_direction

        all_handle = [i for i in range(len(self.env.agents))]

        others_pos = []
        others_directions = []
        others_moving = []

        for id in all_handle:
            if id != agentID:
                if self.env.agents[id].position is None:
                    if self.env.dones[id] is True:
                        others_pos.append((-3, -3))
                        others_directions.append(0)
                        others_moving.append(self.env.agents[id].moving)
                    else:
                        others_pos.append(self.env.agents[id].initial_position)
                        others_directions.append(self.env.agents[id].initial_direction)
                        others_moving.append(self.env.agents[id].moving)
                else:  # position not None
                    others_pos.append(self.env.agents[id].position)
                    others_directions.append(self.env.agents[id].direction)
                    others_moving.append(self.env.agents[id].moving)

        visit_first_junction = False
        visit_first_junction_waiting = False
        visit_first_stopping = False

        # start to compute quantities ---------------------------------------
        full_path = self._cheat_expert(start_pos, heading, agentID)

        if full_path is None:
            return False, [-1,  # single agent path length
                           min(1, count_first_block),  # num_block within the first junction
                           count_all_block / num_agent,  # num_block all along the path
                           count_junction_block,  # num_block standing on junctions
                           -1]

        distance_to_next_junction = self.disttonextjunction(full_path, actual_pos, actual_direction)
        for num_step in range(len(full_path) - 1):
            checking_cell = full_path[num_step].position  # checking cell is the cell we want to check blocking
            checking_cell_dir = full_path[num_step].direction
            if isJuntion(checking_cell):
                count_junction += 1
            if self.StateClassifier(checking_cell, checking_cell_dir) == 2:
                visit_first_junction_waiting = True
            if (checking_cell in others_pos) is True:
                idx = others_pos.index(checking_cell)
                crash_buffer.append(checking_cell)
                # test if there is other agents stepping on the stopping point
                for direction in range(4):
                    if self.StateClassifier(checking_cell, direction) in [1, 4] \
                            and not visit_first_stopping:
                        if checking_cell in others_pos:
                            visit_first_stopping = True

                if others_directions[
                    idx] == checking_cell_dir and others_moving == 0 and not visit_first_junction_waiting:
                    # same heading, not moving, so waiting
                    count_waiting += 1

                elif (others_directions[idx] + checking_cell_dir) % 2 == 0:
                    # opposite heading, so blocking
                    count_all_block += 1
                    if self.StateClassifier(checking_cell, checking_cell_dir) == 2:
                        count_junction_block += 1

                    if not visit_first_junction:
                        count_first_block += 1

                else:  # neither same direction or opposite direction, meaning that an agent staying at a junction
                    # but that is not a junction for the current direction (non-decision point)
                    count_all_block += 1
                    count_junction_block += 1
                    if not visit_first_junction:
                        count_first_block += 1

            if self.StateClassifier(checking_cell, checking_cell_dir) == 2:
                visit_first_junction = True

        return True, [len(full_path),  # single agent path length
                      min(1, count_first_block),  # num_block within the first junction
                      count_all_block/num_agent,  # num_block all along the path
                      count_junction_block/count_junction if count_junction else 0,  # num_block standing on junctions
                      distance_to_next_junction/len(full_path)
                      ]


if __name__ == '__main__':
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_generators import rail_from_manual_specifications_generator
    from flatland.envs.schedule_generators import random_schedule_generator
    from flatland.envs.rail_generators import random_rail_generator

    from flatland.utils.rendertools import RenderTool

    specs = [[(7, 0), (0, 0), (7, 0), (7, 0), (0, 0), (0, 0)],
             [(6, 270), (1, 90), (5, 0), (5, 0), (8, 90), (0, 0)],
             [(1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0)],
             [(6, 270), (1, 90), (4, 0), (3, 0), (2, 90), (7, 90)],
             [(1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0)],
             [(8, 270), (1, 90), (2, 90), (9, 90), (0, 0), (0, 0)]]
    num_agent = 8


    # Env = RailEnv(width=10, height=10,
    # rail_generator=random_rail_generator(seed=4),
    # (nr_start_goal=9,nr_extra=1,min_dist=5,max_dist=99999,seed=8),
    # schedule_generator=random_schedule_generator(seed=1),
    # obs_builder_object=StateMaskingObs(),
    # number_of_agents=num_agent)
    # env_renderer = RenderTool(Env)
    def Complex_params():
        grid_width = 10  # min(int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )),

        # int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )))
        grid_height = 10  # min(int(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1])),

        # nt(np.random.uniform(ENVIRONMENT_SIZE[0], ENVIRONMENT_SIZE[1] )))
        rnd_start_goal = 8 + np.random.randint(0,
                                               3)  # int(np.random.uniform(num_workers, num_workers+1+episode_difficulty ))
        # int(np.random.uniform( num_workers , min(grid_width,grid_height))),

        rnd_extra = 6  # int(np.random.uniform(0 , 1+2*episode_difficulty ))
        # int(np.random.uniform( 0 , min(grid_width,grid_height))))
        rnd_min_dist = int(0.75 * min(grid_height,
                                      grid_width))  # int(np.random.uniform( episode_difficulty , 4+2*episode_difficulty ))
        rnd_max_dist = 99999  # int(np.random.uniform(3+episode_difficulty, 6+2*episode_difficulty))
        rnd_seed = 1

        return grid_width, grid_height, rnd_start_goal, rnd_extra, rnd_min_dist, rnd_max_dist, rnd_seed


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
                      obs_builder_object=StateMaskingObs(),
                      number_of_agents=8)
    gameEnv.reset(regenerate_rail=False, regenerate_schedule=True)
    env_renderer = RenderTool(gameEnv)

    # print("agent0_init:", Env.agents[0].initial_position, Env.agents[0].initial_direction)
    # print("agent1_init:", Env.agents[1].initial_position, Env.agents[1].initial_direction)
    # print('env done!')
    count = 1
    observations = []
    j = 0
    for t in range(500):
        gameEnv.reset(regenerate_rail=True, regenerate_schedule=True)
        print('done:', t)
        for i in range(50):
            # env_renderer.render_env(show=True, frames=True, show_observations=True)
            obs, _, _, _ = gameEnv.step({0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2})
            print(obs)
            env_renderer.render_env(show=True, frames=True, show_observations=True)
            input()
            # print("agent0:", gameEnv.agents[2].position, " ", gameEnv.agents[2].direction)
            observations.append(obs)
            # print("agent1:", Env.agents[1].position, " ", Env.agents[1].direction)
        # print(obs[0])
        # print(obs[1])
        # env_renderer.render_env(show=True, frames=True, show_observations=False)
        # a=int(input('action: '))
        # obs = gameEnv.reset(True,True)
        # if j>=3 :
        #    print(observations[j-3][2]) 
        # j+=1
        # input()

        # time.sleep(0.5)
        # print('done ' , count)
        # count+=1
