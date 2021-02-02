from flatland.evaluators.client import FlatlandRemoteClient
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
import numpy as np
import time
import heapq
import copy

from queue import Queue

EPS = 0.0001
INFINITY = 1000000007
SAFE_LAYER = 4  # not used anymore
START_TIME_LIMIT = 60
REPLAN_LIMIT = 250
MAX_TIME_ONE = 25


#####################################################################
# Instantiate a Remote Client
#####################################################################
# remote_client = FlatlandRemoteClient()

#####################################################################
# Define your custom controller
#
# which can take an observation, and the number of agents and 
# compute the necessary action for this step for all (or even some)
# of the agents
#####################################################################

def getStepsToExitCell(v):
    return int(1 / v + EPS)


class Agent:  # agent general and instant information
    def __init__(self, agentId, env):
        self.start_i = -1  # start (from previous reset)
        self.start_j = -1
        self.fin_i = -1  # finish
        self.fin_j = -1
        self.current_pos = 0  # current position of a personal plan
        self.actions = []  # personal plan
        self.obligations = None
        self.agentId = agentId  # ID (with the same order, as flatland has)
        self.spawned = False  # spawned = entered the simulation, also is True when the agent finished
        self.malfunctioning = False  # helps the controller recognize the first step of a malunction

    def getAgent(self, env):  # read the agent info from RailEnv structure to more solution-friendly format
        if (env.agents[self.agentId].position == None):
            self.start_i = env.agents[self.agentId].initial_position[0]
            self.start_j = env.agents[self.agentId].initial_position[1]
        else:
            self.start_i = env.agents[self.agentId].position[0]  # read start, finish, direction from system
            self.start_j = env.agents[self.agentId].position[1]

        self.fin_i = env.agents[self.agentId].target[0]
        self.fin_j = env.agents[self.agentId].target[1]

        self.dir = env.agents[self.agentId].direction
        self.stepsToExitCell = getStepsToExitCell(
            env.agents[self.agentId].speed_data["speed"])  # number of steps required to
        # move to next cell


class Agents:  # this class contais the information about all agents
    def __init__(self):
        self.allAgents = []  # array of agents
        self.size = 0

    def getAgents(self, env):
        self.size = env.get_num_agents()
        if (self.allAgents == []):
            for ind in range(self.size):
                self.allAgents.append(Agent(ind, env))
                self.allAgents[ind].getAgent(env)
        else:
            for ind in range(self.size):
                self.allAgents[ind].getAgent(env)

    def reset_agent(self, number, new_i, new_j):
        self.allAgents[number].actions = []
        self.allAgents[number].current_pos = 0
        self.allAgents[number].start_i = new_i
        self.allAgents[number].start_j = new_j


class Node:  # low-level code: Node class of a search process
    # this structure is typically used in A* algorithms
    def __init__(self, i, j, dir):
        self.i = i
        self.j = j
        self.t = 0
        self.f = 0
        self.g = 0
        self.h = 0
        self.dir = dir
        self.spawned = False
        self.parent = None

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir and self.spawned == other.spawned

    def __ne__(self, other):
        return not (
                self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir and self.spawned == other.spawned)

    def __lt__(self, other):
        if (self.f == other.f):
            if (self.g == other.g):
                if (self.spawned == other.spawned):
                    if (self.i == other.i):
                        if (self.j == other.j):
                            return self.dir < other.dir
                        return self.j < other.j
                    return self.i < other.i
                return self.spawned < other.spawned
            return self.g > other.g
        return self.f < other.f

    def __hash__(self):
        return hash((self.i, self.j, self.t, self.dir, self.spawned))


class Node_h:
    def __init__(self, x, y, dir, time):
        self.i = x
        self.j = y
        self.dir = dir
        self.time = time


class Global_H:
    def __init__(self, env):
        self.database = dict()
        self.env = env
        for ind in range(env.get_num_agents()):
            self.start_agent(ind)

    def get_neighbors(self, curNode):  # actually, the same procedure with Isearch class
        available = self.env.rail.get_transitions(*[curNode.i, curNode.j], curNode.dir)
        answer = []
        if (available[0] == True):
            answer.append(Node_h(curNode.i - 1, curNode.j, 0, curNode.time + 1))
        if (available[1] == True):
            answer.append(Node_h(curNode.i, curNode.j + 1, 1, curNode.time + 1))
        if (available[2] == True):
            answer.append(Node_h(curNode.i + 1, curNode.j, 2, curNode.time + 1))
        if (available[3] == True):
            answer.append(Node_h(curNode.i, curNode.j - 1, 3, curNode.time + 1))
        return answer

    def get_dir(self, position):  # get a direction (orientation) of a trainstation
        answer = []
        for dest in range(4):
            available = self.env.rail.get_transitions(*position, dest)
            if (sum(available) > 0):
                answer.append(dest)
        return answer

    def start_agent(self, number):
        correct_dir = self.get_dir(self.env.agents[number].target)
        queue = Queue()
        for dest in correct_dir:
            start = Node_h(self.env.agents[number].target[0], self.env.agents[number].target[1], dest, 0)
            queue.put(start)
        while (not queue.empty()):
            current = queue.get()
            candidates = self.get_neighbors(current)
            for node in candidates:
                if ((number, current.i, current.j, (node.dir + 2) % 4) not in self.database):
                    self.database[(number, current.i, current.j, (node.dir + 2) % 4)] = current.time
                    queue.put(node)

    def get_heuristic(self, agentId, x, y, dir):  # output
        if (agentId, x, y, dir) in self.database:
            return self.database[(agentId, x, y, dir)]
        else:
            return INFINITY


class ISearch:  # this is the main path finding class which contains the information about agent paths, rservations,
    # performs the start planning, rebuilds paths after malfunction occurences etc.
    def __init__(self, env, heuristic):
        self.lppath = []  # path of low-level nodes
        for ind in range(env.get_num_agents()):
            self.lppath.append([])
        self.reservations = dict()  # reservated cells
        self.maxTime = 5000
        self.additional_reserve = 6  # magic constant
        self.heuristic = heuristic

    def startallAgents(self, env, control_agent, order, time_limit,
                       current_step):  # preparations and performing A* in the first turn

        # path exists is a feedback for high-level class
        path_exists = []
        for i in range(env.get_num_agents()):
            path_exists.append(False)

        start_time = time.time()
        for i in range(len(order)):  # execute A* with every single agent
            agent = control_agent.allAgents[order[i]]
            if agent.spawned:
                path_exists[agent.agentId] = True
                continue
            path_exists[agent.agentId] = self.startSearch(agent, env, current_step, self.heuristic)
            if (int(
                    time.time()) - start_time > time_limit):  # sometimes start planning takes too much time, so it is reasonable to
                # put it off (for slow agents) for a later time
                break
        return path_exists

    def checkReservation(self, i, j, t):  # low-level code: reservations info
        return (t, i, j) in self.reservations

    def get_occupator(self, i, j, t):
        if (t, i, j) in self.reservations:
            return self.reservations[(t, i, j)]
        else:
            return None

    def startSearch(self, agent, env, current_step, heuristic):

        # start of A* algorithm - usual implementation
        startNode = agent.obligations
        finNode = Node(agent.fin_i, agent.fin_j, agent.dir)

        openHeap = []
        openCopy = set()

        pathFound = False

        heapq.heappush(openHeap, startNode)
        start_search_time = time.time()

        while (not pathFound) and len(openHeap) > 0:

            curNode = heapq.heappop(openHeap)

            if (curNode.t >= self.maxTime or curNode.h == INFINITY or time.time() - start_search_time >= MAX_TIME_ONE):
                break

            if (curNode.i == finNode.i and curNode.j == finNode.j):
                finNode = curNode
                pathFound = True
                break

            else:
                successors = self.findSuccessors(curNode, agent, env, heuristic)
                for i in range(len(successors)):
                    scNode = successors[i]

                    foundInOpen = False
                    if (scNode in openCopy):
                        foundInOpen = True

                    scNode.parent = curNode

                    if (foundInOpen == False):
                        heapq.heappush(openHeap, scNode)
                        openCopy.add(scNode)

        if pathFound:
            self.makePrimaryPath(finNode, startNode, agent)
            self.makeFlatlandFriendlyPath(agent)
            return True
        else:
            return False

    def correct_point(self, scNode, agent):  # is comparable with FLATland
        for step in range(agent.stepsToExitCell):
            if self.checkReservation(scNode.i, scNode.j, scNode.t + step) and self.get_occupator(scNode.i, scNode.j,
                                                                                                 scNode.t + step) != agent.agentId:
                return False
        if self.checkReservation(scNode.i, scNode.j, scNode.t + agent.stepsToExitCell):
            other_number = self.get_occupator(scNode.i, scNode.j, scNode.t + agent.stepsToExitCell)
            if other_number < agent.agentId:
                return False
        if self.checkReservation(scNode.i, scNode.j, scNode.t - 1):
            other_number = self.get_occupator(scNode.i, scNode.j, scNode.t - 1)
            if other_number > agent.agentId:
                return False
        return True

    def findSuccessors(self, curNode, agent, env,
                       heuristic):  # find neighbors of current cell, which we are able to visit
        position = [curNode.i, curNode.j]
        available = env.rail.get_transitions(*position, curNode.dir)
        inter_answer = []
        if (available[0] == True):
            inter_answer.append(Node(curNode.i - 1, curNode.j, 0))
        if (available[1] == True):
            inter_answer.append(Node(curNode.i, curNode.j + 1, 1))
        if (available[2] == True):
            inter_answer.append(Node(curNode.i + 1, curNode.j, 2))
        if (available[3] == True):
            inter_answer.append(Node(curNode.i, curNode.j - 1, 3))
        inter_answer.append(Node(curNode.i, curNode.j, curNode.dir))
        successors = []

        if (
                curNode.spawned == False):  # if the agents is not spawned there are two opportunities - to remain unspawned and to enter the simulation
            not_spawned = Node(curNode.i, curNode.j, curNode.dir)
            not_spawned.g = curNode.g + 1
            not_spawned.t = curNode.t + 1
            not_spawned.h = curNode.h
            not_spawned.f = curNode.f + 1
            not_spawned.spawned = False
            successors.append(not_spawned)

            spawned_on_this_turn = copy.deepcopy(not_spawned)
            spawned_on_this_turn.spawned = True
            spawned_on_this_turn.h -= 1
            spawned_on_this_turn.f -= 1
            if (self.correct_point(spawned_on_this_turn, agent) == True):
                successors.append(spawned_on_this_turn)
            return successors

        for scNode in inter_answer:
            scNode.h = heuristic.get_heuristic(agent.agentId, scNode.i, scNode.j, scNode.dir) * agent.stepsToExitCell
            scNode.spawned = True

            if scNode.i == curNode.i and scNode.j == curNode.j:
                scNode.t = curNode.t + 1
                scNode.g = curNode.g + 1
            else:
                scNode.t = curNode.t + agent.stepsToExitCell
                scNode.g = curNode.g + agent.stepsToExitCell
            scNode.f = scNode.g + scNode.h

            if not self.correct_point(scNode, agent):
                continue
            successors.append(scNode)
        return successors

    def delete_path(self, number):  # delete path except of some start steps
        for ind in range(1, len(self.lppath[number])):
            curNode = self.lppath[number][ind]
            for step in range(path_finder.control_agent.allAgents[number].stepsToExitCell):
                if (curNode.t + step, curNode.i, curNode.j) in self.reservations and self.reservations[
                    (curNode.t + step, curNode.i, curNode.j)] == number:
                    del self.reservations[(curNode.t + step, curNode.i, curNode.j)]
        self.lppath[number] = []

    def need_enter(self, scNode, agent,
                   first_agent):  # according to strange flatland rules, some mistakes (when agent tries to enter occupied cell)
        # are unavoidable, so we need to track these cases
        if self.checkReservation(agent.obligations.i, agent.obligations.j, agent.obligations.t):
            other_number = self.get_occupator(agent.obligations.i, agent.obligations.j, agent.obligations.t)
            if other_number != agent.agentId and (other_number < agent.agentId or other_number == first_agent):
                return False
        if self.checkReservation(agent.obligations.i, agent.obligations.j,
                                 agent.obligations.t - 1) and self.get_occupator(agent.obligations.i,
                                                                                 agent.obligations.j,
                                                                                 agent.obligations.t - 1) > agent.agentId:
            return False
        return True

    def replan_agent(self, agent, env, current_step, calculated, start_replanning_time,
                     second_queue):  # re-builds path for current agent
        # this is the most sophisticated and
        # the worst part of a project
        self.delete_all(agent.agentId)
        self.lppath[agent.agentId] = []
        if (agent.spawned == False):
            for step in range(current_step, agent.obligations.t):
                agent.actions.append(4)
            second_queue.append(agent.agentId)
            return []
        passers_by = []

        # check if there are any agents, which walk through the malfunctioning or re-planned agent
        # while it is staying on the start point (according to current position in RailEnv)
        # if there are any, add them to queue [passers_by] add delete their path

        for step in range(current_step, agent.obligations.t):
            if self.checkReservation(agent.start_i, agent.start_j, step) and self.get_occupator(agent.start_i,
                                                                                                agent.start_j,
                                                                                                step) != agent.agentId:
                passers_by.append(self.get_occupator(agent.start_i, agent.start_j, step))
                self.delete_path(passers_by[-1])
            self.reservations[(step, agent.start_i, agent.start_j)] = agent.agentId
            agent.actions.append(4)

        if self.checkReservation(agent.start_i, agent.start_j, agent.obligations.t):
            other_number = self.get_occupator(agent.start_i, agent.start_j, agent.obligations.t)
            if other_number < agent.agentId:
                passers_by.append(other_number)
                self.delete_path(passers_by[-1])

        if self.checkReservation(agent.obligations.i, agent.obligations.j, agent.obligations.t):
            agent_this_turn = self.get_occupator(agent.obligations.i, agent.obligations.j, agent.obligations.t)
        else:
            agent_this_turn = -1

        if self.checkReservation(agent.obligations.i, agent.obligations.j, agent.obligations.t - 1):
            agent_prev_turn = self.get_occupator(agent.obligations.i, agent.obligations.j, agent.obligations.t - 1)
        else:
            agent_prev_turn = -2

        # sometimes the agent will automatically try to enter the occupied cell
        # here this case is tracked

        if agent_prev_turn == agent_this_turn or (agent_prev_turn > agent.agentId):
            while not self.need_enter(agent.obligations, agent, agent_this_turn):
                if self.checkReservation(agent.start_i, agent.start_j, agent.obligations.t) and self.get_occupator(
                        agent.start_i, agent.start_j, agent.obligations.t) != agent.agentId:
                    passers_by.append(self.get_occupator(agent.start_i, agent.start_j, agent.obligations.t))
                    self.delete_path(passers_by[-1])

                self.reservations[(agent.obligations.t, agent.start_i, agent.start_j)] = agent.agentId
                agent.actions.append(4)
                agent.obligations.t += 1

                if self.checkReservation(agent.start_i, agent.start_j, agent.obligations.t):
                    other_number = self.get_occupator(agent.start_i, agent.start_j, agent.obligations.t)
                    if other_number < agent.agentId:
                        passers_by.append(other_number)
                        self.delete_path(passers_by[-1])

        # sometimes the algorithm enters the infinite loop, so add a few steps of waiting to this agent to break that
        # unfortunatelly, this approach doesn`t solve the stability problem perfectly

        if (calculated[agent.agentId] >= 2):  # calculated parameter is a measure of
            for step in range(agent.obligations.t, agent.obligations.t + self.additional_reserve * (
                    calculated[agent.agentId] - 1) + agent.stepsToExitCell + 1):
                if self.checkReservation(agent.obligations.i, agent.obligations.j, step) and self.get_occupator(
                        agent.obligations.i, agent.obligations.j, step) != agent.agentId:
                    passers_by.append(self.get_occupator(agent.obligations.i, agent.obligations.j, step))
                    self.delete_path(passers_by[-1])
                self.reservations[(step, agent.obligations.i, agent.obligations.j)] = agent.agentId
            agent.obligations.t = agent.obligations.t + self.additional_reserve * (calculated[agent.agentId] - 1)
            for step in range(self.additional_reserve * (calculated[agent.agentId] - 1)):
                agent.actions.append(4)

        # check if the start point of the agent is correct (when the agents starts to make any decisions)

        if self.checkReservation(agent.obligations.i, agent.obligations.j, agent.obligations.t - 1):
            other_number = self.get_occupator(agent.obligations.i, agent.obligations.j, agent.obligations.t - 1)
            if other_number > agent.agentId:
                passers_by.append(other_number)
                self.delete_path(passers_by[-1])

        for step in range(agent.stepsToExitCell):
            if self.checkReservation(agent.obligations.i, agent.obligations.j,
                                     agent.obligations.t + step) and self.get_occupator(agent.obligations.i,
                                                                                        agent.obligations.j,
                                                                                        agent.obligations.t + step) != agent.agentId:
                passers_by.append(
                    self.get_occupator(agent.obligations.i, agent.obligations.j, agent.obligations.t + step))
                self.delete_path(passers_by[-1])

        if self.checkReservation(agent.obligations.i, agent.obligations.j, agent.obligations.t + agent.stepsToExitCell):
            other_number = self.get_occupator(agent.obligations.i, agent.obligations.j,
                                              agent.obligations.t + agent.stepsToExitCell)
            if other_number < agent.agentId:
                passers_by.append(other_number)
                self.delete_path(passers_by[-1])

        # if the start point is correct, but it still inpossible to build any path
        # then take the first agent, which goes through the start point and delete it`s path

        path_exists = self.startSearch(agent, env, current_step)
        while path_exists == False:
            if (time.time() - start_replanning_time >= REPLAN_LIMIT):
                break
            agent_dead = False
            for step in range(agent.obligations.t + agent.stepsToExitCell, self.maxTime):
                if self.checkReservation(agent.obligations.i, agent.obligations.j, step) and self.get_occupator(
                        agent.obligations.i, agent.obligations.j, step) != agent.agentId:
                    if len(passers_by) > 0 and self.get_occupator(agent.obligations.i, agent.obligations.j, step) == \
                            passers_by[-1]:
                        self.delete_all(passers_by[-1])
                    else:
                        passers_by.append(self.get_occupator(agent.obligations.i, agent.obligations.j, step))
                        self.delete_path(passers_by[-1])
                    calculated[agent.agentId] += 1
                    break
                if step == self.maxTime - 1:
                    agent_dead = True

            if agent_dead:

                for step in range(agent.obligations.t - 1, agent.obligations.t + agent.stepsToExitCell):
                    if self.checkReservation(agent.obligations.i, agent.obligations.j, step):
                        other_number = self.get_occupator(agent.obligations.i, agent.obligations.j, step)
                        if other_number != agent.agentId:
                            self.delete_all(other_number)

            if agent_dead and not path_exists:
                break

            path_exists = self.startSearch(agent, env, current_step)
        return passers_by  # these agents will be add to replanning queue

    def delete_all(self, number):  # delete whole path with all reservations
        to_delete = []
        for cell in self.reservations:
            if self.reservations[cell] == number:
                to_delete.append(cell)
        for cell in to_delete:
            del self.reservations[cell]

    def makePrimaryPath(self, curNode, startNode, agent):  # path of nodes

        wait_action = False
        while curNode != startNode:
            self.lppath[agent.agentId].append(curNode)
            if not wait_action:
                for step in range(agent.stepsToExitCell):
                    self.reservations[(curNode.t + step, curNode.i, curNode.j)] = agent.agentId
            elif curNode.spawned == True:
                self.reservations[(curNode.t, curNode.i, curNode.j)] = agent.agentId
            if curNode.i == curNode.parent.i and curNode.j == curNode.parent.j:
                wait_action = True
            else:
                wait_action = False
            curNode = curNode.parent

        self.lppath[agent.agentId].append(curNode)
        if not wait_action:
            for step in range(agent.stepsToExitCell):
                self.reservations[(curNode.t + step, curNode.i, curNode.j)] = agent.agentId
        elif curNode.spawned == True:
            self.reservations[(curNode.t, curNode.i, curNode.j)] = agent.agentId

        self.lppath[agent.agentId] = self.lppath[agent.agentId][::-1]

    def makeFlatlandFriendlyPath(self, agent):
        for ind in range(1, len(self.lppath[agent.agentId])):
            if (self.lppath[agent.agentId][ind].i == self.lppath[agent.agentId][ind - 1].i and
                    self.lppath[agent.agentId][ind].j == self.lppath[agent.agentId][ind - 1].j):
                if (self.lppath[agent.agentId][ind - 1].spawned == False and self.lppath[agent.agentId][
                    ind].spawned == True):
                    agent.actions.append(2)
                else:
                    agent.actions.append(4)
            elif abs(self.lppath[agent.agentId][ind].dir - self.lppath[agent.agentId][ind - 1].dir) % 2 == 0:
                for step in range(agent.stepsToExitCell):
                    agent.actions.append(2)
            elif ((self.lppath[agent.agentId][ind - 1].dir + 1) % 4 == self.lppath[agent.agentId][ind].dir):
                for step in range(agent.stepsToExitCell):
                    agent.actions.append(3)
            else:
                for step in range(agent.stepsToExitCell):
                    agent.actions.append(1)


def build_start_order(env,
                      heuristic):  # custom desine of start agents order, firstly sort by speed, then by distance to
    # finishes
    # it`s prooved that this gets minimum summary of path length (on average)
    # which is mostly suitable for challenge rules
    answer = []
    queue = []
    for speed_value in range(5):
        queue.append([])
    for ind in range(len(env.agents)):
        x1, y1 = env.agents[ind].initial_position
        x2, y2 = env.agents[ind].target
        potential = heuristic.get_heuristic(ind, x1, y1, env.agents[ind].direction)
        queue[getStepsToExitCell(env.agents[ind].speed_data['speed'])].append([potential, ind])
    for speed_value in range(1, 5):
        queue[speed_value].sort()
    for speed_value in range(1, 5):
        for ind in range(len(queue[speed_value])):
            answer.append(queue[speed_value][ind][1])

    return answer


class Solver:  # main class
    def __init__(self, env, heuristic):  # initialization of a new simulation
        self.env = env
        self.control_agent = Agents()
        self.control_agent.getAgents(env)
        self.answer_build = False
        self.search = ISearch(env, heuristic)
        self.current_step = 0
        self.maxStep = 8 * (env.width + env.height + 20)
        self.prev_action = [2] * self.env.get_num_agents()
        self.current_order = build_start_order(self.env, heuristic)
        self.overall_time = 0
        self.heuristic = heuristic
        if (self.env.height + self.env.width) // 2 <= 60:  # small map
            self.maxTime = 620
            REPLAN_LIMIT = 250
        elif (self.env.height + self.env.width) // 2 <= 100:  # medium map
            self.maxTime = 1000
            REPLAN_LIMIT = 300
        else:  # large map
            self.maxTime = 1300
            REPLAN_LIMIT = 380

    def make_obligation(self,
                        number):  # in fact this is a start Node (which the agent is obligated to reach before it starts to make any decisions)
        if (self.env.agents[number].position != None):
            start_i, start_j = self.env.agents[number].position
        else:
            start_i, start_j = self.env.agents[number].initial_position
        direction = self.env.agents[number].direction
        self.control_agent.allAgents[number].obligations = Node(start_i, start_j, self.env.agents[number].direction)
        agent = self.control_agent.allAgents[number]
        self.control_agent.allAgents[number].obligations.h = self.control_agent.allAgents[
                                                                 number].stepsToExitCell * self.heuristic.get_heuristic(
            agent.agentId, start_i, start_j, direction) + (not agent.spawned)
        self.control_agent.allAgents[number].obligations.spawned = agent.spawned
        self.control_agent.allAgents[number].obligations.f = self.control_agent.allAgents[number].obligations.h
        if (self.env.agents[number].speed_data['position_fraction'] == 0.0):
            self.control_agent.allAgents[number].obligations.t = self.current_step + \
                                                                 self.env.agents[number].malfunction_data['malfunction']
        else:
            current_direction = self.env.agents[number].direction
            if (self.prev_action[number] == 1):
                current_direction -= 1
            elif (self.prev_action[number] == 3):
                current_direction += 1
            current_direction %= 4
            self.control_agent.allAgents[number].obligations.dir = current_direction
            if (current_direction == 0 and self.control_agent.allAgents[number].obligations.i > 0):
                self.control_agent.allAgents[number].obligations.i -= 1
            elif (current_direction == 1 and self.control_agent.allAgents[number].obligations.j < self.env.width - 1):
                self.control_agent.allAgents[number].obligations.j += 1
            elif (current_direction == 2 and self.control_agent.allAgents[number].obligations.i < self.env.height - 1):
                self.control_agent.allAgents[number].obligations.i += 1
            elif (current_direction == 3 and self.control_agent.allAgents[number].obligations.j > 0):
                self.control_agent.allAgents[number].obligations.j -= 1
            remain = self.env.agents[number].malfunction_data['malfunction'] + max(int(
                (1 - self.env.agents[number].speed_data['position_fraction'] + EPS) /
                self.env.agents[number].speed_data['speed']), 1)
            self.control_agent.allAgents[number].obligations.t = self.current_step + remain

    def set_obligations(self):
        for ind in range(self.env.get_num_agents()):
            self.make_obligation(ind)
            if (self.env.agents[ind].position != None):
                self.control_agent.allAgents[ind].start_i, self.control_agent.allAgents[ind].start_j = self.env.agents[
                    ind].position
            else:
                self.control_agent.allAgents[ind].start_i, self.control_agent.allAgents[ind].start_j = self.env.agents[
                    ind].initial_position

    def build_on_the_start(self):  # prepare paths before the simulation begins
        self.set_obligations()
        path_exists = self.search.startallAgents(self.env, self.control_agent, self.current_order, START_TIME_LIMIT,
                                                 self.current_step)
        new_order = []
        for ind in range(len(self.current_order)):
            if (path_exists[self.current_order[ind]] == False):
                new_order.append(self.current_order[ind])
        self.current_order = copy.deepcopy(new_order)
        self.answer_build = True

    def build_medium(self):  # when the map is too big
        # start planning takes too much time
        # but slow agents can enter the simulation later
        # if we do that here we save a lot of time
        self.set_obligations()
        path_exists = self.search.startallAgents(self.env, self.control_agent, self.current_order, START_TIME_LIMIT * 3,
                                                 self.current_step)

    def print_step(self):  # send report to FLATland
        _action = {}
        for ind in range(self.env.get_num_agents()):
            agent = self.control_agent.allAgents[ind]
            position = self.env.agents[ind].position
            if agent.current_pos < len(agent.actions):
                if (agent.actions[agent.current_pos] != 4):
                    self.prev_action[ind] = agent.actions[agent.current_pos]
                _action[ind] = agent.actions[agent.current_pos]
                self.control_agent.allAgents[ind].current_pos += 1
        self.current_step += 1
        return _action

    def update_malfunctions(self):  # check if there are any malfunctions in this turn and re-planning control
        for ind in range(self.env.get_num_agents()):
            if (self.env.agents[ind].malfunction_data['malfunction'] > 1 and self.control_agent.allAgents[
                ind].malfunctioning == False):
                self.calculated = [0] * self.env.get_num_agents()
                replanning_queue = []  # malfunction cause some conflict between agents
                # re-build paths one-by-one starting with malfunctioning agent
                # any re-build can cause other conflicts, so keep
                # re-planning until there won`t be any conflicts (and all path will be valid)
                replanning_queue.append(ind)
                start_replanning_time = time.time()
                pos = 0
                second_queue = []  # queue of not spawned agents - we can build their paths later
                while pos < len(replanning_queue):
                    if time.time() - start_replanning_time >= REPLAN_LIMIT:
                        break
                    current = replanning_queue[pos]
                    if (self.env.agents[current].position == None):
                        if self.control_agent.allAgents[current].spawned == True:
                            pos += 1
                            continue
                        self.control_agent.reset_agent(current, self.env.agents[current].initial_position[0],
                                                       self.env.agents[current].initial_position[1])
                    else:
                        self.control_agent.reset_agent(current, self.env.agents[current].position[0],
                                                       self.env.agents[current].position[1])
                    self.make_obligation(current)
                    additional = self.search.replan_agent(self.control_agent.allAgents[current], self.env,
                                                          self.current_step, self.calculated, start_replanning_time,
                                                          second_queue)
                    for i in range(len(additional)):
                        replanning_queue.append(additional[i])
                    pos += 1
                for number in second_queue:
                    path_exists = self.search.startSearch(self.control_agent.allAgents[number], self.env,
                                                          self.current_step)
                malfunction_pos = self.env.agents[replanning_queue[0]].position
                if malfunction_pos == None:
                    malfunction_pos = self.env.agents[replanning_queue[0]].initial_position
                closest = []
                # take a few closest agents and re-plan them due to some rails were clear as the result of malfunction
                # it added a few percent to the challenge result
                for ind in range(self.env.get_num_agents()):
                    agent = self.control_agent.allAgents[ind]
                    malfunction_just_this_turn = (self.env.agents[ind].malfunction_data['malfunction'] > 1 and
                                                  self.control_agent.allAgents[ind].malfunctioning == False)
                    if ind not in replanning_queue and not (agent.spawned == True and self.env.agents[
                        ind].position == None) and not malfunction_just_this_turn:
                        pos = self.env.agents[ind].position
                        if pos == None:
                            pos = self.env.agents[ind].initial_position
                        closest.append([abs(malfunction_pos[0] - pos[0]) + abs(malfunction_pos[1] - pos[1]), ind])
                closest.sort()
                closest_number = 0
                if self.maxTime == 620:
                    closest_number = 6
                elif self.overall_time < self.maxTime // 2 and self.current_step > self.maxStep // 2:
                    closest_number = 4
                for ind in range(min(closest_number, len(closest))):
                    number = closest[ind][1]
                    agent = self.control_agent.allAgents[number]
                    agent.getAgent(self.env)
                    agent.current_pos = 0
                    agent.actions = []
                    self.search.lppath[number] = []
                    self.search.delete_all(number)
                    self.make_obligation(number)

                    if agent.spawned:
                        if self.search.checkReservation(agent.obligations.i, agent.obligations.j, agent.obligations.t):
                            agent_this_turn = self.search.get_occupator(agent.obligations.i, agent.obligations.j,
                                                                        agent.obligations.t)
                        else:
                            agent_this_turn = -1

                        if self.search.checkReservation(agent.obligations.i, agent.obligations.j,
                                                        agent.obligations.t - 1):
                            agent_prev_turn = self.search.get_occupator(agent.obligations.i, agent.obligations.j,
                                                                        agent.obligations.t - 1)
                        else:
                            agent_prev_turn = -2

                        if agent_prev_turn == agent_this_turn or (agent_prev_turn > agent.agentId):
                            while not self.search.need_enter(agent.obligations, agent, agent_this_turn):
                                self.search.reservations[
                                    (agent.obligations.t, agent.start_i, agent.start_j)] = agent.agentId
                                agent.obligations.t += 1

                    for step in range(self.current_step, agent.obligations.t):
                        if agent.spawned:
                            self.search.reservations[(step, agent.start_i, agent.start_j)] = number
                        agent.actions.append(4)
                    path_exists = self.search.startSearch(self.control_agent.allAgents[number], self.env,
                                                          self.current_step)
        # don`t forget to remember active malfunctions
        for ind in range(self.env.get_num_agents()):
            self.control_agent.allAgents[ind].malfunctioning = (
                    self.env.agents[ind].malfunction_data['malfunction'] > 1)


def my_controller(env, path_finder):
    for ind in range(path_finder.env.get_num_agents()):
        if path_finder.env.agents[ind].position != None:
            path_finder.control_agent.allAgents[ind].spawned = True
    if path_finder.answer_build == False:
        path_finder.build_on_the_start()
    if (path_finder.maxStep // 8) * 3 == path_finder.current_step:
        path_finder.build_medium()
    if path_finder.current_step != 0 and path_finder.overall_time <= path_finder.maxTime:
        path_finder.update_malfunctions()
    return path_finder.print_step()


def expert_get_next_action(env):
    heuristic = Global_H(env)
    path_finder = Solver(env, heuristic)
    return my_controller(env, path_finder)
