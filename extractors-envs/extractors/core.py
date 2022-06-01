import numpy as np

# state of agents (including communication and internal/mental state)
class AgentState(object):
    def __init__(self):
        self.anger = None
        self.fear = None
        self.health = None

# action of the agent
class Action(object):
    def __init__(self):
        self.act = None
    
    @ property
    def agent_action_spaces(self):
        action_spaces = {}
        action_spaces[0] = "Raise hand"
        action_spaces[1] = "Give away password"
        action_spaces[2] = "Try to disarm the criminal"
        return action_spaces
    
    @ property
    def adversary_action_spaces(self):
        action_spaces = {}
        action_spaces[0] = "Use gun to point at employee"
        action_spaces[1] = "Use gun to point at customer"
        action_spaces[2] = "Shoot at ceiling"
        action_spaces[3] = "Ask for password"
        action_spaces[4] = "Kill employee"
        action_spaces[5] = "Kill customer"
        action_spaces[6] = "Run away"
        action_spaces[7] = "Hit employee"
        action_spaces[8] = "Hit customer"
        return action_spaces

    @ property
    def neutral_action_spaces(self):
        action_spaces = {}
        action_spaces[0] = "Raise hand"
        action_spaces[1] = "Run away"
        action_spaces[2] = "Try to disarm the criminal"
        return action_spaces

# properties and state of physical world entity
class Agent(object):
    def __init__(self):
        # name 
        self.name = ''

        # state
        self.state = AgentState()
        # action
        self.action = Action()

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents (can change at execution-time!)
        self.agents = []

    # update state of the world
    # 修改各agent的状态
    def step(self):
        self.update_states()

    def update_states(self):
        for _, agent in enumerate(self.agents):
            if agent.adversary:
                if agent.action == 0:
                    agent.state.anger -= 1
                    agent.state.fear += 0
                    agent.state.health += 0
                elif agent.action == 1:
                    agent.state.anger -= 1
                    agent.state.fear += 0
                    agent.state.health += 0
                elif agent.action == 2:
                    agent.state.anger -= 1
                    agent.state.fear += 0
                    agent.state.health += 0
                elif agent.action == 3:
                    agent.state.anger += 1
                    agent.state.fear += 0
                    agent.state.health += 0
                elif agent.action == 4:
                    agent.state.anger -= 10
                    agent.state.fear += 0
                    agent.state.health += 0
                elif agent.action == 5:
                    agent.state.anger -= 10
                    agent.state.fear += 0
                    agent.state.health += 0
                elif agent.action == 6:
                    agent.state.anger -= 10
                    agent.state.fear += 0
                    agent.state.health -= 5
                elif agent.action == 7:
                    agent.state.anger -= 2
                    agent.state.fear += 0
                    agent.state.health -= 5
                elif agent.action == 8:
                    agent.state.anger -= 2
                    agent.state.fear += 0
                    agent.state.health -= 5
                else:
                    raise NotImplementedError
            elif agent.neutral:
                if agent.action == 0:
                    agent.state.anger -= 0
                    agent.state.fear -= 1
                    agent.state.health += 0
                elif agent.action == 1:
                    agent.state.anger += 0
                    agent.state.fear -= 10
                    agent.state.health = -201 # 设定为特定值
                elif agent.action == 2:
                    agent.state.anger += 0
                    agent.state.fear += 1
                    agent.state.health -= 5
                else:
                    raise NotImplementedError
            else:
                if agent.action == 0:
                    agent.state.anger -= 0
                    agent.state.fear -= 1
                    agent.state.health += 0
                elif agent.action == 1:
                    agent.state.anger += 0
                    agent.state.fear -= 5
                    agent.state.health += 0
                elif agent.action == 2:
                    agent.state.anger += 0
                    agent.state.fear += 1
                    agent.state.health -= 5
                else:
                    raise NotImplementedError
            self.update_others_states(agent)
    
    def update_others_states(self, agent):
        for _, other in enumerate(self.agents):
            if agent == other: continue
            if agent.adversary:
                if other.neutral:
                    if agent.action == 0:
                        other.state.anger += 0
                        other.state.fear += 1
                        other.state.health += 0
                    elif agent.action == 1:
                        other.state.anger += 0
                        other.state.fear += 2
                        other.state.health += 0
                    elif agent.action == 2:
                        other.state.anger += 0
                        other.state.fear += 1
                        other.state.health += 0
                    elif agent.action == 3:
                        other.state.anger += 0
                        other.state.fear += 0
                        other.state.health += 0
                    elif agent.action == 4:
                        other.state.anger += 0
                        other.state.fear += 10
                        other.state.health += 0
                    elif agent.action == 5:
                        other.state.anger += 0
                        other.state.fear += 0
                        other.state.health = -202 # 设定为特殊值
                    elif agent.action == 6:
                        other.state.anger += 0
                        other.state.fear -= 10
                        other.state.health += 0
                    elif agent.action == 7:
                        other.state.anger += 0
                        other.state.fear += 1
                        other.state.health += 0
                    elif agent.action == 8:
                        other.state.anger += 0
                        other.state.fear += 2
                        other.state.health -= 5
                else:
                        if agent.action == 0:
                            other.state.anger += 0
                            other.state.fear += 2
                            other.state.health += 0
                        elif agent.action == 1:
                            other.state.anger += 0
                            other.state.fear += 1
                            other.state.health += 0
                        elif agent.action == 2:
                            other.state.anger += 0
                            other.state.fear += 1
                            other.state.health += 0
                        elif agent.action == 3:
                            other.state.anger += 0
                            other.state.fear += -1
                            other.state.health += 0
                        elif agent.action == 4:
                            other.state.anger += 0
                            other.state.fear += 0
                            other.state.health -= 100
                        elif agent.action == 5:
                            other.state.anger += 0
                            other.state.fear += 10
                            other.state.health += 0
                        elif agent.action == 6:
                            other.state.anger += 0
                            other.state.fear -= 10
                            other.state.health += 0
                        elif agent.action == 7:
                            other.state.anger += 0
                            other.state.fear += 2
                            other.state.health -= 5
                        elif agent.action == 8:
                            other.state.anger += 0
                            other.state.fear += 1
                            other.state.health += 0
            elif agent.neutral:
                if other.adversary:
                    if agent.action == 0:
                        other.state.anger -= 1
                        other.state.fear += 0
                        other.state.health += 0
                    elif agent.action == 1:
                        other.state.anger = 0
                        other.state.fear += 0
                        other.state.health += 0
                    else:
                        other.state.anger += 5
                        other.state.fear += 0
                        other.state.health += 0
                else:
                    if agent.action == 0:
                        other.state.anger += 0
                        other.state.fear += 0
                        other.state.health += 0
                    elif agent.action == 1:
                        other.state.anger += 0
                        other.state.fear += 0
                        other.state.health += 0
                    else:
                        other.state.anger += 0
                        other.state.fear += 0
                        other.state.health += 0
            else:
                if other.adversary:
                    if agent.action == 0:
                        other.state.anger -= 1
                        other.state.fear += 0
                        other.state.health += 0
                    elif agent.action == 1:
                        other.state.anger -= 10
                        other.state.fear += 0
                        other.state.health += 0
                    else:
                        other.state.anger += 5
                        other.state.fear += 0
                        other.state.health += 0
                else:
                    if agent.action == 0:
                        other.state.anger += 0
                        other.state.fear += 0
                        other.state.health += 0
                    elif agent.action == 1:
                        other.state.anger += 0
                        other.state.fear += 0
                        other.state.health += 0
                    else:
                        other.state.anger += 0
                        other.state.fear += 0
                        other.state.health += 0