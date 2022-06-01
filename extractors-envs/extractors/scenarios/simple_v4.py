import numpy as np
from extractors.core import World, Agent
from extractors.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
       
        num_agents = 3
        world.num_agents = num_agents
        num_adversaries = 1
        num_neutrals = 1

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.adversary = True if i < num_adversaries else False
            agent.neutral = True if i >= num_adversaries and i < num_adversaries + num_neutrals else False
        
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.anger = 20.0
            agent.state.fear = 20.0
            agent.state.health = 100.0
        
    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary and not agent.neutral]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    
    # return all neutral agents
    def neutrals(self, world):
        return [agent for agent in world.agents if agent.neutral]

    def reward(self, agent, world):
        # TODO
        if agent.adversary:
            return self.adversary_reward(agent, world)
        elif agent.neutral:
            return self.neutral_reward(agent, world)
        return self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # TODO
        pos_rew = max(agent.state.health, 0) * 0.1 + 5.0

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        adv_rew = sum(a.state.fear * 0.1 for a in adversary_agents)

        return pos_rew + adv_rew

    def adversary_reward(self, agent, world):
        # TODO
        pos_rew = -agent.state.anger * 0.1

        good_agents = self.good_agents(world)
        good_rew = sum(a.state.fear * 0.1 for a in good_agents)

        # 每回合固定的负向奖励，用于催促快速达成目标
        neg_rew = -1.0
        return pos_rew + good_rew + neg_rew

    def neutral_reward(self, agent, world):
        pos_rew = 5.0 if agent.state.health > 0 else 0
        return pos_rew

    def observation(self, agent, world):
        # obs放自己的状态值和其他的agent的状态值
        other_state = []
        agent_state = np.array([agent.state.anger, agent.state.fear, agent.state.health])
        for other in world.agents:
            if other is agent: continue
            other_state.append(np.array([other.state.anger, other.state.fear, other.state.health]))
        return np.concatenate(agent_state + other_state)

    def done(self, agent, world):
        # 如果做了相应动作，角色/整个游戏结束
        # 如果状态值过高过低，有概率结束游戏
        '''
        | Done val | Done state                             |
        |----------|----------------------------------------|
        | 0        | false                                  |
        | 1        | Employee give away pw actively         |
        | 2        | Criminal run away actively             |
        | 3        | Criminal kill people actively          |
        | 4        | Employee give away pw coz of low hp    |
        | 5        | Criminal kill people coz of high anger |
        | 6        | Employee disarms criminal              |
        | 7        | Customer disarms criminal              |
        '''
        done = 0
        if agent.adversary:
            if agent.action == 4: # Kill people or Run away
                done = 3
            elif agent.action == 6:
                done = 2
            elif agent.state.anger > 80 and np.random.uniform() < 0.05:
                done = 5
        elif not agent.neutral:
            if agent.action == 1:
                done = 1
            elif agent.state.health < 40 and np.random.uniform() < 0.05:
                done = 4
            elif agent.action == 2 and np.random.uniform() < 0.02:
                done = 6
        else:
            if agent.action == 2 and np.random.uniform() < 0.02:
                done = 7
        return done
