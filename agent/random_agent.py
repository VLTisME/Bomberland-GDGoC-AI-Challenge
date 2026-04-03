import random

class RandomAgent:
    team_id = "RandomAgent"
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
    
    def act(self, obs):
        return random.randint(0, 5)