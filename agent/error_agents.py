# simulates agents that violate the rules

class TimeOutAgent:
    team_id = "TimeOutAgent"
    def __init__(self):
        pass
    
    def act(self, obs):
        while True:
            pass
        
class InvalidActionAgent:
    team_id = "InvalidActionAgent"
    def __init__(self):
        pass
    
    def act(self, obs):
        return 6

class NoBombAgent:
    team_id = "NoBombAgent"
    def __init__(self):
        pass
    
    def act(self, obs):
        return 5

# class WrongDeviceAgent:
#     def __init__(self):
#         pass
    
#     def act(self, obs):
#         import torch
#         x = torch.tensor([1.0], device='cuda')
#         return 0