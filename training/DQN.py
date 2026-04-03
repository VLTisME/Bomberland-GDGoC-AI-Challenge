import argparse
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from engine import *
from .reward import compute_reward
from .utils import plot_loss, plot_rewards, plot_win_rates, plot_moving_average
from agent import SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent

class ReplayBuffer:
    """Pre-allocated numpy circular buffer — sample() is pure array indexing, no Python objects."""
    def __init__(self, capacity: int, map_shape, aux_dim: int):
        self.capacity  = capacity
        self.pos       = 0
        self.size      = 0
        self.map_shape = tuple(map_shape)
        self.aux_dim   = int(aux_dim)
        self.map_states      = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.aux_states      = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.next_map_states = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.next_aux_states = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity,              dtype=np.int64)
        self.rewards     = np.zeros(capacity,              dtype=np.float32)
        self.dones       = np.zeros(capacity,              dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, map_state, aux_state, action, reward, next_map_state, next_aux_state, done):
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.map_states[self.pos]      = map_state
        self.aux_states[self.pos]      = aux_state
        self.next_map_states[self.pos] = next_map_state
        self.next_aux_states[self.pos] = next_aux_state
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = done

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.map_states[idx],
            self.aux_states[idx],
            self.next_map_states[idx],
            self.next_aux_states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
        )

class DQNModel(nn.Module):
    """
    Two-branch DQN:
      - Conv2D branch for spatial map/object channels
      - MLP branch for auxiliary scalar features
    """
    def __init__(self, map_shape, aux_dim, output_dim):
        super().__init__()
        c, h, w = map_shape
        self.map_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_dim = self.map_encoder(dummy).reshape(1, -1).size(1)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(conv_out_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, map_x, aux_x):
        map_feat = self.map_encoder(map_x).reshape(map_x.size(0), -1)
        aux_feat = self.aux_encoder(aux_x)
        feat = torch.cat([map_feat, aux_feat], dim=1)
        return self.head(feat)

class DQNAgent:
    """
    Agent class for DQN training and evaluation.
    Args:
        agent_id: int
        input_dim: int
        num_actions: int
        lr: float
        device: str
        pretrained_model: str
    Returns:
        None
    """
    team_id = "DQNAgent"
    
    def __init__(self, agent_id: int, input_spec, num_actions: int, lr: float=1e-3, device: str="cpu", pretrained_model=None):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.device = device
        self.gamma = 0.99
        self.lr = lr
        self.global_step = 0
        self.epsilon = 1.0

        # Networks: Q-Network (learning) and Target-Network (stable target)
        if pretrained_model:
            self.load_agent(pretrained_model)
        else:
            self.map_shape = tuple(input_spec[0])
            self.aux_dim = int(input_spec[1])
            self.q_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)

        self.target_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict()) # Sync weights initially
        
        self.loss_fn = nn.MSELoss()

    def act(self, map_state, aux_state, epsilon=0.0):
        """
        Take an action based on the state.
        Args:
            map_state: np.ndarray
            aux_state: np.ndarray
            epsilon: float
        Returns:
            action: int
        """
        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        map_tensor = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.q_net(map_tensor, aux_tensor).argmax().item()
            
        # action with the highest predicted Q-value
        return action

    def train_step(self, map_state, aux_state, next_map_state, next_aux_state, action, reward, done):
        """
        Train the DQN agent for one step.
        Args:
            state: np.ndarray
            action: int
            reward: float
            next_state: np.ndarray
            done: bool
        Returns:
            None
        """
        # torch.from_numpy is zero-copy; only move to device when not CPU
        map_state_t      = torch.from_numpy(map_state)
        aux_state_t      = torch.from_numpy(aux_state)
        next_map_state_t = torch.from_numpy(next_map_state)
        next_aux_state_t = torch.from_numpy(next_aux_state)
        action_t     = torch.from_numpy(action).unsqueeze(1)
        reward_t     = torch.from_numpy(reward).unsqueeze(1)
        done_t       = torch.from_numpy(done).unsqueeze(1)
        if self.device != "cpu":
            map_state_t      = map_state_t.to(self.device)
            aux_state_t      = aux_state_t.to(self.device)
            next_map_state_t = next_map_state_t.to(self.device)
            next_aux_state_t = next_aux_state_t.to(self.device)
            action_t     = action_t.to(self.device)
            reward_t     = reward_t.to(self.device)
            done_t       = done_t.to(self.device)

        # 2. Calculate current Q-values: Q(s, a)
        # gather() extracts the Q-value for the specific action taken
        q_values = self.q_net(map_state_t, aux_state_t).gather(1, action_t)

        # max(1)[0] gets the max Q-value for the next state
            # ~ max_a' {Q(s', a', weights)}
        # If done=1, the future reward is 0.
            # Q*(s, a) = E[r + gamma * max_a' {Q*(s', a')}]
            # ~ Q(s, a) = r + gamma * max_a' {Q(s', a', weights)} if not done else Q(s, a) = r
        # inference_mode is stricter than no_grad: disables autograd engine entirely
        with torch.no_grad():
            max_next_q = self.target_net(next_map_state_t, next_aux_state_t).max(1)[0].unsqueeze(1)
            target_q   = reward_t + self.gamma * max_next_q * (1 - done_t)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad(set_to_none=True)  # skip memset, just nullify refs
        loss.backward()
        self.optimizer.step()
        self.global_step += 1
        return loss.item()
        
    def update_target_network(self):
        """Copies the learned weights into the target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def load_agent(self, pretrained_model):
        checkpoint = torch.load(pretrained_model, map_location=self.device)
        input_spec = checkpoint.get("input_spec", checkpoint.get("input_shape", checkpoint["input_dim"]))
        self.map_shape = tuple(input_spec[0])
        self.aux_dim = int(input_spec[1])
        self.num_actions = checkpoint["num_actions"]
        self.q_net = DQNModel(self.map_shape, self.aux_dim, self.num_actions).to(self.device)
        self.q_net.load_state_dict(checkpoint["model_state_dict"])
        self.lr = checkpoint["lr"]
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1e-08, weight_decay=1e-5)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epsilon = checkpoint["epsilon"]
    

BOMB_MAX_TIMER = 7  # matches Bomb.__init__ default timer

def encode_obs(obs, agent_ids):
    """
    Returns:
      map_feat: spatial tensor for Conv2D branch, shape (C, H, W)
      aux_feat: scalar tensor for auxiliary branch, shape (A,)

    agent_ids: int (user's player id) or list/tuple [user_id, opp_id].
    When a single int is given the enemy is inferred as the other
    player in a 2-player game (1 - user_id).
    """
    if obs is None:
        raise ValueError("obs should not be None")

    # Normalise agent_ids to (user_id, opp_id)
    user_id = int(agent_ids[0])
    opp_id  = int(agent_ids[1]) if len(agent_ids) > 1 else (1 - user_id)

    grid    = obs["map"]      # (H, W)
    players = obs["players"]  # (num_players, 5)
    bombs   = obs["bombs"]    # (N, 4), N may be 0
    H, W    = grid.shape

    # One-hot map: grass, wall, box, item_radius, item_capacity
    map_channels = []
    for v in [Map.GRASS, Map.WALL, Map.BOX, Map.ITEM_RADIUS, Map.ITEM_CAPACITY]:
        map_channels.append((grid == v).astype(np.float32))
    # Player position masks
    my_x, my_y, my_alive, my_bombs_left, my_radius_bonus = players[user_id]
    ox,   oy,   opp_alive, _,            _               = players[opp_id]
    my_pos  = np.zeros((H, W), dtype=np.float32)
    opp_pos = np.zeros((H, W), dtype=np.float32)
    if int(my_alive)  == 1:
        my_pos[int(my_x), int(my_y)] = 1.0
    if int(opp_alive) == 1:
        opp_pos[int(ox), int(oy)]    = 1.0

    # Bomb channels — bombs is a numpy array, not a list of Bomb objects
    bomb_timer = np.zeros((H, W), dtype=np.float32)
    bomb_owned = np.zeros((H, W), dtype=np.float32)
    for b in bombs:
        bx, by, timer, owner_id = b
        bx, by = int(bx), int(by)
        t = float(timer) / BOMB_MAX_TIMER  # normalise by default max timer
        bomb_timer[bx, by] = max(bomb_timer[bx, by], t)
        bomb_owned[bx, by] = 1.0 if int(owner_id) == user_id else 0.0

    scalar = np.array([
        float(my_bombs_left)   / Player.MAX_BOMB_CAPACITY,
        float(my_radius_bonus) / Player.MAX_BOMB_RADIUS,
        float(opp_alive),
    ], dtype=np.float32)

    map_feat = np.stack([
        *map_channels,          # 5 channels
        my_pos,                 # 1 channel
        opp_pos,                # 1 channel
        bomb_timer,             # 1 channel
        bomb_owned,             # 1 channel
    ], axis=0).astype(np.float32)  # (9, H, W)
    return map_feat, scalar

def save_model_fn(model, optimizer, global_step, epsilon, lr, input_spec, num_actions, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epsilon": epsilon,
        "lr": lr,
        "input_dim": input_spec,  # backward-compatible key name
        "input_shape": input_spec,
        "input_spec": input_spec,
        "num_actions": num_actions,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model_fn(checkpoint):
    """
    Args:
        checkpoint: dict
    Returns:
        model: DQNModel
        optimizer: optim.Adam
        global_step: int
        epsilon: float
        lr: float
    """
    input_spec = checkpoint.get("input_spec", checkpoint.get("input_shape", checkpoint["input_dim"]))
    map_shape = tuple(input_spec[0])
    aux_dim = int(input_spec[1])
    model = DQNModel(map_shape, aux_dim, checkpoint["num_actions"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=checkpoint["lr"], eps=1e-08, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["global_step"], checkpoint["epsilon"], checkpoint["lr"]

def train_dqn(user_id=0, enemy_type="simple", num_episodes=100, max_steps=500, seed=86, save_model=True, pretrained_model=None):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    if enemy_type == "simple":
        enemy_agent = SimpleRuleAgent(1)
    elif enemy_type == "smarter":
        enemy_agent = SmarterRuleAgent(1)
    elif enemy_type == "tactical":
        enemy_agent = TacticalRuleAgent(1)
    elif enemy_type == "genius":
        enemy_agent = GeniusRuleAgent(1)
    elif enemy_type == "box_farmer":
        enemy_agent = BoxFarmerAgent(1)
    else:
        raise ValueError(f"Invalid enemy type: {enemy_type}")

    # hyperparam
    epsilon_start      = 1.0
    epsilon_min        = 0.05
    epsilon_decay      = 0.995
    epsilon            = epsilon_start
    batch_size         = 64
    lr                 = 1e-3

    dummy_obs = env.reset(seed=seed)
    agent_ids = [user_id, enemy_agent.agent_id]
    sample_state = encode_obs(dummy_obs, agent_ids=agent_ids)
    input_spec = (sample_state[0].shape, sample_state[1].shape[0])
    num_actions = 6

    user_agent = DQNAgent(user_id, input_spec, num_actions, lr=lr, device="cuda" if torch.cuda.is_available() else "cpu", pretrained_model=pretrained_model)
    buffer = ReplayBuffer(capacity=10_000, map_shape=input_spec[0], aux_dim=input_spec[1])

    global_step = 0
    loss_history = []
    reward_history = []
    win_history = []
    with tqdm(total=num_episodes, desc="Training DQN") as pbar:
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            prev_obs = None
            total_reward = 0

            map_state, aux_state = encode_obs(obs, agent_ids)

            for _ in range(max_steps):
                # 1. Action
                user_action  = user_agent.act(map_state, aux_state, epsilon=epsilon)
                enemy_action = enemy_agent.act(obs)
                actions = [None, None]
                actions[user_id]              = user_action
                actions[enemy_agent.agent_id] = enemy_action

                # 2. Environment Step
                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated

                # 3. Reward
                r = compute_reward(prev_obs, next_obs, agent_id=user_id)
                total_reward += r
                reward_history.append(r)
                if done:
                    win_history.append(1 if next_obs["players"][user_id][2] else 0)
                
                # 4. Buffer Push
                next_map_state, next_aux_state = encode_obs(next_obs, agent_ids)
                buffer.push(map_state, aux_state, user_action, r, next_map_state, next_aux_state, done)

                # 5. Train
                global_step += 1
                if len(buffer) >= batch_size:
                    sampled_map_state, sampled_aux_state, sampled_next_map_state, sampled_next_aux_state, sampled_action, sampled_reward, sampled_done = buffer.sample(batch_size)
                    loss = user_agent.train_step(
                        sampled_map_state,
                        sampled_aux_state,
                        sampled_next_map_state,
                        sampled_next_aux_state,
                        sampled_action,
                        sampled_reward,
                        sampled_done,
                    )
                    loss_history.append(loss)

                # 6. Update
                prev_obs  = obs
                obs       = next_obs
                map_state = next_map_state
                aux_state = next_aux_state

                # 7. Done
                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if ep % 10 == 0:
                user_agent.update_target_network()
            pbar.update(1)
            pbar.set_postfix(reward=f"{total_reward:.2f}", epsilon=f"{epsilon:.3f}")

    model_folder = f"ckpts/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed"
    if save_model:
        model_path = f"{model_folder}/{user_agent.global_step}_global_step.pth"
        save_model_fn(user_agent.q_net, 
                    user_agent.optimizer, 
                    user_agent.global_step, 
                    user_agent.epsilon, 
                    user_agent.lr, 
                    input_spec,
                    num_actions,
                    model_path)
        
    plot_loss(loss_history=loss_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_loss.png")
    plot_rewards(reward_history=reward_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_rewards.png")
    plot_win_rates(win_history=win_history, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_win_rates.png")
    plot_moving_average(data=reward_history, window_size=10, save_path=f"{model_folder}/dqn_{enemy_type}_{num_episodes}_episodes_{max_steps}_steps_{seed}_seed_moving_average.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enemy_type", type=str, default="simple", choices=["simple", "smarter", "tactical", "genius", "box_farmer"])
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=86, help="Random seed for reproducibility")
    parser.add_argument("--save_model", action="store_true", help="Save model")
    parser.add_argument("--load_model", type=str, default=None, help="Load model")
    parser.add_argument("--skip_training", action="store_true", help="Skip training")
    args = parser.parse_args()
    
    print("Skip training? ", args.skip_training)
    if not args.skip_training:
        train_dqn(enemy_type=args.enemy_type, 
                    num_episodes=args.num_episodes, 
                    max_steps=args.max_steps, 
                    seed=args.seed, 
                    save_model=args.save_model,
                    pretrained_model=args.load_model)
    