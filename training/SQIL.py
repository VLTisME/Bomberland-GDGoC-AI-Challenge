"""Improved Imitation Learning: Behavioral Cloning Pre-training + DQfD Fine-tuning.

Phase 1 — Behavioral Cloning: supervised cross-entropy on expert actions with
          class weights, data augmentation, and validation monitoring.
Phase 2 — DQfD Fine-tuning: combined TD loss + large-margin supervised loss
          with real environment rewards and permanent demo replay.
"""

import argparse
import sys
import random
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from engine import BomberEnv, Map, Player
from reward import compute_reward
from utils import plot_loss, plot_rewards, plot_moving_average
from agent import (
    SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent,
    GeniusRuleAgent, BoxFarmerAgent,
)

# ======================================================================
# Reusable components (self-contained to avoid DQN.py import issues)
# ======================================================================

BOMB_MAX_TIMER = 7
NUM_ACTIONS = 6

AGENT_LOOKUP = {
    "genius":     GeniusRuleAgent,
    "tactical":   TacticalRuleAgent,
    "smarter":    SmarterRuleAgent,
    "simple":     SimpleRuleAgent,
    "box_farmer": BoxFarmerAgent,
}


def _make_agent(name: str, agent_id: int):
    cls = AGENT_LOOKUP.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent type: {name}")
    return cls(agent_id)


class ReplayBuffer:
    """Pre-allocated numpy circular buffer."""

    def __init__(self, capacity: int, map_shape, aux_dim: int):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.map_shape = tuple(map_shape)
        self.aux_dim = int(aux_dim)
        self.map_states = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.aux_states = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.next_map_states = np.zeros((capacity, *self.map_shape), dtype=np.float32)
        self.next_aux_states = np.zeros((capacity, self.aux_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.size

    def push(self, map_state, aux_state, action, reward,
             next_map_state, next_aux_state, done):
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.map_states[self.pos] = map_state
        self.aux_states[self.pos] = aux_state
        self.next_map_states[self.pos] = next_map_state
        self.next_aux_states[self.pos] = next_aux_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.map_states[idx], self.aux_states[idx],
            self.next_map_states[idx], self.next_aux_states[idx],
            self.actions[idx], self.rewards[idx], self.dones[idx],
        )


class DQNModel(nn.Module):
    """Two-branch DQN: Conv2D for spatial features + MLP for scalars."""

    def __init__(self, map_shape, aux_dim, output_dim):
        super().__init__()
        c, _, _ = map_shape

        class _SE(nn.Module):
            def __init__(self, channels: int, reduction: int = 16):
                super().__init__()
                hidden = max(channels // reduction, 4)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
                self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

            def forward(self, x):
                s = self.pool(x)
                s = F.relu(self.fc1(s), inplace=True)
                s = torch.sigmoid(self.fc2(s))
                return x * s

        class _ResBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)
                self.se = _SE(out_ch)
                self.act = nn.ReLU(inplace=True)

                self.proj = None
                if stride != 1 or in_ch != out_ch:
                    self.proj = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_ch),
                    )

            def forward(self, x):
                identity = x
                out = self.act(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = self.se(out)
                if self.proj is not None:
                    identity = self.proj(identity)
                out = out + identity
                return self.act(out)

        def _make_stage(in_ch: int, out_ch: int, blocks: int, stride: int):
            layers = [_ResBlock(in_ch, out_ch, stride=stride)]
            for _ in range(blocks - 1):
                layers.append(_ResBlock(out_ch, out_ch, stride=1))
            return nn.Sequential(*layers)

        base = 64
        self.map_stem = nn.Sequential(
            nn.Conv2d(c, base, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.map_stage1 = _make_stage(base, base, blocks=2, stride=1)
        self.map_stage2 = _make_stage(base, base * 2, blocks=2, stride=2)
        self.map_stage3 = _make_stage(base * 2, base * 4, blocks=2, stride=2)
        self.map_pool = nn.AdaptiveAvgPool2d(1)
        map_feat_dim = base * 4
        self.map_dropout = nn.Dropout(p=0.1)

        self.aux_encoder = nn.Sequential(
            nn.Linear(aux_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(map_feat_dim + 64, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, map_x, aux_x):
        x = self.map_stem(map_x)
        x = self.map_stage1(x)
        x = self.map_stage2(x)
        x = self.map_stage3(x)
        map_feat = self.map_pool(x).flatten(1)
        map_feat = self.map_dropout(map_feat)
        aux_feat = self.aux_encoder(aux_x)
        return self.head(torch.cat([map_feat, aux_feat], dim=1))


def encode_obs(obs, agent_ids):
    """Encode raw observation into (map_feat [C,H,W], aux_feat [A,]).

    agent_ids can be:
    - [user_id, opp_id] (legacy 2-agent encoding)
    - [user_id, opp1_id, opp2_id, opp3_id] (4-agent encoding)
    - [user_id] (will auto-fill opponents from obs["players"])
    """
    if obs is None:
        raise ValueError("obs should not be None")
    user_id = int(agent_ids[0])
    players = obs["players"]
    num_players = int(players.shape[0])

    if len(agent_ids) > 1:
        ordered_ids = [int(i) for i in agent_ids]
    else:
        ordered_ids = [user_id] + [i for i in range(num_players) if i != user_id]

    opp_ids = [i for i in ordered_ids[1:] if i != user_id]
    while len(opp_ids) < 3:
        opp_ids.append(-1)

    grid = obs["map"]
    bombs = obs["bombs"]
    H, W = grid.shape

    map_channels = [(grid == v).astype(np.float32)
                    for v in (Map.GRASS, Map.WALL, Map.BOX, Map.ITEM_RADIUS, Map.ITEM_CAPACITY)]

    my_x, my_y, my_alive, my_bombs_left, my_radius_bonus = players[user_id]
    my_pos = np.zeros((H, W), dtype=np.float32)
    if int(my_alive) == 1:
        my_pos[int(my_x), int(my_y)] = 1.0

    opp_pos_planes = []
    opp_alive_flags = []
    for oid in opp_ids[:3]:
        plane = np.zeros((H, W), dtype=np.float32)
        alive_flag = 0.0
        if 0 <= oid < num_players:
            ox, oy, o_alive, _, _ = players[oid]
            if int(o_alive) == 1:
                plane[int(ox), int(oy)] = 1.0
                alive_flag = 1.0
        opp_pos_planes.append(plane)
        opp_alive_flags.append(alive_flag)

    bomb_timer = np.zeros((H, W), dtype=np.float32)
    bomb_owned = np.zeros((H, W), dtype=np.float32)
    for b in bombs:
        bx, by, timer, owner_id = b
        bx, by = int(bx), int(by)
        bomb_timer[bx, by] = max(bomb_timer[bx, by], float(timer) / BOMB_MAX_TIMER)
        bomb_owned[bx, by] = 1.0 if int(owner_id) == user_id else 0.0

    scalar = np.array([
        float(my_bombs_left) / Player.MAX_BOMB_CAPACITY,
        float(my_radius_bonus) / Player.MAX_BOMB_RADIUS,
        *opp_alive_flags[:3],
    ], dtype=np.float32)

    map_feat = np.stack([
        *map_channels,
        my_pos,
        *opp_pos_planes[:3],
        bomb_timer,
        bomb_owned,
    ],
                        axis=0).astype(np.float32)
    return map_feat, scalar


def save_model_fn(model, optimizer, global_step, epsilon, lr, input_spec,
                  num_actions, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step, "epsilon": epsilon, "lr": lr,
        "input_dim": input_spec, "input_shape": input_spec,
        "input_spec": input_spec, "num_actions": num_actions,
    }, path)
    print(f"Model saved to {path}")


# ======================================================================
# 1. Data Augmentation
# ======================================================================

_HFLIP_ACTION = np.array([0, 2, 1, 3, 4, 5], dtype=np.int64)
_VFLIP_ACTION = np.array([0, 1, 2, 4, 3, 5], dtype=np.int64)
_HVFLIP_ACTION = np.array([0, 2, 1, 4, 3, 5], dtype=np.int64)


def augment_transition(map_state, aux_state, action):
    """Return 3 augmented copies: h-flip, v-flip, hv-flip.

    map_state: (C, H, W), aux_state: (A,), action: int
    Returns list of (map, aux, action) tuples.
    """
    h_map = np.flip(map_state, axis=1).copy()
    v_map = np.flip(map_state, axis=2).copy()
    hv_map = np.flip(np.flip(map_state, axis=1), axis=2).copy()

    return [
        (h_map, aux_state.copy(), int(_HFLIP_ACTION[action])),
        (v_map, aux_state.copy(), int(_VFLIP_ACTION[action])),
        (hv_map, aux_state.copy(), int(_HVFLIP_ACTION[action])),
    ]


# ======================================================================
# 2. Demonstration Collection (enhanced: win-filtering + real rewards)
# ======================================================================

def collect_demonstrations(expert_type, opponent_type, num_episodes,
                           max_steps, seed, augment=True):
    """Collect expert demonstrations, keeping only winning episodes.

    Returns
    -------
    bc_data : dict  with keys 'map', 'aux', 'action' — numpy arrays for BC
    demo_buffer : ReplayBuffer  filled with real rewards for DQfD
    input_spec : tuple
    """
    env = BomberEnv(max_steps=max_steps, seed=seed)
    expert_id = 0
    expert = _make_agent(expert_type, agent_id=expert_id)

    # BomberEnv always has 4 players (0..3). Fill all opponents.
    opp_ids = [i for i in range(4) if i != expert_id]
    opponents = [_make_agent(opponent_type, agent_id=i) for i in opp_ids]
    agent_ids = [expert_id, *opp_ids]

    dummy_obs = env.reset(seed=seed)
    sample = encode_obs(dummy_obs, agent_ids)
    input_spec = (sample[0].shape, sample[1].shape[0])

    bc_maps, bc_auxs, bc_actions = [], [], []
    capacity = num_episodes * max_steps * (4 if augment else 1)
    demo_buffer = ReplayBuffer(capacity=capacity,
                               map_shape=input_spec[0], aux_dim=input_spec[1])

    wins, total_eps = 0, 0
    for ep in tqdm(range(num_episodes), desc="Collecting demos"):
        obs = env.reset(seed=seed + ep)
        map_state, aux_state = encode_obs(obs, agent_ids)
        prev_obs = None

        ep_transitions = []

        for _ in range(max_steps):
            actions = [None] * 4
            expert_action = expert.act(obs)
            actions[expert_id] = expert_action
            for opp in opponents:
                actions[opp.agent_id] = opp.act(obs)

            next_obs, terminated, truncated = env.step(actions)
            done = terminated or truncated

            next_map_state, next_aux_state = encode_obs(next_obs, agent_ids)
            r = compute_reward(prev_obs, next_obs, agent_id=0)

            ep_transitions.append((
                map_state, aux_state, expert_action, r,
                next_map_state, next_aux_state, float(done),
            ))

            prev_obs = obs
            obs = next_obs
            map_state = next_map_state
            aux_state = next_aux_state
            if done:
                break

        total_eps += 1
        alive_final = next_obs["players"][:, 2].astype(np.int8)
        expert_alive = int(alive_final[expert_id]) == 1
        opp_alive_any = any(int(alive_final[i]) == 1 for i in opp_ids)
        expert_won = expert_alive and (not opp_alive_any)

        if not expert_won:
            continue
        wins += 1

        for ms, axs, act, rew, nms, naxs, d in ep_transitions:
            demo_buffer.push(ms, axs, act, rew, nms, naxs, d)
            bc_maps.append(ms)
            bc_auxs.append(axs)
            bc_actions.append(act)

            if augment:
                for aug_ms, aug_axs, aug_act in augment_transition(ms, axs, act):
                    bc_maps.append(aug_ms)
                    bc_auxs.append(aug_axs)
                    bc_actions.append(aug_act)

    print(f"Demo collection: {wins}/{total_eps} winning episodes, "
          f"{len(bc_maps)} BC samples (augmented), "
          f"{len(demo_buffer)} replay transitions")

    bc_data = {
        "map": np.array(bc_maps, dtype=np.float32),
        "aux": np.array(bc_auxs, dtype=np.float32),
        "action": np.array(bc_actions, dtype=np.int64),
    }
    return bc_data, demo_buffer, input_spec


# ======================================================================
# 3. Behavioral Cloning Pre-training
# ======================================================================

# STOP is common in expert data, BOMB is rare but critical
BC_CLASS_WEIGHTS = torch.tensor([0.3, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=torch.float32)


def pretrain_bc(q_net, bc_data, device, bc_epochs=15, batch_size=128,
                lr=1e-3, val_ratio=0.1):
    """Train q_net via cross-entropy on expert actions.

    Returns the best model state_dict (lowest val loss) and training history.
    """
    n = len(bc_data["action"])
    idx = np.random.permutation(n)
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    weights = BC_CLASS_WEIGHTS.to(device)
    optimizer = optim.AdamW(q_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_state = None
    bc_loss_history = []

    for epoch in range(bc_epochs):
        # ---- Train ----
        q_net.train()
        np.random.shuffle(train_idx)
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        action_correct = np.zeros(NUM_ACTIONS, dtype=np.int64)
        action_total = np.zeros(NUM_ACTIONS, dtype=np.int64)

        for start in range(0, len(train_idx), batch_size):
            bi = train_idx[start:start + batch_size]
            m = torch.from_numpy(bc_data["map"][bi]).to(device)
            a = torch.from_numpy(bc_data["aux"][bi]).to(device)
            labels = torch.from_numpy(bc_data["action"][bi]).to(device)

            logits = q_net(m, a)
            loss = F.cross_entropy(logits, labels, weight=weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(bi)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(bi)

            for act_id in range(NUM_ACTIONS):
                mask = labels == act_id
                action_total[act_id] += mask.sum().item()
                action_correct[act_id] += ((preds == labels) & mask).sum().item()

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        bc_loss_history.append(train_loss)

        # ---- Validate ----
        q_net.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for start in range(0, len(val_idx), batch_size):
                bi = val_idx[start:start + batch_size]
                m = torch.from_numpy(bc_data["map"][bi]).to(device)
                a = torch.from_numpy(bc_data["aux"][bi]).to(device)
                labels = torch.from_numpy(bc_data["action"][bi]).to(device)
                logits = q_net(m, a)
                loss = F.cross_entropy(logits, labels, weight=weights)
                val_loss_sum += loss.item() * len(bi)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(bi)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in q_net.state_dict().items()}

        act_names = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]
        per_act = " | ".join(
            f"{act_names[i]}: {action_correct[i] / max(action_total[i], 1):.2f}"
            for i in range(NUM_ACTIONS)
        )
        print(f"  BC Epoch {epoch + 1:>2}/{bc_epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | {per_act}")

    if best_state is not None:
        q_net.load_state_dict(best_state)
        q_net.to(device)
    print(f"  BC done — best val loss: {best_val_loss:.4f}")
    return bc_loss_history


# ======================================================================
# 4. DQfD Agent (TD loss + large-margin supervised loss)
# ======================================================================

class DQfDAgent:
    """DQN agent with an additional large-margin classification loss for DQfD."""
    
    team_id = "DQfDAgent"

    def __init__(self, agent_id, input_spec, num_actions, lr=1e-3,
                 device="cpu", margin=0.8, pretrained_model=None):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.device = device
        self.gamma = 0.99
        self.lr = lr
        self.margin = margin
        self.global_step = 0
        self.epsilon = 1.0

        if pretrained_model:
            self.load_agent(pretrained_model)
        else:
            self.map_shape = tuple(input_spec[0])
            self.aux_dim = int(input_spec[1])
            self.q_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr,
                                        eps=1e-8, weight_decay=1e-5)
        self.target_net = DQNModel(self.map_shape, self.aux_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())

    def act(self, map_state, aux_state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        m = torch.from_numpy(map_state).unsqueeze(0).to(self.device)
        a = torch.from_numpy(aux_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(m, a).argmax().item()

    def _to_device(self, *arrays):
        tensors = [torch.from_numpy(a) for a in arrays]
        if self.device != "cpu":
            tensors = [t.to(self.device) for t in tensors]
        return tensors

    def dqfd_train_step(self, demo_batch, env_batch, lambda_bc):
        """Combined DQfD update: TD loss on full batch + margin loss on demo half.

        demo_batch / env_batch: each is a tuple of numpy arrays
            (map, aux, next_map, next_aux, action, reward, done)

        Returns (total_loss, td_loss, margin_loss) as floats.
        """
        d_ms, d_ax, d_nms, d_nax, d_act, d_rew, d_done = demo_batch
        e_ms, e_ax, e_nms, e_nax, e_act, e_rew, e_done = env_batch
        half = len(d_act)

        all_ms = np.concatenate([d_ms, e_ms])
        all_ax = np.concatenate([d_ax, e_ax])
        all_nms = np.concatenate([d_nms, e_nms])
        all_nax = np.concatenate([d_nax, e_nax])
        all_act = np.concatenate([d_act, e_act])
        all_rew = np.concatenate([d_rew, e_rew])
        all_done = np.concatenate([d_done, e_done])

        (ms_t, ax_t, nms_t, nax_t, act_t,
         rew_t, done_t) = self._to_device(
            all_ms, all_ax, all_nms, all_nax, all_act, all_rew, all_done)
        act_t = act_t.long().unsqueeze(1)
        rew_t = rew_t.unsqueeze(1)
        done_t = done_t.unsqueeze(1)

        # -- TD loss (full batch) --
        q_all = self.q_net(ms_t, ax_t)
        q_taken = q_all.gather(1, act_t)

        with torch.no_grad():
            max_next_q = self.target_net(nms_t, nax_t).max(1)[0].unsqueeze(1)
            target_q = rew_t + self.gamma * max_next_q * (1 - done_t)

        td_loss = F.mse_loss(q_taken, target_q)

        # -- Large-margin classification loss (demo half only) --
        q_demo = q_all[:half]
        demo_actions = act_t[:half]
        margins = torch.full_like(q_demo, self.margin)
        margins.scatter_(1, demo_actions, 0.0)
        max_q_margin = (q_demo + margins).max(dim=1)[0]
        expert_q = q_demo.gather(1, demo_actions).squeeze(1)
        margin_loss = (max_q_margin - expert_q).clamp(min=0).mean()

        loss = td_loss + lambda_bc * margin_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        return loss.item(), td_loss.item(), margin_loss.item()

    def load_agent(self, pretrained_model):
        ckpt = torch.load(pretrained_model, map_location=self.device)
        self.map_shape, self.aux_dim = ckpt["input_shape"]
        self.actions_dim = int(ckpt["num_actions"])
        self.q_net = DQNModel(self.map_shape, self.aux_dim, self.actions_dim).to(self.device)
        self.q_net.load_state_dict(ckpt["model_state_dict"])

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=ckpt.get("lr", self.lr), eps=1e-8, weight_decay=1e-5)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.epsilon = ckpt.get("epsilon", 1.0)
        self.lr = ckpt.get("lr", self.lr)
        print(f"Loaded pretrained model from {pretrained_model} "
              f"(global_step={self.global_step}, epsilon={self.epsilon:.3f})")

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# ======================================================================
# 5. Main training loop
# ======================================================================

def train_dqfd(
    user_id=0,
    expert_type="genius",
    enemy_type="simple",
    num_players=4,
    demo_episodes=100,
    bc_epochs=15,
    num_episodes=300,
    max_steps=500,
    seed=86,
    lambda_bc_init=1.0,
    lambda_bc_final=0.1,
    margin=0.8,
    save_model=True,
    pretrained_model=None,
):
    # ------------------------------------------------------------------
    # Phase 0: Collect demonstrations
    # ------------------------------------------------------------------
    bc_data, demo_buffer, input_spec = collect_demonstrations(
        expert_type=expert_type,
        opponent_type=enemy_type,
        num_episodes=demo_episodes,
        max_steps=max_steps,
        seed=seed,
        augment=True,
    )

    if len(demo_buffer) == 0:
        print("WARNING: no winning episodes collected — increase demo_episodes "
              "or choose a weaker opponent")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQfDAgent(user_id, input_spec, NUM_ACTIONS,
                      lr=1e-3, device=device, margin=margin)

    if pretrained_model:
        ckpt = torch.load(pretrained_model, map_location=device)
        agent.q_net.load_state_dict(ckpt["model_state_dict"])
        agent.target_net.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded pretrained weights from {pretrained_model}")

    # ------------------------------------------------------------------
    # Phase 1: Behavioral Cloning pre-training
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 1: Behavioral Cloning Pre-training")
    print(f"{'='*60}")
    bc_loss_history = pretrain_bc(
        agent.q_net, bc_data, device,
        bc_epochs=bc_epochs, batch_size=128, lr=1e-3,
    )
    agent.target_net.load_state_dict(agent.q_net.state_dict())
    agent.optimizer = optim.Adam(agent.q_net.parameters(), lr=1e-3,
                                 eps=1e-8, weight_decay=1e-5)

    # ------------------------------------------------------------------
    # Phase 2: DQfD fine-tuning with environment interaction
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 2: DQfD Fine-tuning")
    print(f"{'='*60}")

    env = BomberEnv(max_steps=max_steps, seed=seed)

    if num_players not in (2, 4):
        raise ValueError("num_players must be 2 or 4 for this script")

    if num_players == 2:
        enemy_agents = [_make_agent(enemy_type, agent_id=(1 - user_id))]
    else:
        enemy_ids = [i for i in range(4) if i != user_id]
        enemy_agents = [_make_agent(enemy_type, agent_id=i) for i in enemy_ids]

    agent_ids = [user_id, *[e.agent_id for e in enemy_agents]]

    epsilon_start = 0.3
    epsilon_min = 0.05
    epsilon_decay = 0.995
    epsilon = epsilon_start
    batch_size = 64
    half_batch = batch_size // 2

    env_buffer = ReplayBuffer(capacity=20_000, map_shape=input_spec[0],
                              aux_dim=input_spec[1])

    loss_history = []
    td_loss_history = []
    margin_loss_history = []
    reward_history = []
    ep_reward_history = []
    
    tag = (f"dqfd_{expert_type}_{enemy_type}_"
           f"{num_episodes}ep_{max_steps}steps_{seed}seed")
    model_folder = f"ckpts/{tag}"

    with tqdm(total=num_episodes, desc="DQfD Training") as pbar:
        for ep in range(num_episodes):
            obs = env.reset(seed=seed + ep)
            map_state, aux_state = encode_obs(obs, agent_ids)
            prev_obs = None
            total_reward = 0.0

            progress = ep / max(num_episodes - 1, 1)
            lambda_bc = lambda_bc_init + (lambda_bc_final - lambda_bc_init) * progress

            for _ in range(max_steps):
                user_action = agent.act(map_state, aux_state, epsilon=epsilon)
                actions = [None] * (2 if num_players == 2 else 4)
                actions[user_id] = user_action
                for e in enemy_agents:
                    actions[e.agent_id] = e.act(obs)

                next_obs, terminated, truncated = env.step(actions)
                done = terminated or truncated
                next_map_state, next_aux_state = encode_obs(next_obs, agent_ids)

                r = compute_reward(prev_obs, next_obs, agent_id=user_id)
                total_reward += r
                reward_history.append(r)

                env_buffer.push(map_state, aux_state, user_action, r,
                                next_map_state, next_aux_state, float(done))

                if (len(env_buffer) >= half_batch
                        and len(demo_buffer) >= half_batch):
                    demo_batch = demo_buffer.sample(half_batch)
                    env_batch = env_buffer.sample(half_batch)

                    total_l, td_l, margin_l = agent.dqfd_train_step(
                        demo_batch, env_batch, lambda_bc)
                    loss_history.append(total_l)
                    td_loss_history.append(td_l)
                    margin_loss_history.append(margin_l)

                prev_obs = obs
                obs = next_obs
                map_state = next_map_state
                aux_state = next_aux_state

                if done:
                    break

            ep_reward_history.append(total_reward)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if ep % 10 == 0:
                agent.update_target_network()
            pbar.update(1)
            pbar.set_postfix(
                reward=f"{total_reward:.2f}",
                eps=f"{epsilon:.3f}",
                lbc=f"{lambda_bc:.2f}",
            )
            # save the model
            if ep % 1000 == 0:
                path = f"{model_folder}/{agent.global_step}_global_step.pth"
                save_model_fn(agent.q_net, agent.optimizer, agent.global_step,
                            agent.epsilon, agent.lr, input_spec, NUM_ACTIONS, path)

    # ------------------------------------------------------------------
    # Save & plot
    # ------------------------------------------------------------------

    if save_model:
        path = f"{model_folder}/{agent.global_step}_global_step.pth"
        save_model_fn(agent.q_net, agent.optimizer, agent.global_step,
                      agent.epsilon, agent.lr, input_spec, NUM_ACTIONS, path)

    plot_loss(bc_loss_history,
              save_path=f"{model_folder}/{tag}_bc_loss.png")
    plot_loss(loss_history,
              save_path=f"{model_folder}/{tag}_dqfd_loss.png")
    plot_rewards(reward_history,
                 save_path=f"{model_folder}/{tag}_rewards.png")
    plot_moving_average(ep_reward_history, window_size=10,
                        save_path=f"{model_folder}/{tag}_moving_avg.png")


# ======================================================================
# 6. CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Imitation Learning: BC Pre-training + DQfD Fine-tuning")
    parser.add_argument("--seed", type=int, default=86,
                        help="Random seed for reproducibility")

    parser.add_argument("--expert_type", type=str, default="genius",
                        choices=list(AGENT_LOOKUP.keys()),
                        help="Expert rule agent to imitate")
    parser.add_argument("--enemy_type", type=str, default="simple",
                        choices=list(AGENT_LOOKUP.keys()),
                        help="Opponent during demo collection and RL training")
    parser.add_argument("--num_players", type=int, default=4, choices=[2, 4],
                        help="Train in 2-player or 4-player mode (env has 4 by default)")
    parser.add_argument("--demo_episodes", type=int, default=100,
                        help="Episodes of expert play to collect")
    parser.add_argument("--bc_epochs", type=int, default=15,
                        help="Behavioral cloning pre-training epochs")

    parser.add_argument("--num_episodes", type=int, default=300,
                        help="DQfD RL training episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--lambda_bc", type=float, default=1.0,
                        help="Initial weight for supervised margin loss")
    parser.add_argument("--margin", type=float, default=0.8,
                        help="Large-margin value for DQfD classification loss")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for DQfD training")

    parser.add_argument("--save_model", action="store_true",
                        help="Save model checkpoint after training")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a pretrained model checkpoint")
    args = parser.parse_args()

    train_dqfd(
        expert_type=args.expert_type,
        enemy_type=args.enemy_type,
        demo_episodes=args.demo_episodes,
        bc_epochs=args.bc_epochs,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        lambda_bc_init=args.lambda_bc,
        margin=args.margin,
        num_players=args.num_players,
        save_model=args.save_model,
        pretrained_model=args.load_model,
    )
