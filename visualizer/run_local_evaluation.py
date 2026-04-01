import os
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch

parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from engine import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, TacticalRuleAgent, GeniusRuleAgent, BoxFarmerAgent
from training import DQNAgent, encode_obs

class Viewer:
	def __init__(self, width=13, height=13, cell_size=42, fps=8):
		self.width = width
		self.height = height
		self.cell_size = cell_size
		self.fps = fps

		self.top_bar = 60
		self.screen_width = width * cell_size
		self.screen_height = height * cell_size + self.top_bar

		pygame.init()
		self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
		pygame.display.set_caption("Bomberland Enhanced Viewer")
		self.clock = pygame.time.Clock()
		self.font_info = pygame.font.SysFont(None, 24)
		self.font_small = pygame.font.SysFont(None, 20)
		self.explosion_overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
		self.explosion_overlay.fill((255, 140, 0, 130))

	def draw_grid(self, grid):
		for row in range(self.height):
			for col in range(self.width):
				rect = pygame.Rect(
					col * self.cell_size,
					row * self.cell_size + self.top_bar,
					self.cell_size,
					self.cell_size,
				)
				cell_type = int(grid[row, col])
				if cell_type == 1:
					pygame.draw.rect(self.screen, (80, 80, 80), rect)
					pygame.draw.rect(self.screen, (40, 40, 40), rect, 2)
				elif cell_type == 2:
					pygame.draw.rect(self.screen, (139, 69, 19), rect)
					pygame.draw.rect(self.screen, (101, 67, 33), rect, 2)
					pygame.draw.line(self.screen, (101, 67, 33), (rect.left, rect.top), (rect.right, rect.bottom), 2)
					pygame.draw.line(self.screen, (101, 67, 33), (rect.right, rect.top), (rect.left, rect.bottom), 2)
				elif cell_type == 3:
					pygame.draw.rect(self.screen, (225, 225, 225), rect)
					pygame.draw.circle(self.screen, (255, 0, 0), rect.center, self.cell_size // 4)
					text = self.font_small.render("R", True, (255, 255, 255))
					self.screen.blit(text, (rect.centerx - 5, rect.centery - 8))
				elif cell_type == 4:
					pygame.draw.rect(self.screen, (225, 225, 225), rect)
					pygame.draw.circle(self.screen, (0, 0, 255), rect.center, self.cell_size // 4)
					text = self.font_small.render("C", True, (255, 255, 255))
					self.screen.blit(text, (rect.centerx - 5, rect.centery - 8))
				else:
					pygame.draw.rect(self.screen, (144, 238, 144), rect)
					pygame.draw.rect(self.screen, (120, 200, 120), rect, 1)

	def draw_players(self, players):
		colors = [(220, 50, 50), (50, 50, 220), (30, 150, 30), (200, 140, 0)]
		for i, p in enumerate(players):
			if p[2] != 1:
				continue
			center = (
				int(p[1]) * self.cell_size + self.cell_size // 2,
				int(p[0]) * self.cell_size + self.top_bar + self.cell_size // 2,
			)
			pygame.draw.circle(self.screen, colors[i % len(colors)], center, self.cell_size // 3)
			img = self.font_small.render(str(i), True, (255, 255, 255))
			self.screen.blit(img, (center[0] - 5, center[1] - 8))
			stats_text = f"B:{int(p[3])} R:{int(p[4])}"
			stats_img = self.font_small.render(stats_text, True, (0, 0, 0))
			self.screen.blit(stats_img, (center[0] - 16, center[1] + 12))

	def draw_bombs(self, bombs):
		for b in bombs:
			if b[2] <= 0:
				continue
			center = (
				int(b[1]) * self.cell_size + self.cell_size // 2,
				int(b[0]) * self.cell_size + self.top_bar + self.cell_size // 2,
			)
			pygame.draw.circle(self.screen, (20, 20, 20), center, self.cell_size // 4)
			timer_img = self.font_small.render(str(int(b[2])), True, (255, 255, 255))
			self.screen.blit(timer_img, (center[0] - 5, center[1] - 8))

	def draw_header(self, episode_idx, total_episodes, step_idx, total_steps, paused):
		pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.screen_width, self.top_bar))
		status = "PAUSED" if paused else "PLAYING"
		text = (
			f"Ep {episode_idx + 1}/{total_episodes} | "
			f"Step {step_idx}/{max(total_steps - 1, 0)} | {status}"
		)
		help_text = "[A/D] Step [W/S] Ep [SPACE] Pause [ESC] Quit"
		self.screen.blit(self.font_info.render(text, True, (245, 245, 245)), (10, 5))
		self.screen.blit(self.font_small.render(help_text, True, (210, 210, 210)), (10, 35))

	def _in_bounds(self, row, col):
		return 0 <= row < self.height and 0 <= col < self.width

	def _blast_tiles(self, grid, bx, by, radius):
		tiles = {(bx, by)}
		for drow, dcol in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
			for r in range(1, radius + 1):
				tr, tc = bx + drow * r, by + dcol * r
				if not self._in_bounds(tr, tc):
					break
				cell = int(grid[tr, tc])
				if cell == 1:
					break
				tiles.add((tr, tc))
				if cell == 2:
					break
		return tiles

	def _explosion_tiles_from_transition(self, prev_obs, obs):
		if prev_obs is None:
			return set()

		prev_bombs = prev_obs["bombs"]
		curr_bombs = obs["bombs"]
		curr_positions = {(int(b[0]), int(b[1])) for b in curr_bombs}
		prev_players = prev_obs["players"]
		prev_grid = prev_obs["map"]

		tiles = set()
		for b in prev_bombs:
			bx, by, timer, owner_id = int(b[0]), int(b[1]), int(b[2]), int(b[3])
			exploded = timer <= 1 or (bx, by) not in curr_positions
			if not exploded:
				continue
			radius = 1
			if 0 <= owner_id < len(prev_players):
				radius = 1 + int(prev_players[owner_id][4])
			tiles.update(self._blast_tiles(prev_grid, bx, by, radius))
		return tiles

	def draw_explosions(self, explosion_tiles):
		for row, col in explosion_tiles:
			px = col * self.cell_size
			py = row * self.cell_size + self.top_bar
			self.screen.blit(self.explosion_overlay, (px, py))
			center = (px + self.cell_size // 2, py + self.cell_size // 2)
			pygame.draw.circle(self.screen, (255, 220, 120), center, self.cell_size // 6)

	def render(self, obs, prev_obs, episode_idx, total_episodes, step_idx, total_steps, paused):
		self.screen.fill((245, 245, 245))
		self.draw_grid(obs["map"])
		explosion_tiles = self._explosion_tiles_from_transition(prev_obs, obs)
		self.draw_explosions(explosion_tiles)
		self.draw_players(obs["players"])
		self.draw_bombs(obs["bombs"])
		self.draw_header(episode_idx, total_episodes, step_idx, total_steps, paused)
		pygame.display.flip()
		self.clock.tick(self.fps)

	def close(self):
		pygame.quit()


def str2bool(value):
	if isinstance(value, bool):
		return value
	value = str(value).strip().lower()
	if value in {"true", "1", "yes", "y", "t"}:
		return True
	if value in {"false", "0", "no", "n", "f"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def make_agents(model_paths, seed=None):
	agents = [None] * len(model_paths)
	names = [None] * len(model_paths)

	if seed is not None:
		random.seed(seed)

	for i, path in enumerate(model_paths):
		if path != "None":
			checkpoint = torch.load(path, map_location="cpu")
			input_dim = checkpoint["input_dim"]
			num_actions = checkpoint["num_actions"]
			agents[i] = DQNAgent(i, input_dim, num_actions, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu", pretrained_model=path)
			agents[i].load_agent(pretrained_model=path)
			names[i] = os.path.basename(path)
		else:
			x = random.randint(0, 6)
			if x == 0:
				names[i] = "RandomAgent"
				agents[i] = RandomAgent(i)
			elif x == 1:
				names[i] = "SimpleRuleAgent"
				agents[i] = SimpleRuleAgent(i)
			elif x == 2:
				names[i] = "SmarterRuleAgent"
				agents[i] = SmarterRuleAgent(i)
			elif x == 3:
				names[i] = "GeniusRuleAgent"
				agents[i] = GeniusRuleAgent(i)
			elif x == 4:
				names[i] = "BoxFarmerAgent"
				agents[i] = BoxFarmerAgent(i)
			else:
				names[i] = "TacticalRuleAgent"
				agents[i] = TacticalRuleAgent(i)

	return agents, names


def clone_obs(obs):
	return {
		"map": np.array(obs["map"], copy=True),
		"players": np.array(obs["players"], copy=True),
		"bombs": np.array(obs["bombs"], copy=True),
	}


ACTION_MIRROR = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 5}


def rotate_map_180(map_feat):
	"""Flip spatial features both vertically and horizontally (180-degree rotation)
	so a bottom-right agent sees the board as if it were at top-left."""
	return np.flip(map_feat, axis=(1, 2)).copy()


def simulate_episodes(model_paths, num_episodes=10, max_steps=500, seed=None):
	env = BomberEnv(max_steps=max_steps)
	agents, names = make_agents(model_paths, seed=seed)
	episodes = []

	for episode in range(num_episodes):
		episode_seed = None if seed is None else seed + episode
		obs = env.reset(seed=episode_seed)
		done = False
		step = 0
		trajectory = [clone_obs(obs)]

		while not done and step < max_steps:
			actions = []
			for i in range(len(agents)):
				if isinstance(agents[i], DQNAgent):
					opp_id = 1 - i
					map_feat, aux_feat = encode_obs(obs, agent_ids=[i, opp_id])
					if i != 0:
						map_feat = rotate_map_180(map_feat)
					action = agents[i].act(map_feat, aux_feat, epsilon=0.05)
					if i != 0:
						action = ACTION_MIRROR[action]
					actions.append(action)
				else:
					actions.append(agents[i].act(obs))
			obs, terminated, truncated = env.step(actions)
			trajectory.append(clone_obs(obs))
			done = terminated or truncated
			step += 1

		episodes.append(trajectory)

	return episodes, names


def run_simple_viewer(model_paths, num_episodes=10, max_steps=100, seed=None, autoplay=True):
	episodes, agent_names = simulate_episodes(
		model_paths=model_paths,
		num_episodes=num_episodes,
		max_steps=max_steps,
		seed=seed,
	)
	if not episodes:
		print("No episodes to display.")
		return

	first_obs = episodes[0][0]
	viewer = Viewer(width=first_obs["map"].shape[1], height=first_obs["map"].shape[0])

	print("Agents:", ", ".join(agent_names))
	print("Controls: A/D step, W/S episode, SPACE pause/play, ESC quit")

	episode_idx = 0
	step_idx = 0
	paused = not autoplay
	last_tick = time.time()

	running = True
	while running:
		now = time.time()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_SPACE:
					paused = not paused
				elif event.key == pygame.K_d:
					step_idx = min(step_idx + 1, len(episodes[episode_idx]) - 1)
					paused = True
				elif event.key == pygame.K_a:
					step_idx = max(step_idx - 1, 0)
					paused = True
				elif event.key == pygame.K_s:
					episode_idx = min(episode_idx + 1, len(episodes) - 1)
					step_idx = 0
				elif event.key == pygame.K_w:
					episode_idx = max(episode_idx - 1, 0)
					step_idx = 0

		if not paused and (now - last_tick) >= (1 / max(viewer.fps, 1)):
			if step_idx < len(episodes[episode_idx]) - 1:
				step_idx += 1
			else:
				paused = True
			last_tick = now

		current_obs = episodes[episode_idx][step_idx]
		prev_obs = episodes[episode_idx][step_idx - 1] if step_idx > 0 else None
		viewer.render(
			obs=current_obs,
			prev_obs=prev_obs,
			episode_idx=episode_idx,
			total_episodes=len(episodes),
			step_idx=step_idx,
			total_steps=len(episodes[episode_idx]),
			paused=paused,
		)

	viewer.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_paths", nargs="+", default=["None", "None"])
	parser.add_argument("--num_episodes", type=int, default=10)
	parser.add_argument("--max_steps", type=int, default=500)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument("--autoplay", type=str2bool, default=True)
	args = parser.parse_args()

	run_simple_viewer(
		model_paths=args.model_paths,
		num_episodes=args.num_episodes,
		max_steps=args.max_steps,
		seed=args.seed,
		autoplay=args.autoplay,
	)
