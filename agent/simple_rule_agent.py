import random
from collections import deque


class SimpleRuleAgent:
    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    team_id = "SimpleRuleAgent"

    def __init__(self, agent_id):
        self.agent_id = int(agent_id)

    def act(self, observation):
        grid = observation["map"]
        players = observation["players"]
        bombs = observation["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}

        occupied = {
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        }
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]

        blocked = set(occupied) | bomb_positions
        blocked.discard(my_pos)

        danger_soon, danger_now = self._danger_tiles(grid, bombs, players, default_radius=2)
        valid_actions = self._valid_actions(grid, my_pos, blocked)

        # 1) Escape danger first
        if my_pos in danger_now or my_pos in danger_soon:
            move = self._move_to_targets(grid, my_pos, blocked, self._safe_tiles(grid, danger_soon), danger_soon)
            if move is not None:
                return move
            safe_moves = [a for a in valid_actions if self._next_pos(my_pos, a) not in danger_now]
            return random.choice(safe_moves) if safe_moves else 0

        # 2) Pick nearby items
        item_tiles = self._item_tiles(
            grid,
            prefer_capacity=int(bombs_left) <= 1,
            prefer_radius=int(bomb_bonus) <= 1,
        )
        if item_tiles:
            move = self._move_to_targets(grid, my_pos, blocked, item_tiles, danger_soon)
            if move is not None:
                return move

        # 3) Place bomb to break adjacent boxes or hit adjacent enemy (if escape exists)
        if bombs_left > 0 and my_pos not in bomb_positions:
            adjacent_enemy = any(abs(ex - my_pos[0]) + abs(ey - my_pos[1]) == 1 for ex, ey in enemies)
            boxes_hit = self._count_boxes_in_blast(grid, my_pos, bomb_radius)
            if (adjacent_enemy or boxes_hit > 0) and self._can_escape_after_placing(
                grid, my_pos, blocked, danger_now, bomb_radius
            ):
                return 5

        # 4) Move toward a tile where bombing would break boxes
        box_spots = self._box_bomb_spots(grid, blocked)
        if box_spots:
            move = self._move_to_targets(grid, my_pos, blocked, box_spots, danger_soon)
            if move is not None:
                return move

        # 5) Fallback safe movement
        safe_moves = [a for a in valid_actions if self._next_pos(my_pos, a) not in danger_soon]
        return random.choice(safe_moves) if safe_moves else 0

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and grid[x, y] in [0, 3, 4]

    def _valid_actions(self, grid, my_pos, occupied):
        actions = [0]
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(my_pos, a)
            if self._passable(grid, nx, ny) and (nx, ny) not in occupied:
                actions.append(a)
        return actions

    def _blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = grid[x, y]
                if cell == 1:
                    break
                tiles.add((x, y))
                if cell == 2:
                    break
        return tiles

    def _danger_tiles(self, grid, bombs, players, default_radius=2):
        danger_soon = set()
        danger_now = set()
        for b in bombs:
            bx, by, timer = int(b[0]), int(b[1]), int(b[2])
            owner_id = int(b[3]) if len(b) > 3 else -1
            if timer <= 0:
                continue
            radius = default_radius
            if 0 <= owner_id < len(players):
                radius = max(1, int(players[owner_id][4]) + 1)
            blast = self._blast_tiles(grid, bx, by, radius)
            danger_soon |= blast
            if timer <= 1:
                danger_now |= blast
        return danger_soon, danger_now

    def _safe_tiles(self, grid, danger_soon):
        safe = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] in [0, 3, 4] and (x, y) not in danger_soon:
                    safe.add((x, y))
        return safe

    def _move_to_targets(self, grid, start, occupied, targets, danger_soon):
        if not targets:
            return None
        q = deque([(start, None)])
        seen = {start}
        while q:
            pos, first_action = q.popleft()
            if pos in targets and first_action is not None:
                return first_action
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos in occupied:
                    continue
                if npos in danger_soon:
                    continue
                seen.add(npos)
                q.append((npos, a if first_action is None else first_action))
        return None

    def _item_tiles(self, grid, prefer_capacity=False, prefer_radius=False):
        preferred_values = set()
        if prefer_radius:
            preferred_values.add(3)
        if prefer_capacity:
            preferred_values.add(4)

        preferred_tiles = {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if grid[x, y] in preferred_values
        }
        if preferred_tiles:
            return preferred_tiles

        return {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if grid[x, y] in [3, 4]
        }

    def _count_boxes_in_blast(self, grid, my_pos, radius):
        cnt = 0
        for x, y in self._blast_tiles(grid, my_pos[0], my_pos[1], radius):
            if grid[x, y] == 2:
                cnt += 1
        return cnt

    def _move_to_nearest_safe(self, grid, start, occupied, danger, search_depth=8):
        q = deque([(start, 0, None)])
        seen = {start}
        while q:
            pos, d, first_action = q.popleft()
            if pos not in danger and d > 0:
                return first_action
            if d >= search_depth:
                continue
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if not self._passable(grid, nx, ny) or npos in occupied or npos in seen:
                    continue
                seen.add(npos)
                q.append((npos, d + 1, a if first_action is None else first_action))
        return None

    def _can_escape_after_placing(self, grid, my_pos, occupied, existing_danger, bomb_radius):
        my_blast = self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        combined_danger = set(existing_danger) | my_blast
        return self._move_to_nearest_safe(grid, my_pos, occupied, combined_danger) is not None

    def _box_bomb_spots(self, grid, occupied):
        targets = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] != 2:
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if self._passable(grid, nx, ny) and (nx, ny) not in occupied:
                        targets.add((nx, ny))
        return targets