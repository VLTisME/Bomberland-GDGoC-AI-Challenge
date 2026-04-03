import random
from collections import deque


class GeniusRuleAgent:
    """
    Actions:
    0: STOP
    1: LEFT
    2: RIGHT
    3: UP
    4: DOWN
    5: PLACE_BOMB
    """

    MOVES = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    team_id = "GeniusRuleAgent"
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.escape_mode = False

    ############################################################

    def act(self, obs):

        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]

        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        bomb_radius = max(1, int(bomb_bonus) + 1)

        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}

        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]

        blocked = set(bomb_positions)
        blocked.discard(my_pos)

        danger_soon, danger_now = self._danger_tiles(grid, bombs, players)

        valid_actions = self._valid_actions(grid, my_pos, blocked)

        ########################################################
        # ESCAPE MODE
        ########################################################

        if self.escape_mode or my_pos in danger_soon:

            escape = self._move_to_nearest_safe(
                grid, my_pos, blocked, danger_soon, search_depth=10
            )

            if escape is not None:
                if my_pos not in danger_soon:
                    self.escape_mode = False
                return escape

        ########################################################
        # ITEM COLLECTION
        ########################################################

        item_tiles = self._item_tiles(grid)

        if item_tiles:

            move = self._move_toward_targets(
                grid, my_pos, item_tiles, blocked, danger_soon
            )

            if move is not None:
                return move

        ########################################################
        # BOMB ENEMY
        ########################################################

        if bombs_left > 0 and my_pos not in bomb_positions:

            if self._can_hit_enemy_with_bomb(
                grid, my_pos, enemies, bomb_radius
            ):

                escape = self._move_to_nearest_safe(
                    grid,
                    my_pos,
                    blocked,
                    danger_soon
                    | self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius),
                    search_depth=6,
                )

                if escape is not None:
                    self.escape_mode = True
                    return 5

        ########################################################
        # FARM BOXES
        ########################################################

        if bombs_left > 0 and my_pos not in bomb_positions:

            boxes = self._count_boxes_in_blast(
                grid, my_pos, bomb_radius
            )

            if boxes >= 1:

                escape = self._move_to_nearest_safe(
                    grid,
                    my_pos,
                    blocked,
                    danger_soon
                    | self._blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius),
                    search_depth=6,
                )

                if escape is not None:
                    self.escape_mode = True
                    return 5

        ########################################################
        # MOVE TO BOX SPOTS
        ########################################################

        box_spots = self._box_bomb_spots(grid, blocked)

        if box_spots:

            move = self._move_toward_targets(
                grid, my_pos, box_spots, blocked, danger_soon
            )

            if move is not None:
                return move

        ########################################################
        # ENEMY PRESSURE
        ########################################################

        if enemies:

            move = self._move_toward_targets(
                grid, my_pos, set(enemies), blocked, danger_soon
            )

            if move is not None:
                return move

        ########################################################
        # RANDOM SAFE MOVE
        ########################################################

        safe_moves = [
            a for a in valid_actions
            if self._next_pos(my_pos, a) not in danger_soon
        ]

        return random.choice(safe_moves) if safe_moves else 0

    ############################################################
    # BASIC HELPERS
    ############################################################

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and grid[x, y] in [0, 3, 4]

    def _valid_actions(self, grid, my_pos, blocked):

        actions = [0]

        for a in [1, 2, 3, 4]:

            nx, ny = self._next_pos(my_pos, a)

            if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                actions.append(a)

        return actions

    ############################################################
    # DANGER MODEL
    ############################################################

    def _blast_tiles(self, grid, bx, by, radius):

        tiles = {(bx, by)}

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

            for r in range(1, radius + 1):

                x = bx + dx * r
                y = by + dy * r

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

    ############################################################
    # BFS ESCAPE
    ############################################################

    def _move_to_nearest_safe(self, grid, start, blocked, danger, search_depth=8):

        q = deque([(start, 0, None)])
        seen = {start}

        while q:

            pos, d, first_action = q.popleft()

            if pos not in danger and d > 0:
                return first_action

            if d >= search_depth:
                continue

            for a in [1, 2, 3, 4, 0]:

                nx, ny = self._next_pos(pos, a)

                if a != 0:

                    if not self._passable(grid, nx, ny):
                        continue

                    if (nx, ny) in blocked:
                        continue

                npos = (nx, ny)

                if npos in seen:
                    continue

                seen.add(npos)

                q.append((npos, d + 1, a if first_action is None else first_action))

        return None

    ############################################################
    # PATHFINDING
    ############################################################

    def _move_toward_targets(self, grid, start, targets, blocked, danger):

        q = deque([(start, None)])
        seen = {start}

        while q:

            pos, first_action = q.popleft()

            if pos in targets and first_action is not None:
                return first_action

            for a in [1, 2, 3, 4]:

                nx, ny = self._next_pos(pos, a)

                if not self._passable(grid, nx, ny):
                    continue

                if (nx, ny) in blocked:
                    continue

                if (nx, ny) in danger:
                    continue

                npos = (nx, ny)

                if npos in seen:
                    continue

                seen.add(npos)

                q.append((npos, a if first_action is None else first_action))

        return None

    ############################################################
    # STRATEGY HELPERS
    ############################################################

    def _count_boxes_in_blast(self, grid, pos, radius):

        tiles = self._blast_tiles(grid, pos[0], pos[1], radius)

        return sum(grid[x, y] == 2 for x, y in tiles)

    def _can_hit_enemy_with_bomb(self, grid, my_pos, enemies, radius):

        mx, my = my_pos

        for ex, ey in enemies:

            if mx == ex and abs(ey - my) <= radius:
                return True

            if my == ey and abs(ex - mx) <= radius:
                return True

        return False

    def _box_bomb_spots(self, grid, blocked):

        spots = set()

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):

                if grid[x, y] != 2:
                    continue

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                    nx, ny = x + dx, y + dy

                    if self._passable(grid, nx, ny) and (nx, ny) not in blocked:
                        spots.add((nx, ny))

        return spots

    def _item_tiles(self, grid):

        return {
            (x, y)
            for x in range(grid.shape[0])
            for y in range(grid.shape[1])
            if grid[x, y] in [3, 4]
        }