from PIL import Image, ImageDraw


TOP_BAR_HEIGHT = 48


def _draw_text(draw, position, text, fill=(0, 0, 0)):
    draw.text(position, text, fill=fill)


def blast_tiles(grid, bx, by, radius):
    tiles = {(bx, by)}
    for drow, dcol in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for r in range(1, radius + 1):
            tr, tc = bx + drow * r, by + dcol * r
            if not (0 <= tr < len(grid) and 0 <= tc < len(grid[0])):
                break
            cell = int(grid[tr][tc])
            if cell == 1:
                break
            tiles.add((tr, tc))
            if cell == 2:
                break
    return tiles


def explosion_tiles_from_transition(prev_obs, curr_obs):
    if prev_obs is None:
        return set()

    prev_bombs = prev_obs.get("bombs", [])
    curr_bombs = curr_obs.get("bombs", [])
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
        tiles.update(blast_tiles(prev_grid, bx, by, radius))
    return tiles


def render_match_frame(obs, prev_obs=None, cell_size=40, top_bar_height=TOP_BAR_HEIGHT):
    width = len(obs["map"][0])
    height = len(obs["map"])

    img = Image.new("RGBA", (width * cell_size, height * cell_size + top_bar_height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    draw.rectangle([0, 0, width * cell_size, top_bar_height], fill=(30, 30, 30))
    _draw_text(draw, (10, 14), f"Step {int(obs.get('_step', 0))}", fill=(245, 245, 245))

    board_top = top_bar_height
    draw.rectangle([0, board_top, width * cell_size, board_top + height * cell_size], fill=(144, 238, 144))

    grid = obs["map"]
    for row in range(height):
        for col in range(width):
            cell = int(grid[row][col])
            rect = [
                col * cell_size,
                board_top + row * cell_size,
                (col + 1) * cell_size,
                board_top + (row + 1) * cell_size,
            ]
            if cell == 1:
                draw.rectangle(rect, fill=(80, 80, 80), outline=(40, 40, 40))
            elif cell == 2:
                draw.rectangle(rect, fill=(139, 69, 19), outline=(101, 67, 33))
                draw.line((rect[0], rect[1], rect[2], rect[3]), fill=(101, 67, 33), width=2)
                draw.line((rect[2], rect[1], rect[0], rect[3]), fill=(101, 67, 33), width=2)
            elif cell == 3:
                draw.rectangle(rect, fill=(225, 225, 225))
                cx = rect[0] + cell_size // 2
                cy = rect[1] + cell_size // 2
                draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(255, 0, 0))
            elif cell == 4:
                draw.rectangle(rect, fill=(225, 225, 225))
                cx = rect[0] + cell_size // 2
                cy = rect[1] + cell_size // 2
                draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(0, 0, 255))

    explosion_tiles = explosion_tiles_from_transition(prev_obs, obs)
    for row, col in explosion_tiles:
        px = col * cell_size
        py = board_top + row * cell_size
        draw.rectangle([px, py, px + cell_size, py + cell_size], fill=(255, 140, 0, 90))
        draw.ellipse([px + 10, py + 10, px + cell_size - 10, py + cell_size - 10], fill=(255, 220, 120))

    for b in obs.get("bombs", []):
        bx, by, timer = int(b[0]), int(b[1]), int(b[2])
        pulse = 2 if (int(obs.get('_step', 0)) + timer) % 2 == 0 else 0
        cx = by * cell_size + cell_size // 2
        cy = board_top + bx * cell_size + cell_size // 2
        radius = max(7, cell_size // 4 + pulse)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(20, 20, 20), outline=(0, 0, 0))
        _draw_text(draw, (cx - 5, cy - 8), str(timer), fill=(255, 255, 255))

    colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    for i, p in enumerate(obs.get("players", [])):
        if not p[2]:
            continue
        row, col = int(p[0]), int(p[1])
        rect = [
            col * cell_size + 5,
            board_top + row * cell_size + 5,
            (col + 1) * cell_size - 5,
            board_top + (row + 1) * cell_size - 5,
        ]
        draw.ellipse(rect, fill=colors[i % len(colors)], outline=(0, 0, 0))
        _draw_text(draw, (col * cell_size + 15, board_top + row * cell_size + 15), str(i), fill=(0, 0, 0))

    return img.convert("RGB")