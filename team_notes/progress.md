# Progress Tracking

Theo dõi tiến độ phát triển agent theo từng thay đổi có kiểm chứng.

## Quy ước ghi log

Mỗi mục nên ghi đủ:

- Ngày thực hiện
- Người thực hiện
- Thay đổi chính
- Cách kiểm thử
- Kết quả
- Việc tiếp theo

## Log thay đổi

### 2026-05-24 - Khởi tạo workspace team

**Người thực hiện:** Hưng

**Thay đổi chính:**

- Clone lại repository từ fork `hunglee149/Bomberland-GDGoC-AI-Challenge`.
- Thêm remote `upstream` trỏ về repo gốc `VLTisME/Bomberland-GDGoC-AI-Challenge`.
- Tạo cấu trúc làm việc riêng cho team:
  - `submission/`
  - `experiments/`
  - `team_notes/`
  - `scripts/team/`
- Thêm template agent tối thiểu tại `submission/agent.py`.

**Cách kiểm thử:**

```bash
python -m scripts.participant.run_local_match --agent_paths submission RandomAgent RandomAgent RandomAgent --num_episodes 1 --max_steps 5
```

**Kết quả:**

- Agent trong `submission/` load được.
- Match chạy xong không lỗi.

**Việc tiếp theo:**

- Thiết kế baseline rule-based đầu tiên.
- Thêm benchmark script cho team.
- Ghi kết quả so sánh với các baseline chính thức.

### 2026-05-24 - Tạo branch làm việc

**Người thực hiện:** Codex

**Thay đổi chính:**

- Kiểm tra branch hiện tại là `main`.
- Fetch `origin` và `upstream`.
- Xác nhận `main` đang đồng bộ với:
  - `origin/main`
  - `upstream/main`
- Tạo branch làm việc `team/setup-progress-tracking` từ `main`.

**Cách kiểm thử:**

```bash
git fetch origin
git fetch upstream
git rev-list --left-right --count main...origin/main
git rev-list --left-right --count main...upstream/main
git checkout -b team/setup-progress-tracking
```

**Kết quả:**

- `main...origin/main`: `0 0`
- `main...upstream/main`: `0 0`
- Đã chuyển sang branch `team/setup-progress-tracking`.

**Việc tiếp theo:**

- Commit cấu trúc workspace team và progress tracking.
- Push branch lên `origin`.
- Tạo pull request vào `main` nếu cần review trong nhóm.

## Backlog

- [ ] Tạo agent rule-based có khả năng né bom cơ bản.
- [ ] Thêm danger map dựa trên bomb timer và radius.
- [ ] Thêm BFS tìm đường tới item/box/enemy.
- [ ] Viết script benchmark nhiều seed trong `scripts/team/`.
- [ ] Ghi lại kết quả benchmark vào `experiments/`.
- [ ] Nghiên cứu reward design nếu thử DQN/RL.
