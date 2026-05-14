# Agent Development Guide

This guide explains how to build, test, and submit your Bomberland AI agent.

## 🤖 Baseline Agents
You can find several baseline agents in this directory to use as a starting point:
*   `random_agent.py`: Simple random actions.
*   `simple_rule_agent.py`: Avoids bombs and places bombs.
*   `tactical_rule_agent.py`: Uses BFS for pathfinding and targets enemies.
*   `dqn_agent/`: A Deep Q-Network implementation (not chosen as one of the official baseline agents).

## 🛠️ Developing Your Agent
Your agent must be a Python class named `Agent` inside a file named `agent.py`. It must implement an `act` method:

```python
class Agent:
    def __init__(self):
        # Initialize your model or state here
        pass

    def act(self, observation):
        # observation contains the game state
        # return an action (0-5)
        return 0 
```

### Constraints
*   **Time Limit**: 100ms per step.
*   **Resources**: CPU-only evaluation. No GPU access.
*   **Isolation**: No network access or file writing during the match.

## Local Training (Kaggle)

Follow the provided steps to train DQN on Kaggle Platform: 
1. Get github access token from github.com > Settings > Developer settings > Personal Access Tokens > Generate new token > Choose scope for the key (allow read repo) > Copy the key
2. Create a new notebook on kaggle and add to Settings > Secrets > Key: any name for the key such as "dqn", Value: Paste the key here
3. Paste the following code in 4 cells:
```
# Cell 1: Get token based on the key in Step 2
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("dqn") 

# Cell 2: Change the "your github username here"
!git clone https://{your github username here}:{secret_value_0}@github.com/VLTisME/Bomberland-GDGoC-AI-Challenge.git

# Cell 3: Inspect to see any error
%cd /kaggle/working/Bomberland-GDGoC-AI-Challenge
%ls

# Cell 4: Train 10000 episodes versus a tactical baseline agent
!python /kaggle/working/Bomberland-GDGoC-AI-Challenge/agent/dqn_agent/agent.py --enemy_type tactical --num_episodes 10000 --save_model
```

## 🧪 Local Testing

Guidance for local testing before submitting your agent. All participant scripts are located in the `scripts/participant/` folder.

### 1. Evaluate Agent Performance
To get a quick estimate of your agent's TrueSkill rating, run the ranking script. It will play matches against random baseline bots and compute your estimated win rate and leaderboard score.
```bash
python -m scripts.participant.estimate_rankings --agent_path path/to/your/agent/ --num_matches 100
```

### 2. Run Headless or Visual Matches
Use the local match script to pit specific agents against each other or watch them play.
```bash
python -m scripts.participant.run_local_match --agent_paths path/to/your/agent/ None None None --visualize true
```

#### Arguments:
*   `--agent_paths`: Expects exactly 4 arguments representing the 4 players. You can pass:
    *   **A folder path** (e.g., `agent/dqn_agent/`): Perfect for Deep Reinforcement Learning agents. It will automatically load `agent.py` inside that folder, allowing your agent to load its weights relative to itself.
    *   **A file path** (e.g., `agent/random_agent.py`): Perfect for rule-based agents that don't need external weights.
    *   **A baseline name** (e.g., `TacticalRuleAgent`): Explicitly loads a built-in bot.
    *   `None` or `Random`: Automatically loads a random baseline bot.
*   `--visualize true`: Opens the PyGame window to watch the match live. Set to `false` for fast headless testing.

### 3. Replay Saved Matches
If you downloaded a match log (`.json`) from the Google Drive, you can replay it:
```bash
python -m scripts.participant.replay_viewer path/to/log.json
```

## 📤 Submission Process
1.  **Package**: Create a `.zip` file containing:
    *   `agent.py` (Required)
    *   Any weights or models (e.g., `.pth` files)
2.  **Submit**: Use the Official Submission Form with your **Team ID** and **Token**.
3.  **Feedback**: Once submitted, your agent will immediately play 25 matches. You can check the leaderboard for your updated rating.
