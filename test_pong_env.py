import numpy as np
from pong_env import VectorizedPongEnv

def test_pong_env():
    n_games = 50000
    steps_per_game = 2000
    env = VectorizedPongEnv(n_envs=n_games)
    
    obs = env.reset()
    
    for _ in range(steps_per_game):
        actions = np.zeros(n_games)  # Stationary player paddle
        obs, rewards, done, _ = env.step(actions)

    # Calculate win ratios
    player_scores = env.scores[:, 0]
    ai_scores = env.scores[:, 1]
    
    ai_wins = np.sum(ai_scores > player_scores)
    player_wins = np.sum(player_scores > ai_scores)
    ties = np.sum(player_scores == ai_scores)

    ai_win_ratio = ai_wins / n_games
    player_win_ratio = player_wins / n_games
    tie_ratio = ties / n_games

    print(f"Games played: {n_games}")
    print(f"AI win ratio: {ai_win_ratio:.2f}")
    print(f"Player win ratio: {player_win_ratio:.2f}")
    print(f"Tie ratio: {tie_ratio:.2f}")
    print(f"Average AI score: {np.mean(ai_scores):.2f}")
    print(f"Average player score: {np.mean(player_scores):.2f}")

if __name__ == "__main__":
    test_pong_env()
