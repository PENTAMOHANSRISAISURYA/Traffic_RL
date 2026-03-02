import os
import sys
import time
import numpy as np

# ── Make sure imports work from project root ──────────────────────
# Add parent directory to path so we can import traffic_env and q_learning
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_env import TrafficEnv, GREEN_TIME_OPTIONS, WEATHER_LABELS
from q_learning import QLearningAgent

# ─────────────────────────────────────────────────────────────────
#  TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────

EPISODES        = 500     # number of training episodes
PRINT_EVERY     = 50      # print progress every N episodes
SAVE_EVERY      = 100     # save Q-table every N episodes
RESULTS_DIR     = '../results'
QTABLE_PATH     = 'q_table.pkl'


# ─────────────────────────────────────────────────────────────────
#  HELPER — print a clean section header
# ─────────────────────────────────────────────────────────────────

def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────
#  PHASE 1 — RUN FIXED BASELINE (Traditional System)
#  Run BEFORE training so we have a fair comparison number
# ─────────────────────────────────────────────────────────────────

def run_baseline(env):
    header("PHASE 1 — Fixed Baseline (Traditional 20s Signal)")
    print("  Simulating traditional fixed-time traffic signal...")
    print("  Rules: every lane gets 20s green, strict rotation,")
    print("         empty lanes served, no weather awareness.\n")

    baseline_wait = env.run_fixed_baseline(fixed_green=20)
    print(f"  Total Waiting Time (Fixed 20s) : {baseline_wait:.2f}")
    print(f"  This is the benchmark your RL agent must beat.")
    return baseline_wait


# ─────────────────────────────────────────────────────────────────
#  PHASE 2 — TRAIN RL AGENT (Weather-Aware Q-Learning)
# ─────────────────────────────────────────────────────────────────

def train(env, agent):
    header(f"PHASE 2 — Training RL Agent ({EPISODES} Episodes)")
    print(f"  Learning Rate  (α) : {agent.alpha}")
    print(f"  Discount       (γ) : {agent.gamma}")
    print(f"  Epsilon Start      : {agent.epsilon}")
    print(f"  Epsilon End        : {agent.epsilon_end}")
    print(f"  Epsilon Decay      : {agent.epsilon_decay}")
    print(f"  Weather Mode       : random (60% Clear, 25% Rain, 15% Fog)\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_time     = time.time()

    # Track metrics per episode for plotting later
    all_rewards    = []
    all_waits      = []
    all_epsilons   = []
    weather_counts = {0: 0, 1: 0, 2: 0}

    for episode in range(1, EPISODES + 1):

        state        = env.reset()
        total_reward = 0.0
        total_wait   = 0.0
        steps        = 0

        weather_counts[env.weather] += 1

        # ── Episode loop ──────────────────────────────────────────
        while not env.done:

            # Agent picks action (weather-aware exploration)
            action = agent.choose_action(state, weather=state[4])

            # Environment responds
            next_state, reward, done, info = env.step(action)

            # Agent learns from this experience
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            total_wait   += info['total_waiting']
            steps        += 1
            state         = next_state

            if done:
                break

        # ── End of episode ────────────────────────────────────────
        agent.decay_epsilon()
        agent.record_episode(total_reward)

        all_rewards.append(total_reward)
        all_waits.append(total_wait)
        all_epsilons.append(agent.epsilon)

        # ── Print progress ────────────────────────────────────────
        if episode % PRINT_EVERY == 0:
            avg_reward = np.mean(all_rewards[-PRINT_EVERY:])
            avg_wait   = np.mean(all_waits[-PRINT_EVERY:])
            elapsed    = time.time() - start_time
            print(
                f"  Episode {episode:4d}/{EPISODES} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Wait: {avg_wait:8.2f} | "
                f"ε: {agent.epsilon:.4f} | "
                f"Q-States: {len(agent.q_table):4d} | "
                f"Time: {elapsed:.1f}s"
            )

        # ── Save Q-table periodically ─────────────────────────────
        if episode % SAVE_EVERY == 0:
            agent.save(QTABLE_PATH)

    # ── Training complete ─────────────────────────────────────────
    total_time = time.time() - start_time
    header("Training Complete!")
    print(f"  Total Episodes     : {EPISODES}")
    print(f"  Total Time         : {total_time:.1f}s")
    print(f"  Final Epsilon      : {agent.epsilon:.4f}")
    print(f"  Q-Table Size       : {len(agent.q_table)} states learned")
    print(f"\n  Weather distribution across episodes:")
    print(f"    Clear : {weather_counts[0]} episodes")
    print(f"    Rain  : {weather_counts[1]} episodes")
    print(f"    Fog   : {weather_counts[2]} episodes")

    agent.save(QTABLE_PATH)

    return all_rewards, all_waits, all_epsilons


# ─────────────────────────────────────────────────────────────────
#  PHASE 3 — EVALUATE TRAINED AGENT
#  Test the agent in 3 scenarios and compare against baseline
# ─────────────────────────────────────────────────────────────────

def evaluate(env, agent, baseline_wait):
    header("PHASE 3 — Evaluating Trained Agent")

    # Turn off exploration — agent uses Q-table only
    original_epsilon = agent.epsilon
    agent.epsilon    = 0.0

    results = {}

    scenarios = [
        ('clear',   'Clear Weather  (0% exploration)'),
        ('adverse', 'Adverse Weather (Rain/Fog)      '),
        ('random',  'Mixed Weather  (realistic)      '),
    ]

    for mode, label in scenarios:
        env_eval   = TrafficEnv(weather_mode=mode)
        state      = env_eval.reset()
        total_wait = 0.0
        steps      = 0

        while not env_eval.done:
            action                         = agent.choose_action(state, weather=state[4])
            next_state, _, done, info      = env_eval.step(action)
            total_wait                    += info['total_waiting']
            steps                         += 1
            state                          = next_state
            if done:
                break

        improvement = ((baseline_wait - total_wait) / baseline_wait) * 100
        results[mode] = {
            'total_wait' : total_wait,
            'steps'      : steps,
            'improvement': improvement
        }

        status = "✅" if improvement > 0 else "❌"
        print(f"\n  {status} {label}")
        print(f"     Total Waiting Time : {total_wait:.2f}")
        print(f"     vs Fixed Baseline  : {baseline_wait:.2f}")
        print(f"     Improvement        : {improvement:.1f}%")

    # Restore epsilon
    agent.epsilon = original_epsilon

    return results


# ─────────────────────────────────────────────────────────────────
#  PHASE 4 — SAVE RESULTS FOR PLOTTING
# ─────────────────────────────────────────────────────────────────

def save_results(all_rewards, all_waits, all_epsilons, baseline_wait, eval_results):
    header("PHASE 4 — Saving Results")

    import csv
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save training metrics
    train_path = os.path.join(RESULTS_DIR, 'training_metrics.csv')
    with open(train_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'total_wait', 'epsilon'])
        for i, (r, w, e) in enumerate(zip(all_rewards, all_waits, all_epsilons), 1):
            writer.writerow([i, round(r, 2), round(w, 2), round(e, 4)])
    print(f"  Training metrics saved → {train_path}")

    # Save comparison results
    comp_path = os.path.join(RESULTS_DIR, 'comparison.csv')
    with open(comp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'total_wait', 'improvement_pct'])
        writer.writerow(['Fixed Baseline (Traditional)', round(baseline_wait, 2), '0.00'])
        for mode, data in eval_results.items():
            writer.writerow([
                f'RL Agent ({mode})',
                round(data['total_wait'], 2),
                round(data['improvement'], 2)
            ])
    print(f"  Comparison results saved  → {comp_path}")
    print(f"\n  Run  python results/plots.py  to generate graphs!")


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    header("Weather-Aware RL Traffic Signal Control — Training Pipeline")
    print("  YOLOv8 vehicle detection + Q-Learning signal control")
    print("  Solves: empty lane waste, fixed timing, starvation,")
    print("          weather blindness, fixed rotation\n")

    # Create environment and agent
    env   = TrafficEnv(weather_mode='random')
    agent = QLearningAgent(n_actions=4)

    # Check if saved Q-table exists — resume training if so
    if os.path.exists(QTABLE_PATH):
        print(f"  Found existing Q-table at {QTABLE_PATH}")
        ans = input("  Resume previous training? (y/n): ").strip().lower()
        if ans == 'y':
            agent.load(QTABLE_PATH)
            print("  Resuming training from saved state.\n")
        else:
            print("  Starting fresh training.\n")

    # Phase 1 — Baseline
    baseline_wait = run_baseline(env)

    # Phase 2 — Train
    all_rewards, all_waits, all_epsilons = train(env, agent)

    # Phase 3 — Evaluate
    eval_results = evaluate(env, agent, baseline_wait)

    # Phase 4 — Save
    save_results(all_rewards, all_waits, all_epsilons, baseline_wait, eval_results)

    header("Pipeline Complete!")
    print("  Next step → python results/plots.py")
    print("  to generate your comparison graphs.\n")