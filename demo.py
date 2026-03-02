import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_agent.traffic_env      import TrafficEnv, GREEN_TIME_OPTIONS, WEATHER_LABELS
from rl_agent.q_learning import QLearningAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QTABLE_CANDIDATES = [
    os.path.join(BASE_DIR, 'rl_agent', 'rl_agent', 'q_table.pkl'),
    os.path.join(BASE_DIR, 'rl_agent', 'q_table.pkl'),
]

# ─────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def divider(char='─', width=70):
    print(char * width)

def header(title):
    divider('═')
    print(f"  {title}")
    divider('═')


def print_lane_bar(label, count, max_count=14, color_code=None):
    """Prints a visual bar showing vehicle queue length."""
    filled = int((count / max_count) * 20)
    bar    = '█' * filled + '░' * (20 - filled)
    print(f"    {label:5s} | {bar} | {count:2d} vehicles")


def show_intersection_snapshot(lane_counts, priority_lane, green_time, weather):
    """Prints a simple text visual of the intersection state."""
    w = WEATHER_LABELS.get(weather, 'Clear')
    print(f"\n  Weather: {w}    Priority Lane: {priority_lane}    Green Time: {green_time}s\n")

    # North
    n = lane_counts['NORTH']
    marker = '🟢' if priority_lane == 'NORTH' else '🔴'
    print(f"              {'↑ ' * min(n,5):15s}")
    print(f"           NORTH({n}) {marker}")
    print(f"              |")

    # West / East
    w_c = lane_counts['WEST']
    e_c = lane_counts['EAST']
    wm  = '🟢' if priority_lane == 'WEST' else '🔴'
    em  = '🟢' if priority_lane == 'EAST' else '🔴'
    print(f"  WEST({w_c}) {wm} {'←'*min(w_c,4):8s}[INTER]{'→'*min(e_c,4):8s} {em} EAST({e_c})")

    # South
    s   = lane_counts['SOUTH']
    sm  = '🟢' if priority_lane == 'SOUTH' else '🔴'
    print(f"              |")
    print(f"           SOUTH({s}) {sm}")
    print(f"              {'↓ ' * min(s,5):15s}\n")


# ─────────────────────────────────────────────────────────────────
#  DEMO 1 — TRADITIONAL FIXED SIGNAL
#  Shows the problems with the old system
# ─────────────────────────────────────────────────────────────────

def demo_traditional():
    header("DEMO 1 — Traditional Fixed-Time Signal (The Old Way)")
    print("  Rules:")
    print("  ✗ Every lane gets exactly 20s green — regardless of vehicle count")
    print("  ✗ Strict rotation: NORTH → SOUTH → WEST → EAST")
    print("  ✗ Empty lanes still get green time (wasted!)")
    print("  ✗ No weather awareness")
    print()
    input("  Press Enter to start...\n")

    env      = TrafficEnv(weather_mode='clear')
    env.reset()
    env.weather = 0
    lane_idx = 0
    lanes    = ['NORTH', 'SOUTH', 'WEST', 'EAST']

    divider()
    print(f"  {'Step':>4} | {'Serving':>8} | {'Green':>5} | "
          f"N   S   W   E  | {'Issue'}")
    divider()

    total_wait = 0.0

    for i in range(min(12, len(env.data))):
        row    = env.data.iloc[i]
        counts = {
            'NORTH': int(row['NORTH']),
            'SOUTH': int(row['SOUTH']),
            'WEST' : int(row['WEST']),
            'EAST' : int(row['EAST']),
        }

        current_lane = lanes[lane_idx]
        count_here   = counts[current_lane]

        # Detect issues
        issues = []
        if count_here == 0:
            issues.append("⚠ WASTED — lane is empty!")
        elif count_here >= 8:
            issues.append("⚠ UNDERTIMED — 20s not enough for heavy traffic!")

        # Check if a more congested lane is being ignored
        max_other = max(counts[l] for l in lanes if l != current_lane)
        if max_other > count_here + 4:
            issues.append("⚠ MORE CONGESTED LANE WAITING!")

        issue_str = ' | '.join(issues) if issues else '✓ OK'

        print(
            f"  {i+1:>4} | {current_lane:>8} | {'20s':>5} | "
            f"{counts['NORTH']:>2}  {counts['SOUTH']:>2}  "
            f"{counts['WEST']:>2}  {counts['EAST']:>2}  | "
            f"{issue_str}"
        )

        # Simulate waiting
        for lane in lanes:
            if lane == current_lane:
                cleared = min(counts[lane], 20 / 8.0)
                env.waiting_times[lane] = max(0.0, env.waiting_times[lane] - cleared)
            else:
                env.waiting_times[lane] += counts[lane] * 0.5

        total_wait += sum(env.waiting_times.values())
        lane_idx    = (lane_idx + 1) % 4
        time.sleep(0.4)

    divider()
    print(f"\n  Total Waiting Time (Traditional) : {total_wait:.2f}")
    print(f"\n  Problems observed:")
    print(f"  • Green light given to empty lanes   → wasted time")
    print(f"  • Fixed 20s regardless of congestion → inefficient")
    print(f"  • No priority for busiest lane       → congestion builds")
    print(f"  • No weather adjustment              → dangerous in rain/fog\n")


# ─────────────────────────────────────────────────────────────────
#  DEMO 2 — RL AGENT (YOUR SYSTEM)
#  Shows how the trained agent solves all problems
# ─────────────────────────────────────────────────────────────────

def demo_rl_agent(weather_choice):
    weather_map  = {1: 0, 2: 1, 3: 2}
    weather_name = {1: 'Clear', 2: 'Rain', 3: 'Fog'}
    weather_mode = {1: 'clear', 2: 'adverse', 3: 'adverse'}

    w_int  = weather_map[weather_choice]
    w_name = weather_name[weather_choice]
    w_mode = weather_mode[weather_choice]

    header(f"DEMO 2 — RL Agent (Your System) | Weather: {w_name}")
    print("  Upgrades:")
    print("  ✅ Empty lanes SKIPPED — no wasted green time")
    print("  ✅ Most congested lane served FIRST")
    print("  ✅ Dynamic green time: 10s / 20s / 30s / 40s")
    print("  ✅ Weather-aware — longer green in rain/fog")
    print("  ✅ Starvation prevention — no lane ignored forever")
    print()
    input("  Press Enter to start...\n")

    # Load trained agent
    agent         = QLearningAgent(n_actions=4)
    qtable_path   = next((p for p in QTABLE_CANDIDATES if os.path.exists(p)), QTABLE_CANDIDATES[0])
    loaded        = agent.load(qtable_path)
    agent.epsilon = 0.0   # pure exploitation — no random actions

    if not loaded:
        print("  ⚠ No saved Q-table found! Run main.py first to train.")
        print(f"  Checked: {QTABLE_CANDIDATES[0]}")
        print(f"           {QTABLE_CANDIDATES[1]}")
        return 0

    print(f"  Q-table loaded: {len(agent.q_table)} states learned\n")

    env         = TrafficEnv(weather_mode=w_mode)
    state       = env.reset()
    env.weather = w_int   # force selected weather

    divider()
    print(f"  {'Step':>4} | {'Priority':>8} | {'Green':>5} | "
          f"N   S   W   E  | {'Skipped':<10} | {'Wait':>8} | Notes")
    divider()

    total_wait = 0
    step       = 0

    while not env.done:
        action                         = agent.choose_action(state, weather=state[4])
        next_state, reward, done, info = env.step(action)

        skipped_str = ', '.join(info['empty_lanes']) if info['empty_lanes'] else '—'
        forced      = info.get('starved_lanes', [])
        forced_str  = f'⚠ FORCED({forced[0]})' if forced else ''

        # Highlight smart decisions
        counts = info['lane_counts']
        p_lane = info['priority_lane']
        notes  = []
        if info['empty_lanes']:
            notes.append(f"Skipped empty")
        if info['green_time'] >= 30 and counts.get(p_lane, 0) >= 6:
            notes.append(f"Long green for heavy lane ✓")
        if info['green_time'] == 10 and counts.get(p_lane, 0) <= 3:
            notes.append(f"Short green for light lane ✓")
        if env.weather > 0 and info['green_time'] >= 20:
            notes.append(f"Extended for {w_name} ✓")
        note_str = ' | '.join(notes) if notes else ''

        print(
            f"  {step+1:>4} | "
            f"{str(p_lane):>8} | "
            f"{info['green_time']:>4}s | "
            f"{counts['NORTH']:>2}  {counts['SOUTH']:>2}  "
            f"{counts['WEST']:>2}  {counts['EAST']:>2}  | "
            f"{skipped_str:<10} | "
            f"{info['total_waiting']:>8.1f} | "
            f"{forced_str}{note_str}"
        )

        total_wait += info['total_waiting']
        state       = next_state
        step       += 1
        time.sleep(0.4)

        if done:
            break

    divider()
    print(f"\n  Total Waiting Time (RL Agent) : {total_wait:.2f}")
    return total_wait


# ─────────────────────────────────────────────────────────────────
#  DEMO 3 — SIDE BY SIDE COMPARISON
# ─────────────────────────────────────────────────────────────────

def demo_comparison(rl_wait, weather_choice):
    weather_name = {1: 'Clear', 2: 'Rain', 3: 'Fog'}
    w_name       = weather_name[weather_choice]

    header("DEMO 3 — Final Comparison")

    env      = TrafficEnv(weather_mode='clear')
    baseline = env.run_fixed_baseline(fixed_green=20)

    improvement = ((baseline - rl_wait) / baseline) * 100 if baseline > 0 else 0

    print(f"\n  Weather Condition         : {w_name}")
    print()
    divider()
    print(f"  {'System':<35} | {'Total Wait':>12} | {'Improvement':>12}")
    divider()
    print(f"  {'Traditional Fixed Signal (20s)':<35} | {baseline:>12.2f} | {'baseline':>12}")
    print(f"  {'RL Agent (Your System)':<35} | {rl_wait:>12.2f} | {improvement:>11.1f}%")
    divider()

    print(f"\n  🎯 Result: RL Agent reduces waiting time by {improvement:.1f}%")
    print()

    # Visual bar comparison
    max_val   = max(baseline, rl_wait)
    base_bar  = int((baseline / max_val) * 40)
    rl_bar    = int((rl_wait  / max_val) * 40)

    print(f"  Traditional | {'█' * base_bar} {baseline:.0f}")
    print(f"  RL Agent    | {'█' * rl_bar} {rl_wait:.0f}  ← {improvement:.1f}% less waiting")
    print()

    print("  Key Advantages Demonstrated:")
    print("  ✅ Empty lanes skipped   → no green time wasted")
    print("  ✅ Priority-based        → busiest lane served first")
    print("  ✅ Dynamic green time    → matched to actual queue length")
    print("  ✅ Weather-aware         → longer green in rain/fog")
    print("  ✅ Self-learning         → improves with more data\n")


# ─────────────────────────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    header("Weather-Aware RL Traffic Signal Control — LIVE DEMO")
    print("  Final Year Project Demo")
    print("  YOLOv8 Vehicle Detection + Q-Learning Signal Control\n")

    print("  Select weather condition for RL demo:")
    print("  1. Clear Weather")
    print("  2. Rain")
    print("  3. Fog")
    print()

    while True:
        try:
            choice = int(input("  Enter 1 / 2 / 3: ").strip())
            if choice in [1, 2, 3]:
                break
            print("  Please enter 1, 2, or 3.")
        except ValueError:
            print("  Please enter 1, 2, or 3.")

    print()
    print("  Demo will run in 3 parts:")
    print("  Part 1 → Traditional fixed signal (shows the problems)")
    print("  Part 2 → Your RL agent          (shows the solution)")
    print("  Part 3 → Side-by-side comparison (shows the result)")
    print()
    input("  Press Enter to begin...\n")

    # Part 1 — Traditional
    demo_traditional()
    input("\n  Press Enter for Part 2 — RL Agent...\n")

    # Part 2 — RL Agent
    rl_total_wait = demo_rl_agent(choice)
    input("\n  Press Enter for Part 3 — Comparison...\n")

    # Part 3 — Comparison
    demo_comparison(rl_total_wait, choice)

    divider('═')
    print("  Demo Complete!")
    print("  For full results run: python results/plots.py")
    divider('═')