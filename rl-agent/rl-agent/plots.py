import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

RESULTS_DIR    = '../results'
TRAIN_CSV      = os.path.join(RESULTS_DIR, 'training_metrics.csv')
COMPARISON_CSV = os.path.join(RESULTS_DIR, 'comparison.csv')
ROLLING_WINDOW = 30

COLORS = {
    'rl'       : '#2196F3',
    'baseline' : '#F44336',
    'rolling'  : '#1565C0',
    'clear'    : '#4CAF50',
    'adverse'  : '#FF9800',
    'mixed'    : '#9C27B0',
    'epsilon'  : '#607D8B',
    'grid'     : '#EEEEEE',
}


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────

def load_training_data():
    episodes, rewards, waits, epsilons = [], [], [], []
    with open(TRAIN_CSV) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
            waits.append(float(row['total_wait']))
            epsilons.append(float(row['epsilon']))
    return episodes, rewards, waits, epsilons


def load_comparison_data():
    scenarios, waits, improvements = [], [], []
    with open(COMPARISON_CSV) as f:
        for row in csv.DictReader(f):
            scenarios.append(row['scenario'])
            waits.append(float(row['total_wait']))
            improvements.append(float(row['improvement_pct']))
    return scenarios, waits, improvements


def rolling_average(data, window):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(np.mean(data[start:i+1]))
    return result


# ─────────────────────────────────────────────
#  PLOT 1 — Reward Over Episodes
# ─────────────────────────────────────────────

def plot_rewards(episodes, rewards):
    fig, ax = plt.subplots(figsize=(12, 5))
    rolled  = rolling_average(rewards, ROLLING_WINDOW)

    ax.plot(episodes, rewards,
            color=COLORS['rl'], alpha=0.25, linewidth=0.8, label='Episode Reward')
    ax.plot(episodes, rolled,
            color=COLORS['rolling'], linewidth=2.5,
            label=f'Rolling Average (window={ROLLING_WINDOW})')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

    best_idx = int(np.argmax(rolled))
    ax.annotate(
        f'Best avg: {rolled[best_idx]:.0f}',
        xy=(episodes[best_idx], rolled[best_idx]),
        xytext=(episodes[best_idx] + 20, rolled[best_idx] + 500),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
        fontsize=10
    )

    ax.set_title('Q-Learning Agent — Reward Over Training Episodes',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_facecolor('white')
    ax.grid(True, color=COLORS['grid'], linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'plot1_rewards.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
#  PLOT 2 — Waiting Time Over Episodes
# ─────────────────────────────────────────────

def plot_waiting_time(episodes, waits, baseline_wait):
    fig, ax = plt.subplots(figsize=(12, 5))
    rolled  = rolling_average(waits, ROLLING_WINDOW)

    ax.plot(episodes, waits,
            color=COLORS['rl'], alpha=0.2, linewidth=0.8, label='Episode Waiting Time')
    ax.plot(episodes, rolled,
            color=COLORS['rolling'], linewidth=2.5,
            label=f'Rolling Average (window={ROLLING_WINDOW})')
    ax.axhline(baseline_wait, color=COLORS['baseline'], linewidth=2.0,
               linestyle='--', label=f'Fixed Baseline ({baseline_wait:.0f})')

    ax.set_title('Total Waiting Time Per Episode vs Fixed Baseline',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Waiting Time', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_facecolor('white')
    ax.grid(True, color=COLORS['grid'], linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'plot2_waiting_time.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
#  PLOT 3 — Epsilon Decay Curve
# ─────────────────────────────────────────────

def plot_epsilon(episodes, epsilons):
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(episodes, epsilons, color=COLORS['epsilon'], linewidth=2.5)
    ax.fill_between(episodes, epsilons, alpha=0.15, color=COLORS['epsilon'])

    for target_eps, label in [(0.5,  'Explore→Exploit (ε=0.5)'),
                               (0.1,  'Mostly Exploit   (ε=0.1)'),
                               (0.01, 'Full Exploit     (ε=0.01)')]:
        closest = min(range(len(epsilons)), key=lambda i: abs(epsilons[i] - target_eps))
        ax.axvline(episodes[closest], color='red', linewidth=1.0, linestyle=':', alpha=0.6)
        ax.text(episodes[closest] + 5, target_eps + 0.03, label,
                fontsize=8.5, color='red', alpha=0.8)

    ax.set_title('Epsilon Decay — Exploration vs Exploitation Over Training',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon (ε)', fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.set_facecolor('white')
    ax.grid(True, color=COLORS['grid'], linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'plot3_epsilon.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
#  PLOT 4 — Comparison Bar Chart (KEY RESULT)
# ─────────────────────────────────────────────

def plot_comparison(scenarios, waits, improvements):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bar_colors  = []
    short_labels = []
    for s in scenarios:
        if   'Baseline'  in s:       bar_colors.append(COLORS['baseline']); short_labels.append('Fixed\nBaseline\n(Traditional)')
        elif 'clear'     in s.lower(): bar_colors.append(COLORS['clear']);   short_labels.append('RL Agent\n(Clear\nWeather)')
        elif 'adverse'   in s.lower(): bar_colors.append(COLORS['adverse']); short_labels.append('RL Agent\n(Rain/Fog)')
        else:                          bar_colors.append(COLORS['mixed']);   short_labels.append('RL Agent\n(Mixed\nWeather)')

    # Left: Total waiting time
    bars = ax1.bar(short_labels, waits, color=bar_colors,
                   width=0.5, edgecolor='white', linewidth=1.5)
    for bar, wait in zip(bars, waits):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'{wait:.0f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    ax1.set_title('Total Waiting Time Comparison', fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel('Total Waiting Time', fontsize=11)
    ax1.set_facecolor('white')
    ax1.grid(axis='y', color=COLORS['grid'], linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Improvement %
    rl_labels      = short_labels[1:]
    rl_improvements= improvements[1:]
    rl_colors      = bar_colors[1:]

    bars2 = ax2.bar(rl_labels, rl_improvements, color=rl_colors,
                    width=0.45, edgecolor='white', linewidth=1.5)
    for bar, imp in zip(bars2, rl_improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{imp:.1f}%', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    ax2.set_title('Improvement Over Fixed Baseline (%)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylabel('Improvement (%)', fontsize=11)
    ax2.set_ylim(0, max(rl_improvements) * 1.2)
    ax2.set_facecolor('white')
    ax2.grid(axis='y', color=COLORS['grid'], linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    patches = [
        mpatches.Patch(color=COLORS['baseline'], label='Traditional Fixed Signal'),
        mpatches.Patch(color=COLORS['clear'],    label='RL — Clear Weather'),
        mpatches.Patch(color=COLORS['adverse'],  label='RL — Rain/Fog'),
        mpatches.Patch(color=COLORS['mixed'],    label='RL — Mixed Weather'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle('Weather-Aware RL Traffic Signal vs Traditional Fixed Signal',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'plot4_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
#  PLOT 5 — Weather Performance Breakdown
# ─────────────────────────────────────────────

def plot_weather_breakdown(scenarios, waits, improvements):
    fig, ax = plt.subplots(figsize=(10, 5))

    rl_waits       = [w for s, w in zip(scenarios, waits)        if 'Baseline' not in s]
    rl_improvements= [i for s, i in zip(scenarios, improvements) if 'Baseline' not in s]
    weather_labels = ['Clear\nWeather', 'Rain / Fog\n(Adverse)', 'Mixed\n(Realistic)']
    weather_colors = [COLORS['clear'], COLORS['adverse'], COLORS['mixed']]

    bars = ax.bar(weather_labels, rl_waits, color=weather_colors,
                  width=0.45, edgecolor='white', linewidth=1.5, alpha=0.85)

    baseline_wait = waits[0]
    ax.axhline(baseline_wait, color=COLORS['baseline'], linewidth=2,
               linestyle='--', label=f'Fixed Baseline ({baseline_wait:.0f})')

    for bar, imp, col in zip(bars, rl_improvements, weather_colors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'↓{imp:.1f}%', ha='center', fontsize=12,
                fontweight='bold', color=col)

    ax.set_title('RL Agent Performance Across Weather Conditions',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Total Waiting Time', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_facecolor('white')
    ax.grid(axis='y', color=COLORS['grid'], linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, 'plot5_weather_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  Generating Result Plots")
    print("=" * 55)

    episodes, rewards, waits, epsilons = load_training_data()
    scenarios, comp_waits, improvements = load_comparison_data()
    baseline_wait = comp_waits[0]

    print(f"\n  Training episodes  : {len(episodes)}")
    print(f"  Baseline wait      : {baseline_wait:.2f}\n")

    plot_rewards(episodes, rewards)
    plot_waiting_time(episodes, waits, baseline_wait)
    plot_epsilon(episodes, epsilons)
    plot_comparison(scenarios, comp_waits, improvements)
    plot_weather_breakdown(scenarios, comp_waits, improvements)

    print(f"\n{'='*55}")
    print("  All 5 plots saved to results/ folder!")
    print(f"{'='*55}")
    print("""
  plot1_rewards.png          — reward learning curve
  plot2_waiting_time.png     — waiting time vs baseline
  plot3_epsilon.png          — explore→exploit transition
  plot4_comparison.png       — RL vs Traditional (KEY RESULT)
  plot5_weather_breakdown.png— weather performance breakdown
    """)