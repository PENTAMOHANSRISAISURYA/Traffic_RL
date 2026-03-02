import numpy as np
import pandas as pd
import random
import os

# ─────────────────────────────────────────────────────────────────
#  BEGINNER EXPLANATION
#
#  This file is the "game world" for the RL agent.
#
#  KEY UPGRADES OVER TRADITIONAL SYSTEMS:
#  1. Empty lanes are SKIPPED — no green light wasted on 0-car lanes
#  2. Priority-based selection — most congested lane is served first
#  3. Dynamic green time — agent picks 10s/20s/30s/40s per lane
#  4. Weather-aware — rain/fog changes the learning policy
#
#  HOW IT WORKS:
#  Every cycle, the environment reads real vehicle counts (from YOLO CSV),
#  skips empty lanes, ranks remaining lanes by congestion, and asks the
#  agent: "How long should the most congested lane get green light?"
#  The agent answers, gets a reward, and learns over time.
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(
    os.path.join(BASE_DIR, '..', 'yolo_detection', 'yolo_detection', 'counts_output.csv')
)

# Green time options (seconds) — agent picks one per lane
GREEN_TIME_OPTIONS = [10, 20, 30, 40]

# Weather states
WEATHER_LABELS  = {0: 'Clear', 1: 'Rain', 2: 'Fog'}

# Weather slows vehicles → more waiting time penalty
WEATHER_PENALTY   = {0: 1.0, 1: 1.4, 2: 1.7}

# Bad weather needs longer minimum green time
WEATHER_MIN_GREEN = {0: 10,  1: 20,  2: 20}

LANES     = ['NORTH', 'SOUTH', 'WEST', 'EAST']
MAX_COUNT = 15   # cap per lane (your CSV max was ~13)


# ─────────────────────────────────────────────
#  THE ENVIRONMENT CLASS
# ─────────────────────────────────────────────

class TrafficEnv:
    """
    Upgraded 4-lane intersection environment for Q-Learning.

    UPGRADES vs traditional systems:
    ✅ Empty lanes skipped entirely   — no wasted green time
    ✅ Priority-based lane selection  — most congested served first
    ✅ Dynamic green time             — 10s to 40s per lane
    ✅ Weather in state               — agent learns rain/fog policies

    STATE  = (north_count, south_count, west_count, east_count, weather)
    ACTION = index into GREEN_TIME_OPTIONS  [0→10s, 1→20s, 2→30s, 3→40s]
    REWARD = negative total waiting time    (less wait = higher reward)
    """

    def __init__(self, csv_path=CSV_PATH, weather_mode='random'):
        self.green_time_options = GREEN_TIME_OPTIONS
        self.n_actions          = len(GREEN_TIME_OPTIONS)
        self.weather_mode       = weather_mode
        self.lanes              = LANES
        self.max_count          = MAX_COUNT

        print(f"[TrafficEnv] Loading vehicle counts from: {csv_path}")
        self.data = pd.read_csv(csv_path)
        print(f"[TrafficEnv] Loaded {len(self.data)} rows of traffic data.\n")

        # State space size for Q-table
        # State space size for Q-table
        # 15 counts × 4 lanes × 3 weather × 4 priority lanes
        # Dictionary-based Q-table so only visited states are stored — memory efficient
        self.state_space_size = (MAX_COUNT, MAX_COUNT, MAX_COUNT, MAX_COUNT, 3, 4)

        # Internal state
        self.current_step   = 0
        self.weather        = 0
        self.waiting_times  = {lane: 0.0 for lane in self.lanes}
        self.episode_reward = 0.0
        self.done           = False
        self.priority_lane  = None
        self.skipped_count  = 0
        self.served_count   = 0

        # ✅ Starvation prevention
        # Tracks how many consecutive cycles each lane has been waiting
        # If a lane waits too long it gets forced green — even if not most congested
        self.starvation_counter = {lane: 0 for lane in self.lanes}
        self.max_wait_turns     = 5   # no lane waits more than 5 turns in a row


    # ─────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────
    def reset(self):
        self.current_step   = 0
        self.waiting_times  = {lane: 0.0 for lane in self.lanes}
        self.episode_reward = 0.0
        self.done           = False
        self.priority_lane      = None
        self.skipped_count      = 0
        self.served_count       = 0
        self.starvation_counter = {lane: 0 for lane in self.lanes}

        # Assign weather for this episode
        if self.weather_mode == 'random':
            self.weather = random.choices([0, 1, 2], weights=[60, 25, 15])[0]
        elif self.weather_mode == 'adverse':
            self.weather = random.choices([1, 2], weights=[50, 50])[0]
        else:
            self.weather = 0

        max_start         = max(0, len(self.data) - 20)
        self.current_step = random.randint(0, max_start)

        return self._get_state()


    # ─────────────────────────────────────────
    #  PRIORITY LANE SELECTION
    #  ✅ Skips empty lanes
    #  ✅ Serves most congested lane first
    # ─────────────────────────────────────────
    def _select_priority_lane(self, lane_counts):
        """
        Returns the lane with the most vehicles.
        Completely skips lanes with 0 vehicles.
        This eliminates the traditional problem of green lights
        being given to empty lanes.
        """
        eligible = {lane: count for lane, count in lane_counts.items() if count > 0}
        if not eligible:
            return None
        return max(eligible, key=eligible.get)


    # ─────────────────────────────────────────
    #  STEP
    # ─────────────────────────────────────────
    def step(self, action):
        """
        Agent picks a green time for the priority lane.

        action : 0→10s | 1→20s | 2→30s | 3→40s

        Returns: next_state, reward, done, info
        """
        assert 0 <= action < self.n_actions, f"Invalid action: {action}"

        # Read vehicle counts from CSV
        row = self.data.iloc[self.current_step % len(self.data)]
        lane_counts = {
            'NORTH': min(int(row['NORTH']), MAX_COUNT - 1),
            'SOUTH': min(int(row['SOUTH']), MAX_COUNT - 1),
            'WEST' : min(int(row['WEST']),  MAX_COUNT - 1),
            'EAST' : min(int(row['EAST']),  MAX_COUNT - 1),
        }

        weather_multiplier = WEATHER_PENALTY[self.weather]
        min_green          = WEATHER_MIN_GREEN[self.weather]
        green_time         = self.green_time_options[action]

        # ✅ Update starvation counters
        # Increment counter for all lanes with vehicles
        # Reset counter for any lane that is empty (no cars = no starvation)
        for lane in self.lanes:
            if lane_counts[lane] > 0:
                self.starvation_counter[lane] += 1
            else:
                self.starvation_counter[lane]  = 0

        # ✅ Starvation check — force-serve any lane waiting too long
        # This prevents one dominant lane from blocking all others
        starved = [
            l for l in self.lanes
            if self.starvation_counter[l] >= self.max_wait_turns
            and lane_counts[l] > 0
        ]

        if starved:
            # Serve the most starved lane (waited the longest)
            priority_lane = max(starved, key=lambda l: self.starvation_counter[l])
            self.starvation_counter[priority_lane] = 0   # reset its counter
        else:
            # Normal case — serve most congested lane
            priority_lane = self._select_priority_lane(lane_counts)

        self.priority_lane = priority_lane

        # Track empty lanes that were skipped
        empty_lanes         = [l for l in self.lanes if lane_counts[l] == 0]
        self.skipped_count += len(empty_lanes)

        reward = 0.0

        if priority_lane is None:
            # All lanes empty — no action needed
            reward = 2.0
        else:
            self.served_count += 1

            for lane in self.lanes:

                if lane_counts[lane] == 0:
                    # ✅ Empty lane — skip, zero waiting penalty
                    self.waiting_times[lane] = 0.0
                    continue

                if lane == priority_lane:
                    # ✅ Green lane — clear vehicles proportional to green time
                    cleared = min(lane_counts[lane], green_time / 8.0)
                    self.waiting_times[lane] = max(0.0, self.waiting_times[lane] - cleared)
                else:
                    # Red lane — vehicles accumulate, weather worsens it
                    self.waiting_times[lane] += lane_counts[lane] * 0.5 * weather_multiplier

            total_waiting  = sum(self.waiting_times.values())
            reward         = -total_waiting

            # ── Bonus: right green time for the count ────────────
            count = lane_counts[priority_lane]
            if   count >= 8            and action == 3: reward += 8.0  # high  → 40s
            elif 4 <= count <= 7       and action == 2: reward += 6.0  # med   → 30s
            elif 1 <= count <= 3       and action == 0: reward += 4.0  # low   → 10s

            # ── Penalty: too short green in bad weather ───────────
            if self.weather > 0 and green_time < min_green:
                reward -= 5.0 * self.weather

            # ── Penalty: long green for near-empty lane ───────────
            if count <= 2 and action >= 2:
                reward -= 3.0

        self.episode_reward += reward
        self.current_step   += 1

        if self.current_step >= len(self.data):
            self.done = True

        next_state = self._get_state()

        info = {
            'priority_lane'     : priority_lane,
            'green_time'        : green_time,
            'weather'           : WEATHER_LABELS[self.weather],
            'lane_counts'       : lane_counts,
            'empty_lanes'       : empty_lanes,
            'waiting_times'     : dict(self.waiting_times),
            'total_waiting'     : sum(self.waiting_times.values()),
            'starvation_counter': dict(self.starvation_counter),
            'starved_lanes'     : starved if 'starved' in dir() else [],
        }

        return next_state, reward, self.done, info


    # ─────────────────────────────────────────
    #  GET STATE
    # ─────────────────────────────────────────
    def _get_state(self):
        """
        Returns raw vehicle counts + weather + priority lane index as the state.

        State = (north_count, south_count, west_count, east_count, weather, lane_idx)

        Adding lane_idx tells the agent WHICH lane it is currently deciding for.
        This makes Q-values lane-specific — the agent learns different green time
        policies for each lane based on its position and traffic pattern.
        """
        row = self.data.iloc[self.current_step % len(self.data)]

        # Which lane is currently being served? (0=NORTH,1=SOUTH,2=WEST,3=EAST)
        lane_idx = self.lanes.index(self.priority_lane) if self.priority_lane else 0

        return (
            min(int(row['NORTH']), MAX_COUNT - 1),
            min(int(row['SOUTH']), MAX_COUNT - 1),
            min(int(row['WEST']),  MAX_COUNT - 1),
            min(int(row['EAST']),  MAX_COUNT - 1),
            self.weather,
            lane_idx
        )


    # ─────────────────────────────────────────
    #  FIXED BASELINE — traditional signal
    # ─────────────────────────────────────────
    def run_fixed_baseline(self, fixed_green=20):
        """
        Simulates traditional fixed-time traffic signal:
        - Every lane always gets 20s green (no matter how many cars)
        - Strict rotation: NORTH→SOUTH→WEST→EAST (even if lane is empty)
        - No weather awareness

        Returns total waiting time — compared against RL agent.
        This is what your RL system should beat.
        """
        self.reset()
        self.weather  = 0
        total_wait    = 0.0
        lane_idx      = 0

        for _ in range(len(self.data)):
            row = self.data.iloc[self.current_step % len(self.data)]
            lane_counts = {
                'NORTH': int(row['NORTH']),
                'SOUTH': int(row['SOUTH']),
                'WEST' : int(row['WEST']),
                'EAST' : int(row['EAST']),
            }
            current_lane = self.lanes[lane_idx]

            # Traditional: serves lane even if empty
            for lane in self.lanes:
                if lane == current_lane:
                    cleared = min(lane_counts[lane], fixed_green / 8.0)
                    self.waiting_times[lane] = max(0.0, self.waiting_times[lane] - cleared)
                else:
                    self.waiting_times[lane] += lane_counts[lane] * 0.5

            total_wait       += sum(self.waiting_times.values())
            lane_idx          = (lane_idx + 1) % len(self.lanes)
            self.current_step += 1

            if self.current_step >= len(self.data):
                break

        return total_wait


    # ─────────────────────────────────────────
    #  UTILITY
    # ─────────────────────────────────────────
    def get_state_space_size(self): return self.state_space_size
    def get_n_actions(self):        return self.n_actions

    def render(self, info):
        empty_str    = f" | Skipped: {info['empty_lanes']}"   if info['empty_lanes']   else ""
        starved_str  = f" | ⚠ Forced: {info['starved_lanes']}" if info.get('starved_lanes') else ""
        print(
            f"  Priority: {str(info['priority_lane']):5s} | "
            f"Green: {info['green_time']:2d}s | "
            f"Weather: {info['weather']:5s} | "
            f"N:{info['lane_counts']['NORTH']} "
            f"S:{info['lane_counts']['SOUTH']} "
            f"W:{info['lane_counts']['WEST']} "
            f"E:{info['lane_counts']['EAST']}"
            f"{empty_str}{starved_str} | Wait: {info['total_waiting']:.1f}"
        )


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Testing Upgraded TrafficEnv")
    print("=" * 60)

    env   = TrafficEnv(weather_mode='random')
    state = env.reset()

    print(f"Initial State : {state}")
    print(f"  N={state[0]} S={state[1]} W={state[2]} E={state[3]} "
          f"Weather={WEATHER_LABELS[state[4]]} Lane={LANES[state[5]]}\n")

    print("Running 8 random steps:\n")
    total_reward = 0
    for i in range(8):
        action                         = random.randint(0, env.n_actions - 1)
        next_state, reward, done, info = env.step(action)
        env.render(info)
        print(f"           Action : {GREEN_TIME_OPTIONS[action]}s | Reward: {reward:.2f}\n")
        total_reward += reward
        if done:
            break

    print(f"Total Reward (8 steps): {total_reward:.2f}")
    print("\n" + "=" * 60)
    print("  Running Fixed Baseline (Traditional 20s fixed)...")
    baseline = env.run_fixed_baseline(fixed_green=20)
    print(f"  Fixed Baseline Total Waiting Time: {baseline:.2f}")
    print("  (RL agent should beat this after training)")
    print("=" * 60)