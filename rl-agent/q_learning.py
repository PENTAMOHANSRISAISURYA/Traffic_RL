import numpy as np
import random
import pickle
import os

# ─────────────────────────────────────────────────────────────────
#  BEGINNER EXPLANATION
#
#  This file is the "brain" of the traffic signal controller.
#
#  The agent uses Q-Learning — a simple but powerful algorithm.
#
#  CORE IDEA:
#  The agent maintains a Q-Table — a dictionary that maps every
#  (state, action) pair to a score (Q-value).
#
#  High Q-value = this action was good in this situation
#  Low Q-value  = this action was bad in this situation
#
#  The agent starts knowing nothing (all Q-values = 0).
#  After each step it updates the Q-value using this formula:
#
#  Q(s,a) = Q(s,a) + α × [reward + γ × max(Q(s')) - Q(s,a)]
#
#  Where:
#  α (alpha)  = learning rate  — how fast to learn (0.1)
#  γ (gamma)  = discount       — how much future rewards matter (0.9)
#  ε (epsilon)= exploration    — how often to try random actions
#
#  EXPLORE vs EXPLOIT:
#  Early training  → agent tries random actions (exploring)
#  Late training   → agent uses Q-table (exploiting what it learned)
#  Epsilon slowly decreases from 1.0 → 0.01 over training
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

ALPHA         = 0.1      # Learning rate  — how fast agent updates knowledge
GAMMA         = 0.9      # Discount factor — future rewards matter 90%
EPSILON_START = 1.0      # Start fully random (100% exploration)
EPSILON_END   = 0.01     # End mostly greedy (1% exploration)
EPSILON_DECAY = 0.990    # Multiply epsilon by this after each episode (faster for short episodes)

QTABLE_PATH   = 'rl_agent/q_table.pkl'


# ─────────────────────────────────────────────
#  Q-LEARNING AGENT CLASS
# ─────────────────────────────────────────────

class QLearningAgent:
    """
    A Q-Learning agent that learns optimal traffic signal timing.

    The Q-table is stored as a Python dictionary:
        key   = (state_tuple)          e.g. (3, 1, 8, 5, 0)
        value = numpy array of size 4  e.g. [-12.1, -8.4, -5.2, -3.1]
                one Q-value per action (10s, 20s, 30s, 40s)

    Only visited states are stored → memory efficient.
    """

    def __init__(self, n_actions=4):
        self.n_actions   = n_actions
        self.alpha       = ALPHA
        self.gamma       = GAMMA
        self.epsilon     = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY

        # Q-table as dictionary {state: [q_val_action0, ..., q_val_action3]}
        self.q_table     = {}

        # Tracking metrics for plotting
        self.episode_rewards  = []
        self.episode_epsilons = []
        self.total_steps      = 0


    # ─────────────────────────────────────────
    #  GET Q-VALUES FOR A STATE
    # ─────────────────────────────────────────
    def _get_q_values(self, state):
        """
        Returns Q-values for all actions in a given state.
        If state is new (never seen), initializes with zeros.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]


    # ─────────────────────────────────────────
    #  CHOOSE ACTION (Epsilon-Greedy)
    # ─────────────────────────────────────────
    def choose_action(self, state, weather=None):
        """
        Picks an action using epsilon-greedy strategy:

        With probability epsilon  → random action  (explore)
        With probability 1-epsilon → best action   (exploit)

        In bad weather (rain/fog), random exploration is biased
        toward longer green times since vehicles move slower.
        """
        if random.random() < self.epsilon:
            # EXPLORE
            if weather and weather > 0:
                # Rain=1 or Fog=2 → bias toward longer green times
                # weights: 10s=5%, 20s=30%, 30s=40%, 40s=25%
                return random.choices([0, 1, 2, 3], weights=[5, 30, 40, 25])[0]
            return random.randint(0, self.n_actions - 1)
        else:
            # EXPLOIT: pick the action with highest Q-value
            q_values = self._get_q_values(state)
            return int(np.argmax(q_values))


    # ─────────────────────────────────────────
    #  UPDATE Q-TABLE
    # ─────────────────────────────────────────
    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Bellman equation:

        Q(s,a) = Q(s,a) + α × [reward + γ × max(Q(s')) - Q(s,a)]

        In plain English:
        New Q = Old Q + learning_rate × (what_actually_happened - what_I_expected)

        Parameters:
            state      : current state before action
            action     : action the agent took
            reward     : reward received from environment
            next_state : state after action
            done       : True if episode ended
        """
        current_q  = self._get_q_values(state)[action]

        if done:
            # No future rewards if episode is over
            target_q = reward
        else:
            # Future reward = best possible action in next state
            next_max_q = np.max(self._get_q_values(next_state))
            target_q   = reward + self.gamma * next_max_q

        # Calculate the difference (TD error)
        td_error = target_q - current_q

        # Update Q-value
        self.q_table[state][action] = current_q + self.alpha * td_error

        self.total_steps += 1


    # ─────────────────────────────────────────
    #  DECAY EPSILON
    # ─────────────────────────────────────────
    def decay_epsilon(self):
        """
        Reduces epsilon after each episode.
        Agent gradually shifts from exploring → exploiting.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


    # ─────────────────────────────────────────
    #  RECORD EPISODE METRICS
    # ─────────────────────────────────────────
    def record_episode(self, total_reward):
        """Saves reward and epsilon for this episode (used for plots)."""
        self.episode_rewards.append(total_reward)
        self.episode_epsilons.append(self.epsilon)


    # ─────────────────────────────────────────
    #  SAVE Q-TABLE
    # ─────────────────────────────────────────
    def save(self, path=QTABLE_PATH):
        """Saves the Q-table to disk so training can be resumed later."""
        dir_path = os.path.dirname(path)
        if dir_path:  # only create directory if path contains one
            os.makedirs(dir_path, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'q_table'         : self.q_table,
                'epsilon'         : self.epsilon,
                'episode_rewards' : self.episode_rewards,
                'episode_epsilons': self.episode_epsilons,
                'total_steps'     : self.total_steps,
            }, f)
        print(f"[Agent] Q-table saved → {path}  "
              f"({len(self.q_table)} states learned)")


    # ─────────────────────────────────────────
    #  LOAD Q-TABLE
    # ─────────────────────────────────────────
    def load(self, path=QTABLE_PATH):
        """Loads a previously saved Q-table to resume training or run inference."""
        if not os.path.exists(path):
            print(f"[Agent] No saved Q-table found at {path}. Starting fresh.")
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table          = data['q_table']
        self.epsilon          = data['epsilon']
        self.episode_rewards  = data['episode_rewards']
        self.episode_epsilons = data['episode_epsilons']
        self.total_steps      = data['total_steps']
        print(f"[Agent] Q-table loaded ← {path}  "
              f"({len(self.q_table)} states, "
              f"epsilon={self.epsilon:.4f}, "
              f"episodes={len(self.episode_rewards)})")
        return True


    # ─────────────────────────────────────────
    #  STATS
    # ─────────────────────────────────────────
    def get_stats(self):
        """Returns a summary of the agent's current learning state."""
        rewards = self.episode_rewards
        return {
            'episodes'        : len(rewards),
            'total_steps'     : self.total_steps,
            'epsilon'         : round(self.epsilon, 4),
            'q_table_size'    : len(self.q_table),
            'avg_reward_last10' : round(np.mean(rewards[-10:]), 2) if len(rewards) >= 10 else 'N/A',
            'avg_reward_last50' : round(np.mean(rewards[-50:]), 2) if len(rewards) >= 50 else 'N/A',
            'best_reward'     : round(max(rewards), 2) if rewards else 'N/A',
        }

    def print_stats(self):
        s = self.get_stats()
        print(f"\n  Episodes         : {s['episodes']}")
        print(f"  Total Steps      : {s['total_steps']}")
        print(f"  Epsilon          : {s['epsilon']}")
        print(f"  Q-Table Size     : {s['q_table_size']} states")
        print(f"  Avg Reward(10)   : {s['avg_reward_last10']}")
        print(f"  Avg Reward(50)   : {s['avg_reward_last50']}")
        print(f"  Best Reward      : {s['best_reward']}")


# ─────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Testing QLearningAgent")
    print("=" * 55)

    agent = QLearningAgent(n_actions=4)

    # Simulate a few fake steps to verify update logic
    fake_states = [
        (3, 1, 8, 5, 0, 2),   # Clear, WEST priority
        (2, 3, 7, 4, 1, 3),   # Rain,  EAST priority
        (0, 5, 6, 2, 0, 1),   # Clear, SOUTH priority
        (8, 0, 3, 7, 2, 0),   # Fog,   NORTH priority
    ]

    print("\nSimulating 20 fake update steps...\n")
    for i in range(20):
        state      = random.choice(fake_states)
        weather    = state[4]
        action     = agent.choose_action(state, weather=weather)
        reward     = random.uniform(-50, 5)
        next_state = random.choice(fake_states)
        done       = (i == 19)

        agent.update(state, action, reward, next_state, done)

    agent.record_episode(total_reward=-120.5)
    agent.record_episode(total_reward=-95.3)
    agent.record_episode(total_reward=-78.1)
    agent.decay_epsilon()

    print("Q-Table sample (first 3 states learned):")
    for i, (state, qvals) in enumerate(list(agent.q_table.items())[:3]):
        best_action = int(np.argmax(qvals))
        green_times = [10, 20, 30, 40]
        print(f"  State {state} → Q-values: {np.round(qvals, 2)} "
              f"→ Best action: {green_times[best_action]}s green")

    print()
    agent.print_stats()

    # Test save and load
    agent.save()
    print("\nLoading saved Q-table...")
    agent2 = QLearningAgent(n_actions=4)
    agent2.load()
    agent2.print_stats()

    print("\nQLearningAgent test complete!")