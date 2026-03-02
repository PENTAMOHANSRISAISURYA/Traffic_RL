# 🚦 Weather-Aware RL Traffic Signal Control

> **Final Year Project** — A smart traffic signal system that uses **YOLOv8 vehicle detection** and **Q-Learning (Reinforcement Learning)** to dynamically control traffic signals based on real-time vehicle counts and weather conditions.

🔗 **Repository:** [github.com/PENTAMOHANSRISAISURYA/Traffic_RL](https://github.com/PENTAMOHANSRISAISURYA/Traffic_RL)

---

## 📌 What Is This Project?

Traditional traffic signals give every lane a **fixed green time** (e.g., 30 seconds) regardless of how many vehicles are actually waiting. This wastes time and causes unnecessary congestion.

This project builds a **self-learning traffic signal controller** that:

- Uses a camera to **count vehicles** in each lane using **YOLOv8 + MOG2**
- **Learns** the best green time for each situation using **Q-Learning**
- **Skips empty lanes** entirely — no wasted green time
- Always serves the **most congested lane first**
- Adjusts its decisions based on **weather** (clear, rain, fog)
- Prevents **lane starvation** — no lane is ignored for too long

---

## 📊 Results

| Scenario | Traditional Signal | RL Agent | Improvement |
|---|---|---|---|
| Clear Weather | 12,274 | 1,956 | **84%** ✅ |
| Rain / Fog | 12,274 | 8,533 | **30%** ✅ |
| Mixed Weather | 12,274 | 5,432 | **55%** ✅ |

---

## 🗂️ Project Structure

```
Traffic_RL/
│
├── rl_agent/                       ← ⚠️ Rename from rl-agent to rl_agent (see note below)
│   ├── __init__.py
│   ├── traffic_env.py              ← The intersection environment
│   └── q_learning_agent.py         ← The Q-learning brain
│
├── yolo_detection/
│   └── detect_vehicles.py          ← Vehicle detection pipeline
│
├── videos/
│   └── traffic_sample.mp4          ← Your traffic video (add manually)
│
├── results/
│   ├── plots.py                    ← Generates result graphs
│   ├── training_metrics.csv        ← Reward per episode
│   ├── comparison.csv              ← RL vs baseline comparison
│   └── *.png                       ← Result plots (5 graphs)
│
├── main.py                         ← Full training pipeline
├── demo.py                         ← Live presentation demo
├── test_setup.py                   ← Verify installation
└── requirements.txt                ← Python dependencies
```

> ⚠️ **Important — Folder Rename Required:**
> The folder in this repo is named `rl-agent` (with a dash). Python **cannot import** folders with dashes in their name. You **must** rename it to `rl_agent` (with an underscore) after downloading.
>
> **How to rename on GitHub UI before downloading:**
> 1. Go to the repo → click `rl-agent` folder
> 2. Click any `.py` file inside → click the ✏️ pencil (edit) icon
> 3. At the top, change `rl-agent/filename.py` → `rl_agent/filename.py`
> 4. Scroll down → click **Commit changes**
> 5. Repeat for all files inside the folder
>
> **Or rename locally after downloading:**
> - Windows: Right-click the folder → Rename → type `rl_agent`
> - Mac/Linux: `mv rl-agent rl_agent`

---

## 🧠 How It Works

### The Full Pipeline

```
Traffic Video (camera)
        │
        ▼
YOLOv8 + MOG2 Background Subtraction
(detect_vehicles.py)
        │  counts vehicles per lane, per frame
        ▼
counts_output.csv
        │
        ▼
TrafficEnv  ←  reads counts + simulates weather
        │  state  = (N, S, W, E, weather, lane)
        │  action = green time [10s/20s/30s/40s]
        │  reward = −total waiting time
        ▼
Q-Learning Agent
        │  learns optimal green time per situation
        ▼
Trained Q-Table → Smart Signal Decisions
```

### State — What the Agent Sees

```
state = (north_count, south_count, west_count, east_count, weather, priority_lane)
```

| Element | Values | Meaning |
|---|---|---|
| north / south / west / east count | 0 – 14 | Vehicles waiting in each lane |
| weather | 0 / 1 / 2 | Clear / Rain / Fog |
| priority_lane | 0 / 1 / 2 / 3 | NORTH / SOUTH / WEST / EAST |

### Action — What the Agent Decides

| Action Index | Green Time |
|---|---|
| 0 | 10 seconds |
| 1 | 20 seconds |
| 2 | 30 seconds |
| 3 | 40 seconds |

### Reward — How the Agent Learns

```
reward = −(total waiting time across all lanes)
```

Less waiting = higher reward = agent is doing well. The agent learns to minimize total waiting time across all 4 lanes.

### Q-Learning Formula

```
Q(state, action) = Q(state, action) + α × [reward + γ × max(Q(next_state)) − Q(state, action)]
```

In plain English: **New knowledge = Old knowledge + learning rate × (what actually happened − what I expected)**

---

## ⚙️ Setup — Step by Step

> **Complete beginner? Follow every step carefully. Do not skip any.**

---

### Step 1 — Install Python

Download Python 3.10 or newer from [python.org](https://www.python.org/downloads/).

During installation on Windows, **check "Add Python to PATH"**.

Verify it works — open terminal and type:
```bash
python --version
```
You should see: `Python 3.11.x`

---

### Step 2 — Download This Project

Click the green **Code** button → **Download ZIP** → Extract anywhere on your computer.

Or with Git:
```bash
git clone https://github.com/PENTAMOHANSRISAISURYA/Traffic_RL.git
cd Traffic_RL
```

---

### Step 3 — Rename the Folder ⚠️

> **This step is critical. Without it, nothing will run.**

Find the folder named `rl-agent` inside the project.
Rename it to `rl_agent` (dash → underscore).

- **Windows:** Right-click → Rename
- **Mac/Linux:** `mv rl-agent rl_agent`

---

### Step 4 — Create a Virtual Environment

Open a terminal **inside the project folder** and run:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

You will see `(venv)` at the start of your terminal line. This means it worked.

> ⚠️ You must activate the virtual environment **every time** you open a new terminal.

---

### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

> This takes 3–5 minutes — it downloads PyTorch, YOLOv8, OpenCV and other libraries.

**CPU only (no NVIDIA GPU)?** Run this instead:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

### Step 6 — Add Your Traffic Video

Download a traffic video from YouTube:

```bash
pip install yt-dlp
yt-dlp -f "best[height<=720]" "PASTE_YOUTUBE_URL_HERE" -o "videos/traffic_sample.mp4"
```

**Good YouTube search terms:**
- *"traffic junction cctv top view"*
- *"4 way intersection overhead surveillance"*
- *"busy road junction bird eye view"*

The video should show a **4-way intersection** with clearly visible cars from above or slight angle.

---

### Step 7 — Verify Everything Works

```bash
python test_setup.py
```

**Expected output:**
```
PyTorch Version  : 2.x.x
YOLOv8 loaded successfully!
Video found ✓
Resolution : 640x360
Detection test at 15s: 5 vehicles found
Setup test complete!
Next step: python yolo_detection/detect_vehicles.py
```

---

## 🏃 Running the Project

Run these **in order**. Do not skip steps.

---

### Run 1 — Detect Vehicles

```bash
python yolo_detection/detect_vehicles.py
```

Reads your video, counts vehicles in each lane every 15 frames, saves to CSV.

**Output:**
```
Frame  150 |  11.0s | N:2 S:2 W:4 E:5 | Total:13
Frame  300 |  16.0s | N:3 S:1 W:8 E:5 | Total:17
...
Detection Complete!
Output file : yolo_detection/counts_output.csv
Preview     : yolo_detection/detection_preview.jpg
```

> ⏱ Takes about 3–5 minutes on CPU.

---

### Run 2 — Train the RL Agent

```bash
python main.py
```

Runs 4 phases: baseline → training → evaluation → save results.

**Output:**
```
PHASE 1 — Fixed Baseline: 12274.50

PHASE 2 — Training RL Agent (500 Episodes)
  Episode  50/500 | Avg Reward: -3200 | ε: 0.6050
  Episode 100/500 | Avg Reward: -2800 | ε: 0.3624
  Episode 200/500 | Avg Reward: -2400 | ε: 0.1326
  Episode 500/500 | Avg Reward: -2100 | ε: 0.0100

PHASE 3 — Evaluation
  ✅ Clear Weather   → Improvement: 84.06%
  ✅ Adverse Weather → Improvement: 30.48%
  ✅ Mixed Weather   → Improvement: 55.74%

PHASE 4 — Results saved to results/
```

> ⏱ Takes about 5–10 minutes on CPU.
> 💡 If asked "Resume previous training?" — type `y` to continue, `n` to restart.

---

### Run 3 — Generate Plots

```bash
python results/plots.py
```

Generates 5 graphs in the `results/` folder:

| Graph | Description |
|---|---|
| `plot1_rewards.png` | Reward improving over 500 episodes |
| `plot2_waiting_time.png` | Waiting time vs fixed baseline |
| `plot3_epsilon.png` | Explore → Exploit transition |
| `plot4_comparison.png` | **KEY RESULT** — RL vs Traditional |
| `plot5_weather_breakdown.png` | Performance in each weather condition |

---

### Run 4 — Live Demo (For Presentations)

```bash
python demo.py
```

Select a weather condition (1=Clear, 2=Rain, 3=Fog).

The demo runs in 3 parts:
- **Part 1:** Traditional signal — shows empty lanes getting green, wasted time
- **Part 2:** Your RL agent — shows smart priority-based decisions
- **Part 3:** Side-by-side improvement % comparison

---

## 🔬 Traditional vs RL — Full Comparison

| Problem | Traditional Signal | This System |
|---|---|---|
| Green light with 0 cars | ❌ Always happens | ✅ Empty lanes completely skipped |
| Fixed green time | ❌ Same 20s always | ✅ Dynamic: 10s / 20s / 30s / 40s |
| Fixed rotation (N→S→W→E) | ❌ Ignores congestion | ✅ Most congested lane served first |
| Weather ignored | ❌ No adjustment | ✅ Longer green in rain/fog |
| No learning | ❌ Same forever | ✅ Improves over 500 episodes |
| Lane starvation | ❌ One lane dominates | ✅ Force-served after 5 turns |

---

## 📐 Q-Learning Parameters

| Parameter | Value | Meaning |
|---|---|---|
| α Learning Rate | 0.1 | How fast to update knowledge |
| γ Discount Factor | 0.9 | How much future rewards matter |
| ε Start | 1.0 | Fully random at start (exploring) |
| ε End | 0.01 | 99% smart at end (exploiting) |
| ε Decay | 0.990 | Reduces by 1% per episode |
| Episodes | 500 | Total training rounds |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Main language |
| YOLOv8 (Ultralytics) | Vehicle detection |
| OpenCV | Video processing + MOG2 detection |
| PyTorch | YOLOv8 backend |
| NumPy / Pandas | Data handling |
| Matplotlib | Plotting results |

---

## ❓ Common Errors & Fixes

**`ModuleNotFoundError: No module named 'rl_agent'`**
→ You haven't renamed `rl-agent` to `rl_agent`. See Step 3 of Setup.

**`No module named 'ultralytics'`**
→ Virtual environment is not activated. Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux) first.

**`Video not found at videos/traffic_sample.mp4`**
→ Make sure video is inside the `videos/` folder and named exactly `traffic_sample.mp4`.

**`Vehicles found: 0`**
→ Video likely starts with a black intro screen. The script skips 5 seconds automatically. If still 0, your video angle is too steep — use a slightly angled intersection video.

**`No saved Q-table found`**
→ You haven't trained yet. Run `python main.py` before running `demo.py`.

**`pip is not recognized`**
→ Python is not added to PATH. Reinstall Python and check the "Add to PATH" box.

---

## 🔮 Future Work

- Real-time live camera feed integration
- Deep Q-Network (DQN) for better generalization
- Multi-intersection coordination
- Emergency vehicle preemption
- Pedestrian crossing signal integration
- Larger training video dataset for better convergence

---

## 👤 Author

**Penta Mohan Sri Sai Surya**
Final Year Project — B.Tech (Computer Science / Artificial Intelligence)
2025

---

## 📄 License

MIT License — free to use for academic and research purposes.
