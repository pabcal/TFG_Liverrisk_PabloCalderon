import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 1. Dataset restored to original palette (Pink, Orange, Green, Blue)
tasks = [
    {"id": "—", "name": "LiverRisk TFG (Overall)", "start": "2025-10-01", "end": "2026-08-31", "color": "#E91E63", "bold": True},
    {"id": "1", "name": "Medical Research (Censored Patients)", "start": "2025-10-01", "end": "2025-11-07", "color": "#FB8C00", "bold": False},
    {"id": "2", "name": "Survival Analysis Research", "start": "2025-11-07", "end": "2025-11-25", "color": "#FB8C00", "bold": False},
    {"id": "3", "name": "Data Exploration", "start": "2025-10-20", "end": "2025-11-25", "color": "#4CAF50", "bold": False},
    {"id": "4", "name": "Feature Engineering", "start": "2025-11-25", "end": "2025-12-20", "color": "#4CAF50", "bold": False},
    {"id": "5", "name": "Exploratory Model Testing", "start": "2025-12-20", "end": "2026-01-20", "color": "#1E88E5", "bold": False},
    {"id": "6", "name": "Survival Target Construction", "start": "2026-01-20", "end": "2026-02-15", "color": "#4CAF50", "bold": False},
    {"id": "7", "name": "Model Dev & Hyperparameter Tuning", "start": "2026-02-15", "end": "2026-04-01", "color": "#1E88E5", "bold": False},
    {"id": "8", "name": "Final Training and Evaluation", "start": "2026-04-01", "end": "2026-06-01", "color": "#1E88E5", "bold": False},
    {"id": "9", "name": "Clinical Formula Comparison", "start": "2026-06-01", "end": "2026-07-01", "color": "#FB8C00", "bold": False},
    {"id": "10", "name": "Web Application Development", "start": "2026-07-01", "end": "2026-08-31", "color": "#1E88E5", "bold": False},
]

dependencies = [
    ("1", "2"), ("3", "4"), ("4", "5"), ("5", "6"), 
    ("6", "7"), ("7", "8"), ("8", "9"), ("9", "10")
]

# 2. Setup Figure & Layout with ample left margin for full task names
fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)
plt.subplots_adjust(left=0.32, right=0.96, top=0.88, bottom=0.08)

n_tasks = len(tasks)
y_positions = list(range(n_tasks - 1, -1, -1))

task_coords = {}
y_labels = []

# 3. Plot Bars and Sync Y-Axis Labels 1:1
for t, y in zip(tasks, y_positions):
    start_dt = datetime.strptime(t["start"], "%Y-%m-%d")
    end_dt = datetime.strptime(t["end"], "%Y-%m-%d")
    start_num = mdates.date2num(start_dt)
    end_num = mdates.date2num(end_dt)
    
    task_coords[t["id"]] = {"start": start_num, "end": end_num, "y": y}
    
    # Draw horizontal bar
    bar_height = 0.55 if t["bold"] else 0.45
    ax.barh(y, end_num - start_num, left=start_num, height=bar_height, color=t["color"], align='center', zorder=2)
    
    # Label formatting
    label_text = f"{t['id']:<3}  {t['name']}" if t["id"] != "—" else f"   {t['name']}"
    y_labels.append(label_text)

# Set native Y-axis ticks to guarantee strict center alignment
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=10, fontweight='normal', color='#212121')

# Bold the summary bar label
for tick in ax.get_yticklabels():
    if "LiverRisk TFG" in tick.get_text():
        tick.set_fontweight('bold')

# 4. Draw Clean 90-Degree Orthogonal Dependency Arrows
for src_id, tgt_id in dependencies:
    src, tgt = task_coords[src_id], task_coords[tgt_id]
    ax.plot([src["end"], src["end"], tgt["start"]], [src["y"], tgt["y"], tgt["y"]], 
            color='#757575', linestyle='--', linewidth=1.1, zorder=1)

# 5. Format Timeline Header (X-Axis)
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_xlim(mdates.date2num(datetime(2025, 9, 25)), mdates.date2num(datetime(2026, 9, 5)))
ax.xaxis.set_ticks_position('top')
ax.tick_params(axis='x', rotation=0, labelsize=9.5, colors='#424242')

# 6. Gridlines & Minimalist Spines
ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

for spine in ['left', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.spines['top'].set_color('#BDBDBD')

plt.title("LiverRisk TFG — Project Timeline", fontsize=15, fontweight='bold', loc='left', pad=30, color='#111111')
plt.savefig("gantt_chart_original_colors.png", dpi=300, bbox_inches='tight')
plt.show()