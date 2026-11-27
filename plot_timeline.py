import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

preds = np.load("warhammer_preds.npy")  # Shape: (71 seconds, 20484 brain points)
events = pd.read_csv("warhammer_events.csv")

# Average activation per second across top active brain regions
top_brain_hubs = np.argsort(preds.std(axis=0))[-500:]
timeline = preds[:, top_brain_hubs].mean(axis=1)

# Match each sentence to its average brain activation score
sentences = events[events["type"] == "Sentence"].copy()
scores = []
for _, row in sentences.iterrows():
  s, e = int(row["start"]), min(int(row["stop"]) + 1, len(timeline))
  scores.append(timeline[s:e].mean() if e > s else 0.0)
sentences["brain_score"] = scores

# Print the top 5 highest-activating sentences
print("\n--- TOP SCENES BY PREDICTED BRAIN RESPONSE ---")
for _, row in sentences.sort_values("brain_score", ascending=False).head(5).iterrows():
  print(f"[{row['brain_score']:.3f}] ({row['start']:.1f}s): {row['text']}")

# Save the timeline graph
plt.figure(figsize=(10, 4))
plt.plot(timeline, color="crimson", linewidth=2)
plt.title("Second-by-Second Brain Activation Across the Passage")
plt.xlabel("Time (seconds)")
plt.ylabel("Predicted Cortical Response")
plt.grid(True, alpha=0.3)
plt.savefig("warhammer_timeline.png", dpi=150, bbox_inches="tight")
print("\nSaved timeline graph to warhammer_timeline.png")
