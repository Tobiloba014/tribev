import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import datasets
import numpy as np
import pandas as pd

# 1. Load your saved predictions (71s x 20,484 vertices) and word/sentence timings
preds = np.load("warhammer_preds.npy")
events = pd.read_csv("warhammer_events.csv")

# 2. Load the standard fsaverage5 anatomical cortical atlas (10,242 left + 10,242 right vertices)
atlas = datasets.fetch_atlas_surf_destrieux()
labels = [
    l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in atlas.labels
]

left_preds = preds[:, :10242]
right_preds = preds[:, 10242:]

# 3. Average vertex activity inside each named brain region to create clean 1 Hz signals
signals = {"second": np.arange(preds.shape[0])}

for idx, name in enumerate(labels):
  if name in ("Medial_wall", "Unknown"):
    continue
  left_mask = atlas.map_left == idx
  right_mask = atlas.map_right == idx
  if left_mask.any():
    signals[f"L_{name}"] = left_preds[:, left_mask].mean(axis=1)
  if right_mask.any():
    signals[f"R_{name}"] = right_preds[:, right_mask].mean(axis=1)

df_signals = pd.DataFrame(signals)

# 4. Attach the active sentence to each second so every signal aligns with the text
sentences = events[events["type"] == "Sentence"].reset_index(drop=True)
active_text = []
for sec in df_signals["second"]:
  match = sentences[(sentences["start"] <= sec) & (sentences["stop"] >= sec)]
  active_text.append(match.iloc[0]["text"].strip() if not match.empty else "")
df_signals["text"] = active_text

# 5. Export the universal telemetry file
df_signals.to_csv("warhammer_signals.csv", index=False)
print(
    f"Saved {df_signals.shape[1] - 2} brain region signals across"
    f" {len(df_signals)} seconds to warhammer_signals.csv"
)

# 6. Inspect which regions varied the most across the story and where they peaked
region_cols = [c for c in df_signals.columns if c not in ("second", "text")]
top_regions = df_signals[region_cols].std().sort_values(ascending=False).head(8)

print("\n--- TOP 8 MOST DYNAMIC BRAIN REGIONS IN THIS PASSAGE ---")
for reg, std_val in top_regions.items():
  peak_sec = int(df_signals[reg].idxmax())
  peak_val = df_signals.loc[peak_sec, reg]
  snippet = df_signals.loc[peak_sec, "text"] or "[dramatic pause]"
  print(
      f"{reg:<32} | Peak: {peak_val:+.3f} at {peak_sec:>2}s | Text:"
      f" {snippet[:65]}"
  )

# 7. Plot the top 4 dynamic region signals over time
plt.figure(figsize=(12, 5))
for reg in top_regions.index[:4]:
  plt.plot(df_signals["second"], df_signals[reg], label=reg, linewidth=2)
plt.title("Multi-Region Cortical Signal Telemetry (Warhammer Intro)")
plt.xlabel("Time (seconds)")
plt.ylabel("Predicted Activation")
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, alpha=0.3)
plt.savefig("warhammer_multi_signals.png", dpi=150, bbox_inches="tight")
print("Saved multi-signal chart to warhammer_multi_signals.png")
