import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_parquet("Sporting-CP-vs-SnapYT.App_360.parquet")
cols = [c for c in df.columns if c not in ("second", "words", "sentence")]
mat = df[cols].to_numpy(dtype=np.float64)

# Fix: Keep ONLY positive activation above baseline (do NOT square negative deactivations)
df["pos_drive"] = np.maximum(0.0, mat).mean(axis=1)

# Also isolate the specific Motion + Crowd Audio tracks (MT, MST, A4, A5)
motion_audio_cols = [
    c for c in cols if any(k in c for k in ("_MT", "_MST", "_A4", "_A5"))
]
df["motion_audio"] = df[motion_audio_cols].mean(axis=1)

goals = [
    ("2:21", 141, "Suarez 1st"),
    ("4:12", 252, "Bruno Pen"),
    ("5:18", 318, "Bruno 2nd"),
    ("7:16", 436, "Suarez 2nd"),
]

# Measure the +4s to +20s post-goal blood-flow & replay window
goal_mask = np.zeros(len(df), dtype=bool)
for _, sec, _ in goals:
  goal_mask[sec + 4 : min(len(df), sec + 21)] = True

print(
    f"Positive Cortical Activation -> Post-Goal (+4s..+20s):"
    f" {df.loc[goal_mask, 'pos_drive'].mean():.4f} vs Rest of Match:"
    f" {df.loc[~goal_mask, 'pos_drive'].mean():.4f}"
)
print(
    f"Motion+Audio Track           -> Post-Goal (+4s..+20s):"
    f" {df.loc[goal_mask, 'motion_audio'].mean():+.4f} vs Rest of Match:"
    f" {df.loc[~goal_mask, 'motion_audio'].mean():+.4f}"
)

plt.figure(figsize=(14, 5))
plt.plot(
    df["second"],
    df["motion_audio"],
    color="teal",
    linewidth=1.5,
    label="Visual Motion + Audio Cortex (MT/MST/A4/A5)",
)
plt.axhline(0, color="gray", linestyle=":", linewidth=1)

for label, sec, name in goals:
  plt.axvspan(sec + 4, min(len(df), sec + 20), color="crimson", alpha=0.18)
  plt.axvline(sec, color="crimson", linestyle="--", linewidth=1.2)
  plt.text(
      sec + 5,
      df["motion_audio"].max() * 0.92,
      f"{label}\n{name}",
      color="darkred",
      fontsize=8,
      fontweight="bold",
  )

plt.title("Post-Goal Response Windows (+4s to +20s) — Motion & Audio Cortex")
plt.xlabel("Video Timeline (seconds)")
plt.ylabel("Activation Above Baseline")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.25)
plt.savefig("match_verification_clean.png", dpi=160, bbox_inches="tight")
print("Saved -> match_verification_clean.png")
