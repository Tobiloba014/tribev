import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class NeuralExplorer:
  """Open-ended, un-opinionated multi-scale scene segmenter and 360-region query engine."""

  def __init__(self, parquet_path: str):
    self.parquet_path = Path(parquet_path)
    self.df = pd.read_parquet(self.parquet_path)
    self.meta_cols = {"second", "chunk_id", "words", "sentence"}
    self.region_cols = [c for c in self.df.columns if c not in self.meta_cols]

  def list_regions(self, filter_str: str | None = None) -> list[str]:
    """Lists all available 360 Glasser parcels (optionally filtered by substring)."""
    if not filter_str:
      return self.region_cols
    q = filter_str.lower()
    return [c for c in self.region_cols if q in c.lower()]

  def resolve_regions(self, patterns: list[str] | None = None) -> list[str]:
    """Matches user-supplied parcel patterns against all 360 Glasser columns."""
    if not patterns:
      return self.region_cols

    matched = []
    for pat in patterns:
      pat_low = pat.lower()
      hits = [c for c in self.region_cols if pat_low in c.lower()]
      matched.extend(hits)

    unique_cols = list(dict.fromkeys(matched))
    if not unique_cols:
      raise ValueError(
          f"No regions matched {patterns}. Run with --list-regions to inspect"
          " all 360 parcel names."
      )
    return unique_cols

  def build_signal(
      self,
      regions: list[str] | None = None,
      weights: list[float] | None = None,
  ) -> tuple[np.ndarray, list[str]]:
    """Computes a 1 Hz signal from all regions (global cortical drive) or a user-defined subset."""
    cols = self.resolve_regions(regions)
    matrix = self.df[cols].to_numpy(dtype=np.float64)

    if weights is not None:
      w = np.asarray(weights, dtype=np.float64)
      if len(w) != len(cols):
        raise ValueError(
            f"Number of weights ({len(w)}) must match matched regions"
            f" ({len(cols)}): {cols}"
        )
      signal = (matrix * w).sum(axis=1) / np.abs(w).sum()
    else:
      # When scanning all 360 regions blindly, use RMS amplitude so positive & negative shifts don't cancel out
      if regions is None:
        signal = np.sqrt((matrix**2).mean(axis=1))
      else:
        signal = matrix.mean(axis=1)

    return signal, cols

  def segment_scenes(
      self,
      regions: list[str] | None = None,
      weights: list[float] | None = None,
      smooth_sigma: float = 3.0,
      min_scene_sec: int = 6,
      prominence: float | None = None,
      top_drivers: int = 5,
  ) -> pd.DataFrame:
    """
    Finds multi-second valley-to-peak arcs and reports which specific brain regions
    spiked strongest inside each scene without imposing subjective labels.
    """
    raw_sig, matched_cols = self.build_signal(regions, weights)
    smooth_sig = gaussian_filter1d(raw_sig, sigma=smooth_sigma)

    if prominence is None:
      prominence = max(0.005, float(np.std(smooth_sig) * 0.35))

    peaks, _ = find_peaks(
        smooth_sig, distance=min_scene_sec, prominence=prominence
    )
    if len(peaks) == 0:
      peaks = np.array([int(np.argmax(smooth_sig))])

    n = len(raw_sig)
    all_matrix = self.df[self.region_cols].to_numpy(dtype=np.float64)
    scenes = []

    for idx, p in enumerate(peaks):
      start = p
      while start > 0 and smooth_sig[start - 1] <= smooth_sig[start]:
        start -= 1

      end = p
      while end < n - 1 and smooth_sig[end + 1] <= smooth_sig[end]:
        end += 1

      if idx > 0 and start < scenes[-1]["end_second"]:
        mid = (scenes[-1]["peak_second"] + p) // 2
        scenes[-1]["end_second"] = mid
        start = mid

      seg_raw = raw_sig[start : end + 1]
      duration = max(1, end - start + 1)
      rise_time = max(1, p - start)

      peak_val = float(np.max(seg_raw))
      sustained_mean = float(np.mean(seg_raw))
      buildup_slope = float((raw_sig[p] - raw_sig[start]) / rise_time)
      total_impact_auc = float(np.sum(np.maximum(0.0, seg_raw)))

      # Data-driven discovery: find the exact parcels that activated highest during this scene
      seg_all_regions = all_matrix[start : end + 1].mean(axis=0)
      top_idx = np.argsort(seg_all_regions)[::-1][:top_drivers]
      drivers = [
          f"{self.region_cols[k]} ({seg_all_regions[k]:+.3f})" for k in top_idx
      ]

      slice_df = self.df.iloc[start : end + 1]
      unique_sents = list(
          dict.fromkeys(s for s in slice_df["sentence"] if s.strip())
      )
      excerpt = (
          " ".join(unique_sents)
          if unique_sents
          else " ".join(w for w in slice_df["words"] if w.strip())
      )

      scenes.append({
          "scene_id": idx + 1,
          "start_second": int(start),
          "peak_second": int(p),
          "end_second": int(end),
          "duration_sec": int(duration),
          "peak_intensity": round(peak_val, 4),
          "sustained_mean": round(sustained_mean, 4),
          "buildup_slope": round(buildup_slope, 4),
          "total_impact_auc": round(total_impact_auc, 4),
          "top_active_parcels": ", ".join(drivers),
          "peak_sentence": str(self.df.loc[p, "sentence"]).strip(),
          "full_scene_text": excerpt.strip(),
      })

    scenes_df = pd.DataFrame(scenes)
    scenes_df["composite_score"] = (
        0.45 * scenes_df["peak_intensity"]
        + 0.35 * scenes_df["sustained_mean"]
        + 0.20 * (scenes_df["total_impact_auc"] / scenes_df["duration_sec"].max())
    ).round(4)

    return scenes_df

  def plot_telemetry(
      self,
      scenes_df: pd.DataFrame,
      out_png: str,
      regions: list[str] | None = None,
      weights: list[float] | None = None,
      smooth_sigma: float = 3.0,
  ) -> None:
    raw_sig, cols = self.build_signal(regions, weights)
    smooth_sig = gaussian_filter1d(raw_sig, sigma=smooth_sigma)
    secs = self.df["second"].to_numpy()

    plt.figure(figsize=(13, 5))
    plt.plot(
        secs,
        raw_sig,
        color="gray",
        alpha=0.35,
        linewidth=1,
        label="Raw 1 Hz Signal",
    )
    plt.plot(
        secs,
        smooth_sig,
        color="crimson",
        linewidth=2.2,
        label=f"Smoothed Arc (sigma={smooth_sigma}s)",
    )

    for _, row in scenes_df.iterrows():
      plt.axvspan(
          row["start_second"],
          row["end_second"],
          color="gold",
          alpha=0.18,
      )
      plt.scatter(
          [row["peak_second"]],
          [smooth_sig[row["peak_second"]]],
          color="darkred",
          zorder=5,
      )
      plt.text(
          row["peak_second"],
          smooth_sig[row["peak_second"]] + 0.01,
          f"S{int(row['scene_id'])}",
          fontsize=8,
          ha="center",
          fontweight="bold",
      )

    title_tag = (
        f"Custom ({', '.join(regions)})"
        if regions
        else "All 360 Parcels (Global Cortical Energy)"
    )
    plt.title(f"Neural Telemetry & Multi-Second Arcs — {title_tag}")
    plt.xlabel("Timeline (seconds)")
    plt.ylabel("Cortical Response")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.25)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Open-ended 360-region neural telemetry explorer."
  )
  parser.add_argument("parquet_file", help="Path to *_360.parquet file")
  parser.add_argument(
      "--list-regions",
      nargs="?",
      const="",
      default=None,
      help="List available 360 Glasser region names (optional search filter)",
  )
  parser.add_argument(
      "--regions",
      nargs="+",
      default=None,
      help=(
          "Custom Glasser parcel substrings to track (e.g. --regions PHA1 PGi"
          " AVI). Omit for blind whole-brain discovery."
      ),
  )
  parser.add_argument(
      "--weights",
      nargs="+",
      type=float,
      default=None,
      help="Optional custom weights matching each resolved region column",
  )
  parser.add_argument(
      "--lens-file",
      default=None,
      help=(
          "Optional path to a user-created JSON lens file: {'regions': [...],"
          " 'weights': [...]}"
      ),
  )
  parser.add_argument(
      "--top", type=int, default=5, help="Number of top scenes to output"
  )
  parser.add_argument(
      "--sigma",
      type=float,
      default=3.0,
      help="Smoothing scale in seconds (smaller = finer beats, larger = broad chapters)",
  )
  parser.add_argument(
      "--sort-by",
      default="composite_score",
      choices=[
          "composite_score",
          "peak_intensity",
          "sustained_mean",
          "buildup_slope",
          "total_impact_auc",
      ],
  )
  args = parser.parse_args()

  explorer = NeuralExplorer(args.parquet_file)

  if args.list_regions is not None:
    matches = explorer.list_regions(args.list_regions or None)
    print(f"Found {len(matches)} matching Glasser parcels:")
    for i in range(0, len(matches), 6):
      print("  " + ", ".join(matches[i : i + 6]))
    raise SystemExit(0)

  regions = args.regions
  weights = args.weights
  if args.lens_file:
    spec = json.loads(Path(args.lens_file).read_text(encoding="utf-8"))
    regions = spec.get("regions")
    weights = spec.get("weights")

  scenes = explorer.segment_scenes(
      regions=regions, weights=weights, smooth_sigma=args.sigma
  )
  ranked = scenes.sort_values(args.sort_by, ascending=False)

  stem = Path(args.parquet_file).stem
  tag = "_".join(regions[:3]) if regions else "whole_brain"
  out_parquet = f"{stem}_scenes_{tag}.parquet"
  out_json = f"{stem}_top_scenes_{tag}.json"
  out_png = f"{stem}_arc_{tag}.png"

  ranked.to_parquet(out_parquet, index=False)
  ranked.head(args.top).to_json(out_json, orient="records", indent=2)
  explorer.plot_telemetry(
      scenes_df=scenes,
      out_png=out_png,
      regions=regions,
      weights=weights,
      smooth_sigma=args.sigma,
  )
  print(
      f"\n=== TOP {min(args.top, len(ranked))} SCENES | MODE:"
      f" {regions or 'WHOLE-BRAIN DISCOVERY'} | SORTED BY: {args.sort_by} ==="
  )
  for _, r in ranked.head(args.top).iterrows():
      print(
          f"\n[Scene {int(r['scene_id'])}] {int(r['start_second'])}s ->"
          f" {int(r['end_second'])}s (Peak @ {int(r['peak_second'])}s) | Score:"
          f" {r['composite_score']:.3f} (Peak: {r['peak_intensity']:+.3f},"
          f" Slope: {r['buildup_slope']:+.3f})"
      )
  print(f"  Top Parcels: {r['top_active_parcels']}")
  print(f"  Peak Line  : {r['peak_sentence']}")
  print(f"  Scene Text : {r['full_scene_text'][:140]}...")
  print(f"\nExported scene table : {out_parquet}")
  print(f"Exported scene cards : {out_json}")
  print(f"Exported arc plot    : {out_png}")
