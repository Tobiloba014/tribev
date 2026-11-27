import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
import numpy as np
import pandas as pd


def render_3d_brain(
    stem: str,
    start_sec: int,
    end_sec: int,
    out_png: str,
    hemi: str = "left",
    view: str = "lateral",
    threshold: float = 0.08,
):
  """Renders a 3D inflated cortical surface map from full 20,484 vertices (or reconstructs from 360 parquet)."""
  vert_path = Path(f"{stem}_vertices_fp16.npy")
  parquet_path = Path(f"{stem}_360.parquet")

  if vert_path.exists():
    verts = np.load(vert_path).astype(np.float32)
    stat_vec = verts[start_sec : end_sec + 1].mean(axis=0)
    source_tag = "Full 20,484 Vertices"
  elif parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    atlas = np.load("glasser_360.npz")
    stat_vec = np.zeros(20484, dtype=np.float32)
    row_mean = df.iloc[start_sec : end_sec + 1]
    for col in df.columns:
      if col in atlas.files:
        stat_vec[atlas[col]] = float(row_mean[col].mean())
    source_tag = "360 Glasser Reconstruction"
  else:
    raise FileNotFoundError(f"Could not find {vert_path} or {parquet_path}")

  fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
  if hemi == "left":
    mesh, sulc, data = fs.infl_left, fs.sulc_left, stat_vec[:10242]
  else:
    mesh, sulc, data = fs.infl_right, fs.sulc_right, stat_vec[10242:]

  plotting.plot_surf_stat_map(
      surf_mesh=mesh,
      stat_map=data,
      hemi=hemi,
      view=view,
      bg_map=sulc,
      threshold=threshold,
      cmap="cold_hot",
      title=f"{stem} ({start_sec}s-{end_sec}s) | {hemi.title()} {view.title()} [{source_tag}]",
  )
  plt.savefig(out_png, dpi=200, bbox_inches="tight")
  plt.close()
  print(f"Rendered 3D brain map -> {out_png}")


def diff_scans(stem_a: str, stem_b: str, top_k: int = 10):
  """Compares two scans (e.g. Draft 1 vs Draft 2, or Text vs Text+Music) across all 360 regions."""
  df_a = pd.read_parquet(f"{stem_a}_360.parquet")
  df_b = pd.read_parquet(f"{stem_b}_360.parquet")

  meta = {"second", "chunk_id", "words", "sentence"}
  regions = [c for c in df_a.columns if c not in meta and c in df_b.columns]

  mean_a = df_a[regions].mean()
  mean_b = df_b[regions].mean()
  delta = (mean_b - mean_a).sort_values(ascending=False)

  print(f"\n=== REVISION / MODALITY DIFF: [{stem_b}] minus [{stem_a}] ===")
  print("Biggest Increases (+):")
  for reg, val in delta.head(top_k).items():
    print(f"  {reg:<16} : {val:+.4f}  ({mean_a[reg]:+.3f} -> {mean_b[reg]:+.3f})")

  print("\nBiggest Decreases (-):")
  for reg, val in delta.tail(top_k).iloc[::-1].items():
    print(f"  {reg:<16} : {val:+.4f}  ({mean_a[reg]:+.3f} -> {mean_b[reg]:+.3f})")


def find_similar_moments(stem: str, target_sec: int, top_k: int = 5):
  """Uses the early 1 Hz Transformer latent vectors to find moments with the closest neural/semantic state."""
  lat_path = Path(f"{stem}_latent_1hz.npy")
  df = pd.read_parquet(f"{stem}_360.parquet")
  if not lat_path.exists():
    raise FileNotFoundError(
        f"{lat_path} not found. Re-run bare_engine.py to export latent embeddings."
    )

  lat = np.load(lat_path).astype(np.float32)
  norms = np.linalg.norm(lat, axis=1, keepdims=True) + 1e-8
  lat_norm = lat / norms

  query_vec = lat_norm[target_sec]
  sims = lat_norm @ query_vec

  # Exclude seconds immediately adjacent (+/- 5s) to the target moment
  sims[max(0, target_sec - 5) : min(len(sims), target_sec + 6)] = -1.0
  best_secs = np.argsort(sims)[::-1][:top_k]

  print(
      f"\n=== MOMENTS MOST SIMILAR TO t={target_sec}s"
      f" ('{df.loc[target_sec, 'sentence'][:80]}...') ==="
  )
  for s in best_secs:
    print(
        f"  t={int(s):>4}s | Similarity: {sims[s]:.3f} | Text:"
        f" {df.loc[s, 'sentence'][:110]}"
    )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="3D Brain Renderer, Revision Diff & Latent Similarity Explorer."
  )
  parser.add_argument(
      "stem",
      help="Base name of scan (e.g. 'warhammer_intro' without _360.parquet)",
  )
  parser.add_argument(
      "--render",
      nargs=2,
      type=int,
      metavar=("START_SEC", "END_SEC"),
      help="Render 3D brain surface map for a second range",
  )
  parser.add_argument(
      "--hemi", choices=["left", "right"], default="left", help="Hemisphere"
  )
  parser.add_argument(
      "--view",
      choices=["lateral", "medial", "dorsal", "ventral"],
      default="lateral",
  )
  parser.add_argument(
      "--diff",
      default=None,
      help="Second scan stem to compare against (e.g. --diff revised_chapter)",
  )
  parser.add_argument(
      "--similar-to",
      type=int,
      default=None,
      help="Find seconds with the closest internal Transformer latent state",
  )
  args = parser.parse_args()

  if args.render:
    s0, s1 = args.render
    out_img = f"{args.stem}_brain_{s0}s_{s1}s_{args.hemi}_{args.view}.png"
    render_3d_brain(args.stem, s0, s1, out_img, hemi=args.hemi, view=args.view)

  if args.diff:
    diff_scans(args.stem, args.diff)

  if args.similar_to is not None:
    find_similar_moments(args.stem, args.similar_to)