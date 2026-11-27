import argparse
import gc
from pathlib import Path
import pathlib
import re
import shutil
import time

from nilearn import datasets
import numpy as np
import pandas as pd
import torch
from tribev2 import TribeModel

# Windows compatibility for Linux-serialized checkpoints
pathlib.PosixPath = pathlib.WindowsPath


def clear_stale_locks(cache_dir: str = "C:/tc") -> None:
  """Removes leftover exca lock files from interrupted runs so Windows never throws WinError 11."""
  cache_path = Path(cache_dir)
  if not cache_path.exists():
    return
  for lock_item in cache_path.rglob("*inflight*"):
    if lock_item.is_dir():
      shutil.rmtree(lock_item, ignore_errors=True)
    else:
      lock_item.unlink(missing_ok=True)


def build_reading_timeline(text: str, wpm: int = 250) -> pd.DataFrame:
  """Converts raw text into a timed Word events DataFrame at natural reading speed (no TTS needed)."""
  sec_per_word = 60.0 / wpm
  paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]

  rows = []
  t = 0.0
  seq_id = 0

  for para in paragraphs:
    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()
    ]
    for sent in sentences:
      words = sent.split()
      for w in words:
        dur = sec_per_word + (0.02 * max(0, len(w) - 5))
        rows.append({
            "type": "Word",
            "start": round(t, 3),
            "duration": round(dur, 3),
            "stop": round(t + dur, 3),
            "timeline": "default",
            "subject": "default",
            "text": w,
            "sentence": sent,
            "sequence_id": float(seq_id),
            "language": "english",
            "context": sent,  # Shares full sentence so Llama 3.2 runs once per sentence, not per word
        })
        t += dur
        if w.endswith((".", "!", "?")):
          t += 0.35
        elif w.endswith((",", ";", ":")):
          t += 0.15
      seq_id += 1
    t += 0.65

  return pd.DataFrame(rows)


def split_into_chunks(
    events_df: pd.DataFrame, max_chunk_sec: float = 85.0
) -> list[pd.DataFrame]:
  """Splits timeline into <=85s chunks so every chunk fits cleanly inside TRIBE v2's 100s window."""
  chunks = []
  current_rows = []
  chunk_start = 0.0

  for _, row in events_df.iterrows():
    if (
        current_rows
        and (row["stop"] - chunk_start) > max_chunk_sec
        and row["text"].endswith((".", "!", "?"))
    ):
      chunks.append(pd.DataFrame(current_rows))
      current_rows = []
      chunk_start = row["start"]
    current_rows.append(row)

  if current_rows:
    chunks.append(pd.DataFrame(current_rows))
  return chunks


def configure_model(cache_dir: str = "C:/tc", use_gpu: bool = True) -> TribeModel:
  """Loads TRIBE v2 with single-worker safety so Llama 3.2 only loads once."""
  model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_dir)

  # Prevent Windows multiprocessing spawn crashes (WinError 1455)
  if hasattr(model.data, "num_workers"):
    model.data.num_workers = 0
  for field in vars(model.data).values():
    if hasattr(field, "num_workers"):
      field.num_workers = 0

  # Force single worker on text extractor so duplicate Llama models never load into VRAM
  if hasattr(model.data, "text"):
    txt_ext = model.data.text
    for attr in ("workers", "max_workers", "num_workers"):
      if hasattr(txt_ext, attr):
        setattr(txt_ext, attr, 1)
    if hasattr(txt_ext, "infra") and hasattr(txt_ext.infra, "max_workers"):
      txt_ext.infra.max_workers = 1
    if hasattr(txt_ext, "batch_size"):
      txt_ext.batch_size = 32

    if use_gpu and torch.cuda.is_available():
      txt_ext.device = "cuda"
      if hasattr(txt_ext, "dtype"):
        txt_ext.dtype = "float16"
    else:
      txt_ext.device = "cpu"

  return model


def map_vertices_to_atlas(
    preds: np.ndarray, atlas, labels: list[str]
) -> dict[str, np.ndarray]:
  """Maps (N_sec, 20484) cortical surface vertices onto 148 named anatomical brain regions."""
  left_preds = preds[:, :10242]
  right_preds = preds[:, 10242:]
  region_data = {}

  for idx, name in enumerate(labels):
    if name in ("Medial_wall", "Unknown"):
      continue
    l_mask = atlas.map_left == idx
    r_mask = atlas.map_right == idx
    if l_mask.any():
      region_data[f"L_{name}"] = left_preds[:, l_mask].mean(axis=1)
    if r_mask.any():
      region_data[f"R_{name}"] = right_preds[:, r_mask].mean(axis=1)

  return region_data


def scan_book(
    txt_path: str,
    out_dir: str = None,
    wpm: int = 250,
    save_raw: bool = False,
    cache_dir: str = "C:/tc",
) -> None:
  clear_stale_locks(cache_dir)

  src = Path(txt_path)
  bundle_dir = Path(out_dir) if out_dir else Path(f"{src.stem}.neuro")
  chunks_dir = bundle_dir / "chunks"
  chunks_dir.mkdir(parents=True, exist_ok=True)

  print(f"1. Tokenizing '{src.name}' at {wpm} WPM reading pace...")
  raw_text = src.read_text(encoding="utf-8")
  events_df = build_reading_timeline(raw_text, wpm=wpm)
  chunks = split_into_chunks(events_df, max_chunk_sec=85.0)
  print(
      f"   Total words: {len(events_df):,} | Est. timeline:"
      f" {events_df['stop'].max() / 60:.1f} mins | Chunks: {len(chunks)}"
  )

  print("2. Loading TRIBE v2 & Cortical Atlas...")
  model = configure_model(cache_dir=cache_dir, use_gpu=True)
  atlas = datasets.fetch_atlas_surf_destrieux()
  labels = [
      l.decode("utf-8") if isinstance(l, bytes) else str(l)
      for l in atlas.labels
  ]

  for i, chunk_df in enumerate(chunks):
    chunk_parquet = chunks_dir / f"chunk_{i:04d}.parquet"
    chunk_raw_npy = chunks_dir / f"chunk_{i:04d}_raw.npy"

    if chunk_parquet.exists():
      print(
          f"   [Chunk {i + 1}/{len(chunks)}] Already cached — skipping (resume"
          " mode)."
      )
      continue

    t0 = time.time()
    offset = float(chunk_df["start"].min())
    local_df = chunk_df.copy()
    local_df["start"] -= offset
    local_df["stop"] -= offset

    try:
      if torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=torch.float16):
          preds, _ = model.predict(events=local_df, verbose=False)
      else:
        preds, _ = model.predict(events=local_df, verbose=False)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
      if "out of memory" in str(err).lower() or "cuda" in str(err).lower():
        print(
            f"   [Chunk {i + 1}] GPU VRAM tight — switching extractor to CPU"
            " fallback..."
        )
        torch.cuda.empty_cache()
        gc.collect()
        if hasattr(model.data, "text"):
          model.data.text.device = "cpu"
        preds, _ = model.predict(events=local_df, verbose=False)
      else:
        raise

    n_sec = preds.shape[0]
    global_secs = np.arange(n_sec) + int(round(offset))
    region_cols = map_vertices_to_atlas(preds, atlas, labels)

    active_words, active_sents = [], []
    for sec in global_secs:
      mask = (chunk_df["start"] <= sec + 1.0) & (chunk_df["stop"] >= sec)
      matched = chunk_df[mask]
      active_words.append(
          " ".join(matched["text"].tolist()) if not matched.empty else ""
      )
      active_sents.append(
          matched.iloc[0]["sentence"] if not matched.empty else ""
      )

    out_chunk = pd.DataFrame({
        "second": global_secs,
        "chunk_id": i,
        "words": active_words,
        "sentence": active_sents,
        **region_cols,
    })
    out_chunk.to_parquet(chunk_parquet, index=False)

    if save_raw:
      np.save(chunk_raw_npy, preds.astype(np.float32))

    del preds
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

    print(
        f"   [Chunk {i + 1}/{len(chunks)}] Saved {n_sec}s in"
        f" {time.time() - t0:.1f}s"
    )

  print("3. Merging chunks into unified 1 Hz timeline...")
  all_parts = [
      pd.read_parquet(chunks_dir / f"chunk_{i:04d}.parquet")
      for i in range(len(chunks))
  ]
  full_timeline = pd.concat(all_parts, ignore_index=True)
  final_parquet = bundle_dir / "1_timeline_1hz.parquet"
  full_timeline.to_parquet(final_parquet, index=False)

  if save_raw:
    raw_parts = [
        np.load(chunks_dir / f"chunk_{i:04d}_raw.npy")
        for i in range(len(chunks))
    ]
    np.save(
        bundle_dir / "4_raw_vertices_1hz.npy",
        np.concatenate(raw_parts, axis=0),
    )

  print(f"\nDone! Bundle saved to: {final_parquet}")
  print(
      f"Shape: {full_timeline.shape[0]} seconds x"
      f" {full_timeline.shape[1] - 4} brain regions"
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Stream whole books into 1 Hz neural telemetry."
  )
  parser.add_argument("text_file", help="Path to .txt book or chapter")
  parser.add_argument(
      "--out", default=None, help="Output .neuro directory name"
  )
  parser.add_argument(
      "--wpm", type=int, default=250, help="Reading speed in words per minute"
  )
  parser.add_argument(
      "--save-raw",
      action="store_true",
      help="Also save full 20,484-vertex .npy array",
  )
  args = parser.parse_args()

  scan_book(
      args.text_file,
      out_dir=args.out,
      wpm=args.wpm,
      save_raw=args.save_raw,
  )
