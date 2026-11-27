import json
from pathlib import Path
from nilearn import datasets, surface
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
import streamlit as st

st.set_page_config(
    page_title="NeuroStudio — Open Neural Telemetry",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark studio styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    .stMetric { background-color: #161920; padding: 10px 14px; border-radius: 8px; border: 1px solid #262b36; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_fsaverage_meshes():
  """Loads fsaverage5 inflated left & right cortical meshes once for live 3D rendering."""
  fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
  l_coords, l_faces = surface.load_surf_mesh(fs.infl_left)
  r_coords, r_faces = surface.load_surf_mesh(fs.infl_right)
  l_sulc = surface.load_surf_data(fs.sulc_left)
  r_sulc = surface.load_surf_data(fs.sulc_right)

  # Offset hemispheres slightly so both can render side-by-side without overlapping
  l_coords = l_coords.copy()
  r_coords = r_coords.copy()
  l_coords[:, 0] -= 32.0
  r_coords[:, 0] += 32.0

  atlas_path = Path("glasser_360.npz")
  atlas_dict = dict(np.load(atlas_path)) if atlas_path.exists() else {}

  return {
      "l_coords": l_coords,
      "l_faces": l_faces,
      "l_sulc": l_sulc,
      "r_coords": r_coords,
      "r_faces": r_faces,
      "r_sulc": r_sulc,
      "atlas": atlas_dict,
  }


@st.cache_data
def load_bundle(stem: str):
  """Loads the 360-region parquet, optional 20,484-vertex fp16 array, and 1 Hz latent vectors."""
  df = pd.read_parquet(f"{stem}_360.parquet")
  meta_cols = {"second", "chunk_id", "words", "sentence"}
  region_cols = [c for c in df.columns if c not in meta_cols]

  vert_path = Path(f"{stem}_vertices_fp16.npy")
  verts = np.load(vert_path, mmap_mode="r") if vert_path.exists() else None

  lat_path = Path(f"{stem}_latent_1hz.npy")
  latents = np.load(lat_path).astype(np.float32) if lat_path.exists() else None

  # Check if a matching video or audio file exists in the folder
  media_file = None
  media_type = None
  for ext in (".mp4", ".webm", ".mkv", ".mov"):
    if Path(f"{stem}{ext}").exists():
      media_file = f"{stem}{ext}"
      media_type = "video"
      break
  if media_file is None:
    for ext in (".mp3", ".wav", ".m4a", ".ogg"):
      if Path(f"{stem}{ext}").exists():
        media_file = f"{stem}{ext}"
        media_type = "audio"
        break

  return df, region_cols, verts, latents, media_file, media_type


def get_vertex_vector(
    df: pd.DataFrame,
    verts: np.ndarray | None,
    atlas: dict,
    start_sec: int,
    end_sec: int,
) -> np.ndarray:
  """Gets exact 20,484-vertex vector (or reconstructs from 360 Glasser parcels)."""
  s0 = max(0, min(start_sec, len(df) - 1))
  s1 = max(s0, min(end_sec, len(df) - 1))

  if verts is not None:
    return np.array(verts[s0 : s1 + 1], dtype=np.float32).mean(axis=0)

  stat_vec = np.zeros(20484, dtype=np.float32)
  sub = df.iloc[s0 : s1 + 1]
  for col, idx in atlas.items():
    if col in sub.columns:
      stat_vec[idx] = float(sub[col].mean())
  return stat_vec


def build_3d_brain_figure(
    mesh_data: dict,
    stat_vec: np.ndarray,
    hemi_mode: str = "Both Hemispheres",
    threshold: float = 0.05,
    clim: float = 0.45,
    colorscale: str = "RdBu_r",
) -> go.Figure:
  """Builds a fast, interactive, rotatable 3D Plotly cortical heatmap."""
  fig = go.Figure()

  # Threshold small baseline noise to NaN so underlying sulcal anatomy shows through
  display_vals = stat_vec.copy()
  display_vals[np.abs(display_vals) < threshold] = np.nan

  def add_hemi(coords, faces, vals, name):
    fig.add_trace(
        go.Mesh3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=vals,
            colorscale=colorscale,
            cmin=-clim,
            cmax=clim,
            cmid=0.0,
            showscale=(name == "Left"),
            colorbar=dict(title="Z-Score", thickness=12, len=0.65),
            name=name,
            lighting=dict(
                ambient=0.55, diffuse=0.8, roughness=0.5, specular=0.2
            ),
            flatshading=False,
            hovertemplate=f"{name} Cortex<br>Activation: %{{intensity:.3f}}<extra></extra>",
        )
    )

  if hemi_mode in ("Both Hemispheres", "Left Only"):
    add_hemi(
        mesh_data["l_coords"],
        mesh_data["l_faces"],
        display_vals[:10242],
        "Left",
    )
  if hemi_mode in ("Both Hemispheres", "Right Only"):
    add_hemi(
        mesh_data["r_coords"],
        mesh_data["r_faces"],
        display_vals[10242:],
        "Right",
    )

  fig.update_layout(
      margin=dict(l=0, r=0, t=10, b=0),
      height=430,
      paper_bgcolor="#0e1117",
      scene=dict(
          xaxis=dict(visible=False),
          yaxis=dict(visible=False),
          zaxis=dict(visible=False),
          aspectmode="data",
          camera=dict(eye=dict(x=-1.5, y=0.0, z=0.3)),
      ),
  )
  return fig


# ======================== SIDEBAR: PROJECT & TRACK CONTROLS ========================
st.sidebar.title("🧠 NeuroStudio")
st.sidebar.caption("Open-Ended Multi-Track Neural Telemetry")

parquet_files = sorted(Path(".").glob("*_360.parquet"))
if not parquet_files:
  st.error(
      "No `*_360.parquet` files found in this directory. Run `bare_engine.py`"
      " first."
  )
  st.stop()

stems = [p.name.removesuffix("_360.parquet") for p in parquet_files]
selected_stem = st.sidebar.selectbox("Active Project / Scan", stems)

df, region_cols, verts, latents, media_file, media_type = load_bundle(
    selected_stem
)
mesh_data = load_fsaverage_meshes()
total_secs = len(df)

st.sidebar.markdown("---")
st.sidebar.subheader("3D Brain Viewport Settings")
hemi_mode = st.sidebar.radio(
    "Hemispheres", ["Both Hemispheres", "Left Only", "Right Only"], index=0
)
threshold = st.sidebar.slider(
    "Activity Threshold (|Z|)", 0.0, 0.30, 0.05, 0.01
)
clim = st.sidebar.slider("Color Scale Range (±Z)", 0.10, 1.00, 0.45, 0.05)
colorscale = st.sidebar.selectbox(
    "Color Map", ["RdBu_r", "IceFire", "Hot", "Viridis", "Plasma"], index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Multi-Track Waveform Layers")
default_tracks = [
    c for c in ["L_PGi", "L_AVI", "R_AVI", "L_PH", "L_MT", "L_A5"] if c in region_cols
][:4]
tracked_regions = st.sidebar.multiselect(
    "Select Any of the 360 Cortical Parcels",
    options=region_cols,
    default=default_tracks,
)
smooth_sigma = st.sidebar.slider(
    "Waveform Smoothing (seconds, 0 = raw 1 Hz)", 0.0, 15.0, 2.0, 0.5
)

# Optional Revision / Modality Diff Track
diff_options = ["None"] + [s for s in stems if s != selected_stem]
diff_stem = st.sidebar.selectbox(
    "Overlay / Diff Against Another Scan", diff_options
)

# ======================== TOP PANE: PLAYHEAD + STIMULUS + LIVE 3D BRAIN ========================
st.markdown(f"### Timeline Inspector: `{selected_stem}`")

# Master Playhead Scrubber (DaVinci Resolve style)
col_scrub1, col_scrub2 = st.columns([4, 1])
with col_scrub1:
  playhead_sec = st.slider(
      "Master Playhead (Second)",
      min_value=0,
      max_value=max(0, total_secs - 1),
      value=min(10, total_secs - 1),
      step=1,
  )
with col_scrub2:
  window_span = st.number_input(
      "Window Span (sec)", min_value=1, max_value=120, value=5, step=1
  )

win_end = min(total_secs - 1, playhead_sec + int(window_span) - 1)

# Top Two Columns: Left = Media/Text Context + Live Region Readout | Right = Live 3D Brain
left_top, right_top = st.columns([1.15, 1.0])

with left_top:
  mins, secs = divmod(playhead_sec, 60)
  st.markdown(
      f"**Playhead:** `{mins:02d}:{secs:02d}` (`t = {playhead_sec}s` to"
      f" `{win_end}s`)"
  )

  # Embedded Video / Audio player synced to playhead if media exists
  if media_type == "video":
    st.video(media_file, start_time=playhead_sec)
  elif media_type == "audio":
    st.audio(media_file, start_time=playhead_sec)

  # Active Text Window
  slice_df = df.iloc[playhead_sec : win_end + 1]
  active_sents = list(
      dict.fromkeys(s.strip() for s in slice_df["sentence"] if s.strip())
  )
  active_words = " ".join(w.strip() for w in slice_df["words"] if w.strip())

  if active_sents or active_words:
    st.info(
        f"**Active Passage:** {' '.join(active_sents) if active_sents else active_words}"
    )

  # Live Top Elevated (+) and Suppressed (-) Regions at Playhead Window
  win_means = slice_df[region_cols].mean()
  top_pos = win_means.sort_values(ascending=False).head(6)
  top_neg = win_means.sort_values(ascending=True).head(6)

  c_pos, c_neg = st.columns(2)
  with c_pos:
    st.markdown("**Strongest Elevated (+)**")
    for reg, val in top_pos.items():
      st.write(f"`{reg}` : **{val:+.3f}**")
  with c_neg:
    st.markdown("**Strongest Suppressed (-)**")
    for reg, val in top_neg.items():
      st.write(f"`{reg}` : **{val:+.3f}**")

with right_top:
  stat_vec = get_vertex_vector(
      df, verts, mesh_data["atlas"], playhead_sec, win_end
  )
  res_label = (
      "Full 20,484 Vertices"
      if verts is not None
      else "360 Parcel Reconstruction"
  )
  st.caption(f"Live 3D Cortical Surface ({res_label}) — Rotate & Zoom Freely")
  brain_fig = build_3d_brain_figure(
      mesh_data,
      stat_vec,
      hemi_mode=hemi_mode,
      threshold=threshold,
      clim=clim,
      colorscale=colorscale,
  )
  st.plotly_chart(brain_fig, use_container_width=True)

# ======================== MIDDLE PANE: MULTI-TRACK TIMELINE WAVEFORMS ========================
st.markdown("---")
st.markdown("### Multi-Track Neural Timeline (Zoom, Pan & Inspect Every Second)")

# Range zoom slider for massive books (e.g. 27,000s Bobiverse)
zoom_range = st.slider(
    "Timeline Zoom Window (Seconds)",
    min_value=0,
    max_value=max(1, total_secs - 1),
    value=(0, min(total_secs - 1, 600)),
)
z0, z1 = zoom_range
view_df = df.iloc[z0 : z1 + 1].copy()
mat_view = view_df[region_cols].to_numpy(dtype=np.float64)
pos_energy = np.maximum(0.0, mat_view).mean(axis=1)
neg_suppression = np.minimum(0.0, mat_view).mean(axis=1)
if smooth_sigma > 0 and len(view_df) > 1:
  pos_energy = gaussian_filter1d(pos_energy, sigma=smooth_sigma)
  neg_suppression = gaussian_filter1d(neg_suppression, sigma=smooth_sigma)

n_rows = 1 + len(tracked_regions)
fig_tl = make_subplots(
  rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
  subplot_titles=["Global Cortical Envelope (+ / -)"] + [f"Parcel: {r}" for r in tracked_regions],
)
hover_text = [f"t={int(row.second)}s | {str(row.sentence)[:85]}" for row in view_df.itertuples()]

fig_tl.add_trace(go.Scatter(
  x=view_df["second"], y=pos_energy, name="Positive Activation (+)",
  line=dict(color="#ff4b4b", width=1.6), text=hover_text, hoverinfo="x+y+text",
), row=1, col=1)
fig_tl.add_trace(go.Scatter(
  x=view_df["second"], y=neg_suppression, name="Suppression (-)",
  line=dict(color="#3b82f6", width=1.4), text=hover_text, hoverinfo="x+y+text",
), row=1, col=1)

df_diff = None
if diff_stem != "None":
  df_diff, _, _, _, _, _ = load_bundle(diff_stem)

for row_index, region in enumerate(tracked_regions, start=2):
  values = view_df[region].to_numpy(dtype=np.float64)
  if smooth_sigma > 0 and len(values) > 1:
    values = gaussian_filter1d(values, sigma=smooth_sigma)
  fig_tl.add_trace(go.Scatter(
    x=view_df["second"], y=values, name=region, line=dict(width=1.8),
    text=hover_text, hoverinfo="x+y+text",
  ), row=row_index, col=1)

  if df_diff is not None and region in df_diff.columns:
    diff_slice = df_diff.iloc[z0 : min(z1 + 1, len(df_diff))]
    if not diff_slice.empty:
      diff_values = diff_slice[region].to_numpy(dtype=np.float64)
      if smooth_sigma > 0 and len(diff_values) > 1:
        diff_values = gaussian_filter1d(diff_values, sigma=smooth_sigma)
      fig_tl.add_trace(go.Scatter(
        x=diff_slice["second"], y=diff_values, name=f"{region} [{diff_stem}]",
        line=dict(dash="dot", width=1.5),
      ), row=row_index, col=1)

for row_index in range(1, n_rows + 1):
  fig_tl.add_vline(
    x=playhead_sec, line_width=1.5, line_dash="dash", line_color="gold",
    row=row_index, col=1,
  )
fig_tl.update_layout(
  height=180 + 115 * len(tracked_regions), template="plotly_dark",
  margin=dict(l=40, r=20, t=30, b=20), showlegend=True,
  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_tl, use_container_width=True)

# ======================== BOTTOM PANE: OPEN DISCOVERY & CUSTOM QUERIES ========================
st.markdown("---")
tab_rev, tab_coact, tab_sim, tab_raw = st.tabs([
  "🔍 Reverse-Trigger Finder (Self-Discovery)", "🕸️ Co-Activation Networks",
  "🧬 Latent Similarity Search", "📊 Raw Data & Custom Export",
])

with tab_rev:
  st.markdown("Select regions or combinations of regions to discover moments that cause them to spike or drop.")
  c1, c2, c3, c4 = st.columns([3, 1.5, 1, 1])
  with c1:
    probe_regions = st.multiselect(
      "Target Brain Parcels", options=region_cols,
      default=tracked_regions[:2] if tracked_regions else region_cols[:1], key="probe_regs",
    )
  with c2:
    probe_weights_str = st.text_input("Optional Weights (comma-separated, e.g. 1, -1)", value="")
  with c3:
    search_dir = st.selectbox("Direction", ["Highest (+)", "Lowest / Quietest (-)"])
  with c4:
    block_size = st.number_input("Block Size (s)", 1, 60, 10)

  if probe_regions:
    sub_mat = df[probe_regions].to_numpy(dtype=np.float64)
    composite = sub_mat.mean(axis=1)
    if probe_weights_str.strip():
      try:
        weights = np.array([float(x.strip()) for x in probe_weights_str.split(",")], dtype=np.float64)
        if len(weights) == len(probe_regions) and np.abs(weights).sum() > 0:
          composite = (sub_mat * weights).sum(axis=1) / np.abs(weights).sum()
        else:
          st.warning("Weight count does not match region count; using equal weights.")
      except ValueError:
        st.warning("Invalid weight format; using equal weights.")

    blocks = []
    for block_start in range(0, total_secs, int(block_size)):
      block_end = min(total_secs - 1, block_start + int(block_size) - 1)
      block_slice = df.iloc[block_start : block_end + 1]
      sentences = list(dict.fromkeys(
        str(value).strip() for value in block_slice["sentence"]
        if str(value).strip() and str(value).lower() != "nan"
      ))
      text_context = " ".join(sentences) or " ".join(
        str(value).strip() for value in block_slice["words"]
        if str(value).strip() and str(value).lower() != "nan"
      )
      blocks.append({
        "start_sec": block_start, "end_sec": block_end,
        "time": f"{block_start // 60:02d}:{block_start % 60:02d}",
        "signal_score": round(float(composite[block_start : block_end + 1].mean()), 4),
        "text_or_context": text_context,
      })
    results = pd.DataFrame(blocks).sort_values(
      "signal_score", ascending=(search_dir == "Lowest / Quietest (-)")
    )
    st.dataframe(results.head(20), use_container_width=True)

with tab_coact:
  st.markdown("Discover which regions act together using the 1 Hz correlation matrix.")
  anchor_region = st.selectbox("Choose Anchor Parcel", options=region_cols, index=0)
  correlations = df[region_cols].corrwith(df[anchor_region]).dropna().sort_values(ascending=False)
  col_co1, col_co2 = st.columns(2)
  with col_co1:
    st.markdown(f"Strongest Allies (Rise & Fall With {anchor_region})")
    st.dataframe(correlations.iloc[1:16].rename("correlation").reset_index(), use_container_width=True)
  with col_co2:
    st.markdown(f"Strongest Opponents (Inverse Activity to {anchor_region})")
    st.dataframe(correlations.tail(15).iloc[::-1].rename("correlation").reset_index(), use_container_width=True)

with tab_sim:
  st.markdown(f"Find moments across {selected_stem} closest to the neural state at t = {playhead_sec}s.")
  latent_label = "Internal Transformer Latents (1,152-dim)"
  cortical_label = "360 Glasser Cortical Space"
  vec_source = st.radio(
    "Similarity Space", [latent_label if latents is not None else cortical_label, cortical_label],
    horizontal=True,
  )
  space = latents if latent_label in vec_source and latents is not None else df[region_cols].to_numpy(dtype=np.float32)
  normed = space / np.maximum(np.linalg.norm(space, axis=1, keepdims=True), 1e-8)
  reference = min(playhead_sec, len(normed) - 1)
  similarities = normed @ normed[reference]
  similarities[max(0, reference - 4) : min(total_secs, reference + 5)] = -1.0
  similarity_rows = []
  for second in np.argsort(similarities)[::-1][:15]:
    second = int(second)
    sentence = str(df.iloc[second].get("sentence", ""))
    words = str(df.iloc[second].get("words", ""))
    similarity_rows.append({
      "second": second, "time": f"{second // 60:02d}:{second % 60:02d}",
      "cosine_similarity": round(float(similarities[second]), 4),
      "text": sentence if sentence and sentence.lower() != "nan" else words,
    })
  st.dataframe(pd.DataFrame(similarity_rows), use_container_width=True)

with tab_raw:
  st.markdown("Zero-Loss 1 Hz Telemetry Table — inspect or export any slice.")
  export_start, export_end = st.slider(
    "Export Second Range", 0, max(0, total_secs - 1), (playhead_sec, win_end)
  )
  export_slice = df.iloc[export_start : export_end + 1]
  st.dataframe(export_slice.iloc[:, :15], use_container_width=True)
  c_dl1, c_dl2 = st.columns(2)
  with c_dl1:
    st.download_button(
      "Download Selected Span as JSON", data=export_slice.to_json(orient="records", indent=2),
      file_name=f"{selected_stem}_{export_start}s_{export_end}s.json", mime="application/json",
    )
  with c_dl2:
    st.download_button(
      "Download Selected Span as CSV", data=export_slice.to_csv(index=False),
      file_name=f"{selected_stem}_{export_start}s_{export_end}s.csv", mime="text/csv",
    )
