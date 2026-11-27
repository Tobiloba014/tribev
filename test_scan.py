import matplotlib

matplotlib.use("Agg")  # Renders PNGs directly without Tkinter GUI
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
import numpy as np
import torch
from tribev2 import TribeModel


def main():
  text = """For more than a hundred centuries the Emperor has sat immobile on the Golden Throne of Earth. He is the Master of Mankind. By the might of His inexhaustible armies a million worlds stand against the dark.

Yet, He is a rotting carcass, the Carrion Lord of the Imperium held in life by marvels from the Dark Age of Technology and the thousand souls sacrificed each day so that His may continue to burn.

To be a man in such times is to be one amongst untold billions. It is to live in the cruellest and most bloody regime imaginable. It is to suffer an eternity of carnage and slaughter. It is to have cries of anguish and sorrow drowned by the thirsting laughter of dark gods.

This is a dark and terrible era where you will find little comfort or hope. Forget the power of technology and science. Forget the promise of progress and advancement. Forget any notion of common humanity or compassion.

There is no peace amongst the stars, for in the grim darkness of the far future, there is only war."""

  with open("warhammer_intro.txt", "w", encoding="utf-8") as f:
    f.write(text)

  model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="C:/tc")

  # Prevent Windows from spawning duplicate PyTorch/CUDA worker processes
  if hasattr(model.data, "num_workers"):
      model.data.num_workers = 0
  for field in vars(model.data).values():
      if hasattr(field, "num_workers"):
          field.num_workers = 0

  if hasattr(model.data, "text") and hasattr(model.data.text, "device"):
    model.data.text.device = "cpu"
  if hasattr(model.data, "audio") and hasattr(model.data.audio, "device"):
    model.data.audio.device = "cuda"
  torch.cuda.empty_cache()

  events_df = model.get_events_dataframe(text_path="warhammer_intro.txt")
  preds, segments = model.predict(events=events_df)

  np.save("warhammer_preds.npy", preds)
  events_df.to_csv("warhammer_events.csv", index=False)

  fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
  climax_map = preds[-5:].mean(axis=0)
  left_hemi = climax_map[:10242]

  plotting.plot_surf_stat_map(
      surf_mesh=fsaverage.infl_left,
      stat_map=left_hemi,
      hemi="left",
      view="lateral",
      bg_map=fsaverage.sulc_left,
      threshold=0.1,
      cmap="cold_hot",
      title="Grim Darkness Climax (Left Cortex)",
  )
  plt.savefig("warhammer_climax_brain.png", dpi=200, bbox_inches="tight")
  print("Success! Saved warhammer_climax_brain.png")


if __name__ == "__main__":
  main()
