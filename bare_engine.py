import argparse
import pathlib
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

yaml.UnsafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    lambda loader, node: pathlib.PurePosixPath(*loader.construct_sequence(node)),
)
yaml.UnsafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib._local.PosixPath",
    lambda loader, node: pathlib.PurePosixPath(*loader.construct_sequence(node)),
)


from huggingface_hub import hf_hub_download
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    AutoVideoProcessor,
    BitsAndBytesConfig,
    Wav2Vec2BertModel,
)

from tribev2.model import FmriEncoder


@dataclass
class SimpleBatch:
    data: dict


class BareTribeEngine:
    """Multimodal TRIBE v2 inference engine for text, audio, and video."""

    def __init__(
        self,
        repo_id: str = "facebook/tribev2",
        atlas_path: str = "glasser_360.npz",
        device: str = "auto",
    ):
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        print(f"[BareEngine] Initializing core brain model on {self.device}...")

        config_path = hf_hub_download(repo_id, "config.yaml")
        ckpt_path = hf_hub_download(repo_id, "best.ckpt")

        with open(config_path, "r", encoding="utf-8") as stream:
            cfg = yaml.load(stream, Loader=yaml.UnsafeLoader)

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        build_args = ckpt["model_build_args"]
        state_dict = {
            key.removeprefix("model."): value
            for key, value in ckpt["state_dict"].items()
        }

        brain_cfg = FmriEncoder(**cfg["brain_model_config"])
        brain_cfg.subject_layers.average_subjects = True
        brain_cfg.subject_layers.n_subjects = 0
        if hasattr(brain_cfg.projector, "average_subjects"):
            brain_cfg.projector.average_subjects = True

        self.brain_model = brain_cfg.build(**build_args)
        self.brain_model.load_state_dict(state_dict, strict=True, assign=True)
        self.brain_model.to(self.device).eval()

        self._hooked_latents = []
        self.brain_model.predictor.register_forward_hook(self._capture_latent_hook)

        self.tokenizer = None
        self.llama = None
        self.audio_proc = None
        self.audio_model = None
        self.video_proc = None
        self.video_model = None

        self.atlas_names, self.atlas_matrix = self._load_bundled_atlas(atlas_path)
        print(f"[BareEngine] Core ready ({len(self.atlas_names)} parcels).")

    def _capture_latent_hook(self, module, inputs, output):
        """Capture predictor input features and pool each 200-bin window to 1 Hz."""
        features = inputs[0] if isinstance(inputs, tuple) else inputs
        if not torch.is_tensor(features) or features.ndim != 3:
            return

        pooled = torch.nn.functional.adaptive_avg_pool1d(features, 100)
        # (batch, channels, time) -> (batch*time, channels)
        latent = pooled.permute(0, 2, 1).reshape(-1, pooled.shape[1])
        self._hooked_latents.append(latent.detach().cpu())

    def _load_bundled_atlas(self, atlas_path: str):
        path = Path(atlas_path)
        if not path.exists():
            from tribev2.utils import get_hcp_labels

            left = get_hcp_labels("fsaverage5", False, "left")
            right = get_hcp_labels("fsaverage5", False, "right")
            np.savez_compressed(
                path,
                **({f"L_{key}": value for key, value in left.items()}
                   | {f"R_{key}": value for key, value in right.items()})
            )

        data = np.load(path)
        names = [name for name in data.files if not name.endswith("_?")]
        matrix = torch.zeros((20484, len(names)), dtype=torch.float32)
        for column, name in enumerate(names):
            indices = np.asarray(data[name], dtype=np.int64).ravel()
            if len(indices) == 0:
                continue
            matrix[torch.from_numpy(indices), column] = 1.0 / len(indices)
        return names, matrix.to(self.device)

    def _ensure_text_model(self):
        if self.llama is not None:
            return

        model_id = "meta-llama/Llama-3.2-3B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self.llama = AutoModel.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="cuda:0",
                low_cpu_mem_usage=True,
            )
        else:
            self.llama = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
        self.llama.eval()

    def _ensure_audio_model(self):
        if self.audio_model is not None:
            return
        model_id = "facebook/w2v-bert-2.0"
        self.audio_proc = AutoFeatureExtractor.from_pretrained(model_id)
        self.audio_model = Wav2Vec2BertModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.audio_model.eval()

    def _ensure_video_model(self):
        if self.video_model is not None:
            return
        model_id = "facebook/vjepa2-vitg-fpc64-256"
        self.video_proc = AutoVideoProcessor.from_pretrained(model_id)
        self.video_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.video_model.eval()

    def _tokenize_and_time(self, text: str, wpm: int = 250):
        if wpm <= 0:
            raise ValueError("wpm must be greater than zero")

        sec_per_word = 60.0 / wpm
        paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
        words, starts, stops, sentences = [], [], [], []
        current_time = 0.0

        for paragraph in paragraphs:
            sentence_list = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
                if sentence.strip()
            ]
            for sentence in sentence_list:
                for word in sentence.split():
                    duration = sec_per_word + 0.02 * max(0, len(word) - 5)
                    words.append(word)
                    starts.append(current_time)
                    stops.append(current_time + duration)
                    sentences.append(sentence)
                    current_time += duration
                    if word.endswith((".", "!", "?")):
                        current_time += 0.35
                    elif word.endswith((",", ";", ":")):
                        current_time += 0.15
            current_time += 0.65

        return words, starts, stops, sentences

    @torch.inference_mode()
    def extract_text_grid(self, text: str, wpm: int = 250,
                          window_words: int = 350, stride_words: int = 250):
        self._ensure_text_model()
        words, starts, stops, sentences = self._tokenize_and_time(text, wpm)
        if not words:
            raise ValueError("Input text is empty.")

        word_vectors = torch.zeros((len(words), 2, 3072), dtype=torch.float32)
        computed = torch.zeros(len(words), dtype=torch.bool)

        for start_index in range(0, len(words), stride_words):
            end_index = min(start_index + window_words, len(words))
            chunk_words = words[start_index:end_index]

            spans, pieces, cursor = [], [], 0
            for word in chunk_words:
                if pieces:
                    pieces.append(" ")
                    cursor += 1
                spans.append((cursor, cursor + len(word)))
                pieces.append(word)
                cursor += len(word)

            encoded = self.tokenizer(
                "".join(pieces),
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            hidden_states = self.llama(**encoded, output_hidden_states=True).hidden_states
            group_1 = torch.stack(hidden_states[14:22]).mean(dim=0).float()[0]
            group_2 = torch.stack(hidden_states[21:29]).mean(dim=0).float()[0]

            keep_from = 0 if start_index == 0 else max(0, window_words - stride_words)
            for local_index in range(keep_from, len(chunk_words)):
                global_index = start_index + local_index
                if computed[global_index]:
                    continue
                word_start, word_end = spans[local_index]
                token_indices = [
                    token_index
                    for token_index, (token_start, token_end) in enumerate(offsets)
                    if token_end > word_start and token_start < word_end and token_end > token_start
                ]
                if token_indices:
                    word_vectors[global_index, 0] = group_1[token_indices].mean(dim=0).cpu()
                    word_vectors[global_index, 1] = group_2[token_indices].mean(dim=0).cpu()
                computed[global_index] = True

            if end_index == len(words):
                break

        total_seconds = int(np.ceil(stops[-1])) + 1
        grid = torch.zeros((total_seconds * 2, 2, 3072), dtype=torch.float32)
        for index, (start, stop) in enumerate(zip(starts, stops)):
            bin_index = min(int(0.5 * (start + stop) * 2), len(grid) - 1)
            grid[bin_index] += word_vectors[index]
        return grid, total_seconds, words, starts, stops, sentences

    @torch.inference_mode()
    def extract_audio_grid(self, media_path: str, chunk_sec: float = 25.0):
        self._ensure_audio_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
            wav_path = Path(temporary.name)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(media_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-loglevel",
                    "error",
                    str(wav_path),
                ],
                check=True,
            )
            import wave

            with wave.open(str(wav_path), "rb") as wave_file:
                samples = wave_file.readframes(wave_file.getnframes())
            audio = (
                    np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
            )
        finally:
            wav_path.unlink(missing_ok=True)

        sample_rate = 16000
        total_seconds = max(1, int(np.ceil(len(audio) / sample_rate)))
        grid = torch.zeros((total_seconds * 2, 2, 1024), dtype=torch.float32)
        step = int(chunk_sec * sample_rate)

        for start_sample in range(0, len(audio), step):
            chunk = audio[start_sample: start_sample + step]
            if len(chunk) < 400:
                continue
            inputs = self.audio_proc(
                chunk, sampling_rate=sample_rate, return_tensors="pt"
            )
            features = inputs.input_features.to(
                self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            hs = self.audio_model(
                features, output_hidden_states=True
            ).hidden_states

            g1 = torch.stack(hs[12:19], dim=0).mean(dim=0).squeeze(0).float().cpu()
            g2 = torch.stack(hs[18:25], dim=0).mean(dim=0).squeeze(0).float().cpu()
            combined = torch.stack([g1, g2], dim=1)  # (T_frames, 2, 1024)

            chunk_bins = max(1, int(np.ceil((len(chunk) / sample_rate) * 2.0)))
            pooled = (
                torch.nn.functional.adaptive_avg_pool1d(
                    combined.permute(1, 2, 0).reshape(1, 2048, -1), chunk_bins
                )
                .view(2, 1024, chunk_bins)
                .permute(2, 0, 1)
            )
            begin = int((start_sample / sample_rate) * 2.0)
            end = min(begin + chunk_bins, len(grid))
            grid[begin:end] = pooled[: end - begin]

        return grid, total_seconds

    @torch.inference_mode()
    def extract_video_grid(self, video_path: str, clip_frames: int = 16):
        import cv2

        self._ensure_video_model()
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = max(1, int(np.ceil(frame_count / fps)))
        sampled_frames = []
        stride = max(1, int(round(fps / 8.0)))
        index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if index % stride == 0:
                sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
        capture.release()

        grid = torch.zeros((total_seconds * 2, 2, 1408), dtype=torch.float32)
        for bin_index in range(len(grid)):
            center = int(bin_index * 0.5 * 8.0)
            start = max(0, center - clip_frames // 2)
            clip = sampled_frames[start:start + clip_frames]
            if not clip:
                continue
            while len(clip) < clip_frames:
                clip.append(clip[-1])
            inputs = self.video_proc(clip, return_tensors="pt")
            pixels = inputs.pixel_values_videos.to(
                self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            hidden_states = self.video_model(pixels, output_hidden_states=True).hidden_states
            middle = len(hidden_states) // 2
            three_quarters = (3 * len(hidden_states)) // 4
            group_1 = torch.stack(hidden_states[middle:three_quarters + 1]).mean(dim=(0, 2)).float().cpu()[0]
            group_2 = torch.stack(hidden_states[three_quarters:]).mean(dim=(0, 2)).float().cpu()[0]
            grid[bin_index] = torch.stack([group_1, group_2])
        return grid, total_seconds

    @torch.inference_mode()
    def scan(self, text=None, audio_path=None, video_path=None, wpm=250):
        start_time = time.time()
        grids, durations = {}, []
        words = starts = stops = sentences = []

        if text is not None:
            grids["text"], seconds, words, starts, stops, sentences = self.extract_text_grid(text, wpm)
            durations.append(seconds)
        if audio_path:
            a_grid, a_sec = self.extract_audio_grid(audio_path)
            grids["audio"] = a_grid
            durations.append(a_sec)
            # Free Wav2Vec2-BERT VRAM before loading V-JEPA 2
            del self.audio_model
            self.audio_model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

        if video_path:
            v_grid, v_sec = self.extract_video_grid(video_path)
            grids["video"] = v_grid
            durations.append(v_sec)
            # Free V-JEPA 2 VRAM before running TRIBE v2 brain model
            del self.video_model
            self.video_model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
        if not grids:
            raise ValueError("Provide at least one of text, audio_path, or video_path.")

        total_seconds = max(durations)
        target_bins = ((total_seconds * 2 + 199) // 200) * 200
        windows = target_bins // 200
        batch_dict = {}
        for modality, grid in grids.items():
            if grid.shape[0] < target_bins:
                padding = torch.zeros((target_bins - grid.shape[0], grid.shape[1], grid.shape[2]))
                grid = torch.cat([grid, padding], dim=0)
            else:
                grid = grid[:target_bins]
            batch_dict[modality] = grid.view(windows, 200, grid.shape[1], grid.shape[2]).permute(0, 2, 3, 1).to(self.device)

        self._hooked_latents.clear()
        predictions = []
        for begin in range(0, windows, 8):
            end = min(begin + 8, windows)
            batch = SimpleBatch(data={key: value[begin:end] for key, value in batch_dict.items()})
            prediction = self.brain_model(batch).permute(0, 2, 1).reshape(-1, 20484)
            predictions.append(prediction)

        vertices = torch.cat(predictions, dim=0)[:total_seconds]
        if self._hooked_latents:
            latents = torch.cat(self._hooked_latents, dim=0)[:total_seconds].numpy()
        else:
            latents = np.empty((total_seconds, 0), dtype=np.float32)

        region_scores = (vertices @ self.atlas_matrix).cpu().numpy()
        active_words = [""] * total_seconds
        active_sentences = [""] * total_seconds
        for second in range(total_seconds):
            active = [word for word, start, stop in zip(words, starts, stops) if start < second + 1 and stop >= second]
            matching = [sentence for sentence, start, stop in zip(sentences, starts, stops) if start < second + 1 and stop >= second]
            active_words[second] = " ".join(active)
            active_sentences[second] = matching[0] if matching else ""

        frame = pd.DataFrame(region_scores, columns=self.atlas_names)
        frame.insert(0, "sentence", active_sentences)
        frame.insert(0, "words", active_words)
        frame.insert(0, "second", np.arange(total_seconds))
        print(f"[BareEngine] Completed scan ({total_seconds}s) in {time.time() - start_time:.2f}s")
        return frame, vertices.cpu().numpy().astype(np.float16), latents


def main():
    parser = argparse.ArgumentParser(description="TRIBE v2 text, audio, and video scanner")
    parser.add_argument("--text", help="Path to a .txt file")
    parser.add_argument("--audio", help="Path to an audio file")
    parser.add_argument("--video", help="Path to an .mp4 file")
    parser.add_argument("--prefix", help="Output filename prefix")
    parser.add_argument("--wpm", type=int, default=250)
    args = parser.parse_args()

    primary = args.video or args.audio or args.text
    if not primary:
        parser.error("Provide at least --text, --audio, or --video.")

    text = Path(args.text).read_text(encoding="utf-8") if args.text else None
    engine = BareTribeEngine()
    frame, vertices, latents = engine.scan(text, args.audio, args.video, args.wpm)
    prefix = args.prefix or Path(primary).stem
    frame.to_parquet(f"{prefix}_360.parquet", index=False)
    np.save(f"{prefix}_vertices_fp16.npy", vertices)
    np.save(f"{prefix}_latent_1hz.npy", latents.astype(np.float16))
    print(f"Saved outputs with prefix: {prefix}")


if __name__ == "__main__":
    main()
