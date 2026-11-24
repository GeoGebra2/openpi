"""Stream annotated current-vs-actual-policy camera videos and their time provenance."""

from __future__ import annotations

import json
import logging
import pathlib

import numpy as np


CAMERA_KEYS = ("observation/image", "observation/wrist_image")
CAMERA_NAMES = ("agentview", "wrist")


def comparison_frame(*, current_images, policy_images, step, query_step, source_steps,
                     control_hz, delay_target, query, terminal=None):
    """Annotate copies only; the lower row is the last actual server input."""
    from PIL import Image, ImageDraw

    size = max(224, *(image.shape[1] for image in current_images.values()))
    header, label, gap = 66, 56, 4
    width, height = 2 * size + gap, header + 2 * (size + label) + gap
    canvas = Image.new("RGB", (width, height), (18, 27, 36))
    draw = ImageDraw.Draw(canvas)
    target = CAMERA_NAMES[CAMERA_KEYS.index(delay_target)] if delay_target in CAMERA_KEYS else "none (clean)"
    draw.text((8, 5), f"step={step}  sim t={step / control_hz:.3f}s  delayed camera: {target}", fill="white")
    status = f"QUERY at step {query_step}" if query else f"HOLD input from query step {query_step}"
    if terminal:
        status = f"END: {terminal}  |  last query step {query_step}"
    draw.text((8, 25), status, fill=(255, 208, 112))
    draw.text((8, 45), "Top: live. Bottom: actual input. Simulation time only.", fill="white")
    for column, (key, name) in enumerate(zip(CAMERA_KEYS, CAMERA_NAMES)):
        x = column * (size + gap)
        for row, images in enumerate((current_images, policy_images)):
            y = header + row * (label + size + gap)
            draw.text((x + 6, y + 4), f"{'LIVE' if row == 0 else 'POLICY'} {name}", fill="white")
            if row == 0:
                detail = f"source={step / control_hz:.3f}s  age=0ms"
                extra = "self state: current at each query"
            else:
                source = source_steps[column]
                age = (step - source) * 1000 / control_hz
                at_query = (query_step - source) * 1000 / control_hz
                detail = f"source={source / control_hz:.3f}s  age(now)={age:g}ms"
                extra = f"query age={at_query:g}ms  {'DELAYED' if key == delay_target else 'current'}"
            draw.text((x + 6, y + 21), detail, fill=(255, 208, 112) if row else (168, 225, 230))
            draw.text((x + 6, y + 38), extra, fill="white")
            panel = Image.fromarray(np.asarray(images[key], dtype=np.uint8))
            canvas.paste(panel.resize((size, size)), (x, y + label))
    # H.264 requires even dimensions. Padding does not change the camera pixels.
    if canvas.width % 2 or canvas.height % 2:
        padded = Image.new("RGB", (canvas.width + canvas.width % 2, canvas.height + canvas.height % 2))
        padded.paste(canvas, (0, 0))
        canvas = padded
    return np.asarray(canvas)


class ComparisonVideoRecorder:
    """One encoder per episode/worker; never buffer an entire rollout in RAM."""

    def __init__(self, output_dir, slug, fps, control_hz, delay_target):
        self.output_dir, self.slug = pathlib.Path(output_dir), slug
        self.fps, self.control_hz, self.delay_target = fps, control_hz, delay_target
        if not 0 < fps <= control_hz:
            raise ValueError("Comparison video fps must be positive and no greater than control_hz")
        self.path = self.output_dir / f"{slug}_recording.mp4"
        self.timing_path = self.output_dir / f"{slug}_video_timing.jsonl"
        self.writer, self.timing = None, None
        self.frames, self.error = 0, None
        self.last_step, self.query_step = None, None
        self.policy_images, self.source_steps, self.state_source_step = None, None, None

    def _fail(self, error):
        self.error = repr(error) if self.error is None else self.error
        logging.warning("Video recording failed for %s: %s", self.slug, error)
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                logging.exception("Closing failed video encoder")
            self.writer = None

    def capture(self, *, current_images, policy_obs, step, query, delay, terminal=None):
        """Record before an action, plus the terminal observation after the last action."""
        if query:
            self.policy_images = {key: np.array(policy_obs[key], copy=True) for key in CAMERA_KEYS}
            self.query_step = step
            self.source_steps = [step - delay if self.delay_target in (key, "__all__") else step
                                 for key in CAMERA_KEYS]
            self.state_source_step = step - delay if self.delay_target in ("state", "__all__") else step
        if self.policy_images is None:
            return
        # Each retained frame keeps its simulation time at the requested playback rate.
        slot = int(np.floor(step * self.fps / self.control_hz + 1e-8))
        previous_slot = int(np.floor((step - 1) * self.fps / self.control_hz + 1e-8))
        encode = self.last_step != step and (step == 0 or slot > previous_slot or bool(terminal))
        entry = {"step": step, "time_s": step / self.control_hz, "policy_query": bool(query),
                 "query_step": self.query_step, "image_source_steps": dict(zip(CAMERA_KEYS, self.source_steps)),
                 "image_age_ms": dict(zip(CAMERA_KEYS, [(step - t) * 1000 / self.control_hz
                                                        for t in self.source_steps])),
                 "image_age_at_query_ms": dict(zip(CAMERA_KEYS, [(self.query_step - t) * 1000 / self.control_hz
                                                                 for t in self.source_steps])),
                 "state_source_step": self.state_source_step, "scheduled_delay_steps": delay,
                 "terminal": terminal, "video_frame_index": None}
        try:
            if encode and self.error is None:
                import imageio.v2 as imageio

                if self.writer is None:
                    self.writer = imageio.get_writer(str(self.path), format="FFMPEG", mode="I", fps=self.fps,
                                                    codec="libx264", macro_block_size=1, output_params=["-threads", "1"])
                self.writer.append_data(comparison_frame(
                    current_images=current_images, policy_images=self.policy_images, step=step,
                    query_step=self.query_step, source_steps=self.source_steps, control_hz=self.control_hz,
                    delay_target=self.delay_target, query=query, terminal=terminal))
                entry["video_frame_index"] = self.frames
                self.frames += 1
        except Exception as error:
            self._fail(error)
        try:
            if self.timing is None:
                self.timing = self.timing_path.open("w", encoding="utf-8", buffering=1)
            self.timing.write(json.dumps(entry, allow_nan=False) + "\n")
        except Exception as error:
            self._fail(error)
        self.last_step = step

    def finish(self, outcome):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as error:
                self._fail(error)
            finally:
                self.writer = None
        if self.timing is not None:
            try:
                self.timing.close()
            except Exception as error:
                self._fail(error)
        video = None
        if self.frames and self.error is None:
            destination = self.output_dir / f"{self.slug}_{outcome}.mp4"
            try:
                self.path.replace(destination)
                video = destination.name
            except Exception as error:
                self._fail(error)
        return {"video_path": video, "video_fps": self.fps, "video_frames": self.frames,
                "video_error": self.error, "video_layout": "current_vs_actual_policy_v1",
                "video_timing_path": self.timing_path.name if self.timing is not None else None,
                "video_timebase": "simulation seconds; terminal observation adds at most one frame"}
