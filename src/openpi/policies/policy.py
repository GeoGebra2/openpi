from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import activation_drift
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        diagnostics: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            diagnostics: Whether to run the model's diagnostics forward pass
                         (`sample_actions_with_diagnostics`) and include per-layer activations
                         in the infer output under the "diagnostics" key. JAX models only.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = dict(metadata or {})
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._diagnostics = diagnostics

        if self._is_pytorch_model:
            if diagnostics:
                raise ValueError("diagnostics=True is only supported for JAX models, not PyTorch models.")
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            if diagnostics and not hasattr(model, "sample_actions_with_diagnostics"):
                raise ValueError(
                    f"diagnostics=True requires the model to implement sample_actions_with_diagnostics, "
                    f"but {type(model).__name__} does not."
                )
            if diagnostics:
                # num_steps must be static: the diagnostics forward unrolls the denoise loop.
                self._metadata["activation_drift"] = {
                    "schema_version": activation_drift.SCHEMA_VERSION,
                    "noise_shape": [model.action_horizon, model.action_dim],
                }
                self._metadata["attention"] = {
                    "schema_version": 1,
                    "source_groups": [
                        "base_0_rgb",
                        "left_wrist_0_rgb",
                        "right_wrist_0_rgb",
                        "language",
                        "suffix",
                    ],
                    "query_groups": ["all_action", "first_action"],
                    "noise_shape": [model.action_horizon, model.action_dim],
                }
                self._ordinary_sample_actions = nnx_utils.module_jit(model.sample_actions)
                self._sample_actions = nnx_utils.module_jit(
                    model.sample_actions_with_diagnostics, static_argnames=("num_steps", "capture_attention")
                )
            else:
                self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            if hasattr(model, "num_kv_layers"):
                self._wrist_sample_actions = nnx_utils.module_jit(
                    model.sample_actions, static_argnames=("num_steps", "patch_site", "patch_camera")
                )
                self._metadata["wrist_kv_patch"] = {
                    "schema_version": 1, "num_layers": model.num_kv_layers,
                    "noise_shape": [model.action_horizon, model.action_dim],
                    "source": "left_wrist_0_rgb", "components": ["K", "V"],
                    "denoise_scope": "all_steps", "sample_kwargs": dict(self._sample_kwargs),
                    "sites": ["wrist_kv", "siglip", "prefix_hidden", "action_hidden"],
                }
                self._metadata["token_patch"] = {
                    "schema_version": 1,
                    "num_layers": model.num_kv_layers,
                    "noise_shape": [model.action_horizon, model.action_dim],
                    "cameras": ["base_0_rgb", "left_wrist_0_rgb"],
                    "sites": ["siglip", "wrist_kv", "prefix_hidden"],
                    "mask": "one-dimensional spatial token mask for the selected camera",
                    "tokens_per_camera": 256,
                }
                try:
                    fake_inputs, _ = model.inputs_spec(batch_size=1)
                    fake_image = next(iter(fake_inputs.images.values()))
                    token_count = int(model.PaliGemma.img(fake_image, train=False)[0].shape[1])
                    self._metadata["token_patch"]["tokens_per_camera"] = token_count
                except (AttributeError, StopIteration, TypeError, ValueError):
                    # The runtime endpoint still validates the mask shape from
                    # the actual transformed observation if this optional hint
                    # cannot be computed during policy construction.
                    pass
                self._metadata["token_drift"] = {
                    "schema_version": 1,
                    "noise_shape": [model.action_horizon, model.action_dim],
                    "cameras": ["base_0_rgb", "left_wrist_0_rgb"],
                    "outputs": ["token_cosine_distance", "token_relative_l2", "actions", "reference_actions"],
                }

    @override
    def infer(  # type: ignore[misc]
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        check_consistency: bool = False,
        consistency_atol: float = 1e-4,
        consistency_rtol: float = 1e-4,
        capture_attention: bool = False,
        include_diagnostics: bool = True,
    ) -> dict:
        if capture_attention and not self._diagnostics:
            raise ValueError("Attention capture requires --diagnostics")
        if check_consistency and (not self._diagnostics or not include_diagnostics or noise is None):
            raise ValueError("Consistency checks require diagnostics and explicit shared noise")
        if capture_attention:
            include_diagnostics = True
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if not self._is_pytorch_model:
                expected = (self._model.action_horizon, self._model.action_dim)
                if noise.shape not in (expected, (1, *expected)) or not np.all(np.isfinite(noise)):
                    raise ValueError(
                        f"Expected finite noise with shape {expected} or {(1, *expected)}, got {noise.shape}"
                    )
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        model_diagnostics = None
        run_diagnostics = self._diagnostics and include_diagnostics
        if run_diagnostics:
            actions, model_diagnostics = self._sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                capture_attention=capture_attention,
                **sample_kwargs,
            )
        elif self._diagnostics:
            actions = self._ordinary_sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        consistency = None
        if check_consistency:
            ordinary_actions = self._ordinary_sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
            consistency = activation_drift.compare_actions(
                actions, ordinary_actions, atol=consistency_atol, rtol=consistency_rtol
            )
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        if model_diagnostics is not None:
            # Attach after the output transforms: some of them (e.g. LiberoOutputs) rebuild the
            # output dict with only the "actions" key and would drop the diagnostics.
            # bfloat16 activations aren't supported by the NumPy MessagePack codec.
            converted_diagnostics = jax.tree.map(
                lambda x: np.asarray(x[0, ...], dtype=bool if x.dtype == jnp.bool_ else np.float32), model_diagnostics
            )
            if capture_attention:
                outputs["attention"] = converted_diagnostics["attention"]
                outputs["attention_source_groups"] = self._metadata["attention"]["source_groups"]
            else:
                outputs["diagnostics"] = converted_diagnostics
        if consistency is not None:
            outputs["consistency"] = consistency
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_action(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        """Run an action-only query, suppressing diagnostic activations when enabled."""
        return self.infer(obs, noise=noise, include_diagnostics=False)

    def infer_wrist_patch(
        self, observation: dict, current_wrist_image, noise, mode: str, layers: list[int], site: str = "wrist_kv"
    ) -> dict:
        """Backward-compatible wrapper for patching all wrist image tokens."""
        outputs = self.infer_token_patch(
            observation,
            current_wrist_image,
            noise,
            mode=mode,
            layers=layers,
            site=site,
            camera="left_wrist_0_rgb",
            token_mask=None,
        )
        outputs["wrist_kv_patch"] = {"schema_version": 1, "mode": mode, "layers": layers, "site": site}
        outputs.pop("token_patch", None)
        return outputs

    def infer_token_patch(
        self,
        observation: dict,
        donor_image,
        noise,
        *,
        mode: str,
        layers: list[int],
        site: str = "siglip",
        camera: str = "left_wrist_0_rgb",
        token_mask: np.ndarray | None = None,
    ) -> dict:
        """Run a local image-token counterfactual without returning diagnostics.

        ``observation`` is the delayed recipient input. ``donor_image`` is the
        current image for the selected camera. ``token_mask`` is a boolean mask
        over that camera's spatial SigLIP tokens. The physical rollout caller
        executes only the returned recipient actions.
        """
        if not hasattr(self, "_wrist_sample_actions"):
            raise ValueError("Token patching requires a JAX Pi0 model")
        if mode not in ("none", "self", "clean"):
            raise ValueError("mode must be none, self, or clean")
        if site not in ("wrist_kv", "siglip", "prefix_hidden", "action_hidden"):
            raise ValueError("unknown activation patch site")
        depth = self._model.num_kv_layers
        if (
            not isinstance(layers, list)
            or any(type(layer) is not int or not 0 <= layer < depth for layer in layers)
            or len(layers) != len(set(layers))
            or (mode == "none" and layers)
            or (mode != "none" and not layers)
            or (site == "siglip" and mode != "none" and layers != [0])
        ):
            raise ValueError("Invalid patch layers for this mode/model")
        input_key = {
            "left_wrist_0_rgb": "observation/wrist_image",
            "base_0_rgb": "observation/image",
        }.get(camera)
        if input_key is None or input_key not in observation:
            raise ValueError(f"Token patch endpoint does not support camera {camera!r}")
        noise = np.asarray(noise, dtype=np.float32)
        expected = (self._model.action_horizon, self._model.action_dim)
        if noise.shape != expected or not np.all(np.isfinite(noise)):
            raise ValueError(f"Expected finite shared noise with shape {expected}")
        donor_image = np.asarray(donor_image)
        if donor_image.shape != np.asarray(observation[input_key]).shape:
            raise ValueError("Donor and delayed camera image shapes differ")
        if not np.all(np.isfinite(donor_image)):
            raise ValueError("Donor camera image must be finite")
        if token_mask is not None:
            token_mask = np.asarray(token_mask, dtype=bool)
            if token_mask.ndim != 1 or not np.any(token_mask):
                raise ValueError("token_mask must be a non-empty one-dimensional boolean array")

        def prepare(raw):
            copied = jax.tree.map(lambda x: np.array(x, copy=True) if isinstance(x, np.ndarray) else x, raw)
            return self._input_transform(copied)

        inputs = prepare(observation)
        donor_raw = dict(observation)
        if mode != "self":
            donor_raw[input_key] = donor_image
        donor_inputs = prepare(donor_raw)
        # Enforce identical layout/positions and all non-selected-camera inputs.
        recipient_rest = {**inputs, "image": {k: v for k, v in inputs["image"].items() if k != camera}}
        donor_rest = {
            **donor_inputs,
            "image": {k: v for k, v in donor_inputs["image"].items() if k != camera},
        }
        if jax.tree.structure(recipient_rest) != jax.tree.structure(donor_rest) or any(
            not np.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(recipient_rest), jax.tree.leaves(donor_rest), strict=True)
        ):
            raise ValueError("Donor transformations changed non-wrist inputs or token masks")
        if not np.all(inputs["image_mask"][camera]):
            raise ValueError(f"Selected camera {camera!r} must be valid for this experiment")
        def batch(tree):
            return jax.tree.map(lambda x: jnp.asarray(x)[None, ...], tree)

        observation_model = _model.Observation.from_dict(batch(inputs))
        donor_model = _model.Observation.from_dict(batch(donor_inputs))
        patch_depth = 1 if site == "siglip" else depth
        layer_mask = np.zeros(patch_depth, dtype=bool)
        layer_mask[layers] = True
        self._rng, rng = jax.random.split(self._rng)
        started = time.monotonic()
        actions = self._wrist_sample_actions(
            rng, observation_model, **{**self._sample_kwargs, "noise": jnp.asarray(noise)[None, ...]},
            patch_reference=donor_model, patch_layer_mask=jnp.asarray(layer_mask), patch_site=site,
            patch_camera=camera,
            patch_token_mask=None if token_mask is None else jnp.asarray(token_mask),
        )
        outputs = self._output_transform({"state": np.asarray(inputs["state"]), "actions": np.asarray(actions[0])})
        patch_metadata = {
            "schema_version": 1,
            "mode": mode,
            "layers": layers,
            "site": site,
            "camera": camera,
            "token_masked": token_mask is not None,
            "masked_tokens": int(token_mask.sum()) if token_mask is not None else None,
        }
        outputs["wrist_kv_patch"] = patch_metadata
        outputs["token_patch"] = patch_metadata
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - started) * 1000}
        return outputs

    def infer_attention(self, observation: dict, *, noise: np.ndarray | None = None) -> dict:
        """Run one normal policy query and return compact attention diagnostics."""
        return self.infer(observation, noise=noise, capture_attention=True)

    def infer_token_drift(
        self,
        observation: dict,
        reference_observation: dict,
        noise: np.ndarray,
        *,
        camera: str = "left_wrist_0_rgb",
    ) -> dict:
        """Return spatial SigLIP drift for one camera and matched policy actions.

        The reference and recipient use identical noise and all non-selected
        inputs are supplied by the caller. Only compact per-layer/per-token
        distances are returned; full model activations stay on the server.
        """
        if not self._diagnostics:
            raise ValueError("Token drift requires --diagnostics")
        reference = self.infer(reference_observation, noise=noise, include_diagnostics=True)
        delayed = self.infer(observation, noise=noise, include_diagnostics=True)
        try:
            reference_tokens = np.asarray(reference["diagnostics"]["siglip"][camera]["block_tokens"], dtype=np.float32)
            delayed_tokens = np.asarray(delayed["diagnostics"]["siglip"][camera]["block_tokens"], dtype=np.float32)
        except KeyError as error:
            raise ValueError(f"Diagnostics do not contain SigLIP camera {camera!r}") from error
        if reference_tokens.shape != delayed_tokens.shape or reference_tokens.ndim != 3:
            raise ValueError(f"Unexpected token shapes: {reference_tokens.shape}, {delayed_tokens.shape}")
        delta = delayed_tokens - reference_tokens
        dot = np.sum(reference_tokens * delayed_tokens, axis=-1)
        denom = np.linalg.norm(reference_tokens, axis=-1) * np.linalg.norm(delayed_tokens, axis=-1)
        cosine = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 1e-8)
        relative_l2 = np.linalg.norm(delta, axis=-1) / np.maximum(np.linalg.norm(reference_tokens, axis=-1), 1e-8)
        result = {
            "schema_version": 1,
            "camera": camera,
            "token_cosine_distance": (1.0 - cosine).astype(np.float32),
            "token_relative_l2": relative_l2.astype(np.float32),
            "actions": delayed["actions"],
            "reference_actions": reference["actions"],
            "action_l2": float(np.linalg.norm(np.asarray(delayed["actions"]) - np.asarray(reference["actions"]))),
        }
        return result

    def infer_drift(
        self,
        observation: dict,
        reference_observation: dict,
        noise: np.ndarray,
        *,
        check_consistency: bool = False,
        consistency_atol: float = 1e-4,
        consistency_rtol: float = 1e-4,
    ) -> dict:
        """Compare delayed/current observations locally, using the exact same initial noise."""
        if not self._diagnostics:
            raise ValueError("Paired drift inference requires --diagnostics")
        if noise is None:
            raise ValueError("Paired drift inference requires explicit shared noise")
        if not np.isfinite(consistency_atol) or not np.isfinite(consistency_rtol):
            raise ValueError("Consistency tolerances must be finite")
        if consistency_atol < 0 or consistency_rtol < 0:
            raise ValueError("Consistency tolerances must be nonnegative")
        reference = self.infer(
            reference_observation,
            noise=noise,
            check_consistency=check_consistency,
            consistency_atol=consistency_atol,
            consistency_rtol=consistency_rtol,
        )
        checks = {}
        if check_consistency:
            checks["ordinary_vs_diagnostic"] = reference["consistency"]
            repeated = self.infer(reference_observation, noise=noise)
            checks["identical_input_actions"] = activation_drift.compare_actions(
                repeated["actions"], reference["actions"], atol=consistency_atol, rtol=consistency_rtol
            )
            leaves_a = jax.tree.leaves(reference["diagnostics"])
            leaves_b = jax.tree.leaves(repeated["diagnostics"])
            # Repeated calls run the same executable; deterministic diagnostics must match exactly.
            checks["identical_input_activations"] = {
                "passed": all(np.array_equal(a, b) for a, b in zip(leaves_a, leaves_b, strict=True))
            }
            del repeated, leaves_a, leaves_b
        delayed = self.infer(observation, noise=noise)
        drift = activation_drift.compute_drift(delayed.pop("diagnostics"), reference["diagnostics"])
        delayed["drift"] = drift
        delayed["schema_version"] = activation_drift.SCHEMA_VERSION
        delayed["checks"] = checks
        return delayed

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)
        return self._record(obs, results)

    def infer_drift(self, **kwargs) -> dict:
        results = self._policy.infer_drift(**kwargs)
        return self._record(kwargs, results)

    def _record(self, obs: dict, results: dict) -> dict:
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
