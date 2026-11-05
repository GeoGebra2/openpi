import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def patch_wrist_kv(cache, donor_cache, layer_mask, token_mask):
    """Replace K and V at selected [layer, batch, token] slots; preserve all other slots."""
    if layer_mask.shape != (cache[0].shape[0],) or token_mask.shape != cache[0].shape[1:3]:
        raise ValueError("KV patch mask dimensions do not match the cache")
    selected = layer_mask[:, None, None, None, None] & token_mask[None, :, :, None, None]
    if any(original.shape != donor.shape for original, donor in zip(cache, donor_cache, strict=True)):
        raise ValueError("Donor and recipient KV cache shapes differ")
    return tuple(jnp.where(selected, donor, original) for original, donor in zip(cache, donor_cache, strict=True))


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.num_kv_layers = action_expert_config.depth
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        patch_reference: _model.Observation | None = None,
        patch_layer_mask: at.Bool[at.Array, " l"] | None = None,
        patch_site: str = "wrist_kv",
        patch_camera: str = "left_wrist_0_rgb",
        patch_token_mask: at.Bool[at.Array, " p"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if patch_reference is not None:
            if patch_layer_mask is None:
                raise ValueError("Wrist KV patching requires an explicit layer mask")
            reference = _model.preprocess_observation(None, patch_reference, train=False)
            donor_tokens, donor_mask, donor_ar = self.embed_prefix(reference)
            donor_result = self.PaliGemma.llm(
                [donor_tokens, None], mask=make_attn_mask(donor_mask, donor_ar),
                positions=jnp.cumsum(donor_mask, axis=1) - 1,
                capture_layer_tokens=patch_site == "prefix_hidden",
            )
            donor_cache = donor_result[1]
            # All cameras have equal preprocessed resolution and share the encoder.
            # Prefix order is image tokens followed by language tokens. A local
            # token mask is expressed in the selected camera's spatial token
            # coordinates; the full prefix mask is built below.
            prompt_len = observation.tokenized_prompt.shape[1] if observation.tokenized_prompt is not None else 0
            image_len = prefix_tokens.shape[1] - prompt_len
            names = list(observation.images)
            if not names or image_len % len(names) or patch_camera not in names:
                raise ValueError(f"Cannot identify prefix slots for camera {patch_camera!r}")
            per_image = image_len // len(names)
            start = names.index(patch_camera) * per_image
            indices = jnp.arange(prefix_tokens.shape[1])
            camera_mask = (indices >= start) & (indices < start + per_image)
            if patch_token_mask is None:
                local_mask = jnp.ones((per_image,), dtype=jnp.bool_)
            else:
                if patch_token_mask.ndim != 1 or patch_token_mask.shape != (per_image,):
                    raise ValueError(
                        f"patch_token_mask must have shape {(per_image,)}, got {patch_token_mask.shape}"
                    )
                local_mask = patch_token_mask.astype(jnp.bool_)
            local_indices = jnp.clip(indices - start, 0, per_image - 1)
            token_mask = camera_mask & local_mask[local_indices]
            token_mask = token_mask[None, :] & prefix_mask & donor_mask
            if patch_site == "wrist_kv":
                kv_cache = patch_wrist_kv(kv_cache, donor_cache, patch_layer_mask, token_mask)
            elif patch_site == "siglip":
                if patch_layer_mask.shape != (1,):
                    raise ValueError("SigLIP output patch has one location (index 0)")
                # The prefix image slice is the projected SigLIP output, before
                # any cross-camera/language fusion in the prefix Transformer.
                patched_tokens = jnp.where(
                    (token_mask & patch_layer_mask[0])[..., None], donor_tokens, prefix_tokens
                )
                _, kv_cache = self.PaliGemma.llm(
                    [patched_tokens, None], mask=prefix_attn_mask, positions=positions,
                )
            elif patch_site == "prefix_hidden":
                if patch_layer_mask.shape != (self.num_kv_layers,):
                    raise ValueError("Prefix hidden patch layer count mismatch")
                selected = patch_layer_mask[:, None, None] & token_mask[None, :, :]
                _, kv_cache = self.PaliGemma.llm(
                    [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
                    hidden_patch=[(donor_result[2][0], selected), None],
                )
            elif patch_site == "action_hidden":
                if patch_token_mask is not None:
                    raise ValueError("Local image token masks are not defined for action_hidden patching")
                if patch_layer_mask.shape != (self.num_kv_layers,):
                    raise ValueError("Action hidden patch layer count mismatch")
            else:
                raise ValueError(f"Unknown activation patch site: {patch_site}")
        elif patch_layer_mask is not None:
            raise ValueError("A patch layer mask requires a donor observation")

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            hidden_patch = None
            if patch_reference is not None and patch_site == "action_hidden":
                # Both suffix forwards see exactly the recipient's x_t, t, state
                # and suffix mask. Only the donor's prefix cache uses current wrist.
                _, _, donor_layers = self.PaliGemma.llm(
                    [None, suffix_tokens], mask=full_attn_mask, positions=positions,
                    kv_cache=donor_cache, adarms_cond=[None, adarms_cond], capture_layer_tokens=True,
                )
                action_mask = (
                    jnp.arange(suffix_mask.shape[1]) >= suffix_mask.shape[1] - self.action_horizon
                )[None, :] & suffix_mask
                selected = patch_layer_mask[:, None, None] & action_mask[None, :, :]
                hidden_patch = [None, (donor_layers[1], selected)]

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                hidden_patch=hidden_patch,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @at.typecheck
    def sample_actions_with_diagnostics(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        capture_attention: bool = False,  # noqa: FBT002
    ) -> tuple[_model.Actions, dict[str, Any]]:
        """Variant of `sample_actions` that also returns diagnostics for analysis.

        Replicates `sample_actions` exactly (same preprocessing, KV-cache prefix pass, and Euler
        denoising) but additionally captures:
          - per-camera SigLIP per-block tokens and validity masks,
          - Gemma prefix per-layer tokens, final tokens and group masks,
          - action-expert per-layer tokens at every denoise step,
          - the flow vector field v_t at every denoise step.

        When ``capture_attention`` is true, also records compact action-query attention
        masses grouped by image source, language, and suffix keys.

        All diagnostic arrays carry the leading batch dimension. The denoise loop is unrolled in
        Python (`num_steps` must be static); this method is intended for diagnostics serving only.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        diagnostics: dict[str, Any] = {}

        # Prefix embedding, inlined from `embed_prefix` so per-camera SigLIP block outputs
        # (already returned by the vision encoder, discarded by `embed_prefix`) can be captured.
        input_mask = []
        ar_mask = []
        tokens = []
        siglip_diag: dict[str, Any] = {}
        prefix_groups = []
        for name in observation.images:
            image_tokens, img_out = self.PaliGemma.img(observation.images[name], train=False)
            encoder_out = img_out["encoder"]
            block_keys = sorted(key for key in encoder_out if key.startswith("block"))
            siglip_diag[name] = {
                # [b, layer, token, width]; retain spatial variation until pairwise comparison.
                "block_tokens": jnp.stack([encoder_out[key]["+mlp"] for key in block_keys], axis=1),
                "mask": jnp.broadcast_to(observation.image_masks[name][:, None], image_tokens.shape[:2]),
            }
            prefix_groups.append(name)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    observation.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        if observation.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            prefix_groups.append("language")
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        prefix_tokens = jnp.concatenate(tokens, axis=1)
        prefix_mask = jnp.concatenate(input_mask, axis=1)
        prefix_ar_mask = jnp.array(ar_mask)
        diagnostics["siglip"] = siglip_diag

        # Group masks exclude invalid cameras and padded language tokens from all statistics.
        group_masks = {
            name: jnp.concatenate(
                [mask if i == index else jnp.zeros_like(mask) for i, mask in enumerate(input_mask)], axis=1
            )
            for index, name in enumerate(prefix_groups)
        }
        # Fill the KV cache with a forward pass of the prefix, capturing per-layer tokens.
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_out_list, kv_cache, prefix_layer_tokens = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions, capture_layer_tokens=True
        )
        prefix_out = prefix_out_list[0]
        diagnostics["gemma_prefix"] = {
            "layer_tokens": jnp.swapaxes(prefix_layer_tokens[0], 0, 1),
            "final_tokens": prefix_out,
            "mask": prefix_mask,
            "group_masks": group_masks,
        }

        action_expert_layer_tokens = []
        v_t_steps = []
        attention_source_mass_steps = []
        attention_first_action_steps = []
        # Prefix groups follow the model's deterministic image insertion order, then language.
        # The final group represents suffix keys (state and action tokens).
        prefix_group_masks = jnp.stack([group_masks[name] for name in prefix_groups], axis=1)
        x_t = noise
        time = jnp.float32(1.0)
        # Unrolled version of the `jax.lax.while_loop` in `sample_actions`: exactly `num_steps`
        # iterations starting at t=1.0 (the while-loop's `time >= -dt/2` condition), with f32
        # time accumulation to reproduce its numerics.
        for _ in range(num_steps):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # how the suffix tokens can attend to the prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            suffix_result = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                capture_layer_tokens=True,
                capture_attention=capture_attention,
            )
            if capture_attention:
                (prefix_out, suffix_out), _, suffix_layer_tokens, suffix_attention = suffix_result
            else:
                (prefix_out, suffix_out), _, suffix_layer_tokens = suffix_result
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            action_expert_layer_tokens.append(jnp.swapaxes(suffix_layer_tokens[1], 0, 1))
            v_t_steps.append(v_t)

            # Attention is [layer, batch, kv_head, query_head_per_kv, query, key].
            # Aggregate keys by their source before returning diagnostics; retaining the
            # complete matrix would make every websocket response unnecessarily large.
            if capture_attention:
                suffix_len = suffix_tokens.shape[1]
                action_start = suffix_len - self.action_horizon
                # Prefix groups cover only prefix keys. Add zero columns for suffix keys,
                # then append one explicit group for state/action suffix keys.
                prefix_key_masks = jnp.concatenate(
                    [
                        prefix_group_masks,
                        jnp.zeros(
                            (batch_size, len(prefix_groups), suffix_len), dtype=prefix_group_masks.dtype
                        ),
                    ],
                    axis=-1,
                )
                suffix_key_masks = jnp.concatenate(
                    [
                        jnp.zeros((batch_size, 1, prefix_tokens.shape[1]), dtype=prefix_group_masks.dtype),
                        suffix_mask[:, None, :],
                    ],
                    axis=-1,
                )
                key_group_masks = jnp.concatenate([prefix_key_masks, suffix_key_masks], axis=1)
                source_mass = jnp.einsum(
                    "LBKGTS,BQS->LBKGTQ",
                    suffix_attention.astype(jnp.float32),
                    key_group_masks.astype(jnp.float32),
                )
                attention_source_mass_steps.append(jnp.mean(source_mass[..., action_start:, :], axis=-2))
                attention_first_action_steps.append(source_mass[..., action_start, :])

            x_t = x_t + dt * v_t
            time = time + dt

        diagnostics["action_expert"] = {
            # [b, num_steps, layer, token, width]
            "layer_tokens": jnp.stack(action_expert_layer_tokens, axis=1),
            "mask": suffix_mask,
            "action_mask": jnp.broadcast_to(
                jnp.arange(suffix_mask.shape[1]) >= suffix_mask.shape[1] - self.action_horizon, suffix_mask.shape
            ),
        }
        diagnostics["flow"] = {
            # [b, num_steps, action_horizon, action_dim]
            "v_t": jnp.stack(v_t_steps, axis=1),
        }
        if capture_attention:
            diagnostics["attention"] = {
                # [b, denoise_step, layer, kv_head, query_head, source_group]
                "source_mass": jnp.transpose(
                    jnp.stack(attention_source_mass_steps, axis=1), (2, 1, 0, 3, 4, 5)
                ),
                "first_action_source_mass": jnp.transpose(
                    jnp.stack(attention_first_action_steps, axis=1), (2, 1, 0, 3, 4, 5)
                ),
                # [b, source_group]; useful for checking that padded cameras/language are excluded.
                "source_token_counts": jnp.sum(key_group_masks, axis=-1),
            }
            # Only expose compact attention on this path. JIT can eliminate unused
            # activation outputs instead of materializing/transferring the drift tensors.
            return x_t, {"attention": diagnostics["attention"]}
        return x_t, diagnostics
