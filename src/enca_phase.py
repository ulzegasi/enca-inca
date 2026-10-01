"""Shared opt-in circular phase supervision for ENCAfft2CNN and ENCAfft4CNN."""

import math

import tensorflow as tf


def parse_infer_phase(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError("INFER_PHASE must be true or false (also accepts 1/0 and yes/no).")


def phase_configuration(model, infer_phase, phi_lims=(0.0, 2.0 * math.pi)):
    if model not in {"original", "jupiter"}:
        raise ValueError(f"Unknown SDDE model {model!r}.")
    if not isinstance(infer_phase, bool):
        raise ValueError("infer_phase must be a boolean.")
    if infer_phase and model != "jupiter":
        raise ValueError("INFER_PHASE=true requires MODEL=jupiter.")
    phi_lims = tuple(float(value) for value in phi_lims)
    if len(phi_lims) != 2 or not all(math.isfinite(v) for v in phi_lims) or phi_lims[0] > phi_lims[1]:
        raise ValueError("phi_lims must be a finite ordered pair in radians.")
    linear_names = ["tau", "T", "Nd", "sigma", "Bmax"]
    if model == "jupiter":
        linear_names.append("Aj")
    return {
        "infer_phase": infer_phase,
        "num_linear_parameters": len(linear_names),
        "num_model_parameters": len(linear_names) + int(infer_phase),
        "num_supervised_parameters": len(linear_names) + 2 * int(infer_phase),
        "parameter_names": linear_names + (["phi"] if infer_phase else []),
        "supervised_names": linear_names + (["sin_phi", "cos_phi"] if infer_phase else []),
        "phase_encoding": "sin_cos" if infer_phase else "none",
        "phase_prior": list(phi_lims) if infer_phase else None,
        "phase_reference": "simulator_t0_before_warmup" if infer_phase else None,
        "phase_loss": "range_normalized_pair_mse" if infer_phase else None,
    }


def validate_phase_checkpoint(saved_args, current_args=None):
    """Old metadata means phase marginalized, never permission to add phase."""
    inferred = getattr(saved_args, "infer_phase", False)
    # Earlier phase checkpoints saved phase_prior but had no phi_lims field.
    saved_limits = getattr(saved_args, "phi_lims", None)
    if saved_limits is None:
        saved_limits = getattr(saved_args, "phase_prior", None) or (0.0, 2.0 * math.pi)
    expected = phase_configuration(getattr(saved_args, "model", "original"), inferred, saved_limits)
    if current_args is not None and inferred != current_args.infer_phase:
        raise ValueError("Cannot change INFER_PHASE when resuming; use a fresh run directory.")
    if current_args is not None and tuple(saved_limits) != tuple(getattr(current_args, "phi_lims", (0.0, 2.0 * math.pi))):
        raise ValueError("Cannot change phi_lims when resuming; use a fresh run directory.")
    for key, value in expected.items():
        if not hasattr(saved_args, key):
            if inferred:
                raise ValueError(f"Phase checkpoint is missing {key}; use a fresh run directory.")
            continue
        if getattr(saved_args, key) != value:
            raise ValueError(f"Incompatible phase checkpoint metadata: {key}.")
    if getattr(saved_args, "ndims_latent", 0) < expected["num_supervised_parameters"]:
        raise ValueError("Checkpoint latent width is smaller than its supervised target width.")
    return expected


def supervised_targets(params, infer_phase):
    """Keep physical theta separate from the eight-coordinate training target."""
    params = tf.convert_to_tensor(params, dtype=tf.float32)
    if not infer_phase:
        return params
    tf.debugging.assert_equal(tf.shape(params)[-1], 7)
    phi = params[..., 6:7]
    return tf.concat([params[..., :6], tf.sin(phi), tf.cos(phi)], axis=-1)


def phase_pair_mse(params, latent):
    """One parameter contribution: mean MSE of sin/cos, each scaled by range 2.

    Do not divide by the target value (which may be zero), or force predictions
    onto the unit circle: near-zero vectors can reflect an uncertain phase.
    """
    targets = supervised_targets(params, True)[..., 6:8]
    return tf.reduce_mean(tf.square((latent[..., 6:8] - targets) / 2.0))


def wrapped_phase_error(params, latent):
    """Signed error in [-pi, pi]; report vector norm alongside angle metrics."""
    predicted = tf.atan2(latent[..., 6], latent[..., 7])
    delta = predicted - params[..., 6]
    return tf.atan2(tf.sin(delta), tf.cos(delta))
