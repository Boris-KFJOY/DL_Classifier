# utils/exp.py
from types import SimpleNamespace
import json

def _get(a, k, default=None):
    return (a.get(k, default) if isinstance(a, dict) else getattr(a, k, default))

def build_run_tag(a, seed=None):
    tags = []
    # --- SubBand ---
    use_sb = _get(a, "use_subband", True)
    if not use_sb:
        tags.append("NoSB")
    else:
        sbp = _get(a, "subband_proj", "linear")
        if sbp != "linear":
            tags.append(f"SBProj={sbp}")

    # --- CrossView ---
    if not _get(a, "use_crossview", True):
        tags.append("NoCV")
    elif _get(a, "crossview_alpha", 1.0) != 1.0:
        tags.append(f"CVa{_get(a,'crossview_alpha')}")

    # --- Harmonic PE ---
    if not _get(a, "use_harmonic_pe", True):
        tags.append("NoHPE")
    elif _get(a, "harmonic_beta", 1.0) != 1.0:
        tags.append(f"HPEb{_get(a,'harmonic_beta')}")

    # --- PE mode (非默认) ---
    if _get(a, "pe_mode", "harmonic") != "harmonic":
        tags.append(f"PE={_get(a,'pe_mode')}")

    # --- Seed (可选) ---
    if seed is not None:
        tags.append(f"seed={seed}")

    return "_".join(tags) or "baseline"

def build_model_name(base_name, a, seed=None):
    tag = build_run_tag(a, seed=seed)
    return base_name if tag == "baseline" else f"{base_name}_{tag}"

def annotate_meta(meta: dict, a, seed=None):
    meta = dict(meta)
    meta["ablation_tag"]  = build_run_tag(a, seed=seed)
    meta["ablation_json"] = json.dumps(a if isinstance(a, dict) else vars(a), ensure_ascii=False)
    if seed is not None:
        meta["seed"] = seed
    return meta
