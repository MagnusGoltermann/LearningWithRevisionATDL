import math
from typing import Callable, Dict


def _clamp(value: float, low: float, high: float) -> float:
    # Support inverted ranges by clamping to the unordered bounds
    lo = min(low, high)
    hi = max(low, high)
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def get_threshold_scheduler(args, total_epochs: int) -> Callable[[int, Dict], float]:
    """
    Returns a callable scheduler: tau_t = scheduler(epoch_idx, state)
    state can contain keys like: 'val_loss_hist', 'grad_norm_hist', etc.
    """
    method = getattr(args, "threshold_method", "fixed")
    tau_min = float(getattr(args, "tau_min", 0.1))
    tau_max = float(getattr(args, "tau_max", 0.9))

    # Progress helper in [0,1]
    def progress(epoch_idx: int) -> float:
        if total_epochs <= 1:
            return 1.0
        return max(0.0, min(1.0, epoch_idx / (total_epochs - 1)))

    if method == "fixed":
        def scheduler(epoch_idx: int, state: Dict) -> float:  # noqa: ARG001
            last_tau = (state.get("tau_hist") or [tau_min])[-1]
            # fixed uses start value but still enforce non-increasing
            candidate = _clamp(tau_min, tau_min, tau_max)
            return min(candidate, last_tau)
        return scheduler

    if method == "linear":
        def scheduler(epoch_idx: int, state: Dict) -> float:
            p = progress(epoch_idx)
            candidate = _clamp(tau_min + (tau_max - tau_min) * p, tau_min, tau_max)
            last_tau = (state.get("tau_hist") or [tau_min])[-1]
            return min(candidate, last_tau)
        return scheduler

    if method == "cosine":
        warmup = int(getattr(args, "cosine_warmup_epochs", 0))

        def scheduler(epoch_idx: int, state: Dict) -> float:
            if epoch_idx < warmup and warmup > 0:
                wp = epoch_idx / max(1, warmup)
                candidate = _clamp(tau_min + (tau_max - tau_min) * wp, tau_min, tau_max)
                last_tau = (state.get("tau_hist") or [tau_min])[-1]
                return min(candidate, last_tau)
            # cosine over remaining epochs
            denom = max(1, (total_epochs - max(0, warmup)))
            t = (epoch_idx - warmup) / denom
            cos_term = 0.5 * (1 - math.cos(math.pi * max(0.0, min(1.0, t))))
            candidate = _clamp(tau_min + (tau_max - tau_min) * cos_term, tau_min, tau_max)
            last_tau = (state.get("tau_hist") or [tau_min])[-1]
            return min(candidate, last_tau)
        return scheduler

    if method == "exp":
        k = float(getattr(args, "exp_k", 5.0))

        def scheduler(epoch_idx: int, state: Dict) -> float:
            p = progress(epoch_idx)
            # Smooth exponential rise from tau_min to tau_max
            v = 1.0 - math.exp(-k * p)
            candidate = _clamp(tau_min + (tau_max - tau_min) * v, tau_min, tau_max)
            last_tau = (state.get("tau_hist") or [tau_min])[-1]
            return min(candidate, last_tau)
        return scheduler

    if method == "adaptive_val":
        # Always decrease; decrease faster when validation loss improves
        fast_down = 0.05 * abs(tau_max - tau_min)
        slow_down = 0.02 * abs(tau_max - tau_min)

        def scheduler(epoch_idx: int, state: Dict) -> float:
            hist = state.get("val_loss_hist", [])
            last_tau = (state.get("tau_hist") or [tau_min])[-1]
            if len(hist) < 2:
                return _clamp(last_tau - slow_down, tau_min, tau_max)
            improved = hist[-1] < hist[-2] - 1e-6
            step = fast_down if improved else slow_down
            candidate = last_tau - step
            # enforce monotonic decrease and clamp to bounds
            return _clamp(min(candidate, last_tau), tau_min, tau_max)
        return scheduler

    if method == "adaptive_grad":
        # Always decrease; decrease faster when gradient norm decreases
        fast_down = 0.04 * abs(tau_max - tau_min)
        slow_down = 0.02 * abs(tau_max - tau_min)

        def scheduler(epoch_idx: int, state: Dict) -> float:
            hist = state.get("grad_norm_hist", [])
            last_tau = (state.get("tau_hist") or [tau_min])[-1]
            if len(hist) < 2:
                return _clamp(last_tau - slow_down, tau_min, tau_max)
            decreased = hist[-1] < hist[-2] - 1e-6
            step = fast_down if decreased else slow_down
            candidate = last_tau - step
            return _clamp(min(candidate, last_tau), tau_min, tau_max)
        return scheduler

    if method == "custom":
        # Placeholder: user can later replace via their own import or patch.
        # Enforce non-increasing using provided tau history.
        def scheduler(epoch_idx: int, state: Dict) -> float:  # noqa: ARG001
            last_tau = (state.get("tau_hist") or [tau_min])[-1]
            candidate = _clamp(tau_min, tau_min, tau_max)
            return min(candidate, last_tau)
        return scheduler

    # Fallback to fixed
    def scheduler(epoch_idx: int, state: Dict) -> float:  # noqa: ARG001
        return _clamp(tau_min, tau_min, tau_max)

    return scheduler


