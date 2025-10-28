import math
from typing import Callable, Dict


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
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
            return _clamp(tau_min, tau_min, tau_max)
        return scheduler

    if method == "linear":
        def scheduler(epoch_idx: int, state: Dict) -> float:  # noqa: ARG001
            p = progress(epoch_idx)
            return _clamp(tau_min + (tau_max - tau_min) * p, tau_min, tau_max)
        return scheduler

    if method == "cosine":
        warmup = int(getattr(args, "cosine_warmup_epochs", 0))

        def scheduler(epoch_idx: int, state: Dict) -> float:  # noqa: ARG001
            if epoch_idx < warmup and warmup > 0:
                wp = epoch_idx / max(1, warmup)
                return _clamp(tau_min + (tau_max - tau_min) * wp, tau_min, tau_max)
            # cosine over remaining epochs
            denom = max(1, (total_epochs - max(0, warmup)))
            t = (epoch_idx - warmup) / denom
            cos_term = 0.5 * (1 - math.cos(math.pi * max(0.0, min(1.0, t))))
            return _clamp(tau_min + (tau_max - tau_min) * cos_term, tau_min, tau_max)
        return scheduler

    if method == "exp":
        k = float(getattr(args, "exp_k", 5.0))

        def scheduler(epoch_idx: int, state: Dict) -> float:  # noqa: ARG001
            p = progress(epoch_idx)
            # Smooth exponential rise from tau_min to tau_max
            v = 1.0 - math.exp(-k * p)
            return _clamp(tau_min + (tau_max - tau_min) * v, tau_min, tau_max)
        return scheduler

    if method == "adaptive_val":
        # Heuristic: if validation loss improves, raise tau; else, gently reduce it
        up_step = 0.05 * (tau_max - tau_min)
        down_step = 0.02 * (tau_max - tau_min)

        def scheduler(epoch_idx: int, state: Dict) -> float:
            hist = state.get("val_loss_hist", [])
            last_tau = state.get("tau_hist", [tau_min])[-1]
            if len(hist) < 2:
                # ramp up slowly at the beginning
                return _clamp(last_tau + up_step, tau_min, tau_max)
            improved = hist[-1] < hist[-2] - 1e-6
            if improved:
                return _clamp(last_tau + up_step, tau_min, tau_max)
            return _clamp(last_tau - down_step, tau_min, tau_max)
        return scheduler

    if method == "adaptive_grad":
        # Heuristic: when grad norm decreases → increase tau; when increases → decrease tau
        up_step = 0.04 * (tau_max - tau_min)
        down_step = 0.04 * (tau_max - tau_min)

        def scheduler(epoch_idx: int, state: Dict) -> float:
            hist = state.get("grad_norm_hist", [])
            last_tau = state.get("tau_hist", [tau_min])[-1]
            if len(hist) < 2:
                return _clamp(last_tau + up_step, tau_min, tau_max)
            decreased = hist[-1] < hist[-2] - 1e-6
            if decreased:
                return _clamp(last_tau + up_step, tau_min, tau_max)
            return _clamp(last_tau - down_step, tau_min, tau_max)
        return scheduler

    if method == "custom":
        # Placeholder: user can later replace via their own import or patch
        def scheduler(epoch_idx: int, state: Dict) -> float:  # noqa: ARG001
            return _clamp(tau_min, tau_min, tau_max)
        return scheduler

    # Fallback to fixed
    def scheduler(epoch_idx: int, state: Dict) -> float:  # noqa: ARG001
        return _clamp(tau_min, tau_min, tau_max)

    return scheduler


