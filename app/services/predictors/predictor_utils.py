from __future__ import annotations

from typing import Tuple


def reconstruct_absolute(
    pred_x: float,
    pred_y: float,
    last_pos_norm: Tuple[float, float],
    last_delta_norm: Tuple[float, float],
    target_mode: str,
) -> Tuple[float, float]:
    if target_mode == "absolute":
        return pred_x, pred_y
    if target_mode == "delta":
        return last_pos_norm[0] + pred_x, last_pos_norm[1] + pred_y
    if target_mode == "residual":
        return (
            last_pos_norm[0] + last_delta_norm[0] + pred_x,
            last_pos_norm[1] + last_delta_norm[1] + pred_y,
        )
    return pred_x, pred_y
