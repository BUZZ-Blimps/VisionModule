from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class HSVRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int


@dataclass
class RuntimeConfig:
    green_hsv: HSVRange = field(default_factory=lambda: HSVRange(39, 76, 78, 153, 58, 219))
    purple_hsv: HSVRange = field(default_factory=lambda: HSVRange(106, 133, 30, 189, 62, 180))
    goal_orange_hsv: HSVRange = field(default_factory=lambda: HSVRange(0, 25, 124, 255, 240, 255))
    goal_yellow_hsv: HSVRange = field(default_factory=lambda: HSVRange(27, 37, 107, 255, 197, 255))
    min_area: int = 250
    min_percent_filled: float = 60.0
    ignore_top_ratio: float = 0.0
    green_score_threshold: float = 0.2
    purple_score_threshold: float = 0.2
    green_dilate_iterations: int = 3
    purple_dilate_iterations: int = 3
    include_green: bool = True
    include_purple: bool = True
    include_goal_orange: bool = True
    include_goal_yellow: bool = True
    use_kalman: bool = True
    use_optical_flow: bool = True
    use_lock: bool = True
    goal_score_threshold: float = 0.2
    goal_dilate_iterations: int = 3
    goal_polygon_vertices: int = 4
    goal_min_area: int = 500
    goal_max_aspect_ratio: float = 1.5
    goal_polygon_inner_scale: float = 0.6
    goal_circle_inner_scale: float = 0.6


def _node_block(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(doc, dict):
        return {}
    for value in doc.values():
        if isinstance(value, dict) and "ros__parameters" in value:
            params = value.get("ros__parameters")
            return params if isinstance(params, dict) else {}
    return {}


def _flatten_dict(data: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_dict(value, full))
    else:
        out[prefix] = data
    return out


def _first(flat: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in flat:
            return flat[key]
    return None


def _coalesce(value: Any, default: Any) -> Any:
    return default if value is None else value


def _hsv_range(flat: Dict[str, Any], prefix: str, fallback: HSVRange) -> HSVRange:
    return HSVRange(
        h_min=int(_coalesce(_first(flat, f"{prefix}.h_min"), fallback.h_min)),
        h_max=int(_coalesce(_first(flat, f"{prefix}.h_max"), fallback.h_max)),
        s_min=int(_coalesce(_first(flat, f"{prefix}.s_min"), fallback.s_min)),
        s_max=int(_coalesce(_first(flat, f"{prefix}.s_max"), fallback.s_max)),
        v_min=int(_coalesce(_first(flat, f"{prefix}.v_min"), fallback.v_min)),
        v_max=int(_coalesce(_first(flat, f"{prefix}.v_max"), fallback.v_max)),
    )


def load_runtime_config_file(path: str | Path) -> RuntimeConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        doc = yaml.safe_load(file) or {}

    params = _node_block(doc)
    flat = _flatten_dict(params)
    defaults = RuntimeConfig()

    include_green = _first(flat, "detectors.green_balloon.enabled")
    include_purple = _first(flat, "detectors.purple_balloon.enabled")
    include_goal_orange = _first(flat, "detectors.goal_orange.enabled")
    include_goal_yellow = _first(flat, "detectors.goal_yellow.enabled")
    use_kalman = _first(flat, "tracking.use_kalman")
    use_optical_flow = _first(flat, "tracking.use_optical_flow")
    use_lock = _first(flat, "tracking.use_lock")

    return RuntimeConfig(
        green_hsv=_hsv_range(flat, "detectors.green_balloon.hsv", defaults.green_hsv),
        purple_hsv=_hsv_range(flat, "detectors.purple_balloon.hsv", defaults.purple_hsv),
        goal_orange_hsv=_hsv_range(flat, "detectors.goal_orange.hsv", defaults.goal_orange_hsv),
        goal_yellow_hsv=_hsv_range(flat, "detectors.goal_yellow.hsv", defaults.goal_yellow_hsv),
        min_area=int(_coalesce(_first(
            flat,
            "detectors.green_balloon.filters.min_area",
            "detectors.purple_balloon.filters.min_area",
        ), defaults.min_area)),
        min_percent_filled=float(_coalesce(_first(
            flat,
            "detectors.green_balloon.filters.min_fill",
            "detectors.purple_balloon.filters.min_fill",
        ), defaults.min_percent_filled)),
        ignore_top_ratio=float(_coalesce(_first(
            flat,
            "detectors.green_balloon.filters.ignore_top_ratio",
            "detectors.purple_balloon.filters.ignore_top_ratio",
        ), defaults.ignore_top_ratio)),
        green_score_threshold=float(_coalesce(
            _first(flat, "detectors.green_balloon.score_threshold"),
            defaults.green_score_threshold,
        )),
        purple_score_threshold=float(_coalesce(
            _first(flat, "detectors.purple_balloon.score_threshold"),
            defaults.purple_score_threshold,
        )),
        green_dilate_iterations=int(_coalesce(
            _first(flat, "detectors.green_balloon.filters.dilate_iterations"),
            defaults.green_dilate_iterations,
        )),
        purple_dilate_iterations=int(_coalesce(
            _first(flat, "detectors.purple_balloon.filters.dilate_iterations"),
            defaults.purple_dilate_iterations,
        )),
        include_green=bool(include_green) if include_green is not None else defaults.include_green,
        include_purple=bool(include_purple) if include_purple is not None else defaults.include_purple,
        include_goal_orange=bool(include_goal_orange) if include_goal_orange is not None else defaults.include_goal_orange,
        include_goal_yellow=bool(include_goal_yellow) if include_goal_yellow is not None else defaults.include_goal_yellow,
        use_kalman=bool(use_kalman) if use_kalman is not None else defaults.use_kalman,
        use_optical_flow=bool(use_optical_flow) if use_optical_flow is not None else defaults.use_optical_flow,
        use_lock=bool(use_lock) if use_lock is not None else defaults.use_lock,
        goal_score_threshold=float(_coalesce(_first(
            flat,
            "detectors.goal_orange.score_threshold",
            "detectors.goal_yellow.score_threshold",
        ), defaults.goal_score_threshold)),
        goal_dilate_iterations=int(_coalesce(_first(
            flat,
            "detectors.goal_orange.filters.dilate_iterations",
            "detectors.goal_yellow.filters.dilate_iterations",
        ), defaults.goal_dilate_iterations)),
        goal_polygon_vertices=int(_coalesce(_first(
            flat,
            "detectors.goal_orange.filters.polygon_vertices",
            "detectors.goal_yellow.filters.polygon_vertices",
        ), defaults.goal_polygon_vertices)),
        goal_min_area=int(_coalesce(_first(
            flat,
            "detectors.goal_orange.filters.min_area",
            "detectors.goal_yellow.filters.min_area",
        ), defaults.goal_min_area)),
        goal_max_aspect_ratio=float(_coalesce(_first(
            flat,
            "detectors.goal_orange.filters.max_aspect_ratio",
            "detectors.goal_yellow.filters.max_aspect_ratio",
        ), defaults.goal_max_aspect_ratio)),
        goal_polygon_inner_scale=float(_coalesce(_first(
            flat,
            "detectors.goal_orange.filters.polygon_inner_scale",
            "detectors.goal_yellow.filters.polygon_inner_scale",
        ), defaults.goal_polygon_inner_scale)),
        goal_circle_inner_scale=float(_coalesce(_first(
            flat,
            "detectors.goal_orange.filters.circle_inner_scale",
            "detectors.goal_yellow.filters.circle_inner_scale",
        ), defaults.goal_circle_inner_scale)),
    )


def apply_runtime_config(
    config: RuntimeConfig,
    blob_detector: Any,
    goal_module: Any,
    ball_contour_module: Any | None = None,
) -> None:
    blob_detector.green_lh = config.green_hsv.h_min
    blob_detector.green_uh = config.green_hsv.h_max
    blob_detector.green_ls = config.green_hsv.s_min
    blob_detector.green_us = config.green_hsv.s_max
    blob_detector.green_lv = config.green_hsv.v_min
    blob_detector.green_uv = config.green_hsv.v_max

    blob_detector.purple_lh = config.purple_hsv.h_min
    blob_detector.purple_uh = config.purple_hsv.h_max
    blob_detector.purple_ls = config.purple_hsv.s_min
    blob_detector.purple_us = config.purple_hsv.s_max
    blob_detector.purple_lv = config.purple_hsv.v_min
    blob_detector.purple_uv = config.purple_hsv.v_max

    blob_detector.min_area = config.min_area
    blob_detector.min_percent_filled = config.min_percent_filled
    blob_detector.ignore_top_ratio = config.ignore_top_ratio
    blob_detector.include_green = config.include_green
    blob_detector.include_purple = config.include_purple
    blob_detector.use_kalman = config.use_kalman
    blob_detector.use_optical_flow = config.use_optical_flow
    blob_detector.use_lock = config.use_lock

    if ball_contour_module is not None:
        ball_contour_module.purple_hsv["h_min"] = config.purple_hsv.h_min
        ball_contour_module.purple_hsv["h_max"] = config.purple_hsv.h_max
        ball_contour_module.purple_hsv["s_min"] = config.purple_hsv.s_min
        ball_contour_module.purple_hsv["s_max"] = config.purple_hsv.s_max
        ball_contour_module.purple_hsv["v_min"] = config.purple_hsv.v_min
        ball_contour_module.purple_hsv["v_max"] = config.purple_hsv.v_max

    goal_module.orange_hsv["h_min"] = config.goal_orange_hsv.h_min
    goal_module.orange_hsv["h_max"] = config.goal_orange_hsv.h_max
    goal_module.orange_hsv["s_min"] = config.goal_orange_hsv.s_min
    goal_module.orange_hsv["s_max"] = config.goal_orange_hsv.s_max
    goal_module.orange_hsv["v_min"] = config.goal_orange_hsv.v_min
    goal_module.orange_hsv["v_max"] = config.goal_orange_hsv.v_max

    goal_module.yellow_hsv["h_min"] = config.goal_yellow_hsv.h_min
    goal_module.yellow_hsv["h_max"] = config.goal_yellow_hsv.h_max
    goal_module.yellow_hsv["s_min"] = config.goal_yellow_hsv.s_min
    goal_module.yellow_hsv["s_max"] = config.goal_yellow_hsv.s_max
    goal_module.yellow_hsv["v_min"] = config.goal_yellow_hsv.v_min
    goal_module.yellow_hsv["v_max"] = config.goal_yellow_hsv.v_max

    goal_module.include_orange = config.include_goal_orange
    goal_module.include_yellow = config.include_goal_yellow
    goal_module.score_threshold = config.goal_score_threshold
    goal_module.dilate_iterations = config.goal_dilate_iterations
    goal_module.polygon_N = config.goal_polygon_vertices
    goal_module.goal_min_area = config.goal_min_area
    goal_module.goal_max_aspect_ratio = config.goal_max_aspect_ratio
    goal_module.goal_polygon_inner_scale = config.goal_polygon_inner_scale
    goal_module.goal_circle_inner_scale = config.goal_circle_inner_scale
