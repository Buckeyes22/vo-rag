"""Domain vocabulary + keyword pack for the book_rag leaf.

The shipped default IS the motorsport pack — the book_rag eval baselines
are motorsport, so the
out-of-the-box behaviour must score the existing benchmark. Operators
running non-motorsport corpora can override at the `DomainTagger`
constructor (see `domain_tagger.py`).

The pack is `DOMAIN_VOCABULARY: list[str]` +
`DOMAIN_KEYWORDS: dict[str, tuple[str, ...]]`. It lives apart from
`domain_tagger.py` so the LLM-calling surface and the heuristic surface can
both reach the same pack without importing the other.
"""

from __future__ import annotations

DOMAIN_VOCABULARY: list[str] = [
    "vehicle_dynamics",
    "tire_mechanics",
    "aerodynamics",
    "suspension",
    "braking",
    "powertrain",
    "data_acquisition",
    "telemetry",
    "driver_technique",
    "race_strategy",
    "pit_operations",
    "safety",
    "regulations",
    "track_design",
    "weather_effects",
    "setup_tuning",
    "chassis_design",
    "weight_transfer",
    "cornering",
    "acceleration",
]


DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "vehicle_dynamics": (
        "vehicle dynamics",
        "balance",
        "understeer",
        "oversteer",
        "yaw",
        "rotation",
        "load transfer",
    ),
    "tire_mechanics": ("tire", "tyre", "contact patch", "slip angle", "grip", "traction"),
    "aerodynamics": (
        "aero",
        "aerodynamic",
        "downforce",
        "drag",
        "wing",
        "splitter",
        "diffuser",
    ),
    "suspension": (
        "suspension",
        "spring",
        "damper",
        "shock",
        "anti-roll",
        "roll bar",
    ),
    "braking": ("brake", "braking", "trail braking", "threshold braking", "brake bias"),
    "powertrain": ("engine", "powertrain", "throttle", "gear", "shift", "torque", "rpm"),
    "data_acquisition": (
        "data acquisition",
        "data logger",
        "logging",
        "sensor",
        "channel",
    ),
    "telemetry": ("telemetry", "trace", "lap data", "histogram", "squiggly line"),
    "driver_technique": (
        "driver",
        "driving technique",
        "racing line",
        "apex",
        "turn-in",
        "corner entry",
        "corner exit",
    ),
    "race_strategy": (
        "strategy",
        "race strategy",
        "traffic",
        "passing",
        "overtake",
        "racecraft",
    ),
    "pit_operations": ("pit", "pit stop", "crew", "service"),
    "safety": ("safety", "helmet", "harness", "roll cage", "flag", "incident"),
    "regulations": ("regulation", "rules", "scrutineering", "class", "compliance"),
    "track_design": (
        "track",
        "circuit",
        "corner",
        "chicane",
        "straight",
        "camber",
        "curb",
    ),
    "weather_effects": ("weather", "rain", "wet", "temperature", "wind", "humidity"),
    "setup_tuning": (
        "setup",
        "tuning",
        "adjustment",
        "alignment",
        "camber",
        "toe",
        "caster",
    ),
    "chassis_design": ("chassis", "frame", "monocoque", "stiffness", "rigidity"),
    "weight_transfer": (
        "weight transfer",
        "load transfer",
        "pitch",
        "roll",
        "longitudinal load",
    ),
    "cornering": ("cornering", "corner", "apex", "entry", "mid-corner", "exit"),
    "acceleration": ("acceleration", "accelerate", "launch", "drive off", "exit speed"),
}


__all__ = ["DOMAIN_KEYWORDS", "DOMAIN_VOCABULARY"]
