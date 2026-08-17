"""Các stage của pipeline ingest offline.

Thứ tự: probe -> shot_detect -> media(keyframe) -> scene_group -> asr -> media(cut).
"""

from .asr import assign_script_to_scenes, load_asr_model, transcribe
from .media import cut_scenes, extract_keyframes
from .probe import probe
from .scene_group import (
    boundaries_to_scenes,
    group_shots_into_scenes,
    load_bassl,
    load_resnet_backbone,
)
from .shot_detect import detect_shots, load_transnet

__all__ = [
    "probe",
    "load_transnet",
    "detect_shots",
    "extract_keyframes",
    "cut_scenes",
    "load_resnet_backbone",
    "load_bassl",
    "group_shots_into_scenes",
    "boundaries_to_scenes",
    "load_asr_model",
    "transcribe",
    "assign_script_to_scenes",
]
