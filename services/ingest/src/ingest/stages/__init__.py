"""Các stage của pipeline ingest offline.

Thứ tự: probe -> shot_detect -> media(keyframe) -> scene_group -> asr -> media(cut) -> embed.
"""

from .asr import assign_script_to_scenes, load_asr_model, transcribe
from .embed import (
    assign_scene_idx,
    dump_shards,
    embed_keyframes,
    keyframe_point_id,
    load_siglip,
)
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
    "assign_scene_idx",
    "assign_script_to_scenes",
    "boundaries_to_scenes",
    "cut_scenes",
    "detect_shots",
    "dump_shards",
    "embed_keyframes",
    "extract_keyframes",
    "group_shots_into_scenes",
    "keyframe_point_id",
    "load_asr_model",
    "load_bassl",
    "load_resnet_backbone",
    "load_siglip",
    "load_transnet",
    "probe",
    "transcribe",
]
