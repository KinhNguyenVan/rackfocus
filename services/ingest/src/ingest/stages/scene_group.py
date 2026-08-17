"""Gộp shot thành scene ngữ nghĩa: BaSSL CRN (Contextual Relation Network).

Mỗi shot -> đặc trưng ResNet50 trung bình trên 3 keyframe -> Transformer học ngữ
cảnh chuỗi -> phân loại nhị phân ranh giới giữa 2 shot liền kề. Vượt ngưỡng thì
cắt scene mới.

Phần logic gom scene từ mảng xác suất (`boundaries_to_scenes`) tách riêng, thuần
Python, để test được mà không cần model/GPU.
"""

from __future__ import annotations

import os

import pandas as pd


# =====================================================================
# 1. LOGIC THUẦN (test được, không cần torch)
# =====================================================================
def boundaries_to_scenes(
    shots_df: pd.DataFrame,
    boundary_probs: list[float],
    threshold: float = 0.55,
) -> list[dict]:
    """Gom shot thành scene từ xác suất ranh giới của (num_shots - 1) khoảng trống.

    `boundary_probs[i]` là xác suất có ranh giới scene *giữa* shot i và shot i+1.
    Trả về list scene: {start_frame, end_frame, start_time, end_time, confidence}.
    """
    scenes: list[dict] = []
    start_frame = int(shots_df.iloc[0]["start_frame"])
    start_ts = float(shots_df.iloc[0]["start_ts"])

    for i, prob in enumerate(boundary_probs):
        if prob > threshold:
            scenes.append({
                "start_frame": start_frame,
                "end_frame": int(shots_df.iloc[i]["end_frame"]),
                "start_time": start_ts,
                "end_time": float(shots_df.iloc[i]["end_ts"]),
                "confidence": round(float(prob), 4),
            })
            start_frame = int(shots_df.iloc[i + 1]["start_frame"])
            start_ts = float(shots_df.iloc[i + 1]["start_ts"])

    # Scene cuối luôn kéo tới hết shot cuối.
    scenes.append({
        "start_frame": start_frame,
        "end_frame": int(shots_df.iloc[-1]["end_frame"]),
        "start_time": start_ts,
        "end_time": float(shots_df.iloc[-1]["end_ts"]),
        "confidence": 1.0,
    })
    return scenes


# =====================================================================
# 2. MÔ HÌNH (chỉ import torch khi thực sự chạy)
# =====================================================================
def _build_crn():
    from torch import nn

    class BaSSL_CRN(nn.Module):
        """ResNet50 feature (2048) -> projection 512 -> Transformer -> boundary prob."""

        def __init__(self, input_dim: int = 2048, hidden_dim: int = 512):
            super().__init__()
            self.projection = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.boundary_classifier = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):  # x: (1, num_shots, 2048)
            import torch

            feats = self.projection(x)
            feats = self.transformer(feats).squeeze(0)        # (num_shots, 512)
            # Hiệu tuyệt đối giữa 2 shot liền kề -> tín hiệu đổi ngữ cảnh.
            boundary_feats = torch.abs(feats[:-1] - feats[1:])  # (num_shots-1, 512)
            logits = self.boundary_classifier(boundary_feats).squeeze(-1)
            return self.sigmoid(logits)

    return BaSSL_CRN


def load_resnet_backbone(device):
    """ResNet50 (ImageNet) bỏ lớp fc -> feature 2048. Nạp 1 lần."""
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()
    return backbone.eval().to(device)


def load_bassl(checkpoint_path: str, device):
    """Nạp BaSSL CRN từ checkpoint .ckpt. Thiếu file -> lỗi rõ ràng.

    Random weights cho scene không có nghĩa, nên bắt buộc phải có checkpoint.
    strict=False để bỏ qua vài key lệch do khác version PyTorch.
    """
    import torch

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"BaSSL checkpoint không tồn tại: {checkpoint_path!r}. "
            "Cung cấp đường dẫn .ckpt hợp lệ (xem plan/README ingest).")

    model = _build_crn()().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    return model.eval()


def _extract_shot_features(shots_df: pd.DataFrame, out_dir: str, backbone, device):
    """Feature (num_shots, 2048): trung bình ResNet50 trên 3 keyframe/shot.

    Đọc keyframe theo **frame index** ở cột kf0/kf1/kf2 của shots.csv
    (`keyframes/{idx:06d}.webp`) — chính là file đã ghi ở stage media.
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image

    from .media import KEYFRAMES_DIR

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    with torch.no_grad():
        for _, row in shots_df.iterrows():
            imgs = []
            for col in ("kf0", "kf1", "kf2"):
                p = os.path.join(out_dir, KEYFRAMES_DIR, f"{int(row[col]):06d}.webp")
                if os.path.exists(p):
                    imgs.append(transform(Image.open(p).convert("RGB")))
                else:
                    imgs.append(torch.zeros(3, 224, 224))
            batch = torch.stack(imgs).to(device)      # (3, 3, 224, 224)
            feats.append(backbone(batch).mean(dim=0).cpu())  # (2048,)
    return torch.stack(feats)                          # (num_shots, 2048)


def group_shots_into_scenes(
    shots_df: pd.DataFrame,
    out_dir: str,
    *,
    model,
    backbone,
    device,
    threshold: float = 0.55,
) -> list[dict]:
    """Chạy BaSSL CRN trên feature các shot rồi gom thành scene."""
    import torch

    num_shots = len(shots_df)
    if num_shots == 0:
        return []
    if num_shots == 1:
        return boundaries_to_scenes(shots_df, [], threshold)

    features = _extract_shot_features(shots_df, out_dir, backbone, device)
    with torch.no_grad():
        probs = model(features.unsqueeze(0).to(device))  # (num_shots-1,)
    return boundaries_to_scenes(shots_df, [p.item() for p in probs], threshold)
