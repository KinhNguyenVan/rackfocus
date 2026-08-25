"""Test neighbor_slice_bounds -- logic thuần tính khoảng cắt, không cần S3/boto3 thật.

`get_neighbor_frames` trước đây chỉ nhận 1 anchor. Thêm `to_key` cho ô "xem frames giữa
2 sự kiện" của temporal search: N khung trước sự kiện 1 .. N khung sau sự kiện 2.
`to_frame == from_frame` phải tái hiện đúng hành vi 1-mỏ neo cũ.
"""

import pytest

from app.clients.s3 import neighbor_slice_bounds


def test_single_anchor_matches_old_before_after_behavior():
    frames = [100, 110, 120, 130, 140, 150, 160]
    assert neighbor_slice_bounds(frames, from_frame=130, to_frame=130,
                                 before=2, after=2) == (1, 6)


def test_two_anchors_spans_between_plus_margins():
    frames = [100, 110, 120, 130, 140, 150, 160, 170, 180]
    assert neighbor_slice_bounds(frames, from_frame=120, to_frame=160,
                                 before=1, after=1) == (1, 8)


def test_clamps_before_at_start_of_list():
    frames = [100, 110, 120]
    start, _ = neighbor_slice_bounds(frames, from_frame=100, to_frame=100,
                                     before=10, after=1)
    assert start == 0


def test_clamps_after_at_end_of_list():
    frames = [100, 110, 120]
    _, end = neighbor_slice_bounds(frames, from_frame=120, to_frame=120,
                                   before=1, after=10)
    assert end == 3


def test_order_independent_when_to_frame_precedes_from_frame():
    frames = [100, 110, 120, 130, 140]
    forward = neighbor_slice_bounds(frames, from_frame=110, to_frame=130, before=0, after=0)
    backward = neighbor_slice_bounds(frames, from_frame=130, to_frame=110, before=0, after=0)
    assert forward == backward == (1, 4)


def test_raises_when_frame_not_in_list():
    with pytest.raises(ValueError):
        neighbor_slice_bounds([100, 110, 120], from_frame=999, to_frame=100,
                              before=0, after=0)
