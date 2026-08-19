"""Test phần logic thuần của storage.py (không cần boto3/S3 thật): `parse_s3_uri`."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest.storage import parse_s3_uri


def test_parses_bucket_and_prefix():
    assert parse_s3_uri("s3://my-bucket/models/siglip") == ("my-bucket", "models/siglip")


def test_strips_leading_and_trailing_slashes_from_prefix():
    assert parse_s3_uri("s3://my-bucket/models/siglip/") == ("my-bucket", "models/siglip")


def test_bucket_only_has_empty_prefix():
    assert parse_s3_uri("s3://my-bucket") == ("my-bucket", "")


def test_rejects_non_s3_uri():
    with pytest.raises(ValueError):
        parse_s3_uri("/local/path")


def test_rejects_missing_bucket():
    with pytest.raises(ValueError):
        parse_s3_uri("s3:///prefix")
