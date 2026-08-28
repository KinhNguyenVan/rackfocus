from app.clients.s3 import AWSStorageHelper


class FakeS3:
    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):
        return [{
            "Contents": [
                {"Key": f"{Prefix}000001.webp", "Size": 1},
                {"Key": f"{Prefix}000002.jpg", "Size": 1},
                {"Key": f"{Prefix}scene.mp4", "Size": 1},
            ]
        }]


def helper_for() -> AWSStorageHelper:
    helper = AWSStorageHelper.__new__(AWSStorageHelper)
    helper.bucket = "test-bucket"
    helper.region = "ap-southeast-2"
    helper.s3_client = FakeS3()
    return helper


def test_get_file_urls_filters_images_and_applies_limit():
    helper = helper_for()

    assert helper.get_file_urls("frames/", limit=1) == [
        "https://test-bucket.s3.ap-southeast-2.amazonaws.com/frames/000001.webp"
    ]


def test_get_file_urls_supports_custom_extensions():
    helper = helper_for()

    assert helper.get_file_urls("frames/", extensions=(".mp4",)) == [
        "https://test-bucket.s3.ap-southeast-2.amazonaws.com/frames/scene.mp4"
    ]