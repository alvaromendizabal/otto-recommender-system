import pytest

from otto_recsys.aws.storage import require_bucket, s3_uri


def test_s3_uri() -> None:
    assert (
        s3_uri("/models/example.bin/", bucket="otto-test")
        == "s3://otto-test/models/example.bin"
    )


def test_require_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTTO_BUCKET", "otto-example")
    assert require_bucket() == "otto-example"


def test_require_bucket_rejects_missing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTTO_BUCKET", raising=False)

    with pytest.raises(RuntimeError):
        require_bucket()
