import csv
import json
import zipfile

import pytest

from bertdm.data import ExampleError, load_splits, normalize_cosmos, normalize_wikihop, validate_example
from bertdm.download import download_file, prepare_wikihop


def test_normalization_and_official_split_membership(tmp_path):
    cosmos = {"context": "story", "question": "why?", "label": "2",
              **{f"answer{i}": f"answer {i}" for i in range(4)}}
    assert normalize_cosmos(cosmos) == {"context": "story", "question": "why?",
                                         "choices": ["answer 0", "answer 1", "answer 2", "answer 3"], "label": 2}
    train, valid = tmp_path / "train.csv", tmp_path / "valid.csv"
    for path, label in ((train, "2"), (valid, "1")):
        row = dict(cosmos, label=label)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    splits = load_splits("cosmos", train_file=train, validation_file=valid)
    assert [e["label"] for e in splits["train"]] == [2]
    assert [e["label"] for e in splits["validation"]] == [1]
    with pytest.raises(ValueError, match="different"):
        load_splits("cosmos", train_file=train, validation_file=train)


def test_wikihop_support_and_candidate_order(tmp_path):
    row = {"supports": ["doc two", "doc one"], "query": "relation entity",
           "candidates": ["z", "x", "y"], "answer": "x"}
    normalized = normalize_wikihop(row)
    assert normalized == {"context": "doc two\n\ndoc one", "question": "relation entity",
                           "choices": ["z", "x", "y"], "label": 1}
    train, valid = tmp_path / "train.json", tmp_path / "dev.jsonl"
    train.write_text(json.dumps([row, dict(row, answer="z")]), encoding="utf-8")
    valid.write_text(json.dumps(dict(row, answer="y")) + "\n", encoding="utf-8")
    splits = load_splits("wikihop", train_file=train, validation_file=valid)
    assert [e["label"] for e in splits["train"]] == [1, 0]
    assert [e["label"] for e in splits["validation"]] == [2]
    with pytest.raises(ExampleError, match="exactly one"):
        normalize_wikihop(dict(row, answer="missing"))
    with pytest.raises(ExampleError, match="supports"):
        normalize_wikihop(dict(row, supports=[""]))


@pytest.mark.parametrize("update", [
    {"context": " "}, {"question": None}, {"choices": "abc"}, {"choices": ["a"]},
    {"choices": ["a", ""]}, {"label": -1}, {"label": 2},
    {"label": True}, {"label": 0.5}, {"label": None},
])
def test_malformed_normalized_examples(update):
    with pytest.raises(ExampleError):
        validate_example({"context": "c", "question": "q", "choices": ["a", "b"], "label": 0} | update)


def test_reader_rejections_name_split_and_record(tmp_path):
    train, valid = tmp_path / "train.json", tmp_path / "dev.json"
    train.write_text('[{"choices": []}]', encoding="utf-8")
    valid.write_text("[]", encoding="utf-8")
    with pytest.raises(ExampleError, match="normalized/train.*record 0"):
        load_splits("normalized", train_file=train, validation_file=valid)


def test_official_cosmos_repeated_answer_text_keeps_indexed_slots():
    row = {"context": "story", "question": "why?", "label": "3",
           "answer0": "anxious", "answer1": "scared", "answer2": "scared", "answer3": "regret"}
    normalized = normalize_cosmos(row)
    assert normalized["choices"] == ["anxious", "scared", "scared", "regret"]
    assert normalized["label"] == 3


def test_archive_copies_only_original_official_splits(tmp_path):
    archive = tmp_path / "qangaroo.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("qangaroo_v1.1/wikihop/train.json", '[{"train": true}]')
        stream.writestr("qangaroo_v1.1/wikihop/dev.json", '[{"dev": true}]')
        stream.writestr("qangaroo_v1.1/wikihop_masked/train.json", "masked")
        stream.writestr("../../outside.txt", "unsafe")
    paths = prepare_wikihop(archive, tmp_path / "data")
    assert json.loads(paths["train"].read_text()) == [{"train": True}]
    assert json.loads(paths["validation"].read_text()) == [{"dev": True}]
    assert not (tmp_path / "outside.txt").exists()
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert len(manifest["splits"]["train"]["sha256"]) == 64


def test_download_streaming_cache_hash_and_failure(tmp_path, monkeypatch):
    import hashlib
    import io

    from bertdm import download

    destination = tmp_path / "download.json"
    payload = b'{"data": 42}'
    called = []

    def open_url(request, timeout):
        called.append(request.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr(download.urllib.request, "urlopen", open_url)
    digest = hashlib.sha256(payload).hexdigest()
    assert download_file("https://example.org/data", destination, expected_sha256=digest).read_bytes() == payload
    download_file("https://example.org/data", destination, expected_sha256=digest)
    assert len(called) == 1
    with pytest.raises(ValueError, match="SHA256"):
        download_file("https://example.org/data", destination, expected_sha256="bad")
    with pytest.raises(ValueError, match="SHA256"):
        download_file("https://example.org/data", tmp_path / "bad", expected_sha256="bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="HTTPS"):
        download_file("http://example.org/data", destination)


def test_native_parquet_reader_preserves_splits_and_lists(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    row = {"supports": ["doc two", "doc one"], "query": "relation entity",
           "candidates": ["z", "x", "y"], "answer": "x"}
    train, valid = tmp_path / "train.parquet", tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist([row]), train)
    pq.write_table(pa.Table.from_pylist([dict(row, answer="z")]), valid)
    splits = load_splits("wikihop", train_file=train, validation_file=valid)
    assert splits["train"] == [normalize_wikihop(row)]
    assert splits["validation"][0]["label"] == 0


def test_wikihop_download_uses_pinned_original_parquet_splits(tmp_path, monkeypatch):
    from bertdm import download

    urls = []

    def download_stub(url, destination):
        urls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"test")
        return destination

    monkeypatch.setattr(download, "download_file", download_stub)
    paths = download.download_wikihop(tmp_path / "data")
    assert set(paths) == {"train", "validation"}
    assert all(download.WIKIHOP_REVISION in url and "/wikihop/" in url for url in urls)
    assert urls[0].endswith("/train-00000-of-00001.parquet")
    assert urls[1].endswith("/validation-00000-of-00001.parquet")
