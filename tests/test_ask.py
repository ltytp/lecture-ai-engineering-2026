"""素のLLM呼び出しのテスト（実際のAPIは呼び出さない）。"""

import json

import pytest

from app import ask as ask_module


def test_ask_records_answer_timing_and_tokens(monkeypatch, tmp_path, capsys):
    class Response:
        text = "回答です"
        usage_metadata = type(
            "Usage",
            (),
            {"prompt_token_count": 12, "candidates_token_count": 34},
        )()

    class Models:
        def generate_content(self, *, model, contents):
            assert model == ask_module.MODEL
            assert contents == "質問です"
            return Response()

    class Client:
        models = Models()

        def __init__(self, *, api_key):
            assert api_key == "test-key"

    monkeypatch.setattr(ask_module.genai, "Client", Client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    log_file = tmp_path / "results.jsonl"

    assert ask_module.ask("質問です", log_file) == "回答です"

    record = json.loads(log_file.read_text(encoding="utf-8"))
    assert record["model"] == ask_module.MODEL
    assert record["question"] == "質問です"
    assert record["elapsed_seconds"] >= 0
    assert record["input_tokens"] == 12
    assert record["output_tokens"] == 34
    assert record["status"] == "success"
    assert record["answer"] == "回答です"
    assert "[応答時間]" in capsys.readouterr().out


def test_ask_appends_each_run(monkeypatch, tmp_path):
    class Response:
        text = "回答です"
        usage_metadata = None

    class Models:
        def generate_content(self, **kwargs):
            return Response()

    class Client:
        models = Models()

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(ask_module.genai, "Client", Client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    log_file = tmp_path / "results.jsonl"

    ask_module.ask("1つ目", log_file)
    ask_module.ask("2つ目", log_file)

    records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [record["question"] for record in records] == ["1つ目", "2つ目"]
    assert all(record["input_tokens"] is None for record in records)


def test_ask_records_api_error(monkeypatch, tmp_path):
    class FakeAPIError(Exception):
        code = 429
        message = "rate limit"

    class Models:
        def generate_content(self, **kwargs):
            raise FakeAPIError()

    class Client:
        models = Models()

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(ask_module.genai, "Client", Client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.errors, "APIError", FakeAPIError)
    log_file = tmp_path / "results.jsonl"

    with pytest.raises(FakeAPIError):
        ask_module.ask("質問です", log_file)

    record = json.loads(log_file.read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["error_code"] == 429
    assert record["error"] == "rate limit"
    assert record["elapsed_seconds"] >= 0
