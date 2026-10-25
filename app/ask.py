"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --log-file results.jsonl "LoRAとは何ですか？"

ツールも会話履歴も持たない「素の」呼び出し。
実行日時、モデル名、質問、応答時間、トークン数、回答を JSONL に追記する。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
DEFAULT_LOG_FILE = Path("learning_log/ask_results.jsonl")


def _append_result(log_file: Path, result: dict[str, object]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def ask(question: str, log_file: Path = DEFAULT_LOG_FILE) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    try:
        response = client.models.generate_content(model=MODEL, contents=question)
    except errors.APIError as e:
        elapsed_seconds = time.perf_counter() - start
        _append_result(
            log_file,
            {
                "timestamp": started_at.isoformat(),
                "model": MODEL,
                "question": question,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "status": "error",
                "error_code": e.code,
                "error": e.message,
            },
        )
        print(f"[応答時間] {elapsed_seconds:.3f} 秒")
        print(f"[記録先] {log_file}")
        raise

    elapsed_seconds = time.perf_counter() - start
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    answer = response.text
    _append_result(
        log_file,
        {
            "timestamp": started_at.isoformat(),
            "model": MODEL,
            "question": question,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "status": "success",
            "answer": answer,
        },
    )
    print(f"[応答時間] {elapsed_seconds:.3f} 秒")
    print(f"[トークン] 入力: {input_tokens}, 出力: {output_tokens}")
    print(f"[記録先] {log_file}")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"実行記録のJSONLファイル（既定: {DEFAULT_LOG_FILE}）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question, log_file=args.log_file))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
