import json
import sys
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from src.search import ContextChunk
from src.search import retrieve_context

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT_DIR / "eval" / "eval_queries.json"


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_source: str


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    passed: bool
    source_rank: int | None
    context_chunks: int


def load_eval_cases(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Eval file is empty: {path}")

    try:
        data = json.loads(raw_text)
    except JSONDecodeError as error:
        raise ValueError(f"Eval file has invalid JSON: {path}") from error

    if not isinstance(data, list):
        raise ValueError("Eval file must contain a JSON array")

    return [parse_eval_case(item, index) for index, item in enumerate(data, start=1)]


def parse_eval_case(data: Any, index: int) -> EvalCase:
    if not isinstance(data, dict):
        raise ValueError(f"Eval case #{index}: item must be an object")

    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Eval case #{index}: query must be a non-empty string")

    expected_source = data.get("expected_source")
    if not isinstance(expected_source, str) or not expected_source.strip():
        raise ValueError(f"Eval case #{index}: expected_source must be a non-empty string")

    return EvalCase(
        query=query,
        expected_source=expected_source,
    )


def source_rank(context: list[ContextChunk], expected_source: str) -> int | None:
    for index, chunk in enumerate(context, start=1):
        if chunk.source == expected_source:
            return index

    return None


def run_eval_case(case: EvalCase) -> EvalResult:
    result = retrieve_context(case.query)
    rank = source_rank(result.context, case.expected_source)

    return EvalResult(
        case=case,
        passed=rank is not None,
        source_rank=rank,
        context_chunks=len(result.context),
    )


def print_eval_result(result: EvalResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    rank_text = result.source_rank if result.source_rank is not None else "missing"

    print(f"{status}: {result.case.query}")
    print(f"  expected_source={result.case.expected_source}")
    print(f"  source_rank={rank_text}")
    print(f"  context_chunks={result.context_chunks}")


def main() -> None:
    eval_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EVAL_FILE
    cases = load_eval_cases(eval_file)
    results = [run_eval_case(case) for case in cases]

    for result in results:
        print_eval_result(result)

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed

    print(f"\nSummary: {passed} passed, {failed} failed")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
