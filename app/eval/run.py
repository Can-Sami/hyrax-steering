import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalSummary:
    total: int
    correct_top1: int


def run_eval(dataset_path: str) -> EvalSummary:
    path = Path(dataset_path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    total = len(rows)
    correct = sum(1 for row in rows if row.get('expected_intent') == row.get('predicted_intent'))
    return EvalSummary(total=total, correct_top1=correct)


if __name__ == '__main__':
    summary = run_eval('data/eval/tr_v1.jsonl')
    accuracy = (summary.correct_top1 / summary.total) if summary.total else 0.0
    print(json.dumps({'total': summary.total, 'top1_accuracy': accuracy}, ensure_ascii=True))
