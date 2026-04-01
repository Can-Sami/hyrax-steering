from app.eval.run import run_eval


def test_run_eval_counts_correct_predictions() -> None:
    summary = run_eval('data/eval/tr_v1.jsonl')
    assert summary.total == 2
    assert summary.correct_top1 == 1
