import torch
from transformers import EvalPrediction
from ecological_fallacy.eval.metrics import compute_majoritylabel_metrics

def test_single_task():
    logits = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
    labels = torch.tensor([
            [1],
            [0]
        ])
    result = compute_majoritylabel_metrics(EvalPrediction(logits, labels))
    assert result['majority_macro avg_f1-score'] == 1.0