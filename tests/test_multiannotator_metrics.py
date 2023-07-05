import pytest
import numpy as np

from transformers import EvalPrediction

from ecological_fallacy.datasets import kumar
from ecological_fallacy.eval.metrics import MultiAnnotatorMetrics
from ecological_fallacy.datasets.splits import create_splits
from ecological_fallacy.models.mapping import map_annotators_to_groups

DATA_PATH = "data/processed/kumar/sample_5000_annotators_5_annotations_document.csv"
N = 3000

@pytest.fixture
def all_data():
    return kumar.Dataset.load(
        DATA_PATH,
        n=N
    )

@pytest.fixture
def annotator_indecies_groups_mapping(all_data):
    return map_annotators_to_groups(
        all_data.annotators_mapping,
        all_data.sociodemographic_mapping,
        [
            "age_range",
            "religion_important",
            "lgbtq_status",
            "is_parent",
            "education",
            "race"
        ]
    )

def test_same_entries_across_splits(all_data, annotator_indecies_groups_mapping):
    metrics = MultiAnnotatorMetrics(annotator_indecies_groups_mapping)
    n_annotators = len(annotator_indecies_groups_mapping.keys())
    dataset = all_data.features

    random_seeds = [2803636207]
    for random_seed in random_seeds:
        splits = create_splits(
            dataset, 
            k=5,
            random_state=random_seed,
            create_validation_set = False
        )
        for index, split in enumerate(splits):
            n_examples = split['test'].num_rows
            mock_logits = np.random.rand(n_examples, n_annotators, 2)
            p = EvalPrediction(
                predictions = mock_logits, 
                label_ids= (
                        split['test']['labels'].numpy(),
                        split['test']['annotator_indecies'].numpy()
                    )
                )
            eval_results = metrics.compute(p)
            if index == 0:
                entries_first_split = set(eval_results.keys())
            else:
                entries_this_split = set(eval_results.keys())
                len_union = len(entries_first_split.union(entries_this_split))
                len_first_split = len(entries_first_split)
                assert len_first_split == len_union

def test_contains_predictions_labels(all_data, annotator_indecies_groups_mapping):
    """Tests wheter the results contain the actual predictions and labels in additon to the metrics."""
    metrics = MultiAnnotatorMetrics(annotator_indecies_groups_mapping)
    n_annotators = len(annotator_indecies_groups_mapping.keys())
    dataset = all_data.features

    random_seeds = [2803636207]
    for random_seed in random_seeds:
        splits = create_splits(
            dataset, 
            k=5,
            random_state=random_seed,
            create_validation_set = False
        )
        for index, split in enumerate(splits):
            n_examples = split['test'].num_rows
            mock_logits = np.random.rand(n_examples, n_annotators, 2)
            p = EvalPrediction(
                predictions = mock_logits, 
                label_ids= (
                        split['test']['labels'].numpy(),
                        split['test']['annotator_indecies'].numpy()
                    )
                )
            eval_results = metrics.compute(p)
            assert len(eval_results['predictions_per_annotator']) == n_examples * 5
            assert len(eval_results['labels_per_annotator']) == n_examples * 5
            assert len(eval_results['predictions_majority']) == n_examples
            assert len(eval_results['labels_majority']) == n_examples
            assert np.array(eval_results['labels_raw']).shape == (n_examples, 5)

def test_majority_less_logits_than_annotators():
    """ This is a contreived case. We do not expect to have more annotations per example than annotators in total """
    logits = np.array([
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0]
            ],
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ]
        ])
    labels = np.array([
            [0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1]
        ])
    dummy_annotator_indecies = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    metrics = MultiAnnotatorMetrics()
    result = metrics.compute(EvalPrediction(logits, (labels, dummy_annotator_indecies)))
    assert result['majority_macro avg_f1-score'] == 1.0

def test_majority_many_logits():
    logits = np.array([
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0]
            ],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ]
        ])
    labels = np.array([
            [0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1]
        ])
    dummy_annotator_indecies = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    metrics = MultiAnnotatorMetrics()
    result = metrics.compute(EvalPrediction(logits, (labels, dummy_annotator_indecies)))
    assert result['majority_macro avg_f1-score'] == 1.0

def test_majority_varying_num_labels():
    """ Tests for handling -1 labels when computing metrics. """
    logits = np.array([
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0]
            ],
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ],
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ]
        ])
    # This format results from varying numbers of labels per example
    # The label array has the length of the maximum number of annotations per example
    # If annotations are missing for an example, the array is padded with -1
    # The -1 are ignored when calculating the majority votes
    labels = np.array([
            [1, 1, 1, 0, -1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, -1, -1]
        ])
    dummy_annotator_indecies = np.array([
            [0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, -1, -1]
        ])
    metrics = MultiAnnotatorMetrics()
    result = metrics.compute(EvalPrediction(logits, (labels, dummy_annotator_indecies)))
    assert result['majority_macro avg_f1-score'] == 1.0

def test_majority_varying_num_labels_with_groups():
    """ Tests for handling -1 labels when computing metrics with groups. """
    logits = np.array([
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]
            ],
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ]
        ])
    # This format results from varying numbers of labels per example
    # The label array has the length of the maximum number of annotations per example
    # If annotations are missing for an example, the array is padded with -1
    # The -1 are ignored when calculating the majority votes
    labels = np.array([
            [1, 0, 0, 1, -1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, -1, -1]
        ])
    dummy_annotator_indecies = np.array([
            [0, 1, 2, 3, -1],
            [0, 1, 2, 3, 4],
            [0, 1, 2, -1, -1]
        ])
    
    annotator_indecies_groups_mapping = {
        0: {
            'attribute': 'group_1'
        },
        1: {
            'attribute': 'group_2'
        },
        2: {
            'attribute': 'group_1'
        },
        3: {
            'attribute': 'group_2'
        },
        4: {
            'attribute': 'group_1'
        }
    }
    metrics = MultiAnnotatorMetrics(annotator_indecies_groups_mapping)
    result = metrics.compute(EvalPrediction(logits, (labels, dummy_annotator_indecies)))
    assert result['attribute - group_1 - individual_macro avg_f1-score'] == 1.0
    assert result['attribute - group_2 - individual_macro avg_f1-score'] == 0.0
    

def test_majority_resolve_mode():
    logits = np.array([
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0]
            ],
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0]
            ]
        ])
    labels = np.array([
            [0, 1, 1, 1],
            [1, 1, 0, 0]
        ])
    dummy_annotator_indecies = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
    metrics = MultiAnnotatorMetrics()
    result = metrics.compute(EvalPrediction(logits, (labels, dummy_annotator_indecies)))
    assert result['majority_macro avg_f1-score'] == 1.0