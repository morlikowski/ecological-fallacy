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
            "race",
            "gender"
        ]
    )

def test_same_seed_same_splits(all_data, annotator_indecies_groups_mapping):
    metrics = MultiAnnotatorMetrics(annotator_indecies_groups_mapping)
    n_annotators = len(annotator_indecies_groups_mapping.keys())
    dataset = all_data.features

    annotators = []
    supports = []
    random_seeds = [2923262358, 2923262358]
    for seed_index, random_seed in enumerate(random_seeds):
        annotators.append([])
        supports.append([])
        splits = create_splits(
            dataset, 
            k=5,
            random_state=random_seed,
            create_validation_set = False
        )
        for split_index, split in enumerate(splits):
            annotators[seed_index].append(split['test']['annotator_indecies'])

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
            supports[seed_index].append(eval_results['gender - Female - individual_macro avg_support'])

    for indecies_a, indecies_b in zip(annotators[0], annotators[1]):
        assert (indecies_a == indecies_b).all().item()
    
    for support_a, support_b in zip(supports[0], supports[1]):
        assert support_a == support_b

    for indecies_a, indecies_b in zip(annotators[0], annotators[0][1:]):
        assert not (indecies_a == indecies_b).all().item()

def test_different_seed_different_splits(all_data, annotator_indecies_groups_mapping):
    metrics = MultiAnnotatorMetrics(annotator_indecies_groups_mapping)
    n_annotators = len(annotator_indecies_groups_mapping.keys())
    dataset = all_data.features

    annotators = []
    supports = []
    random_seeds = [2923262358, 165043843]
    for seed_index, random_seed in enumerate(random_seeds):
        annotators.append([])
        supports.append([])
        splits = create_splits(
            dataset, 
            k=5,
            random_state=random_seed,
            create_validation_set = False
        )
        for split in splits:
            annotators[seed_index].append(split['test']['annotator_indecies'])

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
            supports[seed_index].append(eval_results['gender - Female - individual_macro avg_support'])

    for indecies_a, indecies_b in zip(annotators[0], annotators[1]):
        assert (indecies_a != indecies_b).any().item()
    
    for support_a, support_b in zip(supports[0], supports[1]):
        assert support_a != support_b
        # test support diff at max 50 for gender - Female
        assert -50 < support_a - support_b < 50

    for indecies_a, indecies_b in zip(annotators[0], annotators[0][1:]):
        assert (indecies_a != indecies_b).any().item()