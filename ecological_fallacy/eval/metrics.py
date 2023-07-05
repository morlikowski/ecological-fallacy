import numpy as np
import scipy

from transformers import EvalPrediction
from sklearn.metrics import classification_report

from typing import (
    Dict,
    Optional
)

def majority(labels_per_example, missing_label_val = -1):
    if labels_per_example.ndim != 2:
        raise ValueError('ndim of labels per example has to be 2')
    # docs on multiple modes:
    # "If there is more than one such value, only the smallest is returned"
    return np.apply_along_axis(lambda a: scipy.stats.mode(a[a > missing_label_val], nan_policy='omit').mode, 1, labels_per_example).flatten()

from collections import defaultdict
class MultiAnnotatorMetrics:

        def __init__(self, annotator_indecies_groups_mapping: Optional[Dict[int, Dict[str, str]]] = None):
            self.to_groups = {}
            self.attributes = set()
            self.groups_per_attribute = {}
            if annotator_indecies_groups_mapping:
                self.to_groups = annotator_indecies_groups_mapping
                self.attributes = set(next(iter(self.to_groups.values())).keys())
                self.groups_per_attribute = {attribute: set((mapping[attribute] for mapping in self.to_groups.values())) for attribute in self.attributes}

        def compute_single_predictions(self, p: EvalPrediction) -> Dict:
            logits = p.predictions
            labels, annotator_indecies = p.label_ids
            logits_argmax = np.argmax(logits, axis=-1) # num_examples
            argmax_reshaped = logits_argmax.reshape((logits_argmax.shape[0], 1)) # num_examples x 1 
            repeated_majority_predictions = np.repeat(argmax_reshaped, annotator_indecies.shape[-1], axis = 1) # num_examples x num_annotations_per_example
            # repeated prediction for comparision with individual labeling decisions
            predictions_for_known_annotators = repeated_majority_predictions.flatten().astype(int)

            return self._compute_reports(
                logits_argmax, 
                predictions_for_known_annotators, 
                labels, 
                annotator_indecies
            )

        def compute(self, p: EvalPrediction) -> Dict:
            logits = p.predictions
            labels, annotator_indecies = p.label_ids

            logits_argmax = np.argmax(logits, axis=-1)

            majorities = majority(logits_argmax)

            predictions_for_known_annotators = [label for logits, annotators in zip(logits_argmax, annotator_indecies) for label in np.take(logits, annotators[annotators > -1].astype(int))]

            result = self._compute_reports(
                majorities, 
                predictions_for_known_annotators, 
                labels, 
                annotator_indecies
            )
            result['labels_raw'] = labels.tolist()
            result['annotator_indecies'] = annotator_indecies.tolist()

            return result

        
        def _compute_reports(self, majorities, predictions_for_known_annotators, labels, annotator_indecies) -> Dict:

            # if labels are of higher dimensionality, they are interpreted to be label indecies
            # from each annotator for each example with "-1" indicating no label from the respective annotator
            references = majority(labels)

            individual_references = labels[labels > -1] # also flattens array
            known_annotators = annotator_indecies[annotator_indecies > -1] # also flattens array

            group_reports = {}
            for attribute in self.attributes:
                group_label_pairs = [(self.to_groups[a][attribute], l) for a, l in zip(known_annotators, individual_references)]
                labels_per_group = defaultdict(list)
                for group, label in group_label_pairs: 
                    labels_per_group[group].append(label)
                
    
                group_prediction_pairs = [(self.to_groups[a][attribute], p) for a, p in zip(known_annotators, predictions_for_known_annotators)]
                predictions_per_group = defaultdict(list)
                for group, label in group_prediction_pairs: 
                    predictions_per_group[group].append(label)
                
                for group in labels_per_group.keys():
                    group_report = classification_report(
                        y_true=labels_per_group[group], 
                        y_pred=predictions_per_group[group],
                        #TODO make configurable
                        labels = [0,1],
                        output_dict=True
                    )
                    group_reports[f'{attribute} - {group}'] = group_report

            majority_report = classification_report(
                        y_true=references, 
                        y_pred=majorities,
                        #TODO make configurable
                        labels = [0,1],
                        output_dict=True
                    )
            
            individual_report = classification_report(
                        y_true=individual_references, 
                        y_pred=predictions_for_known_annotators,
                        #TODO make configurable
                        labels = [0,1],
                        output_dict=True
                    )

            result = {}

            for attribute, groups in self.groups_per_attribute.items():
                for group in groups:
                    attribute_group = f'{attribute} - {group}'
                    if attribute_group in group_reports:
                        group_report = group_reports[attribute_group]
                        for name, metric in group_report.items():
                            if isinstance(metric, dict):
                                metric_dict = {f'{attribute_group} - individual_{name}_{k}':v for k, v in metric.items()}
                            else:
                                metric_dict = {f'{attribute_group} - individual_{name}': metric}
                            result.update(metric_dict)
                    else:
                        for name, metric in individual_report.items():
                            if isinstance(metric, dict):
                                metric_dict = {f'{attribute_group} - individual_{name}_{k}': float('nan') for k in metric.keys()}
                            else:
                                metric_dict = {f'{attribute_group} - individual_{name}': float('nan')}
                            result.update(metric_dict)

            for name, metric in majority_report.items():
                if isinstance(metric, dict):
                    metric_dict = {f'majority_{name}_{k}':v for k, v in metric.items()}
                else:
                    metric_dict = {f'majority_{name}': metric}
                result.update(metric_dict)

            for name, metric in individual_report.items():
                if isinstance(metric, dict):
                    metric_dict = {f'individual_{name}_{k}':v for k, v in metric.items()}
                else:
                    metric_dict = {f'individual_{name}': metric}
                result.update(metric_dict)

            # delete accuracy / micro averages if present for uniform result format
            # see documenatation on return type https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.classification_report.html
            result = {k: v for k, v in result.items() if 'micro avg' not in k and 'accuracy' not in k}

            result['predictions_per_annotator'] = predictions_for_known_annotators
            result['labels_per_annotator'] = individual_references.tolist()
            result['predictions_majority'] = majorities.tolist()
            result['labels_majority'] = references.tolist()

            return result