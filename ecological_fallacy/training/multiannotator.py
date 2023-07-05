from typing import Dict

import torch
from sklearn.utils.class_weight import compute_class_weight

def compute_label_weights(
        labels, # shape (num_examples, num_annotators)
        annotators_on_example,
        classes=[0,1],
        missing_annotator_val = -1
    ) -> Dict[str, torch.Tensor]:
    annotator_indecies = annotators_on_example[annotators_on_example > missing_annotator_val].unique().int().tolist()
    #TODO does it makes sense to just have it as a tuple based on indecies instead of string key dict?
    label_freq_dict = {}
    for annotator_index in annotator_indecies:
        annotator_labels = labels[annotators_on_example == annotator_index]
        annotator_classes = annotator_labels.unique().int().numpy()
        label_weights = torch.tensor(compute_class_weight(
                'balanced',
                classes=annotator_classes, 
                y=annotator_labels.numpy()
            ), dtype=torch.float32)
        if len(label_weights) < len(classes):
            # set weights for missing classes to zero
            weights = torch.zeros(len(classes))
            for cl, weight in zip(annotator_classes, label_weights):
                weights[cl] = weight
            label_weights = weights
        label_freq_dict[str(annotator_index)] = label_weights
    return label_freq_dict