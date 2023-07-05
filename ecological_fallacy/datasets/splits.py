import numpy as np
import scipy

from sklearn.model_selection import StratifiedKFold
from datasets.dataset_dict import DatasetDict

def majority(labels_per_example, missing_label_val = -1):
    if labels_per_example.ndim != 2:
        raise ValueError('ndim of labels per example has to be 2')
    return np.apply_along_axis(lambda a: scipy.stats.mode(a[a > missing_label_val], nan_policy='omit').mode, 1, labels_per_example).flatten()

def create_splits(
        dataset, 
        k=2, 
        label_column='labels',
        random_state=None,
        create_validation_set = False
    ):
    do_shuffle = True if random_state else False
    folds = StratifiedKFold(
        n_splits=k, 
        shuffle=do_shuffle, 
        random_state=random_state
    )
    labels = dataset[label_column].numpy()
    majority_labels = majority(labels)
    splits = folds.split(np.zeros(dataset.num_rows), majority_labels)
    for train_idxs, eval_idxs in splits:
        train_set = dataset.select(train_idxs)
        eval_set = dataset.select(eval_idxs) # indecies for k distinct sets of rows from the dataset
      
        splits_dict = {
            "train": train_set,
            "test": eval_set
        }

        if create_validation_set:
            val_test_fold = StratifiedKFold(
                n_splits=2, 
                shuffle=do_shuffle, 
                random_state=random_state
            )
            test_idxs, val_idxs = next(val_test_fold.split(
                np.zeros(eval_idxs.shape[0]), 
                majority_labels[eval_idxs]
            ))
            splits_dict.update({
                "dev": eval_set.select(val_idxs),
                "test": eval_set.select(test_idxs)
            })
        
        yield DatasetDict(splits_dict)