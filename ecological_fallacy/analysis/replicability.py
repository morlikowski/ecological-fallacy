import ast

from collections import defaultdict

import numpy as np
import pandas as pd

from boostsa import Bootstrap

from sklearn.metrics import f1_score

from ecological_fallacy.run import map_annotators_to_groups
from ecological_fallacy.datasets import kumar

def calc_k_hat(p_values, alpha=0.05):
    p_values = sorted(p_values)
    k_values = [k for k, p in enumerate(p_values, start=1) if p <= alpha]
    if len(k_values) > 0:
        return max([k for k, p in enumerate(p_values, start=1) if p <= alpha])
    else:
        return -1

def bonferroni(p_values):
    N = len(p_values)
    p_values = sorted(p_values)
    return [(N - u + 1) * p for u, p in enumerate(p_values, start=1)]

def group_labels(to_groups, attribute, annotator_indecies, labels):
    individual_references = labels[labels > -1] # also flattens array
    known_annotators = annotator_indecies[annotator_indecies > -1] 
    group_label_pairs = [(to_groups[a][attribute], l) for a, l in zip(known_annotators, individual_references)]
    labels_per_group = defaultdict(list)
    for group, label in group_label_pairs: 
        labels_per_group[group].append(int(label))
    return labels_per_group

def significance_test_per_group(df, attribute_a, attribute_b, to_groups, attributes, seeds=[2803636207, 165043843, 2923262358], k=4, exclude_list=['Other', 'Prefer not to say']):
    boot = Bootstrap(save_outcomes=False, save_results=False)
    p_dicts = []
    for seed in seeds:
        for split in range(k):
            df_a = df[(df['random_seed'] == seed) & (df['split'] == split) & (df['attributes'] == attribute_a)]
            df_b = df[(df['random_seed'] == seed) & (df['split'] == split) & (df['attributes'] == attribute_b)]

            for attribute in attributes:
                if attribute_a == 'randomized':
                    df_a = df[(df['random_seed'] == seed) & (df['split'] == split) & (df['attributes'] == attribute_b) & (df['do_random_assignment'] == True)]
                else:
                    df_a = df[(df['random_seed'] == seed) & (df['split'] == split) & (df['attributes'] == attribute_a) & (df['do_random_assignment'] == False)]
            
                if attribute_b == 'randomized':
                    df_b = df[(df['random_seed'] == seed) & (df['split'] == split) & (df['attributes'] == attribute_a) & (df['do_random_assignment'] == True)]
                else:
                    df_b = df[(df['random_seed'] == seed) & (df['split'] == split) & (df['attributes'] == attribute_b) & (df['do_random_assignment'] == False)]

                grouped_predictions_a = group_labels(to_groups, attribute, df_a['annotator_indecies'].item(), df_a['predictions_per_annotator'].item())
                grouped_predictions_b = group_labels(to_groups, attribute, df_b['annotator_indecies'].item(), df_b['predictions_per_annotator'].item())
                grouped_labels = group_labels(to_groups, attribute, df_a['annotator_indecies'].item(), df_a['labels_per_annotator'].item())

                for group in grouped_labels.keys():
                    if group in exclude_list:
                        continue
                    print(f'\n\nSeed: {seed} - Split: {split} - {attribute} - Group: {group}')
                    df_results, _ = boot.test(
                        n_loops=1000,
                        sample_size=0.5,
                        targs=grouped_labels[group],
                        h0_preds=grouped_predictions_a[group],
                        h1_preds=grouped_predictions_b[group],
                        verbose=False
                    )
                    p_dicts.append(
                        {
                            'seed': seed,
                            'split': split,
                            'group': group,
                            'p': df_results.loc['h1']['p_f1']
                        }
                    )
    return p_dicts

def k_estimator_groups(group_p_dicts):
    groups = set(d['group'] for d in group_p_dicts)
    k_hats_grouped = {}
    for group in groups:
        p_values = [d['p'] for d in group_p_dicts if d['group'] == group]
        p_bonferroni = bonferroni(p_values)
        k_count = calc_k_hat(p_values)
        k_bonferroni = calc_k_hat(p_bonferroni)
        k_hats_grouped[group] = {
            'k_count': k_count,
            'k_bonferroni': k_bonferroni
        }
    return k_hats_grouped