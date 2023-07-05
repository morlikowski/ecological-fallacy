import numpy as np
import pandas as pd

ALL_GROUP_PREFIX = 'ALL_'

def map_annotators_to_groups(annotators_mapping, sociodemographic_mapping, attributes, do_random_assignment = False):
    annotator_indecies_groups_mapping = {}
    if attributes:
        if do_random_assignment:
            groups_df = pd.DataFrame.from_records(list(sociodemographic_mapping.values()))

            # get maximum size per group from values in mapping
            # replace group name with generic identifier
            group_max_sizes = {a: {f'RANDOM_{i}': size for i, size in enumerate(groups_df[a].value_counts().to_dict().values())} for a in attributes}
            #initalize counter dict starting from zero to track group assignments
            group_assigned_sizes = {attribute: {g: 0 for g in groups.keys()} for attribute, groups in group_max_sizes.items()}
            rng = np.random.default_rng()

        for identifier, index in annotators_mapping.items():
            groups = {}
            for attribute in attributes:
                if attribute.startswith(ALL_GROUP_PREFIX):
                    attribute_value = 'SINGLE_DEFAULT_GROUP'
                else:
                    if do_random_assignment:
                        assignable_groups = list(group_assigned_sizes[attribute].keys())
                        attribute_value = rng.choice(assignable_groups)
                        group_assigned_sizes[attribute][attribute_value] += 1
                        if group_assigned_sizes[attribute][attribute_value] == group_max_sizes[attribute][attribute_value]:
                            del group_assigned_sizes[attribute][attribute_value]
                    else:
                        attribute_value = sociodemographic_mapping[identifier][attribute]
                groups[attribute] = attribute_value
            annotator_indecies_groups_mapping[index] = groups
    return annotator_indecies_groups_mapping