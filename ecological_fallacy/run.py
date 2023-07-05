import json
import torch
import transformers
from transformers.trainer import Trainer
import datasets
from ecological_fallacy.models.mapping import map_annotators_to_groups
from ecological_fallacy.datasets.splits import create_splits

from ecological_fallacy.eval.callbacks import EvalResultsCallback
from ecological_fallacy.eval.metrics import MultiAnnotatorMetrics
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
from pathlib import Path

from ecological_fallacy.models.multiannotator import FixedPredBaseline
from ecological_fallacy.models import PerAnnotatorModelForSequenceClassification

from ecological_fallacy.datasets import kumar

from ecological_fallacy.training import compute_label_weights

# inherit from str for easy serialization to JSON
class Architecture(str, Enum):
    MULTI_TASK = 'multi_task'
    MAJORITY_BASELINE = 'majority_baseline'

# inherit from str for easy serialization to JSON
class EvalSetting(str, Enum):
    TRAIN_TEST_SPLIT = 'TRAIN_TEST_SPLIT'
    K_FOLD = 'K_FOLD'

# inherit from str for easy serialization to JSON
class DatasetType(str, Enum):
    KUMAR = 'KUMAR'

# inherit from str for easy serialization to JSON
class ClassifierType(str, Enum):
    LINEAR_LAYER = 'LINEAR_LAYER'
    HEAD = 'HEAD'

def train(
        experiment_path,
        experiment_index,
        data_path,
        dataset_type = DatasetType.KUMAR,
        pretrained_name_path='distilbert-base-uncased',
        architecture=Architecture.MULTI_TASK,
        classifier_type=ClassifierType.LINEAR_LAYER,
        learning_rate=1e-7,
        num_train_epochs=3,
        n=None,
        attributes=[],
        eval_setting=EvalSetting.TRAIN_TEST_SPLIT,
        k=4,
        eval_while_train=False,
        output_dir='.models/',
        do_random_assignment = False,
        random_seeds = [2803636207]
    ) -> None:
    if torch.cuda.is_available():
        logging.info('CUDA is available')
    else:
        logging.warning('Training on CPU! CUDA not available')

    architecture = Architecture[architecture]
    eval_setting = EvalSetting[eval_setting]
    dataset_type = DatasetType[dataset_type]
    classifier_type = ClassifierType[classifier_type]

    config = {
        'experiment_index': experiment_index,
        'experiment_path': str(experiment_path),
        'data_path': str(data_path),
        'dataset_type': dataset_type,
        'pretrained_name_path': pretrained_name_path,
        'architecture': architecture,
        'classifier_type': classifier_type,
        'learning_rate': learning_rate,
        'num_train_epochs': num_train_epochs,
        'n': n,
        'k': k,
        'attributes': attributes,
        'eval_setting': eval_setting,
        'output_dir': output_dir,
        'eval_while_train': eval_while_train,
        'do_random_assignment': do_random_assignment
    }

    if dataset_type == DatasetType.KUMAR:
        do_eval_groups = True
        all_data = kumar.Dataset.load(
                data_path,
                do_majority_aggregation = False,
                n=n,
                model_name=pretrained_name_path
            )
    else:
        raise ValueError(f'dataset_type needs to be DatasetType.KUMAR, got {dataset_type}')
    
    dataset = all_data.features

    annotator_indecies_groups_mapping = map_annotators_to_groups(
        all_data.annotators_mapping,
        all_data.sociodemographic_mapping,
        attributes,
        do_random_assignment = do_random_assignment
    )

    if do_eval_groups:
        # groups for evaluation, not modelling
        relevant_attributes_indecies_groups_mapping = map_annotators_to_groups(
            all_data.annotators_mapping,
            all_data.sociodemographic_mapping,
            [
                'age_range',
                'religion_important',
                'lgbtq_status',
                'is_parent',
                'education',
                'race',
                'gender'
            ]
        )
        metrics = MultiAnnotatorMetrics(relevant_attributes_indecies_groups_mapping)
    else:
        metrics = MultiAnnotatorMetrics()

    for random_seed in random_seeds:
        if type(all_data.features) == datasets.dataset_dict.DatasetDict:
            # if dataset is split already, use splits
            splits = [
                all_data.features
            ]
        else:
            splits = create_splits(
                dataset, 
                k=k,
                random_state=random_seed,
                create_validation_set = eval_while_train
            )
        for index, split in enumerate(splits):

            config['random_seed'] = random_seed
            config['split'] = index
            
            common_training_args = {
                'output_dir': output_dir,
                'overwrite_output_dir': True,
                'learning_rate': learning_rate,
                'do_train': True,
                'num_train_epochs': num_train_epochs,
                'per_device_train_batch_size': 8,  #NOTE Make configurable if necessary
                'per_device_eval_batch_size': 8,  #NOTE Make configurable if necessary
                'evaluation_strategy': 'epoch' if 'dev' in split else 'no',
                'save_strategy': 'no',
                'logging_steps': 100,  #NOTE Make configurable if necessary
                'save_total_limit': 1,  #NOTE Make configurable if necessary
                'seed': random_seed # also sets data_seed
            }

            if architecture == Architecture.MULTI_TASK:

                label_weights = compute_label_weights(
                    labels=split['train']['labels'],
                    annotators_on_example=split['train']['annotator_indecies'],
                    classes=[0,1] #NOTE Make configurable if necessary
                )

                if classifier_type == ClassifierType.LINEAR_LAYER:
                    model_init = lambda: PerAnnotatorModelForSequenceClassification(
                        pretrained_name_path,
                        annotators_mapping=all_data.annotators_mapping,
                        groups_mapping=annotator_indecies_groups_mapping,
                        label_weights=label_weights,
                        num_labels=2 #NOTE Make configurable if necessary
                    )
                elif classifier_type == ClassifierType.HEAD:
                    raise NotImplementedError()
                else:
                    raise ValueError(f'classifier_type needs to be ClassifierType.LINEAR_LAYER or ClassifierType.HEAD, got {classifier_type}')

                training_args = transformers.TrainingArguments(
                    **common_training_args,
                    label_names = ['labels', 'annotator_indecies']
                )

                trainer = Trainer(
                        model_init = model_init,
                        args=training_args,
                        train_dataset=split['train'],
                        eval_dataset=split['dev'] if 'dev' in split else None,
                        compute_metrics = metrics.compute,
                        callbacks=[
                            EvalResultsCallback(
                                experiment_path=experiment_path,
                                experiment_config = config
                            )
                        ]
                )
                logging.info(f'Running on split {index}')
                trainer.train()
                if 'test' in split:
                    trainer.evaluate(eval_dataset=split['test'])
            
            elif architecture == Architecture.MAJORITY_BASELINE:
                most_common_label = torch.mode(torch.flatten(split['train']['labels'])).values.item()
                model = FixedPredBaseline(
                    label_index = most_common_label,
                    annotators_mapping = all_data.annotators_mapping
                )

                training_args = transformers.TrainingArguments(
                    **common_training_args,
                    label_names = ['labels', 'annotator_indecies']
                )

                trainer = Trainer(
                        model = model,
                        args=training_args,
                        train_dataset=split['train'],
                        eval_dataset=split['dev'] if 'dev' in split else None,
                        compute_metrics = metrics.compute,
                        callbacks=[
                            EvalResultsCallback(
                                experiment_path=experiment_path,
                                experiment_config = config
                            )
                        ]
                )
                if 'test' in split:
                    trainer.evaluate(eval_dataset=split['test'])
            else:
                raise ValueError('Value of argument "architecture" needs to be Architecture.MAJORITY_BASELINE or Architecture.MULTI_TASK')
        
            if eval_setting == EvalSetting.TRAIN_TEST_SPLIT:
                 # Add break to make stratified k-fold effectively stratified train/test split
                break
            elif eval_setting == EvalSetting.K_FOLD:
                pass
            else:
                raise ValueError(f'eval_setting needs to be EvalSetting.TRAIN_TEST_SPLIT or EvalSetting.K_FOLD, got {eval_setting}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train models aware of individual annotators and groups')
    parser.add_argument('experiment_path', help='a path to directory with an experiment configuration config.json')

    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)

    with open(experiment_path / 'config.json') as f:
        config = json.load(f)

    for index, experiment in enumerate(config['settings']):
        train(
            experiment_path,
            experiment_index = index,
            **experiment,
            output_dir="./models/group_specific_layers/"
        )
