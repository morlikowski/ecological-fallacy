import json
from pathlib import Path
import pandas as pd
import numpy as np
import datasets
import transformers

SOCIODEMOGRAPHICS = [
    'gender',
    'identify_as_transgender',
    'race',
    'age_range',
    'lgbtq_status',
    'religion_important',
    'is_parent',
    'political_affilation',
    'education'
]

OTHER_ATTRIBUTES = [ 
    'technology_impact', 
    'uses_media_social', 
    'uses_media_news', 
    'uses_media_forums',
    'toxic_comment_problem'
]

EXPERIENCE = [ 
    'personally_seen_toxic_content', 
    'personally_been_target'
]
class Dataset(object):

    def __init__(self, features, annotators_mapping, sociodemographic_mapping) -> None:
        self.annotators_mapping = annotators_mapping
        self.sociodemographic_mapping = sociodemographic_mapping
        self.features = features

    @classmethod
    def load(
            cls,
            path='kumar_dataset.json',
            metadata_path='kumar_dataset.json',
            do_majority_aggregation = False,
            n=None,
            model_name = 'distilbert-base-uncased'
        ):
        df = cls._read_df(path, n = n, do_majority_aggregation = do_majority_aggregation)

        if do_majority_aggregation:
            dataset = datasets.Dataset.from_pandas(df)
            features = cls._to_features(
                dataset,
                None,
                model_name = model_name
            )
            return cls(features, None, None)
        else:
            annotator_ids = df['worker_id'].explode().unique().tolist()
            sociodemographic_mapping = cls.read_sociodemographic_mapping(
                metadata_path,
                annotator_ids
            )
            annotators_mapping = {annotator: index for index, annotator in enumerate(annotator_ids)}
            df['worker_id'] = df['worker_id'].apply(lambda x: [annotators_mapping[a] for a in x])
            annotations_max_length = df['worker_id'].apply(len).max()
            df['worker_id'] = df['worker_id'].apply(lambda x: x + [-1] * (annotations_max_length - len(x)))
            df['toxic'] = df['toxic'].apply(lambda x: x + [-1] * (annotations_max_length - len(x)))
            dataset = datasets.Dataset.from_pandas(df)
            features = cls._to_features(
                dataset,
                annotators_mapping,
                model_name = model_name
            )
            return cls(features, annotators_mapping, sociodemographic_mapping)

    @classmethod
    def _read_df(cls, path, n = None, do_majority_aggregation: bool = False):
        path = Path(path)
        if path.suffix == '.json':
            df = cls._read_raw_dataset(
                    path=path
            )
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        
        df = cls._group_labels(df, do_majority_aggregation)
        # we want to make sure that the n applies to the level of documents, not individual annotations
        if n:
            df = df.iloc[:n]
        return df

    @classmethod
    def _read_raw_dataset(
            cls,
            path='kumar_dataset.json'
        ):
        with open(path) as f:
            records = []
            for index, line in enumerate(f):
                line_dict = json.loads(line)
                for annotation in line_dict['ratings']:
                    record = {
                        'comment': line_dict['comment'],
                        'id': index,
                        'comment_id': line_dict['comment_id'],
                        'source': line_dict['source']
                    }
                    record.update(annotation)
                    records.append(record)
        df = pd.DataFrame(records)
        df['toxic'] = df['toxic_score'].apply(lambda x: 1 if x > 1 else 0)
        return df

    @classmethod
    def _group_labels(
        cls,
        df,
        do_majority_aggregation: bool = False
    ):  
        annotations = df[['id', 'comment', 'worker_id', 'toxic']] \
                        .groupby(['id']) \
                        .agg({
                            'comment': 'first', 
                            'worker_id': list, 
                            'toxic': (lambda x: list(pd.Series.mode(x))) if do_majority_aggregation else list
                        })
        return annotations
    
    @classmethod
    def _to_features(
        cls,
        dataset,
        annotators_mapping,
        model_name : str
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        max_length = 512
        
        features = dataset.map(
            FeatureConversion(
                tokenizer,
                max_length,
                annotators_mapping
            ),
            batched=True,
            load_from_cache_file=False,
        )

        if isinstance(features, datasets.DatasetDict):
            column_names = features['train'].column_names
        else:
            column_names = features.column_names
            
        to_remove_columns = [column for column in column_names if column not in ['input_ids', 'attention_mask', 'labels', 'annotator_indecies']]
        features = features.remove_columns(to_remove_columns)
        
        features.set_format(
            type="torch", 
            columns=['input_ids', 'attention_mask', 'labels', 'annotator_indecies'] if annotators_mapping else ['input_ids', 'attention_mask', 'labels']
        )
        
        return features

    @classmethod
    def read_sociodemographic_mapping(
            cls,
            path,
            annotators=[]
        ):
        df = cls._read_raw_dataset(
            path=path
        )
        mapping_df = df[['worker_id'] + SOCIODEMOGRAPHICS]
        # Attribute value disambiguation, described in paper Appendix A.1
        mapping = mapping_df.groupby('worker_id').agg(pd.Series.mode).applymap(lambda x: 'Prefer not to say' if type(x) is np.ndarray else x)
        if annotators:
            mapping = mapping[mapping.index.isin(annotators)]
        return mapping.to_dict(orient='index')

class FeatureConversion(object):

    def __init__(self, tokenizer, max_length, annotators_mapping):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.annotators_mapping = annotators_mapping

    def __call__(self, example_batch):
            inputs = example_batch['comment']
            features = self.tokenizer(
                inputs,
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True
            )
            features["labels"] = example_batch['toxic']
            if self.annotators_mapping:
                features["annotator_indecies"] = example_batch['worker_id']
            return features