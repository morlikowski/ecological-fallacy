import torch
import torch.nn as nn

from typing import (
    Optional, 
    Tuple,    
    Union
)

from torch.nn import CrossEntropyLoss

from tqdm import tqdm

from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import AutoModel
class PerAnnotatorModelForSequenceClassification(torch.nn.Module):
    def __init__(
            self, 
            checkpoint, 
            annotators_mapping,
            label_weights, 
            groups_mapping,
            num_labels=2,
            use_return_dict=True,
        ):
        super().__init__()
        self.num_labels = num_labels
        self.use_return_dict = use_return_dict

        self.pretrained_model = AutoModel.from_pretrained(checkpoint)
        self.dim = self.pretrained_model.config.hidden_size

        self.annotators_mapping = annotators_mapping
        self.groups_mapping = groups_mapping

        self.groups = nn.ModuleDict()
        if groups_mapping:
            example_item = next(iter(groups_mapping.values()))
            attributes = tuple(example_item.keys())
            for attribute in attributes:
                self.groups[attribute] = nn.ModuleDict()
                unique_group_indecies = set([groups[attribute] for groups in groups_mapping.values()])
                for index in unique_group_indecies:
                    self.groups[attribute][str(index)] = nn.Linear(self.dim, self.dim)

        self.heads = nn.ModuleDict()
        for annotator_index in tqdm(annotators_mapping.values()):
            self.heads[str(annotator_index)] = nn.Linear(self.dim, self.num_labels)

        self.label_weights = label_weights

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        annotator_indecies: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict
        batch_losses = []
        batch_logits = []
        all_annotator_indecies = set(self.annotators_mapping.values())
        for batch_i in range(labels.shape[0]):
            pretrained_output = self.pretrained_model(
                input_ids=input_ids[batch_i:batch_i+1,:],
                attention_mask=attention_mask[batch_i:batch_i+1,:],
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_state = pretrained_output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)

            example_losses = []
            example_logits = torch.full( (1, len(all_annotator_indecies), self.num_labels), float('nan'), device=next(self.parameters()).device)

            if not self.training:
                # will be ordered correctly based on insertion order from index calculation
                # but can not be sure if unknown calling code, hence explicit ordering
                all_annotator_indecies = list(set(self.annotators_mapping.values()))
                all_annotator_indecies = sorted(all_annotator_indecies)
                for annotator_index in all_annotator_indecies:
                    group_output = pooled_output
                    if self.groups_mapping:
                        for attribute, attribute_layers in self.groups.items():
                            group = self.groups_mapping[annotator_index][attribute]
                            group_output = attribute_layers[str(group)](group_output)
                    logits = self.heads[str(annotator_index)](group_output)  # (bs, num_labels)
                    example_logits[0, annotator_index] = logits
            else:
                not_missing_mask = annotator_indecies[batch_i] > -1
                annotator_indecies_on_example = annotator_indecies[batch_i][not_missing_mask].int().tolist()
                for i, annotator_index in enumerate(annotator_indecies_on_example):
                    group_output = pooled_output
                    if self.groups_mapping:
                        for attribute, attribute_layers in self.groups.items():
                            group = self.groups_mapping[annotator_index][attribute]
                            group_output = attribute_layers[str(group)](group_output)
                    logits = self.heads[str(annotator_index)](group_output)  # (bs, num_labels)
                    if all(torch.isnan(example_logits[0, annotator_index])):
                        # if existing logits are nan, just set our logits
                        example_logits[0, annotator_index] = logits
                    else:
                        raise ValueError('Duplicate annotator indecies')

                    loss = None
                    if labels is not None:
                        # Only supports single label classification
                        loss_fct = CrossEntropyLoss(
                            weight=self.label_weights[str(annotator_index)].to(next(self.parameters()).device), 
                            reduction='none'
                        )
                        label = labels[batch_i, i]
                        label = label.to(next(self.parameters()).device, dtype=torch.long)
                        loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
                        example_losses.append(loss)

            if example_losses:
                batch_losses.append(
                    torch.mean(torch.stack(example_losses)).squeeze()
                )
            batch_logits.append(example_logits)
        
        loss_avg = torch.mean(torch.stack(batch_losses)).squeeze() if batch_losses else torch.tensor(0.0)

        if not return_dict:
            output = (torch.cat(batch_logits),) + pretrained_output[1:]
            return ((loss_avg,) + output) if loss_avg else output
        
        return SequenceClassifierOutput(
            loss=loss_avg,
            logits=torch.cat(batch_logits),
            hidden_states=pretrained_output.hidden_states,
            attentions=pretrained_output.attentions
        )

class FixedPredBaseline(torch.nn.Module):
    def __init__(
            self,
            label_index,
            annotators_mapping,
            num_labels=2,
            use_return_dict=True,
        ):
        super().__init__()
        self.label_index = label_index
        self.num_labels = num_labels
        self.n_annotators =  len(set(annotators_mapping.values()))
        self.use_return_dict = use_return_dict

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        annotator_indecies: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if not self.training:
            n_examples = input_ids.shape[0]
            logits = torch.zeros((n_examples, self.n_annotators, self.num_labels))
            logits[:,:, self.label_index] = 1.0
        else:
            raise Exception('Constant prediction baseline is not meant to be trained.')

        if not return_dict:
            return (logits,)
        
        return SequenceClassifierOutput(
            logits=logits,
            loss = torch.zeros(1)
        )