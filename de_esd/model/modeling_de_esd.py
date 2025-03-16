from __future__ import annotations
import typing as t
import dataclasses
from math import floor

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import ModelOutput
from transformers import PretrainedConfig, PreTrainedModel
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from .configuration_de_esd import DeESDConfig


@dataclasses.dataclass
class DeESDModelOutput(ModelOutput):
    logits: t.Optional[torch.Tensor] = None
    loss: t.Optional[torch.Tensor] = None


class DeESDPretrainedModel(PreTrainedModel):
    config_class = DeESDConfig
    base_model_prefix = "synset_expansion"
    supports_gradient_checkpointing = True
    main_input_name = "synset_embeds"

    def __init__(self, config: DeESDConfig, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        self.synset_set_encoders = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    **config.synset_encoder_config, batch_first=True
                )
                for _ in range(config.num_synset_enc_layers)
            ]
        )
        self.pseudo_set_encoders = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    **config.pseudo_set_encoder_config, batch_first=True
                )
                for _ in range(config.num_pseudo_set_enc_layers)
            ]
        )

    def get_synset_all_hidden_states(
        self,
        synset_embeds: torch.Tensor,
        attention_masks: t.Optional[torch.Tensor] = None,
        src_key_padding_mask: t.Optional[torch.Tensor] = None,
    ) -> t.List[torch.Tensor]:
        projected_synset_embeds = synset_embeds

        all_synset_hidden_states: t.List[torch.Tensor] = []

        for synset_encoder in self.synset_set_encoders:
            projected_synset_embeds = synset_encoder(
                src=projected_synset_embeds,
                src_mask=attention_masks,
                src_key_padding_mask=src_key_padding_mask,
            )
            all_synset_hidden_states.append(projected_synset_embeds)

        return all_synset_hidden_states

    def get_pseudo_set_all_hidden_states(
        self, synset_embeds: torch.Tensor, candidate_embeds: torch.Tensor
    ) -> t.List[torch.Tensor]:
        pseudo_set_embeds = torch.cat([synset_embeds, candidate_embeds], dim=1)

        all_pseudo_set_hidden_states: t.List[torch.Tensor] = []

        for pseudo_encoder in self.pseudo_set_encoders:
            pseudo_set_hidden_states = pseudo_encoder(
                src=pseudo_set_embeds,
            )
            all_pseudo_set_hidden_states.append(pseudo_set_hidden_states)

        return all_pseudo_set_hidden_states

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SetScorer(nn.Module):
    def __init__(self, config: DeESDConfig) -> None:
        super().__init__()
        d_model = config.embedding_dim
        scorer_hidden_dim = config.set_scorer_config["hidden_dim"]
        scorer_dropout = config.set_scorer_config["dropout"]
        self.proj = nn.Sequential(
            nn.Linear(d_model, scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(scorer_hidden_dim, floor(scorer_hidden_dim / 2)),
            nn.ReLU(),
            nn.Dropout(scorer_dropout),
            nn.Linear(floor(scorer_hidden_dim / 2), 1),
        )

    def forward(self, x) -> torch.Tensor:
        return self.proj(x)


class DeESDModel(DeESDPretrainedModel):
    def __init__(self, config: DeESDConfig) -> None:
        super().__init__(config)
        self.config = config

        self.scorer = SetScorer(config)

    def get_scores(
        self,
        features_0: torch.Tensor,
        features_1: torch.Tensor,
    ) -> torch.Tensor:
        scores0: torch.Tensor = self.scorer(features_0)
        scores1: torch.Tensor = self.scorer(features_1)

        scores_diff = scores1 - scores0

        return scores_diff

    def pooling_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # return hidden_states[:, 0, :]
        return torch.mean(hidden_states, dim=1)

    def forward(
        self,
        synset_embeds: torch.Tensor,
        candidate_embeds: torch.Tensor,
        labels: t.Optional[torch.Tensor] = None,
    ) -> DeESDModelOutput:
        all_synset_hidden_states = self.get_synset_all_hidden_states(
            synset_embeds,
        )
        all_pseudo_set_hidden_states = self.get_pseudo_set_all_hidden_states(
            synset_embeds, candidate_embeds
        )

        synset_features = self.pooling_features(all_synset_hidden_states[-1])
        all_features = self.pooling_features(all_pseudo_set_hidden_states[-1])

        scores_diff = self.get_scores(synset_features, all_features)

        if labels is None:
            return DeESDModelOutput(logits=scores_diff)

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(scores_diff.view(-1), labels.view(-1).float())

        return DeESDModelOutput(logits=scores_diff, loss=loss)
