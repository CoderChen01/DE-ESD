from __future__ import annotations
import typing as t
import abc
from collections import defaultdict
import logging
from typing import Any

import torch
import numpy as np

from de_esd.datatypes import SynSetEntry
from de_esd.em_embedding_extractor import (
    BertForEMExtractor,
)
from de_esd.model.modeling_de_esd import (
    DeESDModel,
    DeESDModelOutput,
)

from .base import SynsetCluster, synset_cluster_register

if t.TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@synset_cluster_register("de-esd")
class DeESDCluster(SynsetCluster):
    need_model = True

    def __init__(self, threshold: float) -> None:
        super().__init__(threshold)

        self._synset_all_hidden_states: t.Dict[int, t.List[torch.Tensor]] = (
            defaultdict(list)
        )

    def new_synonym_set(self, entry: SynSetEntry) -> None:
        self._synsets[self.class_num].append(entry)

        if self.class_num in self._synset_all_hidden_states:
            return

        synset_embeds = self._extract_multi_embeddings_em(entry)
        self._synset_all_hidden_states[self.class_num].extend(
            list(synset_embeds)
        )

    def add_new_entry(self, entry: SynSetEntry) -> None:
        # The first input is directly treated as a new synonym set
        if self.class_num == 0:
            self.new_synonym_set(entry)
            return

        # Otherwise, we need to check if the entry should be added to an existing synonym set
        cur_entry_embeds = self._extract_multi_embeddings_em(entry).unsqueeze(
            0
        )
        max_score = -np.inf
        max_class_label = -1
        for (
            class_label,
            synset_embeds,
        ) in self._synset_all_hidden_states.items():
            synset_embeds_tensor = (
                torch.vstack(synset_embeds).unsqueeze(0).to(device=self.device)
            )

            model_rv: DeESDModelOutput = self._model(
                synset_embeds=synset_embeds_tensor,
                candidate_embeds=cur_entry_embeds,
            )
            assert model_rv.logits is not None

            score = t.cast(
                float, torch.sigmoid(model_rv.logits.cpu().detach()).item()
            )
            if score > max_score:
                max_score = score
                max_class_label = class_label

        if max_score < self.threshold:
            self._synset_all_hidden_states[self.class_num].extend(
                list(cur_entry_embeds.squeeze(0))
            )
            self.new_synonym_set(entry)
        else:
            self._synset_all_hidden_states[max_class_label].extend(
                list(cur_entry_embeds.squeeze(0))
            )
            self._synsets[max_class_label].append(entry)

    def load_model(self, model_path: str) -> None:
        if DeESDCluster.static_model is not None:
            self._model = DeESDCluster.static_model
            return
        DeESDCluster.static_model = DeESDModel.to(
            t.cast(DeESDModel, DeESDModel.from_pretrained(model_path)),
            device=self.device,
        )
        if self._em_embedding_extractor is not None:
            logger.warning(
                "EM embedding extractor is not None, will be overwritten"
            )
        DeESDCluster.load_em_embedding_extractor(
            DeESDCluster.static_model.config.em_extractor_name
        )
        DeESDCluster.static_model.eval()
        self._model = DeESDCluster.static_model
