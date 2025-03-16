from __future__ import annotations
import re
import threading
import typing as t

import chromadb
from chromadb.api import ClientAPI
from chromadb import EmbeddingFunction, Embeddings
import torch
import torch.nn.functional as F

from transformers.utils import PaddingStrategy, TensorType
from transformers.tokenization_utils import BatchEncoding

if t.TYPE_CHECKING:
    from de_esd.em_embedding_extractor import BaseEMExtractor


def embed_entity_mentions(
    example,
    extractor: BaseEMExtractor,
) -> t.Dict[str, torch.Tensor]:
    all_sents = [
        *example["synset_embeds"],
        *example["candidate_embeds"],
    ]
    synset_masks = torch.as_tensor(
        [True] * len(example["synset_embeds"])
        + [False] * len(example["candidate_embeds"])
    )

    all_embeds = extractor.batch_embed_entity_mention(all_sents)

    return {
        "synset_embeds": all_embeds[synset_masks],
        "candidate_embeds": all_embeds[~synset_masks],
        "labels": example["labels"],
    }
