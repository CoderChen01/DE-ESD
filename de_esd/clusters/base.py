from __future__ import annotations
import typing as t
import abc
from collections import defaultdict
import logging

import torch
import numpy as np

from de_esd.datatypes import SynSetEntry, EM_EXTRACTOR_INFO
from de_esd.em_embedding_extractor import BaseEMExtractor

logger = logging.getLogger(__name__)

SynSetsType = t.Dict[int, t.List[SynSetEntry]]


class SynsetCluster(abc.ABC):
    need_model: t.ClassVar[bool] = False
    static_model: t.ClassVar[t.Any] = None

    em_extractor_name: t.ClassVar[t.Optional[str]] = None
    _em_embedding_extractor: t.ClassVar[t.Optional[BaseEMExtractor]] = None

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._synsets: SynSetsType = defaultdict(list)
        # self._model = self.load_model()

    def _extract_multi_embeddings_em(self, entry: SynSetEntry) -> torch.Tensor:
        if self._em_embedding_extractor is None:
            raise ValueError(
                f"EM embedding extractor is not initialized for {self.__class__.__name__}"
            )
        return self._em_embedding_extractor.batch_embed_entity_mention(
            entry.entity_ctx
        ).to(device=self.device)

    @property
    def synsets(self) -> SynSetsType:
        return self._synsets

    @property
    def class_num(self) -> int:
        return len(self._synsets)

    @abc.abstractmethod
    def new_synonym_set(self, entry: SynSetEntry) -> None:
        pass

    @abc.abstractmethod
    def add_new_entry(self, entry: SynSetEntry) -> None:
        pass

    @abc.abstractmethod
    def load_model(self, *args, **kwargs) -> t.Any:
        pass

    @classmethod
    def load_em_embedding_extractor(
        cls: t.Type[SynsetCluster], em_extractor_name: str
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name, e_cls, _ = EM_EXTRACTOR_INFO[em_extractor_name]
        cls.em_extractor_name = em_extractor_name
        cls._em_embedding_extractor = e_cls(name, device=device)


SYNSET_CLUSTERS = {}


def synset_cluster_register(
    name: str,
) -> t.Callable[[t.Type[SynsetCluster]], t.Type[SynsetCluster]]:
    def _decorator(cls: t.Type[SynsetCluster]) -> t.Type[SynsetCluster]:
        if name in SYNSET_CLUSTERS:
            raise ValueError(f"Duplicate SynsetCluster name: {name}")
        SYNSET_CLUSTERS[name] = cls
        return cls

    return _decorator


def get_synset_cluster(name: str) -> t.Type[SynsetCluster]:
    if name not in SYNSET_CLUSTERS:
        raise ValueError(f"Unknown SynsetCluster name: {name}")
    return SYNSET_CLUSTERS[name]
