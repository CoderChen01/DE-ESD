from __future__ import annotations
import typing as t
import dataclasses

from de_esd.em_embedding_extractor import (
    GPT2ForEMExtractor,
    XLNetForEMExtractor,
    OPTForEMExtractor,
    BertForEMExtractor,
    RobertaForEMExtractor,
    DistilBertForEMExtractor,
    AlbertForEMExtractor,
    DeBertaForEMExtractor,
    LongFormerForEMExtractor,
    PegasusEMExtractor,
    BartForEMExtractor,
    T5ForEMExtractor,
)


EM_EXTRACTOR_INFO = {
    # Encoder
    "bert-base": ("bert-base-uncased", BertForEMExtractor, 768),
    "bert-large": ("bert-large-uncased", BertForEMExtractor, 1024),
    "roberta-base": ("roberta-base", RobertaForEMExtractor, 768),
    "roberta-large": ("roberta-large", RobertaForEMExtractor, 1024),
    "deberta-base": ("microsoft/deberta-base", DeBertaForEMExtractor, 768),
    "deberta-large": ("microsoft/deberta-large", DeBertaForEMExtractor, 1024),
    # Encoder-Decoder
    "pegasus-base": ("google/pegasus-x-base", PegasusEMExtractor, 768),
    "pegasus-large": ("google/pegasus-large", PegasusEMExtractor, 1024),
    "bart-base": ("facebook/bart-base", BartForEMExtractor, 768),
    "bart-large": ("facebook/bart-large", BartForEMExtractor, 1024),
    "t5-base": ("t5-base", T5ForEMExtractor, 768),
    "t5-large": ("t5-large", T5ForEMExtractor, 1024),
}


class EntityInfoDict(t.TypedDict):
    entity_id: str
    entity_str: str
    entity_ctx: t.List[str]


class RawDataDict(t.TypedDict):
    entity_info_data: t.Dict[str, t.Dict[str, EntityInfoDict]]
    sent_data: t.Dict[str, str]


class DataRowDict(t.TypedDict):
    entity_id: str
    entity_str_set: t.List[str]
    entity_ctx_set: t.List[str]
    candidate_str: str
    candidate_ctx: t.List[str]
    label: t.Literal["Y", "N"]


class SynSetEntryDict(t.TypedDict):
    entity_str: str
    entity_ctx: t.List[str]
    entity_id: t.Optional[str]


@dataclasses.dataclass
class SynSetEntry:
    entity_str: str
    entity_ctx: t.List[str]
    entity_id: t.Optional[str] = None

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, SynSetEntry):
            __value = t.cast(SynSetEntry, __value)
            if self.entity_id is not None and __value.entity_id is not None:
                return (
                    self.entity_id == __value.entity_id
                    and self.entity_str == __value.entity_str
                )
            return self.entity_str == __value.entity_str
        else:
            return False
