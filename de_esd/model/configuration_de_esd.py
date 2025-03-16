import typing as t
import dataclasses
from enum import Enum

from transformers import PretrainedConfig
from transformers.utils import logging

from de_esd.datatypes import EM_EXTRACTOR_INFO

logger = logging.get_logger(__name__)


@dataclasses.dataclass
class TransformerEncoderConfig:
    nhead: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    bias: bool = True
    norm_first: bool = True
    d_model: int = dataclasses.field(init=False, default=768)

    def set_model_dim(self, d_model: int) -> None:
        self.d_model = d_model


@dataclasses.dataclass
class SetScorerConfig:
    hidden_dim: int = 2048
    dropout: float = 0.3


class DeESDConfig(PretrainedConfig):
    model_type = "de_esd"

    def __init__(
        self,
        em_extractor_name: str = "bert-base",
        initializer_range: float = 0.02,
        temperature: float = 1.0,
        num_synset_enc_layers: int = 1,
        num_pseudo_set_enc_layers: int = 1,
        synset_encoder_config: t.Optional[TransformerEncoderConfig] = None,
        pseudo_set_encoder_config: t.Optional[TransformerEncoderConfig] = None,
        set_scorer_config: t.Optional[SetScorerConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = EM_EXTRACTOR_INFO[em_extractor_name][2]

        # synset encoder
        if synset_encoder_config is None:
            synset_encoder_config = TransformerEncoderConfig()
            logger.info(
                "synset_tf_encoder_config is None. Initializing the TransformerEncoderConfig with default values."
            )
        if dataclasses.is_dataclass(synset_encoder_config):
            synset_encoder_config.set_model_dim(self.embedding_dim)
            self.synset_encoder_config = dataclasses.asdict(
                synset_encoder_config
            )
        else:
            self.synset_encoder_config = t.cast(t.Dict, synset_encoder_config)
        self.num_synset_enc_layers = num_synset_enc_layers

        # pseudo-set encoder
        if pseudo_set_encoder_config is None:
            pseudo_set_encoder_config = TransformerEncoderConfig()
            logger.info(
                "pseudo_set_tf_encoder_config is None. Initializing the TransformerEncoderConfig with default values."
            )
        if dataclasses.is_dataclass(pseudo_set_encoder_config):
            pseudo_set_encoder_config.set_model_dim(self.embedding_dim)
            self.pseudo_set_encoder_config = dataclasses.asdict(
                pseudo_set_encoder_config
            )
        else:
            self.pseudo_set_encoder_config = t.cast(
                t.Dict, pseudo_set_encoder_config
            )
        self.num_pseudo_set_enc_layers = num_pseudo_set_enc_layers

        # set scorer
        if set_scorer_config is None:
            set_scorer_config = SetScorerConfig()
            logger.info(
                "set_scorer_config is None. Initializing the ScorerConfig with default values."
            )
        if dataclasses.is_dataclass(set_scorer_config):
            self.set_scorer_config = dataclasses.asdict(set_scorer_config)
        else:
            self.set_scorer_config = t.cast(t.Dict, set_scorer_config)

        self.em_extractor_name = em_extractor_name
        self.initializer_range = initializer_range
        self.temperature = temperature
