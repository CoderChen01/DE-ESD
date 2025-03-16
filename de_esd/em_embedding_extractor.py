from __future__ import annotations
import re
import typing as t
import abc

import torch
from torch._C import device
from transformers import PreTrainedModel
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoModel
from transformers import T5EncoderModel


class BaseEMExtractor(abc.ABC):
    model: PreTrainedModel
    tokenizer: t.Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    def __init__(
        self, pretrained_model_name: str, device: torch.device
    ) -> None:
        self.device = device
        self.pretrained_model_name = pretrained_model_name
        self._init_model()
        self._init_tokenizer()
        self.model.eval()

    def tokenize(self, text: str) -> t.Tuple[torch.Tensor, torch.Tensor]:
        entity_str_matches = list(re.finditer(r"\[START\](.*?)\[END\]", text))

        start_tokens = self.tokenizer.tokenize(
            text[: entity_str_matches[0].span()[0]]
        )
        start_word_masks = [0] * len(start_tokens)

        middle_tokens = []
        middle_word_masks = []
        for s, e in zip(entity_str_matches[:-1], entity_str_matches[1:]):
            if not middle_tokens:
                tmp = self.tokenizer.tokenize(s.group(1))
                middle_tokens.extend(tmp)
                middle_word_masks.extend([1] * len(tmp))
            tmp = self.tokenizer.tokenize(text[s.span()[1] : e.span()[0]])
            middle_tokens.extend(tmp)
            middle_word_masks.extend([0] * len(tmp))
            tmp = self.tokenizer.tokenize(e.group(1))
            middle_tokens.extend(tmp)
            middle_word_masks.extend([1] * len(tmp))

        if not middle_tokens:
            tmp = self.tokenizer.tokenize(entity_str_matches[0].group(1))
            middle_tokens.extend(tmp)
            middle_word_masks.extend([1] * len(tmp))

        end_tokens = self.tokenizer.tokenize(
            text[entity_str_matches[-1].span()[1] :]
        )
        end_word_masks = [0] * len(end_tokens)

        tokens, masks = self._truncate_token(
            start_tokens + middle_tokens + end_tokens,
            start_word_masks + middle_word_masks + end_word_masks,
        )

        ctx_tokens, ctx_word_masks = self._add_special_token(tokens, masks)

        return torch.as_tensor(
            self.tokenizer.convert_tokens_to_ids(ctx_tokens),
            device=self.device,
        ), torch.as_tensor(ctx_word_masks, device=self.device)

    def batch_tokenize(
        self, texts: t.List[str]
    ) -> t.Tuple[t.Dict[str, torch.Tensor], torch.Tensor]:
        batch_input_ids = []
        batch_word_masks = []
        for text in texts:
            input_ids, word_masks = self.tokenize(text)
            batch_input_ids.append(input_ids)
            batch_word_masks.append(word_masks)

        max_len = max(len(x) for x in batch_input_ids)
        batch_input_ids = torch.stack(
            [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - len(x)), mode="constant", value=0
                )
                for x in batch_input_ids
            ]
        )
        batch_attention_masks = (batch_input_ids != 0).int()
        batch_word_masks = torch.stack(
            [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - len(x)), mode="constant", value=0
                )
                for x in batch_word_masks
            ]
        )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_masks,
        }, batch_word_masks

    def _init_model(self) -> None:
        self.model = AutoModel.from_pretrained(self.pretrained_model_name).to(
            self.device
        )

    def _init_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name
        )

    @abc.abstractmethod
    def _add_special_token(
        self, tokens: t.List[str], word_masks: t.List[int]
    ) -> t.Tuple[t.List[str], t.List[int]]:
        pass

    def embed_entity_mention(self, text: str) -> torch.Tensor:
        input_ids, word_masks = self.tokenize(text)
        input_ids = input_ids.unsqueeze(0)
        word_masks = word_masks.unsqueeze(0)
        with torch.no_grad():
            output = self._model_encoding(input_ids=input_ids)
            em_embeddings = self._calc_em_embedding(
                output, word_masks
            ).squeeze(0)
        return em_embeddings

    def batch_embed_entity_mention(self, texts: t.List[str]) -> torch.Tensor:
        input_data, word_masks = self.batch_tokenize(texts)
        with torch.no_grad():
            output = self._model_encoding(**input_data)
            em_embeddings = self._calc_em_embedding(output, word_masks)
        return em_embeddings

    def _calc_em_embedding(
        self, output: t.Any, word_masks: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(output, "encoder_last_hidden_state"):
            output.encoder_last_hidden_state[~word_masks.bool()] = 0
            em_embedding = torch.mean(output.encoder_last_hidden_state, dim=1)
        else:
            output.last_hidden_state[~word_masks.bool()] = 0
            em_embedding = torch.mean(output.last_hidden_state, dim=1)
        return em_embedding

    @property
    def _model_encoding(self) -> PreTrainedModel:
        return self.model

    def _truncate_token(
        self, tokens: t.List[str], masks: t.List[int]
    ) -> t.Tuple[t.List[str], t.List[int]]:

        if not hasattr(self.model.config, "max_position_embeddings"):
            return tokens, masks

        max_len = self.model.config.max_position_embeddings

        num_need_to_trunc = 0
        if len(tokens) > max_len:
            num_need_to_trunc = len(tokens) - max_len + 2

        if num_need_to_trunc == 0:
            return tokens, masks

        new_tokens = []
        new_masks = []

        for t, m in zip(tokens, masks):
            if m == 0 and num_need_to_trunc != 0:
                num_need_to_trunc -= 1
                continue
            new_tokens.append(t)
            new_masks.append(m)

        return new_tokens, new_masks


# Decoder
class GPTLikeForEMExtractor(BaseEMExtractor):
    def _add_special_token(
        self, tokens: t.List[str], word_masks: t.List[int]
    ) -> t.Tuple[t.List[str], t.List[int]]:
        ctx_tokens = (
            [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]
        )
        ctx_word_masks = [0] + word_masks + [0]
        return ctx_tokens, ctx_word_masks


class GPT2ForEMExtractor(GPTLikeForEMExtractor):
    pass


class XLNetForEMExtractor(GPTLikeForEMExtractor):
    pass


class OPTForEMExtractor(GPTLikeForEMExtractor):
    pass


class GPTJForEMExtractor(GPTLikeForEMExtractor):
    pass


class BloomForEMExtractor(GPTLikeForEMExtractor):
    pass


# Encoder
class BertLikeForEMExtractor(BaseEMExtractor):
    def _add_special_token(
        self, tokens: t.List[str], word_masks: t.List[int]
    ) -> t.Tuple[t.List[str], t.List[int]]:
        ctx_tokens = (
            [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        )
        ctx_word_masks = [0] + word_masks + [0]
        return ctx_tokens, ctx_word_masks


class BertForEMExtractor(BertLikeForEMExtractor):
    pass


class RobertaForEMExtractor(BertLikeForEMExtractor):
    pass


class DistilBertForEMExtractor(BertLikeForEMExtractor):
    pass


class AlbertForEMExtractor(BertLikeForEMExtractor):
    pass


class DeBertaForEMExtractor(BertLikeForEMExtractor):
    pass


class LongFormerForEMExtractor(BertLikeForEMExtractor):
    pass


# Encoder-Decoder
class BartForEMExtractor(BaseEMExtractor):
    def _add_special_token(
        self, tokens: t.List[str], word_masks: t.List[int]
    ) -> t.Tuple[t.List[str], t.List[int]]:
        tokens = (
            [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]
        )
        word_masks = [0] + word_masks + [0]
        return tokens, word_masks


class PegasusEMExtractor(BaseEMExtractor):
    def _add_special_token(
        self, tokens: t.List[str], word_masks: t.List[int]
    ) -> t.Tuple[t.List[str], t.List[int]]:
        tokens = tokens + [self.tokenizer.eos_token]
        word_masks = word_masks + [0]
        return tokens, word_masks

    @property
    def _model_encoding(self) -> PreTrainedModel:
        return self.model.encoder


class T5ForEMExtractor(PegasusEMExtractor):
    def _init_model(self) -> None:
        self.model = T5EncoderModel.from_pretrained(self.pretrained_model_name).to(  # type: ignore
            self.device  # type: ignore
        )
