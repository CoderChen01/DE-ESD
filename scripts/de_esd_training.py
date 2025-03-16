from __future__ import annotations
import os
import sys
import re
import logging
import typing as t
from pathlib import Path
import dataclasses
from functools import partial

import numpy as np
import torch
import datasets
import evaluate
import transformers
from transformers import (
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_dataset, Features, Sequence, Value, ClassLabel


from de_esd.model.modeling_de_esd import DeESDModel
from de_esd.model.configuration_de_esd import (
    SetScorerConfig,
    DeESDConfig,
    TransformerEncoderConfig,
)
from de_esd.em_embedding_extractor import BaseEMExtractor
from de_esd.utils import embed_entity_mentions
from de_esd.datatypes import EM_EXTRACTOR_INFO

if t.TYPE_CHECKING:
    from datasets import DatasetDict
    from torch.utils.data import Dataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.1")

require_version(
    "datasets>=2.14.6",
    "To fix: poetry install",
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DatasetArguments:
    dataset_path: str = dataclasses.field(
        metadata={"help": "Path to the dataset to use."}
    )


@dataclasses.dataclass
class ModelArguments:
    model_name_or_path: t.Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: t.Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: t.Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = dataclasses.field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    initializer_range: float = dataclasses.field(
        default=0.02,
        metadata={
            "help": "The standard deviation of the truncated_normal_initializer."
        },
    )
    em_extractor_name: str = dataclasses.field(
        default="bert-base",
        metadata={
            "help": "The name of the pretrained model for EM embedding extraction."
        },
    )
    temperature: float = dataclasses.field(
        default=1.0,
        metadata={"help": "The temperature of the softmax."},
    )


@dataclasses.dataclass
class DualEncoderArguments:
    _argument_group_name = "dual_encoder"

    num_layers: int = dataclasses.field(
        default=2,
        metadata={"help": "The number of layers in the synset encoder."},
    )
    nhead: int = dataclasses.field(
        default=8,
        metadata={"help": "The number of heads in the synset encoder."},
    )
    dim_feedforward: int = dataclasses.field(
        default=2048,
        metadata={"help": "The hidden dimension of the synset encoder."},
    )
    dropout: float = dataclasses.field(
        default=0.1,
        metadata={"help": "The dropout rate of the synset encoder."},
    )
    layer_norm_eps: float = dataclasses.field(
        default=1e-5,
        metadata={
            "help": "The epsilon of the layer norm of the synset encoder."
        },
    )
    bias: bool = dataclasses.field(
        default=True,
        metadata={"help": "Whether to use bias in the synset encoder."},
    )
    norm_first: bool = dataclasses.field(
        default=True,
        metadata={"help": "Whether to use norm first in the synset encoder."},
    )


@dataclasses.dataclass
class SetScorerArguments:
    _argument_group_name = "set_scorer"

    set_scorer_hidden_dim: int = dataclasses.field(
        default=2048,
        metadata={"help": "The hidden dimension of the scorer."},
    )
    set_scorer_dropout: float = dataclasses.field(
        default=0.5,
        metadata={"help": "The dropout rate of the scorer."},
    )


def data_collator(
    examples: t.List[t.Dict[str, t.Any]],
    em_embedding_extractor: BaseEMExtractor,
) -> t.Dict[str, torch.Tensor]:
    rv = [
        embed_entity_mentions(example, em_embedding_extractor)
        for example in examples
    ]

    synset_embeds = [x["synset_embeds"] for x in rv]
    candidate_embeds = [x["candidate_embeds"] for x in rv]

    return {
        "synset_embeds": torch.stack(synset_embeds).to("cpu"),
        "candidate_embeds": torch.stack(candidate_embeds).to("cpu"),
        "labels": torch.as_tensor([x["labels"] for x in rv]).to("cpu"),
    }


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DualEncoderArguments, SetScorerArguments, DatasetArguments, TrainingArguments))  # type: ignore

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (
            model_args,
            dual_encoder_args,
            set_scorer_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            dual_encoder_args,
            set_scorer_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()

    data_args = t.cast(DatasetArguments, data_args)

    model_args = t.cast(ModelArguments, model_args)

    dual_encoder_args = t.cast(DualEncoderArguments, dual_encoder_args)
    set_scorer_args: SetScorerArguments = t.cast(
        SetScorerArguments, set_scorer_args
    )

    training_args = t.cast(TrainingArguments, training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if (
            last_checkpoint is None
            and len(os.listdir(training_args.output_dir)) > 0
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None
            and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Initialize model configuration
    if model_args.model_name_or_path is None:
        dual_encoder_config = TransformerEncoderConfig(
            nhead=dual_encoder_args.nhead,
            dim_feedforward=dual_encoder_args.dim_feedforward,
            dropout=dual_encoder_args.dropout,
            layer_norm_eps=dual_encoder_args.layer_norm_eps,
            bias=dual_encoder_args.bias,
            norm_first=dual_encoder_args.norm_first,
        )
        config = DeESDConfig(
            em_extractor_name=model_args.em_extractor_name,
            initializer_range=model_args.initializer_range,
            temperature=model_args.temperature,
            num_synset_enc_layers=dual_encoder_args.num_layers,
            synset_encoder_config=dual_encoder_config,
            num_pseudo_set_enc_layers=dual_encoder_args.num_layers,
            pseudo_set_encoder_config=dual_encoder_config,
            set_scorer_config=SetScorerConfig(
                hidden_dim=set_scorer_args.set_scorer_hidden_dim,
                dropout=set_scorer_args.set_scorer_dropout,
            ),
        )
    else:
        config = DeESDConfig.from_pretrained(model_args.model_name_or_path)
        config = t.cast(DeESDConfig, config)

    # Initialize or load the model
    if model_args.model_name_or_path is None:
        model = DeESDModel(config=config)
    else:
        model = DeESDModel.from_pretrained(
            model_args.model_name_or_path, config=config
        )

    model = t.cast(DeESDModel, model)

    # Load dataset
    if data_args.dataset_path is None:
        raise ValueError("dataset_path must be specified")
    features = Features(
        {
            "entity_id": Value("string"),
            "entity_ctx_set": Sequence(Value("string")),
            "entity_str_set": Sequence(Value("string")),
            "candidate_ctx": Sequence(Value("string")),
            "candidate_str": Value("string"),
            "label": ClassLabel(num_classes=2, names=["N", "Y"]),
        }
    )
    dataset = t.cast(
        "DatasetDict",
        load_dataset(
            "json",
            data_files={
                "train": str(Path(data_args.dataset_path) / "train.jsonl"),
                "val": str(Path(data_args.dataset_path) / "val.jsonl"),
            },
            features=features,
        ),
    )
    dataset = dataset.rename_columns(
        {
            "entity_ctx_set": "synset_embeds",
            "candidate_ctx": "candidate_embeds",
            "label": "labels",
        }
    )

    # Get the metric function
    acc_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    mae_metric = evaluate.load("mae")

    def preprocess_logits_for_metric(logits, labels) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        return preds

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(eval_predication: EvalPrediction):
        preds = np.squeeze(
            (
                eval_predication.predictions[0]
                if isinstance(eval_predication.predictions, tuple)
                else eval_predication.predictions
            )
        )
        pred_labels = (preds > 0.5).astype(int)
        labels = eval_predication.label_ids

        p = precision_metric.compute(
            predictions=pred_labels, references=labels, zero_division=0
        )
        r = recall_metric.compute(predictions=pred_labels, references=labels)
        f1 = f1_metric.compute(predictions=pred_labels, references=labels)
        acc = acc_metric.compute(predictions=pred_labels, references=labels)
        mae = mae_metric.compute(predictions=preds, references=labels)

        result = {}
        result.update(p if p is not None else {})
        result.update(r if r is not None else {})
        result.update(f1 if f1 is not None else {})
        result.update(acc if acc is not None else {})
        result.update(mae if mae is not None else {})

        return result

    train_dataset = dataset["train"]
    eval_dataset = dataset["val"]

    train_dataset = t.cast("Dataset", train_dataset)
    eval_dataset = t.cast("Dataset", eval_dataset)

    # Initialize our Trainer
    name, extractor_cls, dim = EM_EXTRACTOR_INFO[model_args.em_extractor_name]
    em_embedding_extractor = extractor_cls(name, device="cuda")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=partial(
            data_collator,
            em_embedding_extractor=em_embedding_extractor,
        ),
        preprocess_logits_for_metrics=preprocess_logits_for_metric,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
