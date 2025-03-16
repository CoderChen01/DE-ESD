import typing as t
from pathlib import Path
import json
import jsonlines
import logging
import datetime

import click
from tqdm import tqdm
import torch
from transformers import AutoConfig
import pandas as pd
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)

from de_esd.model.configuration_de_esd import DeESDConfig
from de_esd.clusters.base import get_synset_cluster
from de_esd.datatypes import SynSetEntry, SynSetEntryDict, EM_EXTRACTOR_INFO

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_offline_data(data_path: Path) -> t.List[SynSetEntry]:
    # load data
    with jsonlines.open(data_path, "r") as reader:
        return [SynSetEntry(**d) for d in reader]


def load_gt_data(gt_path: Path) -> t.Tuple[t.List[str], t.List[int]]:
    gt_data = pd.read_csv(gt_path)
    return gt_data["entity_mention"].tolist(), gt_data["entity_class"].tolist()


def eval_prf1(preds: t.List[int], gt: t.List[int]):
    # Pair-based clustering metrics
    def pair_set(labels):
        S = set()
        cluster_ids = np.unique(labels)
        for cluster_id in cluster_ids:
            cluster = np.where(labels == cluster_id)[0]
            n = len(cluster)  # number of elements in this cluster
            if n >= 2:
                for i in range(n):
                    for j in range(i + 1, n):
                        S.add((cluster[i], cluster[j]))
        return S

    F_S = pair_set(gt)
    F_K = pair_set(preds)
    if len(F_K) == 0:
        pair_recall = 0
        pair_precision = 0
        pair_f1 = 0
    else:
        common_pairs = len(F_K & F_S)
        pair_recall = common_pairs / len(F_S)
        pair_precision = common_pairs / len(F_K)
        eps = 1e-6
        pair_f1 = (
            2
            * pair_precision
            * pair_recall
            / (pair_precision + pair_recall + eps)
        )
    return pair_precision, pair_recall, pair_f1


def format_synsets_to_eval(
    synsets: t.Dict[int, t.List[SynSetEntry]], gt_entity_mentions: t.List[str]
) -> t.Tuple[t.List[str], t.List[int]]:
    entity_mentions: t.List[str] = []
    class_labels: t.List[int] = []
    for gem in gt_entity_mentions:
        em, entity_id = gem.split("||")
        tmp_entry = SynSetEntry(
            entity_str=em, entity_id=entity_id, entity_ctx=[]
        )
        has_gt_entity_mentions = False
        for class_label, synset in synsets.items():
            if tmp_entry in synset:
                entity_mentions.append(gem)
                class_labels.append(class_label)
                has_gt_entity_mentions = True
                break
        if not has_gt_entity_mentions:
            entity_mentions.append(gem)
            class_labels.append(-1)
    return entity_mentions, class_labels


def synset_clustering(
    cluster_name: str,
    threshold: t.Tuple[float, float, float],
    data_path: Path,
    save_path: Path,
    em_extractor_name: t.Optional[str] = None,
    model_path: t.Optional[Path] = None,
    gt_path: t.Optional[Path] = None,
) -> None:
    # load synset expansion predictor
    logger.info("Loading synset expansion predictor...")

    start_t, stop_t, step_t = threshold
    synset_expansion_clusters = []
    thresholds = []
    if stop_t > 0 and step_t > 0 and start_t < stop_t and start_t < stop_t:
        thresholds = np.arange(start_t, stop_t, step_t).tolist()
    else:
        thresholds.append(start_t)

    cluster_cls = get_synset_cluster(cluster_name)

    if em_extractor_name is not None:
        cluster_cls.em_extractor_name = em_extractor_name
        cluster_cls.load_em_embedding_extractor(em_extractor_name)

    for t in thresholds:
        cluster = cluster_cls(t)
        if cluster.need_model:
            if model_path is None:
                raise ValueError(
                    f"Model path is required for {cluster.__class__.__name__}"
                )
            cluster.load_model(model_path)
        synset_expansion_clusters.append(cluster)

    # prepare input
    logger.info("Loading offline data...")
    offline_data = load_offline_data(data_path)

    # create metric result file
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    metric_file_name = None
    if gt_path is not None:
        if model_path is not None:
            metric_file_name = (
                save_path
                / f"metric_{cluster_name}_{cluster_cls.em_extractor_name}_{model_path.stem}_{data_path.stem}_{datetime_now}.csv"
            )
        else:
            metric_file_name = (
                save_path
                / f"metric_{cluster_name}_{cluster_cls.em_extractor_name}_{data_path.stem}_{datetime_now}.csv"
            )

    # cluster
    logger.info("Clustering...")
    results = []
    for synset_expansion_cluster in synset_expansion_clusters:
        for d in tqdm(
            offline_data,
            f"Clustering, Threshold: {synset_expansion_cluster.threshold}",
        ):
            synset_expansion_cluster.add_new_entry(d)
            if synset_expansion_cluster.class_num != 0:
                logger.info(
                    f"Current number of synsets: {synset_expansion_cluster.class_num}"
                )

        # metric and saving
        if gt_path is None or metric_file_name is None:
            continue

        logger.info("Calculating metric...")
        gt_entity_mentions, gt_class_lables = load_gt_data(gt_path)

        _, class_labels = format_synsets_to_eval(
            synset_expansion_cluster.synsets, gt_entity_mentions
        )

        round_2 = lambda x: round(x * 100, 2)
        ari = adjusted_rand_score(
            labels_true=np.array(gt_class_lables),
            labels_pred=np.array(class_labels),
        )
        ari = round_2(ari)

        fmi = fowlkes_mallows_score(
            labels_true=np.array(gt_class_lables),
            labels_pred=np.array(class_labels),
        )
        fmi = round_2(fmi)

        nmi = normalized_mutual_info_score(
            labels_true=np.array(gt_class_lables),
            labels_pred=np.array(class_labels),
        )
        nmi = round_2(nmi)

        p, r, f1 = eval_prf1(class_labels, gt_class_lables)
        p = round_2(p)
        r = round_2(r)
        f1 = round_2(f1)

        result = {
            "Threshold": synset_expansion_cluster.threshold,
            "ARI": ari,
            "FMI": fmi,
            "NMI": nmi,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
        logger.info(
            f"Threshold: {synset_expansion_cluster.threshold}, ARI: {result['ARI']}, FMI: {result['FMI']}, NMI: {result['NMI']}"
        )
        results.append(result)

    if metric_file_name is not None:
        metric_df = pd.DataFrame(results)

        max_fmi_index = metric_df["FMI"].idxmax()
        metric_df["IsMaxFMI"] = metric_df.index == max_fmi_index

        max_ari_index = metric_df["ARI"].idxmax()
        metric_df["IsMaxARI"] = metric_df.index == max_ari_index

        max_nmi_index = metric_df["NMI"].idxmax()
        metric_df["IsMaxNMI"] = metric_df.index == max_nmi_index

        metric_with_max_fmi = metric_df.loc[max_fmi_index].to_dict()
        logger.info(f"Metric with max FMI: {metric_with_max_fmi}")

        metric_df.to_csv(metric_file_name, index=False)

    # save result
    logger.info("Saving clustering result...")
    if model_path is not None:
        result_save_dir = (
            save_path
            / f"result-{cluster_name}-{cluster_cls.em_extractor_name}-{model_path.stem}-{data_path.stem}-{datetime_now}"
        )
    else:
        result_save_dir = (
            save_path
            / f"result-{cluster_name}-{cluster_cls.em_extractor_name}-{data_path.stem}-{datetime_now}"
        )
    result_save_dir.mkdir(parents=True, exist_ok=True)
    for threshold, synset_expansion_cluster in zip(
        thresholds, synset_expansion_clusters
    ):
        f = open(
            result_save_dir / f"result-threshold-{threshold}.csv",
            "w",
        )
        for class_label, synset in synset_expansion_cluster.synsets.items():
            for entry in synset:
                if entry.entity_id is None:
                    f.write(f"{entry.entity_str},{class_label}\n")
                else:
                    f.write(
                        f"{entry.entity_str}||{entry.entity_id},{class_label}\n"
                    )
        f.close()


@click.command()
@click.option("-c", "--cluster-name", required=True, type=str)
@click.option(
    "-t",
    "--threshold",
    type=(float, float, float),
)
@click.option(
    "-d",
    "--data-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s", "--save-path", type=click.Path(file_okay=False, path_type=Path)
)
@click.option("-e", "--em-extractor-name", type=str, required=False)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    required=False,
)
@click.option(
    "-g",
    "--gt-path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def cli(
    cluster_name: str,
    threshold: t.Tuple[float, float, float],
    data_path: Path,
    save_path: Path,
    em_extractor_name: t.Optional[str],
    model_path: t.Optional[Path],
    gt_path: t.Optional[Path],
):
    save_path.mkdir(parents=True, exist_ok=True)
    synset_clustering(
        cluster_name,
        threshold,
        data_path,
        save_path,
        em_extractor_name,
        model_path,
        gt_path,
    )


if __name__ == "__main__":
    cli()
