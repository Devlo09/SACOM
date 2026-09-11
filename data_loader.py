

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _numeric_key(name: str) -> Tuple[int, str]:
    suffix = name.split("_", 1)[1]
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _stratified_split(
    labels: np.ndarray,
    ratios: Sequence[float],
    seed: int,
) -> Dict[str, np.ndarray]:
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split_ratios must contain train/val/test values summing to 1.")
    rng = np.random.default_rng(seed)
    split_parts: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    for class_value in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_value)
        rng.shuffle(class_indices)
        if class_indices.size < 3:
            raise ValueError("Every class needs at least three samples for a 7:1:2 split.")
        train_count = max(1, int(np.floor(class_indices.size * ratios[0])))
        val_count = max(1, int(np.floor(class_indices.size * ratios[1])))
        if train_count + val_count >= class_indices.size:
            train_count = class_indices.size - 2
            val_count = 1
        split_parts["train"].append(class_indices[:train_count])
        split_parts["val"].append(class_indices[train_count : train_count + val_count])
        split_parts["test"].append(class_indices[train_count + val_count :])
    result = {}
    for split_name, pieces in split_parts.items():
        values = np.concatenate(pieces).astype(np.int64)
        rng.shuffle(values)
        result[split_name] = values
    return result


def _stratified_group_split(
    labels: np.ndarray,
    record_ids: np.ndarray,
    ratios: Sequence[float],
    seed: int,
) -> Dict[str, np.ndarray]:


    record_ids = np.asarray(record_ids).reshape(-1)
    if record_ids.shape[0] != labels.shape[0]:
        raise ValueError("record_ids must contain one value per sample.")
    unique_records, sample_to_record = np.unique(record_ids, return_inverse=True)
    record_labels = np.empty(unique_records.size, dtype=labels.dtype)
    for record_index in range(unique_records.size):
        values = np.unique(labels[sample_to_record == record_index])
        if values.size != 1:
            raise ValueError(
                "Each acquisition record must contain exactly one class for "
                "stratified record-level splitting."
            )
        record_labels[record_index] = values[0]

    record_splits = _stratified_split(record_labels, ratios, seed)
    return {
        split: np.flatnonzero(np.isin(sample_to_record, selected_records)).astype(np.int64)
        for split, selected_records in record_splits.items()
    }


def _standardize(array: np.ndarray, train_indices: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    train = array[train_indices]
    if array.ndim == 2:
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
    elif array.ndim == 3:
        mean = train.mean(axis=(0, 2), keepdims=True)
        std = train.std(axis=(0, 2), keepdims=True)
    else:
        raise ValueError("Each x_* array must have shape [N, D] or [N, C, L].")
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (array - mean) / std
    return normalized.astype(np.float32), {"mean": mean, "std": std}


@dataclass
class MultimodalDataBundle:
    modalities: List[np.ndarray]
    labels: np.ndarray
    available_mask: np.ndarray
    indices: Dict[str, np.ndarray]
    modality_names: List[str]
    label_values: np.ndarray
    normalization: List[Dict[str, np.ndarray]]
    split_strategy: str

    @property
    def input_shapes(self) -> List[Tuple[int, ...]]:
        return [tuple(array.shape[1:]) for array in self.modalities]

    @property
    def num_classes(self) -> int:
        return int(len(self.label_values))


def load_multimodal_npz(
    path: str | Path,
    split_ratios: Sequence[float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> MultimodalDataBundle:


    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    record_ids: Optional[np.ndarray] = None
    with np.load(path, allow_pickle=False) as archive:
        modality_keys = sorted(
            [key for key in archive.files if key.startswith("x_")], key=_numeric_key
        )
        if len(modality_keys) < 2:
            raise ValueError("Dataset must contain at least x_0 and x_1.")
        raw_modalities = [np.asarray(archive[key], dtype=np.float32) for key in modality_keys]
        if "labels" not in archive.files:
            raise ValueError("Dataset must contain a labels array.")
        raw_labels = np.asarray(archive["labels"]).reshape(-1)
        sample_count = raw_labels.shape[0]
        if any(array.shape[0] != sample_count for array in raw_modalities):
            raise ValueError("All modalities and labels must have the same sample count.")

        label_values, labels = np.unique(raw_labels, return_inverse=True)
        labels = labels.astype(np.int64)
        if "observed_mask" in archive.files:
            available_mask = np.asarray(archive["observed_mask"], dtype=bool)
            expected = (sample_count, len(raw_modalities))
            if available_mask.shape != expected:
                raise ValueError(f"observed_mask must have shape {expected}.")
        else:
            available_mask = np.ones((sample_count, len(raw_modalities)), dtype=bool)
        if np.any(available_mask.sum(axis=1) == 0):
            raise ValueError("Every sample must contain at least one available modality.")

        if all(name in archive.files for name in ("train_idx", "val_idx", "test_idx")):
            indices = {
                "train": np.asarray(archive["train_idx"], dtype=np.int64),
                "val": np.asarray(archive["val_idx"], dtype=np.int64),
                "test": np.asarray(archive["test_idx"], dtype=np.int64),
            }
            split_strategy = "provided_indices"
        elif "record_ids" in archive.files:
            record_ids = np.asarray(archive["record_ids"]).reshape(-1)
            indices = _stratified_group_split(
                labels,
                record_ids,
                split_ratios,
                seed,
            )
            split_strategy = "stratified_record_level"
        else:
            indices = _stratified_split(labels, split_ratios, seed)
            split_strategy = "stratified_sample_level"

        if "record_ids" in archive.files and record_ids is None:
            record_ids = np.asarray(archive["record_ids"]).reshape(-1)

        if "modality_names" in archive.files:
            modality_names = [str(value) for value in np.asarray(archive["modality_names"])]
            if len(modality_names) != len(raw_modalities):
                raise ValueError("modality_names must match the number of x_* arrays.")
        else:
            modality_names = [f"modality_{index}" for index in range(len(raw_modalities))]

    normalized_modalities = []
    normalization = []
    for raw in raw_modalities:
        clean = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        normalized, statistics = _standardize(clean, indices["train"])
        normalized_modalities.append(normalized)
        normalization.append(statistics)

    all_indices = np.concatenate([indices["train"], indices["val"], indices["test"]])
    if (
        all_indices.size != sample_count
        or np.unique(all_indices).size != sample_count
        or np.any(all_indices < 0)
        or np.any(all_indices >= sample_count)
    ):
        raise ValueError("train_idx, val_idx, and test_idx must form a disjoint full partition.")
    if record_ids is not None:
        record_sets = {
            split: set(record_ids[split_indices].tolist())
            for split, split_indices in indices.items()
        }
        if (
            record_sets["train"] & record_sets["val"]
            or record_sets["train"] & record_sets["test"]
            or record_sets["val"] & record_sets["test"]
        ):
            raise ValueError("Acquisition records must not cross dataset splits.")
    return MultimodalDataBundle(
        modalities=normalized_modalities,
        labels=labels,
        available_mask=available_mask,
        indices=indices,
        modality_names=modality_names,
        label_values=label_values,
        normalization=normalization,
        split_strategy=split_strategy,
    )


def _add_awgn(value: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    demeaned = value - value.mean()
    signal_power = float(np.mean(demeaned**2))
    if signal_power < 1e-12:
        signal_power = float(np.mean(value**2)) + 1e-12
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=value.shape).astype(np.float32)
    return value + noise


class SACOMNPZDataset(Dataset):
    def __init__(
        self,
        bundle: MultimodalDataBundle,
        split: str,
        seed: int,
        pseudo_missing_probability: float = 0.0,
        perturb_probability: float = 0.0,
        perturb_snr_range: Sequence[float] = (10.0, 30.0),
        amplitude_range: Sequence[float] = (0.8, 1.2),
        forced_observed: Optional[Sequence[int]] = None,
        evaluation_snr_db: Optional[float] = None,
        degraded_modalities: Optional[Sequence[int]] = None,
    ) -> None:
        if split not in bundle.indices:
            raise ValueError(f"Unknown split: {split}")
        self.bundle = bundle
        self.split = split
        self.indices = bundle.indices[split]
        self.seed = int(seed)
        self.epoch = 0
        self.pseudo_missing_probability = float(pseudo_missing_probability)
        self.perturb_probability = float(perturb_probability)
        self.perturb_snr_range = tuple(float(value) for value in perturb_snr_range)
        self.amplitude_range = tuple(float(value) for value in amplitude_range)
        self.forced_observed = None if forced_observed is None else tuple(forced_observed)
        self.evaluation_snr_db = evaluation_snr_db
        self.degraded_modalities = (
            None if degraded_modalities is None else set(int(value) for value in degraded_modalities)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.indices.size)

    def _rng(self, local_index: int) -> np.random.Generator:
        split_offset = {"train": 0, "val": 1_000_000_000, "test": 2_000_000_000}[self.split]
        return np.random.default_rng(
            self.seed + split_offset + self.epoch * max(1, len(self)) + int(local_index)
        )

    def __getitem__(self, local_index: int) -> Dict[str, object]:
        sample_index = int(self.indices[local_index])
        rng = self._rng(local_index)
        clean = [array[sample_index].copy() for array in self.bundle.modalities]
        inputs = [value.copy() for value in clean]
        available = self.bundle.available_mask[sample_index].copy()
        observed = available.copy()

        if self.forced_observed is not None:
            requested = np.zeros_like(observed)
            requested[list(self.forced_observed)] = True
            observed &= requested

        if self.pseudo_missing_probability > 0.0:
            drop = rng.random(observed.shape[0]) < self.pseudo_missing_probability
            observed &= ~drop

        if not observed.any():
            candidates = np.flatnonzero(available)
            if self.forced_observed is not None:
                forced_candidates = np.intersect1d(candidates, np.asarray(self.forced_observed))
                if forced_candidates.size:
                    candidates = forced_candidates
            if candidates.size == 0:
                raise ValueError("Forced modality pattern removes every available modality.")
            observed[int(rng.choice(candidates))] = True

        for modality_index in np.flatnonzero(observed):
            if self.split == "train" and rng.random() < self.perturb_probability:
                scale = rng.uniform(*self.amplitude_range)
                snr = rng.uniform(*self.perturb_snr_range)
                inputs[modality_index] = _add_awgn(
                    inputs[modality_index] * scale, snr, rng
                )
            if self.evaluation_snr_db is not None:
                should_degrade = (
                    self.degraded_modalities is None
                    or modality_index in self.degraded_modalities
                )
                if should_degrade:
                    inputs[modality_index] = _add_awgn(
                        inputs[modality_index], self.evaluation_snr_db, rng
                    )

        # Missing inputs are zeroed. Clean targets remain available only for
        # representation/predictor losses and are never used in fused inference.
        for modality_index in np.flatnonzero(~observed):
            inputs[modality_index] = np.zeros_like(inputs[modality_index])

        return {
            "modalities": [torch.from_numpy(value).float() for value in inputs],
            "clean_modalities": [torch.from_numpy(value).float() for value in clean],
            "label": torch.tensor(self.bundle.labels[sample_index], dtype=torch.long),
            "available_mask": torch.from_numpy(available),
            "observed_mask": torch.from_numpy(observed),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EpochRandomSampler(Sampler[int]):


    def __init__(self, data_source: Dataset, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def make_data_loader(
    dataset: SACOMNPZDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = EpochRandomSampler(dataset, seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        # Re-spawning workers propagates dataset.set_epoch into worker copies.
        persistent_workers=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )
