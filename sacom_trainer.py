

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from metrics import classification_metrics
from sacom_model import SACOM
from utils import append_jsonl, load_checkpoint, restore_rng_state, save_checkpoint, write_json


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(values.dtype)
    return torch.sum(values * mask_float) / mask_float.sum().clamp_min(1.0)


class SACOMTrainer:
    def __init__(
        self,
        model: SACOM,
        config: Dict[str, Any],
        device: torch.device,
        run_dir: str | Path,
        metadata: Dict[str, Any],
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata

        train_config = config["train"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
        )
        self.loss_weights = train_config["loss_weights"]
        self.grad_clip = float(train_config.get("grad_clip", 0.0))
        self.best_path = self.run_dir / "checkpoints" / "best.pt"
        self.last_path = self.run_dir / "checkpoints" / "last.pt"
        self.log_path = self.run_dir / "history.jsonl"

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "modalities": [value.to(self.device, non_blocking=True) for value in batch["modalities"]],
            "clean_modalities": [
                value.to(self.device, non_blocking=True) for value in batch["clean_modalities"]
            ],
            "label": batch["label"].to(self.device, non_blocking=True),
            "available_mask": batch["available_mask"].to(self.device, non_blocking=True).bool(),
            "observed_mask": batch["observed_mask"].to(self.device, non_blocking=True).bool(),
            "sample_index": batch["sample_index"].to(self.device, non_blocking=True),
        }

    def compute_losses(
        self,
        output: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        labels = batch["label"]
        observed_mask = batch["observed_mask"]
        available_mask = batch["available_mask"]
        batch_size, modality_count = observed_mask.shape

        reconstruction_error = []
        for reconstruction, target in zip(
            output["reconstructions"], batch["clean_modalities"]
        ):
            reconstruction_error.append(
                torch.mean((reconstruction - target) ** 2, dim=tuple(range(1, target.ndim)))
            )
        reconstruction_error_tensor = torch.stack(reconstruction_error, dim=1)
        loss_ae = _masked_average(reconstruction_error_tensor, observed_mask)

        semantic_logits = output["semantic_logits"]
        repeated_labels = labels.unsqueeze(1).expand(-1, modality_count).reshape(-1)
        auxiliary_error = F.cross_entropy(
            (
                semantic_logits
                / max(float(self.model.semantic_temperature), 1e-8)
            ).reshape(-1, semantic_logits.shape[-1]),
            repeated_labels,
            reduction="none",
        ).reshape(batch_size, modality_count)
        loss_aux = _masked_average(auxiliary_error, observed_mask)

        support_target = torch.gather(
            output["semantic_probabilities"],
            2,
            labels[:, None, None].expand(-1, modality_count, 1),
        ).squeeze(-1).detach()
        reliability_error = F.binary_cross_entropy(
            output["reliability"].clamp(1e-6, 1.0 - 1e-6),
            support_target,
            reduction="none",
        )
        loss_rel = _masked_average(reliability_error, observed_mask)

        target_latent = output["z_clean"].unsqueeze(2)
        cross_modal_error = torch.mean(
            (output["cross_predictions"] - target_latent) ** 2, dim=-1
        )
        off_diagonal = ~torch.eye(
            modality_count, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        valid_pairs = (
            available_mask.unsqueeze(2)
            & observed_mask.unsqueeze(1)
            & off_diagonal
        )
        loss_xmod = _masked_average(cross_modal_error, valid_pairs)

        loss_cls = F.cross_entropy(output["logits"], labels)
        total = (
            float(self.loss_weights["ae"]) * loss_ae
            + float(self.loss_weights["xmod"]) * loss_xmod
            + float(self.loss_weights["aux"]) * loss_aux
            + float(self.loss_weights["rel"]) * loss_rel
            + float(self.loss_weights["cls"]) * loss_cls
        )
        return {
            "total": total,
            "ae": loss_ae,
            "xmod": loss_xmod,
            "aux": loss_aux,
            "rel": loss_rel,
            "cls": loss_cls,
        }

    def _forward(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        output = self.model(
            batch["modalities"],
            batch["clean_modalities"],
            batch["observed_mask"],
        )
        return output, self.compute_losses(output, batch)

    def train_epoch(self, loader: Iterable[Dict[str, Any]], epoch: int) -> Dict[str, float]:
        self.model.train()
        if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        totals: Dict[str, float] = {}
        seen = 0
        progress = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
        for raw_batch in progress:
            batch = self._move_batch(raw_batch)
            self.optimizer.zero_grad(set_to_none=True)
            output, losses = self._forward(batch)
            losses["total"].backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            reliable_genuine = output["teacher_mask"] & batch["available_mask"]
            self.model.update_prototypes(output["z_input"].detach(), reliable_genuine)

            batch_size = int(batch["label"].shape[0])
            seen += batch_size
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
            progress.set_postfix(loss=f"{totals['total'] / seen:.4f}")
        return {name: value / max(seen, 1) for name, value in totals.items()}

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Dict[str, Any]],
        split: str,
        save_predictions_to: Optional[str | Path] = None,
    ) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        seen = 0
        labels_all: List[np.ndarray] = []
        predictions_all: List[np.ndarray] = []
        detail: Dict[str, List[np.ndarray]] = {
            "sample_index": [],
            "labels": [],
            "predictions": [],
            "observed_mask": [],
            "teacher_mask": [],
            "reliability": [],
            "completion_confidence": [],
            "raw_fusion_weights": [],
            "fusion_weights": [],
            "path_multiplicity": [],
            "reliability_budgets": [],
            "pre_projection_contributions": [],
            "pre_projection_violations": [],
            "cross_path_contributions": [],
            "budget_violations": [],
        }
        for raw_batch in tqdm(loader, desc=f"[{split}]", leave=False):
            batch = self._move_batch(raw_batch)
            output, losses = self._forward(batch)
            predictions = torch.argmax(output["logits"], dim=1)
            batch_size = int(batch["label"].shape[0])
            seen += batch_size
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

            labels_np = batch["label"].cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            labels_all.append(labels_np)
            predictions_all.append(predictions_np)
            detail["sample_index"].append(batch["sample_index"].cpu().numpy())
            detail["labels"].append(labels_np)
            detail["predictions"].append(predictions_np)
            detail["observed_mask"].append(batch["observed_mask"].cpu().numpy())
            for name in (
                "teacher_mask",
                "reliability",
                "completion_confidence",
                "raw_fusion_weights",
                "fusion_weights",
                "path_multiplicity",
                "reliability_budgets",
                "pre_projection_contributions",
                "pre_projection_violations",
                "cross_path_contributions",
                "budget_violations",
            ):
                detail[name].append(output[name].detach().cpu().numpy())

        labels = np.concatenate(labels_all)
        predictions = np.concatenate(predictions_all)
        result = classification_metrics(labels, predictions)
        result.update({f"loss_{name}": value / max(seen, 1) for name, value in totals.items()})
        arrays = {name: np.concatenate(values) for name, values in detail.items()}
        observed = arrays["observed_mask"].astype(bool)
        tolerance = float(self.model.projection_tolerance)
        pre_violation = arrays["pre_projection_violations"]
        post_violation = arrays["budget_violations"]
        pre_active = np.any((pre_violation > tolerance) & observed, axis=1)
        post_active = np.any((post_violation > tolerance) & observed, axis=1)
        pre_excess = np.sum(pre_violation, axis=1)
        weight_adjustment_l1 = np.sum(
            np.abs(arrays["fusion_weights"] - arrays["raw_fusion_weights"]), axis=1
        )
        weight_adjustment_l2 = np.linalg.norm(
            arrays["fusion_weights"] - arrays["raw_fusion_weights"], axis=1
        )
        safe_budgets = np.where(observed, arrays["reliability_budgets"], 1.0)
        budget_ratios = np.where(
            observed,
            arrays["pre_projection_contributions"] / np.maximum(safe_budgets, 1e-12),
            0.0,
        )
        max_budget_ratio = budget_ratios.max(axis=1)

        result.update(
            {
                "single_teacher_rate": float(
                    np.mean(arrays["teacher_mask"].sum(axis=1) == 1)
                ),
                "pre_projection_violation_rate": float(np.mean(pre_active)),
                "post_projection_violation_rate": float(np.mean(post_active)),
                "mean_pre_projection_excess": float(np.mean(pre_excess)),
                "mean_active_pre_projection_excess": float(
                    np.mean(pre_excess[pre_active]) if np.any(pre_active) else 0.0
                ),
                "mean_active_budget_ratio": float(
                    np.mean(max_budget_ratio[pre_active]) if np.any(pre_active) else 1.0
                ),
                "mean_active_weight_adjustment_l1": float(
                    np.mean(weight_adjustment_l1[pre_active])
                    if np.any(pre_active)
                    else 0.0
                ),
                "mean_active_weight_adjustment_l2": float(
                    np.mean(weight_adjustment_l2[pre_active])
                    if np.any(pre_active)
                    else 0.0
                ),
                "max_budget_violation": float(post_violation.max(initial=0.0)),
            }
        )
        if save_predictions_to is not None:
            output_path = Path(save_predictions_to)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **arrays)
        return result

    def fit(
        self,
        train_loader: Iterable[Dict[str, Any]],
        val_loader: Iterable[Dict[str, Any]],
        resume_from: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        train_config = self.config["train"]
        start_epoch = 1
        best_metric = -float("inf")
        stale_epochs = 0
        if resume_from is not None:
            checkpoint = load_checkpoint(resume_from, self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            restore_rng_state(checkpoint["rng_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_metric = float(checkpoint["best_metric"])

        epochs = int(train_config["epochs"])
        patience = int(train_config["patience"])
        started = time.time()
        for epoch in range(start_epoch, epochs + 1):
            train_result = self.train_epoch(train_loader, epoch)
            val_result = self.evaluate(val_loader, "val")
            record = {
                "epoch": epoch,
                "elapsed_seconds": time.time() - started,
                "train": train_result,
                "val": val_result,
                "tau_min": float(self.model.tau_min.item()),
            }
            append_jsonl(self.log_path, record)
            current_metric = float(val_result["accuracy"])
            improved = current_metric > best_metric + 1e-12
            if improved:
                best_metric = current_metric
                stale_epochs = 0
                save_checkpoint(
                    self.best_path,
                    self.model,
                    self.optimizer,
                    epoch,
                    best_metric,
                    self.config,
                    self.metadata,
                )
            else:
                stale_epochs += 1
            save_checkpoint(
                self.last_path,
                self.model,
                self.optimizer,
                epoch,
                best_metric,
                self.config,
                self.metadata,
            )
            print(
                f"Epoch {epoch:03d} | train_loss={train_result['total']:.4f} "
                f"| val_acc={val_result['accuracy']:.4f} "
                f"| val_macro_f1={val_result['macro_f1']:.4f}"
            )
            if patience > 0 and stale_epochs >= patience:
                print(f"Early stopping after {stale_epochs} stale epochs.")
                break

        summary = {
            "best_validation_accuracy": best_metric,
            "best_checkpoint": str(self.best_path),
            "last_checkpoint": str(self.last_path),
            "training_seconds": time.time() - started,
        }
        write_json(self.run_dir / "train_summary.json", summary)
        return summary

    def load_model(self, path: str | Path) -> Dict[str, Any]:
        checkpoint = load_checkpoint(path, self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def calibrate_tau_min(
        self,
        val_loader: Iterable[Dict[str, Any]],
        candidates: Sequence[float],
    ) -> Dict[str, Any]:
        original = float(self.model.tau_min.item())
        trials = []
        best_tau = original
        best_accuracy = -float("inf")
        for candidate in candidates:
            self.model.set_tau_min(float(candidate))
            result = self.evaluate(val_loader, f"val_tau_{candidate:.3f}")
            trials.append({"tau_min": float(candidate), **result})
            if result["accuracy"] > best_accuracy:
                best_accuracy = result["accuracy"]
                best_tau = float(candidate)
        self.model.set_tau_min(best_tau)
        calibration = {
            "selected_tau_min": best_tau,
            "validation_accuracy": best_accuracy,
            "trials": trials,
        }
        write_json(self.run_dir / "tau_min_calibration.json", calibration)
        return calibration
