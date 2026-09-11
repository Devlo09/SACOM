
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from data_loader import SACOMNPZDataset, load_multimodal_npz, make_data_loader
from sacom_model import SACOM
from sacom_trainer import SACOMTrainer
from utils import apply_overrides, load_config, resolve_device, set_global_seed, write_json

def _parse_modality_list(
    value: Optional[str], modality_names: Sequence[str]
) -> Optional[Sequence[int]]:
    if value is None or value.strip().lower() in {"", "all"}:
        return None
    result = []
    for token in value.split(","):
        token = token.strip()
        if token.isdigit():
            index = int(token)
        else:
            if token not in modality_names:
                raise ValueError(f"Unknown modality '{token}'. Choices: {modality_names}")
            index = modality_names.index(token)
        if not 0 <= index < len(modality_names):
            raise ValueError(f"Modality index out of range: {index}")
        result.append(index)
    return sorted(set(result))


def _build_model(config: Dict[str, Any], input_shapes, num_classes: int) -> SACOM:
    model_config = config["model"]
    return SACOM(
        input_shapes=input_shapes,
        num_classes=num_classes,
        latent_dim=int(model_config["latent_dim"]),
        conv_channels=model_config["conv_channels"],
        kernel_sizes=model_config["kernel_sizes"],
        predictor_hidden=int(model_config["predictor_hidden"]),
        reliability_hidden=int(model_config["reliability_hidden"]),
        semantic_temperature=float(model_config["semantic_temperature"]),
        disagreement_temperature=float(model_config["disagreement_temperature"]),
        fusion_temperature=float(model_config["fusion_temperature"]),
        tau_min=float(model_config["tau_min"]),
        kappa=float(model_config["kappa"]),
        prototype_momentum=float(model_config["prototype_momentum"]),
        projection_tolerance=float(model_config["projection_tolerance"]),
        use_source_attribution_constraint=bool(
            model_config.get("use_source_attribution_constraint", True)
        ),
        completion_path_repeats=int(model_config.get("completion_path_repeats", 1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SACOM reproducible implementation"
    )
    parser.add_argument("--mode", choices=("train", "test", "all"), default="all")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--observed_modalities",
        default="all",
        help="Comma-separated indices or names; default keeps every modality.",
    )
    parser.add_argument(
        "--degraded_modalities",
        default=None,
        help="Comma-separated observed modalities to corrupt at test time.",
    )
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument(
        "--completion_path_repeats",
        type=int,
        default=None,
        help="Replicate each missing-modality completion path L times (paper Appendix A).",
    )
    parser.add_argument(
        "--disable_sac",
        action="store_true",
        help="Use initial quality weights without source-attribution projection.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any YAML key, e.g. --set train.epochs=2",
    )
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.set)
    if args.data_path is not None:
        config["data"]["path"] = args.data_path
    if args.seed is not None:
        config["seed"] = args.seed
    if args.completion_path_repeats is not None:
        if args.completion_path_repeats < 1:
            parser.error("--completion_path_repeats must be at least one")
        config["model"]["completion_path_repeats"] = args.completion_path_repeats
    if args.disable_sac:
        config["model"]["use_source_attribution_constraint"] = False
    seed = int(config["seed"])
    set_global_seed(seed, bool(config.get("deterministic", True)))
    device = resolve_device(str(config.get("device", "auto")))

    bundle = load_multimodal_npz(
        config["data"]["path"],
        split_ratios=config["data"]["split_ratios"],
        seed=seed,
    )
    observed = _parse_modality_list(args.observed_modalities, bundle.modality_names)
    degraded = _parse_modality_list(args.degraded_modalities, bundle.modality_names)
    data_config = config["data"]
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config["num_workers"])

    train_dataset = SACOMNPZDataset(
        bundle,
        "train",
        seed=seed,
        pseudo_missing_probability=float(data_config["pseudo_missing_probability"]),
        perturb_probability=float(data_config["perturb_probability"]),
        perturb_snr_range=data_config["perturb_snr_range"],
        amplitude_range=data_config["amplitude_range"],
    )
    val_dataset = SACOMNPZDataset(
        bundle,
        "val",
        seed=seed,
        pseudo_missing_probability=float(data_config["validation_missing_probability"]),
    )
    test_dataset = SACOMNPZDataset(
        bundle,
        "test",
        seed=seed,
        pseudo_missing_probability=float(data_config["test_missing_probability"]),
        forced_observed=observed,
        evaluation_snr_db=args.snr_db,
        degraded_modalities=degraded,
    )
    train_loader = make_data_loader(train_dataset, batch_size, True, num_workers, seed)
    val_loader = make_data_loader(val_dataset, batch_size, False, num_workers, seed + 1)
    test_loader = make_data_loader(test_dataset, batch_size, False, num_workers, seed + 2)

    model = _build_model(config, bundle.input_shapes, bundle.num_classes)
    run_dir = Path(args.run_dir or config["output"]["run_dir"])
    metadata = {
        "data_path": str(Path(config["data"]["path"]).resolve()),
        "modality_names": bundle.modality_names,
        "input_shapes": bundle.input_shapes,
        "label_values": bundle.label_values,
        "num_classes": bundle.num_classes,
        "split_sizes": {name: int(values.size) for name, values in bundle.indices.items()},
        "split_strategy": bundle.split_strategy,
    }
    trainer = SACOMTrainer(model, config, device, run_dir, metadata)
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "metadata.json", metadata)

    if args.mode in {"train", "all"}:
        trainer.fit(train_loader, val_loader, resume_from=args.resume)

    if args.mode in {"test", "all"}:
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else trainer.best_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Run --mode train first."
            )
        trainer.load_model(checkpoint_path)
        calibration_config = config.get("calibration", {})
        if bool(calibration_config.get("enabled", False)):
            trainer.calibrate_tau_min(val_loader, calibration_config["tau_min_candidates"])
        scenario = {
            "observed_modalities": (
                bundle.modality_names
                if observed is None
                else [bundle.modality_names[index] for index in observed]
            ),
            "degraded_modalities": (
                [] if degraded is None else [bundle.modality_names[index] for index in degraded]
            ),
            "snr_db": args.snr_db,
            "completion_path_repeats": trainer.model.completion_path_repeats,
            "source_attribution_constraint": (
                trainer.model.use_source_attribution_constraint
            ),
            "checkpoint": str(checkpoint_path),
            "tau_min": float(trainer.model.tau_min.item()),
        }
        scenario_name = "test"
        if observed is not None:
            scenario_name += "_obs-" + "-".join(str(value) for value in observed)
        if args.snr_db is not None:
            scenario_name += f"_snr-{args.snr_db:g}"
        if degraded is not None:
            scenario_name += "_deg-" + "-".join(str(value) for value in degraded)
        if trainer.model.completion_path_repeats != 1:
            scenario_name += f"_paths-{trainer.model.completion_path_repeats}"
        if not trainer.model.use_source_attribution_constraint:
            scenario_name += "_no-sac"
        result = trainer.evaluate(
            test_loader,
            "test",
            save_predictions_to=run_dir / "predictions" / f"{scenario_name}.npz",
        )
        payload = {"scenario": scenario, "metrics": result}
        write_json(run_dir / "metrics" / f"{scenario_name}.json", payload)
        print("Test result:")
        for name, value in result.items():
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()

