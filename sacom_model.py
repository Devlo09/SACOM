

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from reliability_projection import (
    compute_cross_path_contributions,
    project_reliability_budget,
)


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = values / max(float(temperature), 1e-8)
    scaled = scaled.masked_fill(~mask, torch.finfo(values.dtype).min)
    result = torch.softmax(scaled, dim=-1)
    return result * mask.to(values.dtype)


class ModalityAutoencoder(nn.Module):

    def __init__(
        self,
        input_shape: Sequence[int],
        latent_dim: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(value) for value in input_shape)
        self.is_vector = len(self.input_shape) == 1
        flat_dim = 1
        for value in self.input_shape:
            flat_dim *= value
        self.flat_dim = flat_dim

        if self.is_vector:
            input_dim = self.input_shape[0]
            hidden = max(128, min(512, input_dim))
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.LayerNorm(hidden),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(hidden, latent_dim),
            )
        else:
            if len(self.input_shape) != 2:
                raise ValueError("Each modality must be [D] or [C, L].")
            if len(conv_channels) != 4 or len(kernel_sizes) != 4:
                raise ValueError("The paper architecture requires four Conv1D blocks.")
            blocks: List[nn.Module] = []
            in_channels = self.input_shape[0]
            for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
                blocks.extend(
                    [
                        nn.Conv1d(
                            in_channels,
                            int(out_channels),
                            kernel_size=int(kernel_size),
                            padding=int(kernel_size) // 2,
                        ),
                        nn.BatchNorm1d(int(out_channels)),
                        nn.LeakyReLU(0.1, inplace=True),
                        nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                    ]
                )
                in_channels = int(out_channels)
            blocks.append(nn.AdaptiveAvgPool1d(1))
            self.encoder_conv = nn.Sequential(*blocks)
            self.encoder_projection = nn.Linear(in_channels, latent_dim)

        decoder_hidden = max(128, min(512, self.flat_dim // 2 if self.flat_dim > 1 else 128))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(decoder_hidden, self.flat_dim),
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        if self.is_vector:
            return self.encoder(value)
        encoded = self.encoder_conv(value).squeeze(-1)
        return self.encoder_projection(encoded)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(latent)
        return reconstruction.reshape(latent.shape[0], *self.input_shape)


class ResidualCrossModalPredictor(nn.Module):

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + self.network(latent)


class ReliabilityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(value)).squeeze(-1)


class SACOM(nn.Module):


    def __init__(
        self,
        input_shapes: Sequence[Sequence[int]],
        num_classes: int,
        latent_dim: int = 128,
        conv_channels: Sequence[int] = (32, 64, 128, 128),
        kernel_sizes: Sequence[int] = (7, 5, 3, 3),
        predictor_hidden: int = 256,
        reliability_hidden: int = 64,
        semantic_temperature: float = 1.0,
        disagreement_temperature: float = 0.5,
        fusion_temperature: float = 1.0,
        tau_min: float = 0.5,
        kappa: float = 1.0,
        prototype_momentum: float = 0.95,
        projection_tolerance: float = 1e-6,
        use_source_attribution_constraint: bool = True,
        completion_path_repeats: int = 1,
    ) -> None:
        super().__init__()
        self.modality_count = len(input_shapes)
        if self.modality_count < 2:
            raise ValueError("SACOM requires at least two modalities.")
        if self.modality_count > 6:
            raise ValueError(
                "The exact active-set projection is intended for at most six modalities."
            )
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.semantic_temperature = float(semantic_temperature)
        self.disagreement_temperature = float(disagreement_temperature)
        self.fusion_temperature = float(fusion_temperature)
        self.kappa = float(kappa)
        self.prototype_momentum = float(prototype_momentum)
        self.projection_tolerance = float(projection_tolerance)
        self.use_source_attribution_constraint = bool(use_source_attribution_constraint)
        self.completion_path_repeats = int(completion_path_repeats)
        if self.completion_path_repeats < 1:
            raise ValueError("completion_path_repeats must be at least one.")

        self.autoencoders = nn.ModuleList(
            [
                ModalityAutoencoder(shape, latent_dim, conv_channels, kernel_sizes)
                for shape in input_shapes
            ]
        )
        self.semantic_heads = nn.ModuleList(
            [nn.Linear(latent_dim, num_classes) for _ in input_shapes]
        )
        self.reliability_heads = nn.ModuleList(
            [
                ReliabilityHead(latent_dim + num_classes, reliability_hidden)
                for _ in input_shapes
            ]
        )
        self.cross_modal_predictors = nn.ModuleDict(
            {
                f"{source}_to_{target}": ResidualCrossModalPredictor(
                    latent_dim, predictor_hidden
                )
                for source in range(self.modality_count)
                for target in range(self.modality_count)
                if source != target
            }
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )

        self.register_buffer("prototypes", torch.zeros(self.modality_count, latent_dim))
        self.register_buffer(
            "prototype_initialized", torch.zeros(self.modality_count, dtype=torch.bool)
        )
        self.register_buffer("tau_min", torch.tensor(float(tau_min)))

    def set_tau_min(self, value: float) -> None:
        self.tau_min.fill_(float(value))

    def _select_reliable_teachers(
        self, reliability: torch.Tensor, observed_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        observed_float = observed_mask.to(reliability.dtype)
        count = observed_float.sum(dim=1).clamp_min(1.0)
        mean = (reliability * observed_float).sum(dim=1) / count
        variance = (
            ((reliability - mean.unsqueeze(1)) ** 2 * observed_float).sum(dim=1) / count
        )
        relative_threshold = mean - self.kappa * torch.sqrt(variance.clamp_min(0.0))
        teacher_mask = (
            observed_mask
            & (reliability >= relative_threshold.unsqueeze(1))
            & (reliability >= self.tau_min)
        )
        return teacher_mask, relative_threshold

    def _cross_modal_predictions(self, latent: torch.Tensor) -> torch.Tensor:
        # Output layout: [batch, target, source, latent].
        target_predictions = []
        for target in range(self.modality_count):
            source_predictions = []
            for source in range(self.modality_count):
                if source == target:
                    source_predictions.append(torch.zeros_like(latent[:, source]))
                else:
                    predictor = self.cross_modal_predictors[f"{source}_to_{target}"]
                    source_predictions.append(predictor(latent[:, source]))
            target_predictions.append(torch.stack(source_predictions, dim=1))
        return torch.stack(target_predictions, dim=1)

    def forward(
        self,
        modalities: Sequence[torch.Tensor],
        clean_modalities: Sequence[torch.Tensor],
        observed_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        if len(modalities) != self.modality_count:
            raise ValueError("Input modality count does not match the model.")
        if observed_mask.dtype != torch.bool:
            observed_mask = observed_mask.bool()
        if torch.any(observed_mask.sum(dim=1) == 0):
            raise ValueError("Every sample must retain at least one observed modality.")

        input_latents = []
        clean_latents = []
        reconstructions: List[torch.Tensor] = []
        unimodal_logits = []
        for index, autoencoder in enumerate(self.autoencoders):
            paired = torch.cat((modalities[index], clean_modalities[index]), dim=0)
            paired_latent = autoencoder.encode(paired)
            input_latent, clean_latent = paired_latent.chunk(2, dim=0)
            input_latents.append(input_latent)
            clean_latents.append(clean_latent)
            reconstructions.append(autoencoder.decode(input_latent))
            unimodal_logits.append(self.semantic_heads[index](input_latent))

        z_input = torch.stack(input_latents, dim=1)
        z_clean = torch.stack(clean_latents, dim=1)
        semantic_logits = torch.stack(unimodal_logits, dim=1)
        semantic_probabilities = torch.softmax(
            semantic_logits / max(self.semantic_temperature, 1e-8), dim=-1
        )

        reliabilities = []
        for index, reliability_head in enumerate(self.reliability_heads):
            head_input = torch.cat(
                (z_input[:, index].detach(), semantic_probabilities[:, index].detach()),
                dim=-1,
            )
            reliabilities.append(reliability_head(head_input))
        reliability = torch.stack(reliabilities, dim=1)
        teacher_mask, relative_threshold = self._select_reliable_teachers(
            reliability, observed_mask
        )

        cross_predictions = self._cross_modal_predictions(z_input)
        teacher_mass = reliability * teacher_mask.to(reliability.dtype)
        teacher_denominator = teacher_mass.sum(dim=1, keepdim=True)
        normalized_teacher_weights = teacher_mass / teacher_denominator.clamp_min(1e-8)
        has_teacher = teacher_denominator.squeeze(1) > 1e-8
        normalized_teacher_weights = normalized_teacher_weights * has_teacher.unsqueeze(1)
        teacher_weights = normalized_teacher_weights.unsqueeze(1).expand(
            -1, self.modality_count, -1
        )

        predicted_latent = torch.sum(
            cross_predictions * teacher_weights.unsqueeze(-1), dim=2
        )
        prediction_error = torch.sum(
            (cross_predictions - predicted_latent.unsqueeze(2)) ** 2, dim=-1
        ) / float(self.latent_dim)
        disagreement = torch.sum(prediction_error * teacher_weights, dim=2)
        teacher_count = teacher_mask.sum(dim=1).clamp_min(1).to(reliability.dtype)
        mean_teacher_reliability = teacher_mass.sum(dim=1) / teacher_count
        completion_confidence = mean_teacher_reliability.unsqueeze(1) * torch.exp(
            -disagreement / max(self.disagreement_temperature, 1e-8)
        )
        completion_confidence = completion_confidence * has_teacher.unsqueeze(1)

        prototypes = self.prototypes.unsqueeze(0).expand(z_input.shape[0], -1, -1)
        completed_latent = (
            completion_confidence.unsqueeze(-1) * predicted_latent
            + (1.0 - completion_confidence).unsqueeze(-1) * prototypes
        )

        quality = torch.where(observed_mask, reliability, completion_confidence)
        path_multiplicity = torch.where(
            observed_mask,
            torch.ones_like(quality),
            torch.full_like(quality, float(self.completion_path_repeats)),
        )
        # L identical completion branches have L equal Softmax terms.  Combining
        # them adds log(L) to the branch logit while retaining a compact M-vector.
        raw_fusion_weights = torch.softmax(
            quality / max(self.fusion_temperature, 1e-8)
            + torch.log(path_multiplicity),
            dim=1,
        )
        reliability_budgets = _masked_softmax(
            reliability, observed_mask, self.fusion_temperature
        )
        pre_projection_contributions = compute_cross_path_contributions(
            raw_fusion_weights,
            observed_mask,
            teacher_weights,
            completion_confidence,
        )
        pre_projection_violations = (
            torch.relu(pre_projection_contributions - reliability_budgets)
            * observed_mask.to(raw_fusion_weights.dtype)
        )
        if self.use_source_attribution_constraint:
            fusion_weights, contributions, budget_violations = (
                project_reliability_budget(
                    raw_fusion_weights,
                    observed_mask,
                    teacher_weights,
                    completion_confidence,
                    reliability_budgets,
                    tolerance=self.projection_tolerance,
                    branch_multiplicity=path_multiplicity,
                )
            )
        else:
            fusion_weights = raw_fusion_weights
            contributions = pre_projection_contributions
            budget_violations = pre_projection_violations

        unified_latent = torch.where(
            observed_mask.unsqueeze(-1), z_input, completed_latent
        )
        fused_latent = torch.sum(fusion_weights.unsqueeze(-1) * unified_latent, dim=1)
        logits = self.classifier(fused_latent)

        return {
            "logits": logits,
            "fused_latent": fused_latent,
            "z_input": z_input,
            "z_clean": z_clean,
            "reconstructions": reconstructions,
            "semantic_logits": semantic_logits,
            "semantic_probabilities": semantic_probabilities,
            "reliability": reliability,
            "teacher_mask": teacher_mask,
            "relative_threshold": relative_threshold,
            "cross_predictions": cross_predictions,
            "teacher_weights": teacher_weights,
            "predicted_latent": predicted_latent,
            "disagreement": disagreement,
            "completion_confidence": completion_confidence,
            "completed_latent": completed_latent,
            "quality": quality,
            "raw_fusion_weights": raw_fusion_weights,
            "fusion_weights": fusion_weights,
            "path_multiplicity": path_multiplicity,
            "reliability_budgets": reliability_budgets,
            "pre_projection_contributions": pre_projection_contributions,
            "pre_projection_violations": pre_projection_violations,
            "cross_path_contributions": contributions,
            "budget_violations": budget_violations,
        }

    @torch.no_grad()
    def update_prototypes(self, latent: torch.Tensor, reliable_mask: torch.Tensor) -> None:
        

        for modality_index in range(self.modality_count):
            selected = reliable_mask[:, modality_index]
            if not torch.any(selected):
                continue
            batch_mean = latent[selected, modality_index].mean(dim=0)
            if not self.prototype_initialized[modality_index]:
                self.prototypes[modality_index].copy_(batch_mean)
                self.prototype_initialized[modality_index] = True
            else:
                self.prototypes[modality_index].mul_(self.prototype_momentum).add_(
                    batch_mean, alpha=1.0 - self.prototype_momentum
                )
