

from __future__ import annotations

from itertools import combinations
from typing import Optional, Tuple

import torch


def _is_true(value: torch.Tensor) -> bool:


    return bool(value.detach().cpu().item())


def _project_single(
    raw: torch.Tensor,
    constraint_matrix: torch.Tensor,
    budgets: torch.Tensor,
    feasible_fallback: torch.Tensor,
    tolerance: float = 1e-6,
    objective_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    modality_count = raw.numel()
    dtype = raw.dtype
    device = raw.device

    if objective_weights is None:
        objective_weights = torch.ones_like(raw)
    if objective_weights.shape != raw.shape:
        raise ValueError("objective_weights must match raw.")
    if _is_true(torch.any(objective_weights <= 0)):
        raise ValueError("objective_weights must be strictly positive.")
    inverse_weights = objective_weights.reciprocal()

    nonnegative_g = -torch.eye(modality_count, dtype=dtype, device=device)
    nonnegative_h = torch.zeros(modality_count, dtype=dtype, device=device)
    if constraint_matrix.numel():
        g_matrix = torch.cat((nonnegative_g, constraint_matrix), dim=0)
        h_vector = torch.cat((nonnegative_h, budgets), dim=0)
    else:
        g_matrix = nonnegative_g
        h_vector = nonnegative_h

    if _is_true(torch.all(g_matrix @ raw <= h_vector + tolerance)):
        return raw

    equality_row = torch.ones((1, modality_count), dtype=dtype, device=device)
    equality_rhs = torch.ones(1, dtype=dtype, device=device)
    inequality_count = g_matrix.shape[0]
    best_candidate = None
    best_distance = float("inf")


    max_active = min(modality_count - 1, inequality_count)
    for active_count in range(1, max_active + 1):
        for active in combinations(range(inequality_count), active_count):
            active_index = torch.tensor(active, dtype=torch.long, device=device)
            c_matrix = torch.cat((equality_row, g_matrix[active_index]), dim=0)
            d_vector = torch.cat((equality_rhs, h_vector[active_index]), dim=0)

            weighted_c_transpose = inverse_weights.unsqueeze(1) * c_matrix.transpose(0, 1)
            gram = c_matrix @ weighted_c_transpose
            multipliers = torch.linalg.pinv(gram) @ (c_matrix @ raw - d_vector)
            candidate = raw - weighted_c_transpose @ multipliers

            # KKT multipliers for active inequalities must be nonnegative.
            if active_count and not _is_true(torch.all(multipliers[1:] >= -tolerance)):
                continue
            if not _is_true(torch.all(g_matrix @ candidate <= h_vector + tolerance)):
                continue
            if not _is_true(torch.abs(candidate.sum() - 1.0) <= 10 * tolerance):
                continue

            distance = float(
                torch.sum(objective_weights * (candidate - raw) ** 2)
                .detach()
                .cpu()
                .item()
            )
            if distance < best_distance:
                best_distance = distance
                best_candidate = candidate


    if best_candidate is None:
        return feasible_fallback
    return best_candidate


def compute_cross_path_contributions(
    fusion_weights: torch.Tensor,
    observed_mask: torch.Tensor,
    teacher_weights: torch.Tensor,
    completion_confidence: torch.Tensor,
) -> torch.Tensor:


    direct = fusion_weights * observed_mask.to(fusion_weights.dtype)
    missing_mask = ~observed_mask
    indirect_coeff = (
        teacher_weights
        * completion_confidence.unsqueeze(-1)
        * missing_mask.unsqueeze(-1).to(fusion_weights.dtype)
    )
    indirect = torch.sum(indirect_coeff * fusion_weights.unsqueeze(-1), dim=1)
    return direct + indirect


def project_reliability_budget(
    raw_weights: torch.Tensor,
    observed_mask: torch.Tensor,
    teacher_weights: torch.Tensor,
    completion_confidence: torch.Tensor,
    reliability_budgets: torch.Tensor,
    tolerance: float = 1e-6,
    branch_multiplicity: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if raw_weights.ndim != 2:
        raise ValueError("raw_weights must have shape [batch, modalities].")
    if observed_mask.shape != raw_weights.shape:
        raise ValueError("observed_mask must match raw_weights.")
    if branch_multiplicity is None:
        branch_multiplicity = torch.ones_like(raw_weights)
    if branch_multiplicity.shape != raw_weights.shape:
        raise ValueError("branch_multiplicity must match raw_weights.")
    if _is_true(torch.any(branch_multiplicity < 1)):
        raise ValueError("branch_multiplicity values must be at least one.")

    batch_size, modality_count = raw_weights.shape
    projected = []
    for batch_index in range(batch_size):
        observed_index = torch.nonzero(observed_mask[batch_index], as_tuple=False).flatten()
        if observed_index.numel() == 0:
            raise ValueError("Every sample must retain at least one observed modality.")

        rows = []
        for source_index in observed_index.tolist():
            row = torch.zeros(
                modality_count,
                dtype=raw_weights.dtype,
                device=raw_weights.device,
            )
            row[source_index] = 1.0
            for target_index in range(modality_count):
                if not _is_true(observed_mask[batch_index, target_index]):
                    row[target_index] = row[target_index] + (
                        teacher_weights[batch_index, target_index, source_index]
                        * completion_confidence[batch_index, target_index]
                    )
            rows.append(row)

        constraint_matrix = torch.stack(rows, dim=0)
        observed_budgets = reliability_budgets[batch_index, observed_index]

        fallback = torch.zeros_like(raw_weights[batch_index])
        fallback = fallback.scatter(0, observed_index, observed_budgets)
        projected.append(
            _project_single(
                raw_weights[batch_index],
                constraint_matrix,
                observed_budgets,
                fallback,
                tolerance=tolerance,
                objective_weights=branch_multiplicity[batch_index].reciprocal(),
            )
        )

    projected_weights = torch.stack(projected, dim=0)
    contributions = compute_cross_path_contributions(
        projected_weights,
        observed_mask,
        teacher_weights,
        completion_confidence,
    )
    violations = torch.relu(contributions - reliability_budgets) * observed_mask.to(
        raw_weights.dtype
    )
    return projected_weights, contributions, violations

