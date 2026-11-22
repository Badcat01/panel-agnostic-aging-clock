"""FT-Mask mortality clock and event-normalized Breslow partial likelihood."""

import torch
from torch import nn


# Keep the pilot feature and organ order for compatibility with its checkpoints.
GROUPS = {
    "liver": [
        "albumin", "alp", "ast", "alt", "ggt", "bilirubin", "total_protein",
        "globulin", "ldh",
    ],
    "kidney": [
        "creatinine", "bun", "uric_acid", "uacr", "potassium", "sodium",
        "chloride", "bicarbonate", "phosphorus", "calcium",
    ],
    "metabolic": [
        "glucose", "hba1c", "triglycerides", "cholesterol", "hdl", "bmi", "waist",
    ],
    "immune_heme": [
        "wbc", "lymph_pct", "mono_pct", "neut_pct", "eos_pct", "crp", "rbc",
        "hgb", "mcv", "mch", "rdw", "platelets", "mpv", "iron",
    ],
    "cardiovascular": ["sbp", "dbp", "pulse"],
}
FEATS = [feature for features in GROUPS.values() for feature in features]
LOGF = [
    "alp", "ast", "alt", "ggt", "bilirubin", "ldh", "uacr", "triglycerides",
    "crp", "creatinine", "glucose", "wbc", "bun",
]
_FEATURE_GROUP = tuple(
    group_idx for group_idx, features in enumerate(GROUPS.values()) for _ in features
)


class FTMask(nn.Module):
    """Encode observed biomarker tokens and an always-visible age token."""

    def __init__(self, nf, d=64, L=3, heads=4, drop=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(nf + 1, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(nf + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d, heads, d * 2, drop, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, m, age):
        observed = m.bool()
        values = torch.cat((age[:, None], x.masked_fill(~observed, 0)), dim=1)
        features = values[..., None] * self.w + self.b
        tokens = torch.cat((self.cls.expand(len(x), -1, -1), features), dim=1)
        always_visible = observed.new_zeros((len(x), 2))
        padding = torch.cat((always_visible, ~observed), dim=1)
        hidden = self.enc(tokens, src_key_padding_mask=padding)
        return self.head(hidden[:, 0]).squeeze(-1)


def augment_mask(mask):
    """Preserve 40% of panels, thin 35% at random, restrict 25% to one organ."""
    if mask.ndim != 2 or mask.shape[1] != len(FEATS):
        raise ValueError(f"Expected a mask with {len(FEATS)} feature columns")
    n = mask.shape[0]
    device = mask.device
    mode = torch.rand((n, 1), device=device)
    drop_probability = torch.rand((n, 1), device=device) * 0.6
    random_keep = torch.rand(mask.shape, device=device) >= drop_probability
    group = torch.randint(len(GROUPS), (n, 1), device=device)
    feature_group = torch.as_tensor(_FEATURE_GROUP, device=device)
    keep = (mode < 0.4) | ((mode < 0.75) & random_keep)
    keep |= (mode >= 0.75) & (group == feature_group)
    return mask * keep


def breslow_loss(scores, times, events):
    """Average negative Cox partial log likelihood over deaths in the full risk set.

    At each event time, all participants still under observation, including those
    censored at that time, contribute to the shared Breslow denominator.
    """
    if scores.ndim != 1 or scores.shape != times.shape or scores.shape != events.shape:
        raise ValueError("Expected equally sized one-dimensional scores, times and events")
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    sorted_times, order = torch.sort(times)
    sorted_scores = scores[order]
    sorted_events = events[order].to(scores.dtype)
    log_risk = torch.logcumsumexp(sorted_scores.flip((0,)), dim=0).flip((0,))
    group_start = torch.searchsorted(sorted_times, sorted_times, right=False)
    contributions = sorted_events * (log_risk[group_start] - sorted_scores)
    return contributions.sum() / sorted_events.sum().clamp_min(1)
