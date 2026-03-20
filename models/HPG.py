"""
Improved Hierarchical Planner-Generator (HPG) for multi-document summarization.

This module is designed as a stronger alternative to models/novel_model.py
without modifying the original implementation.

Core additions over the previous HPG:
1. Salience-aware segment planner:
   - Pools token states into pseudo-document segments
   - Scores segment salience
   - Builds multiple plan tokens from salient segments
2. Plan redundancy control:
   - Auxiliary penalty to encourage diverse plan tokens
3. Plan-conditioned encoder fusion:
   - Token states attend to plan tokens with a learned gate before decoding
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


class SegmentPooler(nn.Module):
    """
    Pools variable-length token sequences into a fixed number of segments.
    This approximates document-level planning even when inputs are flattened.
    """

    def __init__(self, num_segments: int = 16):
        super().__init__()
        self.num_segments = num_segments

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, T, H)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, _, hidden = hidden_states.shape
        seg_states = hidden_states.new_zeros(bsz, self.num_segments, hidden)
        seg_mask = torch.zeros(
            bsz, self.num_segments, dtype=torch.bool, device=hidden_states.device
        )

        for b in range(bsz):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len <= 0:
                continue

            seq = hidden_states[b, :valid_len]  # (L, H)
            boundaries = torch.linspace(
                0, valid_len, steps=self.num_segments + 1, device=hidden_states.device
            ).long()

            for s in range(self.num_segments):
                start = int(boundaries[s].item())
                end = int(boundaries[s + 1].item())

                if start >= valid_len:
                    break
                if end <= start:
                    end = min(start + 1, valid_len)

                seg_states[b, s] = seq[start:end].mean(dim=0)
                seg_mask[b, s] = True

        return seg_states, seg_mask


class SalienceAwarePlanner(nn.Module):
    """
    Produces a set of plan tokens from segment representations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_segments: int = 16,
        num_plan_tokens: int = 6,
        num_heads: int = 8,
        planner_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.segment_pooler = SegmentPooler(num_segments=num_segments)

        self.segment_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.segment_ln = nn.LayerNorm(hidden_size)

        self.salience_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.plan_queries = nn.Parameter(torch.randn(num_plan_tokens, hidden_size) * 0.02)
        self.query_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        planner_block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.plan_refiner = nn.TransformerEncoder(planner_block, num_layers=planner_layers)
        self.dropout = nn.Dropout(dropout)

    def _redundancy_loss(self, plan_tokens: torch.Tensor) -> torch.Tensor:
        # Penalize highly similar plan tokens; encourages diverse planned content.
        bsz, nplan, _ = plan_tokens.size()
        if nplan <= 1:
            return plan_tokens.new_zeros(())

        norm_plan = F.normalize(plan_tokens, dim=-1)
        sim = torch.matmul(norm_plan, norm_plan.transpose(1, 2))
        eye = torch.eye(nplan, device=plan_tokens.device).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        denom = bsz * nplan * (nplan - 1)
        return (off_diag.pow(2).sum() / max(denom, 1)).float()

    def forward(
        self,
        token_states: torch.Tensor,     # (B, T, H)
        attention_mask: torch.Tensor,   # (B, T)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seg_states, seg_mask = self.segment_pooler(token_states, attention_mask)
        key_padding = ~seg_mask  # True => ignore

        attn_out, _ = self.segment_attn(
            seg_states, seg_states, seg_states, key_padding_mask=key_padding
        )
        seg_states = self.segment_ln(seg_states + self.dropout(attn_out))

        salience_logits = self.salience_head(seg_states).squeeze(-1)  # (B, S)
        salience_logits = salience_logits.masked_fill(~seg_mask, -1e4)
        salience = torch.softmax(salience_logits, dim=-1)
        weighted_segments = seg_states * salience.unsqueeze(-1)

        query = self.plan_queries.unsqueeze(0).expand(token_states.size(0), -1, -1)
        plan_tokens, _ = self.query_attn(
            query, weighted_segments, weighted_segments, key_padding_mask=key_padding
        )
        plan_tokens = self.plan_refiner(plan_tokens)

        entropy = -(salience * torch.log(salience + 1e-8)).sum(dim=-1).mean()
        redundancy = self._redundancy_loss(plan_tokens)

        aux = {
            "salience": salience,
            "planner_entropy": entropy,
            "plan_redundancy": redundancy,
        }
        return plan_tokens, aux


class PlanConditionedFusion(nn.Module):
    """
    Injects planned content into token-level encoder states.
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.token_to_plan = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_states: torch.Tensor,   # (B, T, H)
        plan_states: torch.Tensor,    # (B, P, H)
    ) -> torch.Tensor:
        plan_context, _ = self.token_to_plan(
            token_states, plan_states, plan_states
        )
        gate = torch.sigmoid(self.gate(torch.cat([token_states, plan_context], dim=-1)))
        fused = self.ln1(token_states + self.dropout(gate * plan_context))
        fused = self.ln2(fused + self.dropout(self.ffn(fused)))
        return fused


class HierarchicalPlannerGeneratorV2(nn.Module):
    """
    Enhanced HPG architecture.

    Compatible interface:
    - forward(input_ids, attention_mask, labels=None) -> dict with loss/logits/plan
    - generate(input_ids, attention_mask, **kwargs)
    """

    def __init__(
        self,
        base_model_name: str,
        num_segments: int = 16,
        num_plan_tokens: int = 6,
        num_heads: int = 8,
        planner_layers: int = 2,
        dropout: float = 0.1,
        planner_entropy_weight: float = 0.01,
        redundancy_weight: float = 0.03,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        hidden_size = (
            getattr(self.generator.config, "d_model", None)
            or getattr(self.generator.config, "hidden_size", None)
        )
        if hidden_size is None:
            raise ValueError(
                f"Cannot infer hidden size from config for base model: {base_model_name}"
            )

        self.planner = SalienceAwarePlanner(
            hidden_size=hidden_size,
            num_segments=num_segments,
            num_plan_tokens=num_plan_tokens,
            num_heads=num_heads,
            planner_layers=planner_layers,
            dropout=dropout,
        )
        self.fusion = PlanConditionedFusion(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.planner_entropy_weight = planner_entropy_weight
        self.redundancy_weight = redundancy_weight

    def _build_conditioned_encoder(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[BaseModelOutput, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        encoder = self.generator.get_encoder()
        enc_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_states = enc_outputs.last_hidden_state

        plan_states, aux = self.planner(token_states, attention_mask)
        fused_tokens = self.fusion(token_states, plan_states)

        conditioned_states = torch.cat([plan_states, fused_tokens], dim=1)
        plan_mask = attention_mask.new_ones((attention_mask.size(0), plan_states.size(1)))
        conditioned_mask = torch.cat([plan_mask, attention_mask], dim=1)

        conditioned_outputs = BaseModelOutput(last_hidden_state=conditioned_states)
        return conditioned_outputs, conditioned_mask, aux, plan_states

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_outputs, conditioned_mask, aux, plan_states = self._build_conditioned_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        outputs = self.generator(
            encoder_outputs=encoder_outputs,
            attention_mask=conditioned_mask,
            labels=labels,
            return_dict=True,
        )

        loss = outputs.loss
        if loss is not None:
            # Encourage focused salience and diverse plan tokens.
            loss = (
                loss
                + self.planner_entropy_weight * aux["planner_entropy"]
                + self.redundancy_weight * aux["plan_redundancy"]
            )

        return {
            "loss": loss,
            "logits": outputs.logits,
            "plan": plan_states,
            "salience": aux["salience"],
            "planner_entropy": aux["planner_entropy"].detach(),
            "plan_redundancy": aux["plan_redundancy"].detach(),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **gen_kwargs,
    ) -> torch.Tensor:
        encoder_outputs, conditioned_mask, _, _ = self._build_conditioned_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self.generator.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=conditioned_mask,
            **gen_kwargs,
        )


# Convenience alias so this file can be used as a drop-in replacement import:
# from models.HPG import HierarchicalPlannerGenerator
HierarchicalPlannerGenerator = HierarchicalPlannerGeneratorV2

