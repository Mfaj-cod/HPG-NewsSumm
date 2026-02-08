# This script defines a novel hierarchical planner-generator model for abstractive summarization.
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSeq2SeqLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

class DocumentEncoder(nn.Module):
    """
    Shared encoder that encodes each document independently.
    """

    def __init__(self, base_model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Return last hidden state
        return outputs.last_hidden_state


class ContentPlanner(nn.Module):
    """
    Lightweight planner that attends over document representations
    and produces a plan representation.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, doc_embeddings):
        """
        doc_embeddings: (B, N, H) where
            B = batch size
            N = number of documents (or segments)
            H = hidden size
        """
        attn_out, _ = self.attention(doc_embeddings, doc_embeddings, doc_embeddings)
        plan = self.linear(attn_out)
        return plan


class SummaryGenerator(nn.Module):
    """
    Abstractive generator conditioned on document encodings and plan.
    """

    def __init__(self, base_model_name: str):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs


class HierarchicalPlannerGenerator(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()

        self.base_model_name = base_model_name
        self.document_encoder = DocumentEncoder(base_model_name)

        # Get true hidden size from model config (e.g., 768 for LED)
        hidden_size = self.document_encoder.encoder.config.hidden_size

        print(f"[HPG] Using hidden size from backbone: {hidden_size}")

        # Planner always matches backbone dimension
        self.planner = ContentPlanner(hidden_size)

        self.generator = SummaryGenerator(base_model_name)

    def forward(self, input_ids, attention_mask, labels=None):

        # Encode tokens
        doc_repr = self.document_encoder(input_ids, attention_mask)   # (B,T,H)

        # Pool segments -> plan tokens
        pooled = doc_repr.mean(dim=1, keepdim=True)                   # (B,1,H)
        plan = self.planner(pooled)                                   # (B,1,H)

        # REAL INJECTION
        # Prepend plan as prefix embedding
        plan_mask = torch.ones(plan.size()[:2], device=attention_mask.device)

        inputs_embeds = self.generator.model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([plan, inputs_embeds], dim=1)
        attention_mask = torch.cat([plan_mask, attention_mask], dim=1)

        outputs = self.generator.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "plan": plan
        }


    def generate(self, input_ids, attention_mask, **gen_kwargs):

        doc_repr = self.document_encoder(input_ids, attention_mask)
        pooled = doc_repr.mean(dim=1, keepdim=True)
        plan = self.planner(pooled)

        plan_mask = torch.ones(plan.size()[:2], device=attention_mask.device)

        inputs_embeds = self.generator.model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([plan, inputs_embeds], dim=1)
        attention_mask = torch.cat([plan_mask, attention_mask], dim=1)

        return self.generator.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs
        )
