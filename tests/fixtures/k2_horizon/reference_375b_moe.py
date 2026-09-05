# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 IFM. Extracted verbatim class bodies from the released HF source.
# IFM/K2-Horizon-375B-A23B@d33e3ae45281865ebf9f044b12d3635b1d1e17fe
# Full source SHA256: fb09e010956bd51cfa7d4055b4381cff34c9e06164066b49e3546f38b2e6242f
# Imports only; this fixture does not download or construct a full checkpoint.
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

class K2HorizonMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class K2HorizonSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.num_shared_experts = config.num_shared_experts
        self.router_score_func = config.router_score_func
        self.router_scaling_factor = config.router_scaling_factor

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=config.moe_gate_bias)
        self.experts = nn.ModuleList(
            [K2HorizonMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        if config.num_shared_experts > 0:
            self.shared_experts = K2HorizonMLP(
                config=config,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        residuals = hidden_states

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # router_logits = self.gate(hidden_states)
        router_logits = F.linear(hidden_states, self.gate.weight)

        if self.router_score_func == "softmax":
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        else:
            assert self.router_score_func == "sigmoid"
            routing_weights = F.sigmoid(router_logits.to(torch.float32))

        routing_weights_for_choice = routing_weights
        if self.gate.bias is not None:
            routing_weights_for_choice = routing_weights + self.gate.bias.to(routing_weights.dtype)

        _, selected_experts = torch.topk(routing_weights_for_choice, self.top_k, dim=-1)
        routing_weights = torch.gather(routing_weights, dim=-1, index=selected_experts)

        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights * self.router_scaling_factor
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if self.num_shared_experts > 0:
            final_hidden_states = final_hidden_states + self.shared_experts(residuals)

        return final_hidden_states, router_logits
