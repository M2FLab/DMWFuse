import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

class DiffusionExpert(nn.Module):
    def __init__(self, dim, mlp_ratio, num_timesteps):
        super().__init__()
        self.dim = dim
        self.time_emb = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.timestep_range = (0, num_timesteps)

    def forward(self, x, t):
        """严格确保x和t的批量大小一致"""
        assert x.dim() == 2 and t.dim() == 2, f"x和t必须是二维张量，当前维度: x={x.dim()}, t={t.dim()}"
        assert x.size(0) == t.size(0), f"批量大小不匹配: x={x.size(0)}, t={t.size(0)}"
        assert t.size(1) == 1, f"t的第二维度必须为1，当前: {t.size(1)}"

        t_emb = self.time_emb(t)  # [B, 1] → [B, dim]
        x_cat = torch.cat([x, t_emb], dim=-1)  # [B, 2*dim]
        return self.mlp(x_cat)

class Expertset(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        self._device = gates.device

        self._nonzero_mask = gates > 0  # [B*N, num_experts]
        self._nonzero_indices = torch.nonzero(self._nonzero_mask, as_tuple=False)  # [K, 2]
        self._K = self._nonzero_indices.size(0) 

        if self._K == 0:
            self._expert_indices = torch.tensor([], dtype=torch.long, device=self._device)
            self._sample_indices = torch.tensor([], dtype=torch.long, device=self._device)
            self._part_sizes = [0] * num_experts
            return

        self._sample_indices = self._nonzero_indices[:, 0] 
        self._expert_indices = self._nonzero_indices[:, 1] 

        self._part_sizes = []
        for e in range(num_experts):
            count = (self._expert_indices == e).sum().item()
            self._part_sizes.append(count)

        self._nonzero_gates = gates[self._sample_indices, self._expert_indices].unsqueeze(1)  # [K, 1]

    def get_expert_inputs(self, x):
        if self._K == 0:
            return [torch.tensor([], device=self._device) for _ in range(self._num_experts)]
        
        x_selected = x[self._sample_indices]  # [K, dim]
        return torch.split(x_selected, self._part_sizes, dim=0)

    def get_expert_timesteps(self, t):
        if self._K == 0:
            return [torch.tensor([], device=self._device) for _ in range(self._num_experts)]
        
        t_selected = t[self._sample_indices]  # [K, 1]
        return torch.split(t_selected, self._part_sizes, dim=0)

    def aggregate_outputs(self, expert_outputs):
        if self._K == 0:
            return torch.zeros(self._gates.size(0), expert_outputs[0].size(1), device=self._device)
        
        all_outputs = torch.cat(expert_outputs, dim=0)  # [K, dim]
        all_outputs = all_outputs * self._nonzero_gates  # [K, dim]
        aggregated = torch.zeros(self._gates.size(0), all_outputs.size(1), device=self._device)
        aggregated.index_add_(0, self._sample_indices, all_outputs)
        return aggregated


class MoEWithDiffusion(nn.Module):
    def __init__(self, input_size, mlp_ratio, num_experts, num_timesteps, use_experts=2, no=True):
        super().__init__()
        self.num_experts = num_experts
        self.num_timesteps = num_timesteps
        self.k = use_experts
        self.no = no
        self.input_size = input_size

        self.experts = nn.ModuleList([
            DiffusionExpert(input_size, mlp_ratio, num_timesteps)
            for _ in range(num_experts)
        ])
        step_per_expert = num_timesteps // num_experts
        for i in range(num_experts):
            self.experts[i].timestep_range = (i * step_per_expert, (i + 1) * step_per_expert)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size)
        )

        self.gate_weights = nn.Parameter(torch.randn(3 * input_size, num_experts))
        self.noise_weights = nn.Parameter(torch.zeros(3 * input_size, num_experts))
        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def balance_loss(self, x):
        if x.numel() <= 1:
            return torch.tensor(0.0, device=x.device)
        eps = 1e-10
        return x.var() / (x.mean().square() + eps)

    def noisy_gate(self, x, prompt, t, train):
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B*N, 1]
        t_emb = self.time_embedding(t)  # [B*N, input_size]
        x_cat = torch.cat([x, prompt, t_emb], dim=-1)  # [B*N, 3*input_size]
        clean_logits = x_cat @ self.gate_weights  # [B*N, num_experts]

        if self.no and train:
            noise_std = self.softplus(x_cat @ self.noise_weights) + 1e-2
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k, self.num_experts), dim=1)  # [B*N, k]
        gates = torch.zeros_like(logits, device=logits.device)
        gates.scatter_(1, top_indices, self.softmax(top_logits))  

        expert_load = gates.sum(0)  # [num_experts]
        return gates, expert_load

    def forward(self, x, prompt, t):
        """
        x: [B, N, C] 
        prompt: [B, N, C] 
        t: [B, N, 1] 
        """
        B, N, C = x.shape
        x_flat = rearrange(x, "b n c -> (b n) c")  # [B*N, C]
        prompt_flat = rearrange(prompt, "b n c -> (b n) c")  # [B*N, C]
        t_flat = rearrange(t, "b n t -> (b n) t")  # [B*N, 1]

        gates, expert_load = self.noisy_gate(x_flat, prompt_flat, t_flat, self.training)
        expertset = Expertset(self.num_experts, gates)

        expert_x = expertset.get_expert_inputs(x_flat)  
        expert_t = expertset.get_expert_timesteps(t_flat) 

        expert_outputs = []
        for e in range(self.num_experts):
            if expert_x[e].numel() == 0:
                expert_outputs.append(torch.tensor([], device=x.device))
                continue
            expert_outputs.append(self.experts[e](expert_x[e], expert_t[e]))

        output_flat = expertset.aggregate_outputs(expert_outputs)  # [B*N, C]
        output = rearrange(output_flat, "(b n) c -> b n c", b=B, n=N)  # [B, N, C]

        weight_balance = self.balance_loss(gates.sum(0))
        load_balance = self.balance_loss(expert_load)
        moe_loss = weight_balance + load_balance

        return output, moe_loss



class TDMoE(nn.Module):
    def __init__(self, atom_dim, dim, ffn_expansion_factor, num_experts, num_timesteps, use_experts=2):
        super().__init__()
        self.prompt_proj = nn.Linear(atom_dim, dim)
        self.moe = MoEWithDiffusion(
            input_size=dim,
            mlp_ratio=ffn_expansion_factor,
            num_experts=num_experts,
            num_timesteps=num_timesteps,
            use_experts=use_experts
        )

    def forward(self, x, task_prompt, t):
        B, C, H, W = x.shape
        N = H * W
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        prompt_emb = self.prompt_proj(task_prompt).unsqueeze(1).expand(B, N, -1)
        t_broadcast = t.unsqueeze(1).expand(B, N, -1)  # [B, N, 1]

        moe_out, moe_loss = self.moe(x_flat, prompt_emb, t_broadcast)
        output = rearrange(moe_out, "b (h w) c -> b c h w", h=H, w=W)
        return output + x, moe_loss


class DiffusionModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dim = opt["latent_dim"]
        self.num_timesteps = opt["num_timesteps"]
        self.dt = 1.0 / self.num_timesteps
        self.task_moe = TDMoE(
            atom_dim=opt["task_prompt_dim"],
            dim=opt["latent_dim"],
            ffn_expansion_factor=opt["mlp_ratio"],
            num_experts=opt["num_experts"],
            num_timesteps=opt["num_timesteps"],
            use_experts=opt["use_experts"]
        )
        self.beta = opt.get("beta", 0.1)

    def sde_drift(self, x, t):
        return -0.5 * self.beta * x

    def sde_diffusion(self, t):
        return torch.sqrt(self.beta * t)

    def reverse_sde_step(self, x, t, task_prompt):
        if t.dim() == 1:
            t = t.unsqueeze(1)  
        x_denoised, moe_loss = self.task_moe(x, task_prompt, t)
        drift = self.sde_drift(x_denoised, t)
        diffusion = self.sde_diffusion(t)
        noise = torch.randn_like(x)
        x_next = x_denoised + drift * self.dt + diffusion * np.sqrt(self.dt) * noise
        return x_next, moe_loss

    def forward(self, x0, task_prompt):
        B = x0.shape[0]
        x = torch.randn_like(x0)
        total_moe_loss = 0.0

        for i in reversed(range(min(10, self.num_timesteps))):
            t = torch.tensor([i / self.num_timesteps], device=x.device).repeat(B, 1)  # [B, 1]
            x, moe_loss = self.reverse_sde_step(x, t, task_prompt)
            total_moe_loss += moe_loss

        return x, total_moe_loss / min(10, self.num_timesteps)



