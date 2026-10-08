import torch
import torch.nn as nn
import torch.nn.functional as F



class DefaultSelector_(nn.Module):
    """
    默认 selector: 线性投影 + 归一化点积 + 小 MLP 残差
    forward:
    query: [d] or [d_q]  (若为多维，会按第0维循环)
    cand_feats: [M_actual, d]
    cls_prior: [M_actual] or None
    返回 logits: [M_actual]
    """
    def __init__(self, d_feat=768, d_query=None, proj_dim=256, use_prior=True, tau=0.07):
        super().__init__()
        if d_query is None:
            d_query = d_feat
        self.wp = nn.Linear(d_feat, proj_dim)
        self.wt = nn.Linear(d_query, proj_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj_dim * 2 + (1 if use_prior else 0)),
            nn.Linear(proj_dim * 2 + (1 if use_prior else 0), proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
            )
        self.use_prior = use_prior
        self.tau = tau

    def forward(self, query, cand_feats, cls_prior=None):
        # cand_feats: [M, d_feat]
        # query: [d_query] or [1, d_query]
        # returns logits: [M]
        M = cand_feats.shape[0]
        p = F.normalize(self.wp(cand_feats), dim=-1)  # [M,proj_dim]
        q = F.normalize(self.wt(query.unsqueeze(0) if query.dim()==1 else query), dim=-1)  # [1,proj_dim] or [B,proj]
        q = q.squeeze(0)  # [proj_dim]
        dot = torch.matmul(p, q) / self.tau  # [M]
        # mlp residual
        q_exp = q.unsqueeze(0).expand(M, -1)  # [M,proj_dim]
        if cls_prior is None:
            prior = torch.zeros(M, 1, device=cand_feats.device)
        else:
            prior = cls_prior.unsqueeze(1)
        mlp_in = torch.cat([p, q_exp, prior], dim=-1)  # [M, 2proj + 1]
        mlp_logits = self.mlp(mlp_in).squeeze(-1)  # [M]
        logits = dot + mlp_logits
        return logits

class DefaultSelector(torch.nn.Module):
    def __init__(self, d_feat, d_query=None, proj_dim=512, use_prior=True, tau=0.07):
        super().__init__()
        if d_query is None:
            d_query = d_feat
        self.wp = torch.nn.Linear(d_feat, proj_dim)
        self.wt = torch.nn.Linear(d_query, proj_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(proj_dim * 2 + (1 if use_prior else 0)),
            torch.nn.Linear(proj_dim * 2 + (1 if use_prior else 0), proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_dim, 1)
        )
        self.use_prior = use_prior
        self.tau = tau

    def forward(self, query, cand_feats, cls_prior=None):
        Mloc = cand_feats.shape[0]
        p = F.normalize(self.wp(cand_feats), dim=-1)
        q = F.normalize(self.wt(query.unsqueeze(0) if query.dim()==1 else query), dim=-1)
        q = q.squeeze(0)
        dot = torch.matmul(p, q) / self.tau
        q_exp = q.unsqueeze(0).expand(Mloc, -1)
        if cls_prior is None:
            prior = torch.zeros(Mloc, 1, device=cand_feats.device)
        else:
            prior = cls_prior.unsqueeze(1)
        mlp_in = torch.cat([p, q_exp, prior], dim=-1)
        mlp_logits = self.mlp(mlp_in).squeeze(-1)
        logits = dot + mlp_logits
        return logits


class DotPriorSelector(nn.Module):
    def __init__(self, d_feat, d_query=None, use_prior=True):
        super().__init__()
        d_query = d_query or d_feat
        self.use_prior = use_prior
        self.wp = nn.Linear(d_feat, d_query)   # 只做投影对齐
        self.tau = nn.Parameter(torch.tensor(0.5))

    def forward(self, query, cand_feats, cls_prior=None):
        p = F.normalize(self.wp(cand_feats), dim=-1)
        q = F.normalize(query, dim=-1)
        dot = torch.matmul(p, q) / self.tau.clamp(0.3, 1.0)
        if self.use_prior and cls_prior is not None:
            logits = dot + cls_prior * 0.5
        else:
            logits = dot
        return logits


class DefaultSelector_2(nn.Module):
    def __init__(self, d_feat, d_query=None, proj_dim=512, use_prior=True):
        super().__init__()
        d_query = d_query or d_feat
        self.use_prior = use_prior

        # 投影层
        self.wp = nn.Linear(d_feat, proj_dim)
        self.wt = nn.Linear(d_query, proj_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 1)
        )

        # 可学习温度
        self.tau = nn.Parameter(torch.tensor(0.5))

        # dot 分支 LayerNorm
        self.norm_dot = nn.LayerNorm(1)

        self.weight = nn.Linear(3, 1, bias=False)
        # 初始平均加权，避免一开始极端
        with torch.no_grad():
            self.weight.weight[:] = torch.tensor([1.0, 1.0, 1.0])

    def forward(self, query, cand_feats, cls_prior=None):
        Mloc = cand_feats.shape[0]
        device = cand_feats.device

        # 1. 投影 + 归一化
        p = F.normalize(self.wp(cand_feats), dim=-1)
        q = F.normalize(self.wt(query.unsqueeze(0) if query.dim() == 1 else query),
                        dim=-1).squeeze(0)

        # 2. scaled dot + 温度 + LayerNorm
        tau = self.tau.clamp(0.3, 1.0)
        dot = torch.matmul(p, q) / tau
        # dot = self.norm_dot(dot.unsqueeze(-1)).squeeze(-1)

        # 3. MLP 分支
        q_exp = q.unsqueeze(0).expand(Mloc, -1)
        mlp_logits = self.mlp(torch.cat([p, q_exp], dim=-1)).squeeze(-1)

        # 4. prior bias
        if self.use_prior and cls_prior is not None:
            prior_bias = cls_prior * 0.5
        else:
            prior_bias = torch.zeros(Mloc, device=device)

        three = torch.stack([dot, mlp_logits, prior_bias], dim=1)  # [M, 3]
        logits = self.weight(three).squeeze(1)                     # [M]

        # print(logits, logits.shape)
        return logits

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_topk_relaxed(logits, K, tau=0.5, training=True, eps=1e-12):
    """
    logits: 1D tensor [M]
    K: int <= M
    tau: fixed temperature (float)
    training: bool -> if True use Gumbel perturb + relaxed soft K-hot (STE surrogate)
                     if False deterministic hard top-k (no noise)
    Returns:
      y: surrogate weights (sum == K); in eval this is exact K-hot.
      hard: exact K-hot mask (0/1)
      topk_idx: indices of selected positions
    """
    M = logits.numel()
    K = min(int(K), M)
    device = logits.device

    if K <= 0:
        return torch.zeros_like(logits, device=device), torch.zeros_like(logits, device=device), torch.empty(0, dtype=torch.long, device=device)

    if training:
        # 1) sample Gumbel and perturb logits
        g = sample_gumbel((M,), device=device)
        logits_pert = logits + g

        # 2) soft relaxed K-hot: softmax over perturbed logits scaled to sum==K
        y_soft = F.softmax(logits_pert / (tau + eps), dim=0) * float(K)  # sum == K

        # 3) hard top-k according to perturbed logits (sample without replacement)
        topk_vals, topk_idx = torch.topk(logits_pert, K, largest=True)
        hard = torch.zeros_like(logits, device=device)
        hard[topk_idx] = 1.0

        # 4) STE: forward uses hard, backward uses y_soft
        y = hard.detach() - y_soft.detach() + y_soft
        return y, hard, topk_idx
    else:
        # deterministic evaluation: top-k on original logits
        topk_vals, topk_idx = torch.topk(logits, K, largest=True)
        hard = torch.zeros_like(logits, device=device)
        hard[topk_idx] = 1.0
        return hard, hard, topk_idx

def select_features_network_mmr_variable_k_gpu_purefeature_2(
    features, attn_map, query, selector_model, M=100, lambda_mmr=1.0, tau=0.002, method='attn_mass', alpha=0.9, min_k=1, max_k=30,
    min_token_num=1, training=False,
    return_details=False, device=None,
    # --- soft MMR specific ---
    enable_soft_mmr=True, beta_mmr=0.2, mmr_exclude_self=True, mmr_normalize_redundancy=True,
    ):
    """
    MMR + variable k per token (GPU-friendly) + integrated selector + 可微软化 MMR.
    新增 soft-MMR 参数:
    enable_soft_mmr: bool, 是否启用可微软化 MMR（默认 True）
    beta_mmr: float, redundancy 惩罚系数（越大越鼓励多样性）
    mmr_exclude_self: bool, 是否在 S 中置零对角（避免 self contribute）
    mmr_normalize_redundancy: bool, 是否把 redundancy 标准化（/sum(p0) 或 /max）以更稳定调参

    返回与之前一致；若 return_details=True，会在 details 中额外返回 'logits_raw' 和 'logits_mmr' 供分析。
    """

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d_q]
        batch_list = []
        # print("xxxxxxxxxxxxxxx")
        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            idx_candidates = torch.arange(P, device=device)
            
            cand_scores = scores[idx_candidates]
            order = torch.argsort(cand_scores, descending=True)
            idx_candidates = idx_candidates[order][:M]


            M_actual = idx_candidates.numel()
            if M_actual == 0:
                # 没有候选，返回仅 query
                batch_list.append([])
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            cand_scores = scores[idx_candidates]  # descending by construction
            # print(scores.max(), scores.min())
            k_cur = 10

            # ########################
            # 用 selector 在候选池上打分（可训练 head）
            # ########################
            # print(idx_candidates)
            cand_feats = feats_b[idx_candidates]  # [M_actual, d]
            cand_feats_norm = feats_norm_b[idx_candidates]  # [M_actual, d] (已归一化)
            S = torch.matmul(cand_feats_norm, cand_feats_norm.t()) 

            cls_prior = cand_scores  # prior
            # print(cls_prior)
            logits_raw = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]
            # print(logits_raw.shape)

            # print(cand_scores.sum())

            logits_mmr = logits_raw

            if enable_soft_mmr and M_actual > 1 and False: 
                # 计算相似矩阵 S (cosine), 排除对角可选
                S = torch.matmul(cand_feats_norm, cand_feats_norm.t())  # [M_actual, M_actual]

                
                if mmr_exclude_self:
                    # 把对角置0，避免 self contribute
                    try:
                        S = S.clone()
                        S.fill_diagonal_(0.0)
                    except Exception:
                        # 兼容旧 pytorch: 手动减去对角
                        diag = torch.diag(torch.diag(S))
                        S = S - diag

                # base selection probability
                p0 = torch.sigmoid(logits_raw)  # [M_actual]
                # redundancy: S @ p0
                redundancy = torch.matmul(S, p0)  # [M_actual]

                if mmr_normalize_redundancy:
                    # 标准化 redundancy，避免 scale 问题：常见选择是 / (p0.sum()+eps) 或 / redundancy.max()
                    denom = p0.sum().clamp(min=eps)
                    redundancy = redundancy / denom
                    # 也可用 max normalization： redundancy = redundancy / (redundancy.max()+eps)

                # 调整 logits
                logits_mmr = logits_raw - beta_mmr * redundancy
            # print(logits_mmr)
            # logits -> soft scores

            # 替换前（示意）：
            # p = torch.softmax(logits_mmr, 0)
            # topk_vals, topk_pos = torch.topk(p, k_cur, dim=0)
            # hard = torch.zeros_like(p); hard[topk_pos] = 1.0
            # if training: y = (hard - p).detach() + p
            # else: y = hard

            # 替换后：用固定 tau（例如 tau_selector=0.5）
            tau_selector = 0.5  # 训练时/调试时可以改为 0.3/0.2 或 0.8 等
            y, hard, topk_pos = gumbel_topk_relaxed(logits_mmr, K=k_cur, tau=tau_selector, training=training)
            # 使用 y 计算 soft_agg、surrogate_weights 等
            soft_agg = torch.matmul(y, cand_feats)  # [d]

            kept_idx = idx_candidates[topk_pos]  # original global indices

            # soft aggregation
            # soft_agg = torch.matmul(y, cand_feats)  # [d]

            # 保证至少 min_token_num
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 构建返回 token feats（query 在前）
            token_patches = feats_b[kept_idx]  # [k_cur, d]
            token_feats = token_patches
            batch_list.append(token_feats)

            # fill details
            if return_details:
                # details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx
                details['surrogate_weights'][bi][ti] = y if training else y.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

def select_features_network_mmr_variable_k_gpu_purefeature_r(
    features, attn_map, query, selector_model, M=100, lambda_mmr=1.0, tau=0.002, method='attn_mass', alpha=0.9, min_k=1, max_k=30,
    min_token_num=1, training=False, k_cur = 10,
    return_details=False, device=None,
    # --- soft MMR specific ---
    enable_soft_mmr=True, beta_mmr=0.2, mmr_exclude_self=True, mmr_normalize_redundancy=True,
    ):
    """
    MMR + variable k per token (GPU-friendly) + integrated selector + 可微软化 MMR.
    新增 soft-MMR 参数:
    enable_soft_mmr: bool, 是否启用可微软化 MMR（默认 True）
    beta_mmr: float, redundancy 惩罚系数（越大越鼓励多样性）
    mmr_exclude_self: bool, 是否在 S 中置零对角（避免 self contribute）
    mmr_normalize_redundancy: bool, 是否把 redundancy 标准化（/sum(p0) 或 /max）以更稳定调参

    返回与之前一致；若 return_details=True，会在 details 中额外返回 'logits_raw' 和 'logits_mmr' 供分析。
    """

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d_q]
        batch_list = []
        # print("xxxxxxxxxxxxxxx")
        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            idx_candidates = torch.arange(P, device=device)
            
            cand_scores = scores[idx_candidates]
            order = torch.argsort(cand_scores, descending=True)
            idx_candidates = idx_candidates[order][:M]


            M_actual = idx_candidates.numel()
            if M_actual == 0:
                # 没有候选，返回仅 query
                batch_list.append([])
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            cand_scores = scores[idx_candidates]  # descending by construction
            # print(scores.max(), scores.min())
            

            # ########################
            # 用 selector 在候选池上打分（可训练 head）
            # ########################
            # print(idx_candidates)
            cand_feats = feats_b[idx_candidates]  # [M_actual, d]
            cand_feats_norm = feats_norm_b[idx_candidates]  # [M_actual, d] (已归一化)
            S = torch.matmul(cand_feats_norm, cand_feats_norm.t()) 

            cls_prior = cand_scores  # prior
            # print(cls_prior)
            logits_raw = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]
            # print(logits_raw.shape)

            # print(cand_scores.sum())

            logits_mmr = logits_raw

            if enable_soft_mmr and M_actual > 1 and False: 
                # 计算相似矩阵 S (cosine), 排除对角可选
                S = torch.matmul(cand_feats_norm, cand_feats_norm.t())  # [M_actual, M_actual]

                
                if mmr_exclude_self:
                    # 把对角置0，避免 self contribute
                    try:
                        S = S.clone()
                        S.fill_diagonal_(0.0)
                    except Exception:
                        # 兼容旧 pytorch: 手动减去对角
                        diag = torch.diag(torch.diag(S))
                        S = S - diag

                # base selection probability
                p0 = torch.sigmoid(logits_raw)  # [M_actual]
                # redundancy: S @ p0
                redundancy = torch.matmul(S, p0)  # [M_actual]

                if mmr_normalize_redundancy:
                    # 标准化 redundancy，避免 scale 问题：常见选择是 / (p0.sum()+eps) 或 / redundancy.max()
                    denom = p0.sum().clamp(min=eps)
                    redundancy = redundancy / denom
                    # 也可用 max normalization： redundancy = redundancy / (redundancy.max()+eps)

                # 调整 logits
                logits_mmr = logits_raw - beta_mmr * redundancy
            # print(logits_mmr)
            # logits -> soft scores
            p = torch.softmax(logits_mmr, 0)  # [M_actual]
            # print(p)
            # top-k on p
            topk_vals, topk_pos = torch.topk(p, k_cur, dim=0)  # positions in candidate pool
            # print(topk_vals, topk_pos)
            kept_idx = idx_candidates[topk_pos]  # original global indices

            # hard mask & STE surrogate
            hard = torch.zeros_like(p)
            hard[topk_pos] = 1.0
            if training:
                y = (hard - p).detach() + p
            else:
                y = hard


            # soft aggregation
            soft_agg = torch.matmul(y, cand_feats)  # [d]

            # 保证至少 min_token_num
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 构建返回 token feats（query 在前）
            token_patches = feats_b[kept_idx]  # [k_cur, d]
            token_feats = token_patches
            batch_list.append(token_feats)

            # fill details
            if return_details:
                details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx
                details['surrogate_weights'][bi][ti] = y if training else y.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

def select_features_network_mmr_variable_k_gpu_purefeature_finetune_(
    features, attn_map, query, selector_model, M=100, 
    tau=0.002, method='attn_mass', alpha=0.9, min_k=1, max_k=30,
    min_token_num=1, training=False, k_cur=10,
    return_details=False, device=None,
    # --- soft MMR specific (kept disabled as requested) ---
    enable_soft_mmr=False,
    beta_mmr=0.2, mmr_exclude_self=True, mmr_normalize_redundancy=True,
):
    """
    Modified to always return exactly up to k_cur (or min_token_num) features per token,
    even during training, while preserving gradients via soft weights on selected positions.
    """

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs, P, d]
    selected = []
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]
        feats_norm_b = feats_norm[bi]
        attn_b = attn_map[bi]
        q_b = query[bi]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # --- Step 1: Candidate pool ---
            idx_candidates = torch.arange(P, device=device)
            order = torch.argsort(scores, descending=True)
            idx_candidates = idx_candidates[order][:M]
            M_actual = idx_candidates.numel()

            if M_actual == 0:
                empty_tensor = torch.zeros(0, d, device=device)
                batch_list.append(empty_tensor)
                if return_details:
                    for key in details:
                        details[key][bi][ti] = torch.zeros(0, device=device) if key != 'soft_agg' else torch.zeros(d, device=device)
                continue

            # --- Step 2: Candidate features ---
            cand_feats = feats_b[idx_candidates]           # [M_actual, d]
            cls_prior = scores[idx_candidates]

            # --- Step 3: Selector scoring ---
            logits_raw = selector_model(q_b[0], cand_feats, cls_prior=cls_prior)  # ⚠️ fixed: q_b[ti], not q_b[0]
            logits_mmr = logits_raw

            # --- Step 4: Soft prob and top-k ---
            p = torch.softmax(logits_mmr, dim=0)
            k_select = min(k_cur, M_actual)
            topk_vals, topk_pos = torch.topk(p, k_select, dim=0)
            kept_idx_global = idx_candidates[topk_pos]

            # --- Ensure min_token_num ---
            if kept_idx_global.numel() < min_token_num:
                need = min_token_num - kept_idx_global.numel()
                kept_mask = torch.zeros(M_actual, dtype=torch.bool, device=device)
                local_kept = (idx_candidates.unsqueeze(1) == kept_idx_global.unsqueeze(0)).any(dim=1)
                kept_mask = local_kept
                remaining = (~kept_mask).nonzero(as_tuple=True)[0]
                if remaining.numel() > 0:
                    add_pos = remaining[:need]
                    kept_idx_global = torch.cat([kept_idx_global, idx_candidates[add_pos]], dim=0)
                # Update k_select to actual kept number
                k_actual = kept_idx_global.numel()
            else:
                k_actual = kept_idx_global.numel()

            # --- Step 5: Build soft weights ONLY on kept positions ---
            # Create a weight vector of length M_actual, zero everywhere except top-k (and added) positions
            y_full = torch.zeros_like(p)  # [M_actual]
            # Get positions in candidate pool corresponding to kept_idx_global
            # Since kept_idx_global = idx_candidates[kept_local_pos], we need to find kept_local_pos
            # We can reconstruct it by matching
            kept_local_pos = []
            for idx in kept_idx_global:
                pos = (idx_candidates == idx).nonzero(as_tuple=True)[0]
                if pos.numel() > 0:
                    kept_local_pos.append(pos[0])
            kept_local_pos = torch.stack(kept_local_pos) if kept_local_pos else torch.empty(0, dtype=torch.long, device=device)
            
            if training:
                # Use soft weights from p, but only on selected positions
                y_selected = p[kept_local_pos]  # [k_actual]
                # Optional: renormalize? Usually not needed.
                # Create output features: [k_actual, d]
                output_feats = y_selected.unsqueeze(-1) * cand_feats[kept_local_pos]  # ✅ Only k_actual features
                # For STE consistency in details, we can store full y, but output only k
                y_for_details = torch.zeros_like(p)
                y_for_details[kept_local_pos] = y_selected
            else:
                # Hard: just return raw features
                output_feats = feats_b[kept_idx_global]  # [k_actual, d]
                y_for_details = torch.zeros_like(p)
                y_for_details[kept_local_pos] = 1.0

            batch_list.append(output_feats)

            # --- Fill details ---
            if return_details:
                soft_agg = torch.matmul(y_for_details, cand_feats)
                details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx_global
                details['surrogate_weights'][bi][ti] = y_for_details if training else y_for_details.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

def get_3d_grid(grid_size=16, device='cpu'):
    """预计算 16x16x16 → 4096 的 3D 坐标"""
    r = torch.arange(grid_size, device=device)
    grid = torch.stack(torch.meshgrid(r, r, r, indexing='ij'), dim=-1)  # (16,16,16,3)
    return grid.view(-1, 3).float()  # (4096, 3)

def spatially_aware_feature_pooling(
    features,           # (b, 4096, l)
    attention_map,     # (b, 10, 4096)
    k=10,              # output tokens per organ
    M=200,              # candidate pool size
    local_radius=1,    # pooling neighborhood (in voxel units)
    device=None,
    normalize_coords=True,
):
    if device is None:
        device = features.device
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]  # 10
    grid_size = round(total_tokens ** (1/3))
    assert grid_size ** 3 == total_tokens

    # 预计算 3D 坐标（共享）
    coords_3d = get_3d_grid(grid_size, device)  # (4096, 3)
    if normalize_coords:
        coords_3d = coords_3d / (grid_size - 1)  # 归一化到 [0,1]

    # 输出容器
    output_tokens = torch.zeros(b, num_organs, k, l, device=device)
    
    for bi in range(b):
        feats = features[bi]          # (4096, l)
        attn = attention_map[bi]      # (10, 4096)
        
        for oi in range(num_organs):
            # --- Step 1: Top-M candidates ---
            scores = attn[oi]  # (4096,)
            topM_vals, topM_idx = torch.topk(scores, M, dim=-1)  # (M,)
            if topM_idx.numel() == 0:
                continue

            cand_feats = feats[topM_idx]      # (M, l)
            cand_coords = coords_3d[topM_idx] # (M, 3)

            # --- Step 2: Diverse sampling via FPS on combined space ---
            # 联合空间：[norm(coord), norm(feature)]，平衡几何与语义
            feat_norm = F.normalize(cand_feats, dim=-1)
            coord_norm = cand_coords  # already normalized

            # 拼接空间-特征联合表示 (可调权重)
            alpha = 0.5  # 可学习 or 调参
            joint_repr = torch.cat([
                alpha * coord_norm,
                (1 - alpha) * feat_norm
            ], dim=-1)  # (M, 3 + l)

            # Farthest Point Sampling (FPS)
            selected_idx_local = fps_sampling(joint_repr, k)  # (k,) indices in [0, M)

            # Map back to global indices
            selected_global_idx = topM_idx[selected_idx_local]  # (k,)

            # --- Step 3: Local attention-weighted pooling ---
            pooled_feats = []
            for i, center_idx in enumerate(selected_global_idx):
                # 获取中心点3D坐标
                center_coord = coords_3d[center_idx]  # (3,)
                # 计算所有4096点到中心的L∞距离（便于立方体邻域）
                dist = torch.abs(coords_3d - center_coord).max(dim=-1).values  # (4096,)
                # 邻域mask
                mask = dist <= local_radius  # (4096,)
                if mask.sum() == 0:
                    pooled = feats[center_idx]
                else:
                    # 用原始 attention 作为权重（可选：重新归一化）
                    local_attn = attention_map[bi, oi, mask]  # (N_local,)
                    local_feats = feats[mask]                 # (N_local, l)
                    # 归一化权重
                    weights = F.softmax(local_attn, dim=0)    # (N_local,)
                    pooled = torch.sum(weights.unsqueeze(-1) * local_feats, dim=0)  # (l,)
                pooled_feats.append(pooled)
            
            output_tokens[bi, oi] = torch.stack(pooled_feats, dim=0)  # (k, l)

    return output_tokens  # (b, 10, k, l)


def fps_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS)
    points: (N, D) tensor
    returns: (num_samples,) long tensor of indices
    """
    N, D = points.shape
    device = points.device

    if num_samples >= N:
        return torch.arange(N, dtype=torch.long, device=device)

    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)

    # Randomly select first point (make sure it's a scalar int)
    farthest = torch.randint(0, N, (1,), device=device).item()  # 🔥 .item() to get int

    for i in range(num_samples):
        indices[i] = farthest
        centroid = points[farthest]  # (D,)
        dist = torch.norm(points - centroid, dim=1)  # (N,)
        distances = torch.min(distances, dist)
        farthest = torch.argmax(distances).item()  # 🔥 .item() again!

    return indices

def batched_hybrid_fps(coords, features, num_samples, alpha=0.5):
    """
    Optimized Batched FPS that balances Spatial and Semantic distances.
    
    Args:
        coords: (B, N, 3) - Spatial coordinates (normalized [0,1])
        features: (B, N, L) - Semantic features (MUST be normalized)
        num_samples: int - Number of points to sample (k)
        alpha: float - Weight for spatial distance (0.0 to 1.0).
    
    Returns:
        centroids: (B, num_samples) indices
    """
    B, N, D = coords.shape
    device = coords.device
    
    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    # Initialize distances to infinity
    min_dists = torch.full((B, N), float('inf'), device=device)
    
    # Randomly select the first point for each batch
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(num_samples):
        centroids[:, i] = farthest
        
        # 1. Get current pivot data
        curr_coord = coords[batch_indices, farthest, :].unsqueeze(1) # (B, 1, 3)
        curr_feat = features[batch_indices, farthest, :].unsqueeze(1) # (B, 1, L)
        
        # 2. Compute Spatial Distance (Euclidean)
        # Range approx [0, 1.73] for unit cube
        dist_spatial = torch.norm(coords - curr_coord, dim=-1) # (B, N)
        
        # 3. Compute Semantic Distance (Cosine-based)
        # Range [0, 2] (0=same, 1=orthogonal, 2=opposite)
        # Using (1 - dot_product) is faster than calculating euclidean dist on features
        dot_prod = torch.sum(features * curr_feat, dim=-1) 
        dist_semantic = 1.0 - dot_prod
        
        # 4. Weighted Combination
        # This ensures feature length (L) does not dominate the metric
        dist_combined = alpha * dist_spatial + (1 - alpha) * dist_semantic
        
        # 5. Update minimum distance to any selected point
        min_dists = torch.min(min_dists, dist_combined)
        
        # 6. Select point with largest minimum distance
        farthest = torch.argmax(min_dists, dim=-1)

    return centroids

def spatially_aware_feature_pooling_v2(
    features,           # (b, 4096, l)
    attention_map,      # (b, num_organs, 4096)
    k=10,               # output tokens per organ
    M=200,              # candidate pool size
    local_radius=1,     # pooling neighborhood (voxel units)
    alpha=0.5,          # 0.5 = balanced space/semantics
    normalize_coords=True,
    device=None
):
    if device is None:
        device = features.device
        
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    grid_size = round(total_tokens ** (1/3))
    
    # 1. Prepare Grid (Shared)
    coords_3d = get_3d_grid(grid_size, device) # (4096, 3)
    
    # Normalize coords to [0,1] for FPS stability
    coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d

    # ----------------------------------------------------------------
    # Step 1: Vectorized Top-M Selection
    # ----------------------------------------------------------------
    # Flatten Batch and Organs to treat them uniformly: (B*O, 4096)
    # This allows us to process everything in one massive batch
    flat_attn = attention_map.view(-1, total_tokens) 
    
    # Get Top-M indices: (B*O, M)
    topM_vals, topM_idx = torch.topk(flat_attn, M, dim=-1)
    
    # Expand features for gathering: (B, 1, 4096, L) -> (B*O, 4096, L)
    expanded_feats = features.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, l)
    
    # Gather Candidate Features: (B*O, M, L)
    cand_feats = torch.gather(expanded_feats, 1, topM_idx.unsqueeze(-1).expand(-1, -1, l))
    
    # Expand Coords for gathering: (1, 4096, 3) -> (B*O, 4096, 3)
    expanded_coords = coords_3d.unsqueeze(0).expand(b * num_organs, -1, -1)
    expanded_coords_norm = coords_norm.unsqueeze(0).expand(b * num_organs, -1, -1)
    
    # Gather Candidate Coords (Normalized for FPS): (B*O, M, 3)
    cand_coords_norm = torch.gather(expanded_coords_norm, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 3))

    # ----------------------------------------------------------------
    # Step 2: Decoupled Hybrid FPS
    # ----------------------------------------------------------------
    # Normalize features for Cosine Distance calculation
    cand_feats_norm = F.normalize(cand_feats, dim=-1)
    
    # Run FPS on the M candidates
    # Returns indices relative to the M candidates (0 to M-1)
    fps_local_idx = batched_hybrid_fps(cand_coords_norm, cand_feats_norm, k, alpha=alpha)
    
    # Map back to global indices (0 to 4095): (B*O, k)
    selected_global_idx = torch.gather(topM_idx, 1, fps_local_idx)

    # ----------------------------------------------------------------
    # Step 3: Vectorized Local Pooling (Masked)
    # ----------------------------------------------------------------
    # We pool from the ORIGINAL full grid based on the selected centers
    
    # Get Center Coords (Un-normalized for radius calculation): (B*O, k, 3)
    center_coords = torch.gather(expanded_coords, 1, selected_global_idx.unsqueeze(-1).expand(-1, -1, 3))
    
    # Calculate L-infinity distance matrix: (B*O, k, 4096)
    # |center - all_points|
    diff = torch.abs(center_coords.unsqueeze(2) - expanded_coords.unsqueeze(1)) 
    dist_linf = torch.max(diff, dim=-1).values 
    
    # Create Mask: (B*O, k, 4096)
    mask = dist_linf <= local_radius
    
    # Prepare Attention Weights
    # Expand original attention: (B*O, 1, 4096) -> (B*O, k, 4096)
    masked_attn = flat_attn.unsqueeze(1).expand(-1, k, -1).clone()
    
    # Apply mask (set non-neighbors to -inf)
    masked_attn = masked_attn.masked_fill(~mask, float('-inf'))
    
    # Softmax to get weights (B*O, k, 4096)
    weights = F.softmax(masked_attn, dim=-1)
    
    # Handle empty neighborhoods (all -inf -> nan) by zeroing them out
    weights = torch.nan_to_num(weights, nan=0.0)

    # Weighted Sum Pooling: (B*O, k, 4096) @ (B*O, 4096, L) -> (B*O, k, L)
    pooled_feats = torch.bmm(weights, expanded_feats)
    
    # Reshape back to (b, num_organs, k, l)
    output_tokens = pooled_feats.view(b, num_organs, k, l)
    
    return output_tokens



def spatially_aware_feature_pooling_v3(
    features,           # (b, 4096, l)
    attention_map,      # (b, num_organs, 4096)
    question_embedding, # (b, q_dim) [NEW]
    fusion_module,      # nn.Module [NEW]
    k=10,               
    M=200,              
    local_radius=1,     
    alpha=0.5,          
    normalize_coords=True,
    device=None
):
    if device is None:
        device = features.device
        
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    grid_size = round(total_tokens ** (1/3))
    
    # 1. Prepare Grid
    coords_3d = get_3d_grid(grid_size, device)
    coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d

    # ----------------------------------------------------------------
    # Step 1: Vectorized Top-M Selection
    # ----------------------------------------------------------------
    flat_attn = attention_map.view(-1, total_tokens) 
    topM_vals, topM_idx = torch.topk(flat_attn, M, dim=-1)
    
    expanded_feats = features.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, l)
    cand_feats = torch.gather(expanded_feats, 1, topM_idx.unsqueeze(-1).expand(-1, -1, l))
    
    expanded_coords = coords_3d.unsqueeze(0).expand(b * num_organs, -1, -1)
    expanded_coords_norm = coords_norm.unsqueeze(0).expand(b * num_organs, -1, -1)
    cand_coords_norm = torch.gather(expanded_coords_norm, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 3))

    # ----------------------------------------------------------------
    # [NEW] Prepare Question Embedding for the flattened batch
    # ----------------------------------------------------------------
    # question: (b, q_dim) -> (b, num_organs, q_dim) -> (b*num_organs, q_dim)
    # 我们需要让每个 organ 的处理都感知到同一个问题
    flat_question = question_embedding.expand(-1, num_organs, -1).reshape(-1, question_embedding.shape[-1])

    # ----------------------------------------------------------------
    # Step 2: Decoupled Hybrid FPS with Question Awareness
    # ----------------------------------------------------------------
    # 归一化特征 (FPS内部会再次处理融合后的特征，但这里先归一化原始特征是个好习惯)
    cand_feats_norm = F.normalize(cand_feats, dim=-1)
    
    # 调用修改后的 FPS
    # 注意：这里传入了 flat_question 和 fusion_module
    fps_local_idx = batched_hybrid_fps_(
        coords=cand_coords_norm, 
        features=cand_feats_norm, 
        num_samples=k, 
        question_emb=flat_question, # 传入对齐后的问题向量
        fusion_module=fusion_module, # 传入可学习模块
        alpha=alpha
    )
    
    selected_global_idx = torch.gather(topM_idx, 1, fps_local_idx)

    # ----------------------------------------------------------------
    # Step 3: Vectorized Local Pooling (Masked)
    # ----------------------------------------------------------------
    # ... (这部分代码保持不变，因为Pooling是基于位置的聚合) ...
    
    center_coords = torch.gather(expanded_coords, 1, selected_global_idx.unsqueeze(-1).expand(-1, -1, 3))
    diff = torch.abs(center_coords.unsqueeze(2) - expanded_coords.unsqueeze(1)) 
    dist_linf = torch.max(diff, dim=-1).values 
    mask = dist_linf <= local_radius
    
    masked_attn = flat_attn.unsqueeze(1).expand(-1, k, -1).clone()
    masked_attn = masked_attn.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_attn, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    pooled_feats = torch.bmm(weights, expanded_feats)
    output_tokens = pooled_feats.view(b, num_organs, k, l)
    
    return output_tokens


def batched_hybrid_fps_v2___(coords, features, organ_probs, num_samples, 
                          question_emb=None, fusion_module=None, alpha=0.5):
    """
    Args:
        coords: (B, N, 3)
        features: (B, N, L)
        organ_probs: (B, N, Num_Classes) - [NEW] 器官预测概率
        num_samples: int
        question_emb: (B, Q_dim)
        fusion_module: nn.Module
        alpha: float
    """
    B, N, D = coords.shape
    device = coords.device
    
    # -----------------------------------------------------------
    # [NEW] Feature Adaptation with Organ Prior
    # -----------------------------------------------------------
    if question_emb is not None and fusion_module is not None:
        # 我们把 视觉特征 + 器官概率 拼在一起传给融合模块
        # 这样融合模块可以学会：当问题问"Kidney"时，给 organ_probs 中 Kidney 通道高的点更高的权重
        # fusion_module 需要修改输入维度以接受 organ_probs
        metric_features = fusion_module(features, organ_probs, question_emb)
        metric_features = F.normalize(metric_features, dim=-1)
    else:
        metric_features = features

    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float('inf'), device=device)
    
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(num_samples):
        centroids[:, i] = farthest
        
        curr_coord = coords[batch_indices, farthest, :].unsqueeze(1)
        curr_feat = metric_features[batch_indices, farthest, :].unsqueeze(1)
        
        dist_spatial = torch.norm(coords - curr_coord, dim=-1)
        
        dot_prod = torch.sum(metric_features * curr_feat, dim=-1)
        dist_semantic = 1.0 - dot_prod
        
        dist_combined = alpha * dist_spatial + (1 - alpha) * dist_semantic
        
        min_dists = torch.min(min_dists, dist_combined)
        farthest = torch.argmax(min_dists, dim=-1)

    return centroids

def spatially_aware_feature_pooling_v5(
    features,           # (b, 4096, l)
    attention_map,      # (b, num_organs, 4096)
    organ_logits,       # (b, 4096, 10) [NEW] 你的先验 logits
    question_embedding, # (b, q_dim)
    fusion_module,      # nn.Module (用于 FPS 距离计算)
    output_embedder,    # [NEW] SimpleSemanticPositionalEmbedder
    k=10,               
    M=200,              
    local_radius=1,     
    alpha=0.5,          
    normalize_coords=True,
    device=None
):
    if device is None:
        device = features.device
        
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    num_classes = organ_logits.shape[-1]
    grid_size = round(total_tokens ** (1/3))
    
    # -----------------------------------------------------------
    # 0. 预处理 Organ Logits
    # -----------------------------------------------------------
    # 必须先 Softmax，因为 Linear 层处理分布比处理 Logits 更容易收敛
    organ_probs = F.softmax(organ_logits, dim=-1) # (b, 4096, 10)

    # 1. Prepare Grid
    coords_3d = get_3d_grid(grid_size, device)
    coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d

    # -----------------------------------------------------------
    # Step 1: Vectorized Top-M Selection
    # -----------------------------------------------------------
    flat_attn = attention_map.view(-1, total_tokens) 
    topM_vals, topM_idx = torch.topk(flat_attn, M, dim=-1)
    
    # Gather Visual Features
    expanded_feats = features.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, l)
    cand_feats = torch.gather(expanded_feats, 1, topM_idx.unsqueeze(-1).expand(-1, -1, l))
    
    # Gather Coords
    expanded_coords_norm = coords_norm.unsqueeze(0).expand(b * num_organs, -1, -1)
    cand_coords_norm = torch.gather(expanded_coords_norm, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 3))
    
    # [NEW] Gather Organ Probs
    # 我们需要把 organ_probs 也选出来，参与 FPS 和 Pooling
    expanded_organ_probs = organ_probs.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, num_classes)
    cand_organ_probs = torch.gather(expanded_organ_probs, 1, topM_idx.unsqueeze(-1).expand(-1, -1, num_classes))

    # -----------------------------------------------------------
    # Step 2: Question-Guided FPS (带 Organ Prior)
    # -----------------------------------------------------------
    # 这里复用之前的 batched_hybrid_fps_v2
    # 即使不用 Text Embedding，fusion_module 依然可以利用 organ_probs 来调整采样权重
    flat_question = question_embedding.expand(-1, num_organs, -1).reshape(-1, question_embedding.shape[-1])
    cand_feats_norm = F.normalize(cand_feats, dim=-1)
    
    fps_local_idx = batched_hybrid_fps_v2___(
        coords=cand_coords_norm, 
        features=cand_feats_norm, 
        organ_probs=cand_organ_probs, # 传入概率
        num_samples=k, 
        question_emb=flat_question, 
        fusion_module=fusion_module, 
        alpha=alpha
    )
    
    selected_global_idx = torch.gather(topM_idx, 1, fps_local_idx)

    # -----------------------------------------------------------
    # Step 3: Pooling (Visual + Coord + Organ)
    # -----------------------------------------------------------
    # 准备全量数据用于 Pooling
    expanded_coords = coords_3d.unsqueeze(0).expand(b * num_organs, -1, -1)
    
    # 计算 Pooling 权重 (基于距离)
    center_coords = torch.gather(expanded_coords, 1, selected_global_idx.unsqueeze(-1).expand(-1, -1, 3))
    diff = torch.abs(center_coords.unsqueeze(2) - expanded_coords.unsqueeze(1)) 
    dist_linf = torch.max(diff, dim=-1).values 
    mask = dist_linf <= local_radius
    
    masked_attn = flat_attn.unsqueeze(1).expand(-1, k, -1).clone()
    masked_attn = masked_attn.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_attn, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0) # (B*O, k, 4096)

    # --- 执行 Pooling ---
    # 1. 视觉特征聚合
    pooled_feats = torch.bmm(weights, expanded_feats)
    
    # 2. 坐标聚合 (得到局部区域的重心)
    pooled_coords = torch.bmm(weights, expanded_coords.float())
    
    # 3. [NEW] 器官概率聚合 (得到局部区域的平均器官分布)
    # 例如：如果这个区域一半是肝，一半是背景，结果就是 [0.5, 0, ..., 0.5]
    pooled_organ_probs = torch.bmm(weights, expanded_organ_probs)

    # -----------------------------------------------------------
    # Step 4: Final Embedding (Learnable Fusion)
    # -----------------------------------------------------------
    # 这里调用 SimpleSemanticPositionalEmbedder
    # 它会把 (Visual, XYZ, Probs) 融合成最终 Token
    final_tokens = output_embedder(pooled_feats, pooled_coords, pooled_organ_probs)
    
    # Reshape back to (B, Num_Organs, K, Dim)
    output_tokens = final_tokens.view(b, num_organs, k, -1)
    
    return output_tokens




def spatially_aware_feature_pooling_fast(
    features,           # (b, 4096, l)
    attention_map,      # (b, num_organs, 4096)
    organ_logits,       # (b, 4096, 10) [NEW]
    question_embedding, # (b, q_dim)
    fusion_module,      # nn.Module
    output_embedder,    # [NEW] FastPositionEmbedder
    k=10,               
    M=200,              
    local_radius=1,     
    alpha=0.5,          
    normalize_coords=True,
    device=None
):
    if device is None:
        device = features.device
        
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    grid_size = round(total_tokens ** (1/3))
    
    # -----------------------------------------------------------
    # 0. 预处理 (只做一次 Softmax)
    # -----------------------------------------------------------
    organ_probs = F.softmax(organ_logits, dim=-1) # (b, 4096, 10)

    # 1. Prepare Grid
    coords_3d = get_3d_grid(grid_size, device)
    coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d

    # -----------------------------------------------------------
    # Step 1: Vectorized Top-M Selection (保持原速)
    # -----------------------------------------------------------
    flat_attn = attention_map.view(-1, total_tokens) 
    topM_vals, topM_idx = torch.topk(flat_attn, M, dim=-1)
    
    # Gather Visual Features & Coords
    expanded_feats = features.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, l)
    cand_feats = torch.gather(expanded_feats, 1, topM_idx.unsqueeze(-1).expand(-1, -1, l))
    
    expanded_coords_norm = coords_norm.unsqueeze(0).expand(b * num_organs, -1, -1)
    cand_coords_norm = torch.gather(expanded_coords_norm, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 3))
    
    # [优化点]：这里完全不处理 organ_probs，节省大量显存带宽

    # -----------------------------------------------------------
    # Step 2: FPS (保持原速，不加 organ 距离)
    # -----------------------------------------------------------
    # 既然 attention_map 已经筛选出了特定器官的区域，FPS 只需要负责空间覆盖即可
    # 不需要再把 organ_probs 塞进 FPS 的距离计算里，那样太慢
    
    flat_question = question_embedding.expand(-1, num_organs, -1).reshape(-1, question_embedding.shape[-1])
    cand_feats_norm = F.normalize(cand_feats, dim=-1)
    
    # 使用你最原始、最快的 FPS
    fps_local_idx = batched_hybrid_fps_(
        coords=cand_coords_norm, 
        features=cand_feats_norm, 
        num_samples=k, 
        question_emb=flat_question, 
        fusion_module=fusion_module, 
        alpha=alpha
    )
    
    # 得到最终选中的 K 个点的全局索引
    selected_global_idx = torch.gather(topM_idx, 1, fps_local_idx) # (B*Num_Organs, K)

    # -----------------------------------------------------------
    # Step 3: Pooling (Visual + Coord)
    # -----------------------------------------------------------
    expanded_coords = coords_3d.unsqueeze(0).expand(b * num_organs, -1, -1)
    
    center_coords = torch.gather(expanded_coords, 1, selected_global_idx.unsqueeze(-1).expand(-1, -1, 3))
    diff = torch.abs(center_coords.unsqueeze(2) - expanded_coords.unsqueeze(1)) 
    dist_linf = torch.max(diff, dim=-1).values 
    mask = dist_linf <= local_radius
    
    masked_attn = flat_attn.unsqueeze(1).expand(-1, k, -1).clone()
    masked_attn = masked_attn.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_attn, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)

    # Pool 视觉特征
    pooled_feats = torch.bmm(weights, expanded_feats)
    # Pool 坐标
    pooled_coords = torch.bmm(weights, expanded_coords.float())

    # -----------------------------------------------------------
    # Step 4: [极速注入] Organ Prior Injection
    # -----------------------------------------------------------
    # 关键优化：我们不 Pool organ_probs，直接查表！
    # selected_global_idx 是 FPS 选出的中心点索引
    # 我们认为：中心点是肝脏，那它周围这一小圈 pooling 出来的特征就是肝脏特征
    
    # 1. 准备原始 organ_probs (扩展到 B*Num_Organs)
    # (B, 4096, 10) -> (B, Num_Organs, 4096, 10) -> (B*Num_Organs, 4096, 10)
    flat_organ_probs = organ_probs.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, organ_logits.shape[-1])
    
    # 2. 直接 Gather (Index Select)
    # (B*Num_Organs, 4096, 10) gather by (B*Num_Organs, K) -> (B*Num_Organs, K, 10)
    center_organ_probs = torch.gather(
        flat_organ_probs, 
        1, 
        selected_global_idx.unsqueeze(-1).expand(-1, -1, organ_logits.shape[-1])
    )
    
    # -----------------------------------------------------------
    # Step 5: Final Embedding
    # -----------------------------------------------------------
    # 拼接：[Visual(L), Coord(3), Organ(10)] -> MLP -> Output
    final_tokens = output_embedder(pooled_feats, pooled_coords, center_organ_probs)
    
    output_tokens = final_tokens.view(b, num_organs, k, -1)
    
    return output_tokens

def batched_hybrid_fps_(coords, features, num_samples, 
                       question_emb=None, fusion_module=None, alpha=0.5):
    """
    Args:
        coords: (B, N, 3)
        features: (B, N, L)
        num_samples: int
        question_emb: (B, Q_dim) - [NEW] 问题特征
        fusion_module: nn.Module - [NEW] 用于融合问题和视觉特征的模块
        alpha: float
    """
    B, N, D = coords.shape
    device = coords.device
    
    # -----------------------------------------------------------
    # [NEW] Feature Adaptation Step
    # 如果提供了问题和融合模块，先对特征进行变换
    # -----------------------------------------------------------
    if question_emb is not None and fusion_module is not None:
        # features 变成了 "问题感知的特征"
        # 注意：这里我们不改变原features变量，而是生成用于计算距离的 metric_features
        # 这样最后返回的索引是基于问题选择的，但后续Pooling可以用原始特征或变换后的特征
        metric_features = fusion_module(features, question_emb)
        
        # 重新归一化，因为FPS依赖余弦距离，需要模长为1
        metric_features = F.normalize(metric_features, dim=-1)
    else:
        metric_features = features # 假设输入已经是归一化的

    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float('inf'), device=device)
    
    # 随机选择起始点
    
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)
    
    farthest_list = []

    for i in range(num_samples):
        centroids[:, i] = farthest
        
        # 1. 获取当前中心点数据
        curr_coord = coords[batch_indices, farthest, :].unsqueeze(1)
        curr_feat = metric_features[batch_indices, farthest, :].unsqueeze(1) # 使用变换后的特征
        
        # 2. 空间距离
        dist_spatial = torch.norm(coords - curr_coord, dim=-1)
        
        # 3. 语义距离 (基于问题感知特征的余弦距离)
        # 如果两个点在“问题关注的维度”上差异大，它们会被视为距离远，从而更有可能被同时选中
        dot_prod = torch.sum(metric_features * curr_feat, dim=-1)
        dist_semantic = 1.0 - dot_prod
        
        # 4. 加权组合
        dist_combined = alpha * dist_spatial + (1 - alpha) * dist_semantic
        
        # 5. 更新最小距离
        min_dists = torch.min(min_dists, dist_combined)
        
        # 6. 选择最远点
        farthest = torch.argmax(min_dists, dim=-1)

        farthest_list.append(farthest)
    # print(farthest_list)

    return centroids

def spatially_aware_feature_pooling_optimized(
    features,           # (b, 4096, l)
    attention_map,      # (b, num_organs, 4096)
    organ_logits,       # (b, 4096, 10)
    question_embedding, 
    fusion_module,      
    output_embedder,    
    k=10,               
    M=200,              
    local_radius=1,     
    alpha=0.5,          
    normalize_coords=True,
    device=None
):
    if device is None: device = features.device
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    grid_size = round(total_tokens ** (1/3))
    
    # --- 优化点 1: 预计算 Organ Probs ---
    organ_probs = F.softmax(organ_logits, dim=-1) # (b, 4096, 10)
    
    # --- 优化点 2: 利用 Organ Prior 净化 Attention (可选) ---
    # 假设 Query i 强相关于 Organ i
    # 如果不是一一对应，可以跳过这一行，或者只用 attention_map
    refined_attn = attention_map * organ_probs.transpose(1, 2) # bs 10 total_tokens
    flat_attn = refined_attn.view(-1, total_tokens) 

    # --- Step 1: Top-M ---
    topM_vals, topM_idx = torch.topk(flat_attn, M, dim=-1)
    
    # Gather Visual Features
    expanded_feats = features.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, l)
    cand_feats = torch.gather(expanded_feats, 1, topM_idx.unsqueeze(-1).expand(-1, -1, l))
    
    # Gather Coords
    coords_3d = get_3d_grid(grid_size, device)
    coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d
    expanded_coords_norm = coords_norm.unsqueeze(0).expand(b * num_organs, -1, -1)
    cand_coords_norm = torch.gather(expanded_coords_norm, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 3))

    # --- 优化点 3: 同时也 Gather Organ Probs (为 Pooling 做准备) ---
    flat_organ_probs = organ_probs.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, 10)
    cand_organ_probs = torch.gather(flat_organ_probs, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 10))

    # --- Step 2: FPS (保持不变) ---
    flat_question = question_embedding.expand(-1, num_organs, -1).reshape(-1, question_embedding.shape[-1])
    cand_feats_norm = F.normalize(cand_feats, dim=-1)
    
    fps_local_idx = batched_hybrid_fps_(
        coords=cand_coords_norm, 
        features=cand_feats_norm, 
        num_samples=k, 
        question_emb=flat_question, 
        fusion_module=fusion_module, 
        alpha=alpha
    )
    
    # 得到相对于 Top-M 列表的索引，用于 gather 候选特征
    # 注意：这里不需要转回 global idx 再转回来，直接在 cand_ 列表里操作更快
    
    # --- Step 3: Local Pooling (Visual + Coord + Organ) ---
    # 选中的中心点坐标 (B*NO, K, 3)
    center_coords = torch.gather(cand_coords_norm, 1, fps_local_idx.unsqueeze(-1).expand(-1, -1, 3))
    
    # 计算 Top-M 候选点 到 K 个中心点的距离
    # cand_coords_norm: (B*NO, M, 3)
    # center_coords:    (B*NO, K, 3)
    diff = torch.abs(center_coords.unsqueeze(2) - cand_coords_norm.unsqueeze(1)) 
    dist_linf = torch.max(diff, dim=-1).values # (B*NO, K, M)
    
    mask = dist_linf <= (local_radius / (grid_size - 1)) # 注意归一化后的半径
    
    # 这里的 attention score 应该是原始 TopM 的值
    masked_scores = topM_vals.unsqueeze(1).expand(-1, k, -1).clone() # (B*NO, K, M)
    masked_scores = masked_scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_scores, dim=-1) # (B*NO, K, M)
    weights = torch.nan_to_num(weights, nan=0.0)

    # 统一 Pooling
    pooled_feats = torch.bmm(weights, cand_feats)       # (B*NO, K, L)
    pooled_coords = torch.bmm(weights, cand_coords_norm)# (B*NO, K, 3)
    pooled_organ_probs = torch.bmm(weights, cand_organ_probs) # (B*NO, K, 10) - 修正了边界错位问题

    # --- Step 4: Output ---
    final_tokens = output_embedder(pooled_feats, pooled_coords, pooled_organ_probs)
    output_tokens = final_tokens.view(b, num_organs, k, -1)
    
    return output_tokens

def batched_hybrid_fps_s(coords, features, num_samples, 
                       question_emb=None, fusion_module=None, 
                       scorer_module=None, # [NEW] 传入评分模块
                       alpha=0.):
    """
    Args:
        coords: (B, N, 3)
        features: (B, N, L)
        num_samples: int
        question_emb: (B, Q_dim)
        fusion_module: nn.Module 用于特征融合
        scorer_module: nn.Module [NEW] 用于计算可学习的起始分数
        alpha: float
    """
    B, N, D = coords.shape
    device = coords.device
    
    # -----------------------------------------------------------
    # 1. Feature Adaptation (融合问题与视觉)
    # -----------------------------------------------------------
    if question_emb is not None and fusion_module is not None:
        # metric_features: (B, N, L)
        # 这些特征已经包含了问题的信息
        metric_features = fusion_module(features, question_emb)
        
        # 归一化用于后续的余弦距离计算
        metric_features_norm = F.normalize(metric_features, dim=-1)
    else:
        metric_features = features
        metric_features_norm = F.normalize(features, dim=-1)

    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float('inf'), device=device)
    
    # -----------------------------------------------------------
    # 2. [NEW] Learnable Start Point Selection
    # 使用可学习的 Scorer 计算语义相关性分数
    # -----------------------------------------------------------
    if scorer_module is not None:
        # 输入融合后的特征，让网络判断哪个点最重要
        # metric_features 是未归一化的，保留了模长信息，通常对 MLP 更友好
        start_scores = scorer_module(metric_features) # (B, N)
        
        # 选择分数最高的点作为起始点
        farthest = torch.argmax(start_scores, dim=-1)
    else:
        # Fallback: 如果没有提供 scorer，使用随机或默认逻辑
        farthest = torch.randint(0, N, (B,), device=device)
    
    batch_indices = torch.arange(B, device=device)
    
    # -----------------------------------------------------------
    # 3. FPS Loop
    # -----------------------------------------------------------
    for i in range(num_samples):
        centroids[:, i] = farthest
        
        # 获取当前中心点数据
        curr_coord = coords[batch_indices, farthest, :].unsqueeze(1)
        curr_feat = metric_features_norm[batch_indices, farthest, :].unsqueeze(1)
        
        # 空间距离
        dist_spatial = torch.norm(coords - curr_coord, dim=-1)
        
        # 语义距离 (基于融合特征的余弦距离)
        dot_prod = torch.sum(metric_features_norm * curr_feat, dim=-1)
        dist_semantic = 1.0 - dot_prod
        
        # 加权组合
        dist_combined = alpha * dist_spatial + (1 - alpha) * dist_semantic
        
        # 更新最小距离
        min_dists = torch.min(min_dists, dist_combined)
        
        # 选择最远点
        farthest = torch.argmax(min_dists, dim=-1)

    return centroids

# 假设你在外部的模型类中已经定义了 scorer
# self.start_point_scorer = RelevanceScorer(input_dim=feature_dim)

def spatially_aware_feature_pooling_optimized_2(
    features,           
    attention_map,      
    organ_logits,       
    question_embedding, 
    fusion_module,      
    output_embedder,
    scorer_module,      # [NEW] 需要从外部传入训练好的 scorer
    k=10,               
    M=200,              
    local_radius=1,     
    alpha=0.5,          
    normalize_coords=True,
    device=None
):
    if device is None: device = features.device
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    grid_size = round(total_tokens ** (1/3))
    
    if device is None: device = features.device
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    grid_size = round(total_tokens ** (1/3))
    
    # --- 优化点 1: 预计算 Organ Probs ---
    organ_probs = organ_logits # F.softmax(organ_logits, dim=-1) # (b, 4096, 10)
    
    # --- 优化点 2: 利用 Organ Prior 净化 Attention (可选) ---
    # 假设 Query i 强相关于 Organ i
    # 如果不是一一对应，可以跳过这一行，或者只用 attention_map
    print(attention_map.max(), attention_map.min())
    refined_attn = attention_map * organ_probs.transpose(1, 2) # bs 10 total_tokens
    flat_attn = refined_attn.view(-1, total_tokens) 
    print(refined_attn.max(), refined_attn.min())
    # --- Step 1: Top-M ---
    topM_vals, topM_idx = torch.topk(flat_attn, M, dim=-1)

    # 恢复形状用于后续 gather
    topM_idx_reshaped = topM_idx.view(b, num_organs, M)
    
    # Gather Visual Features
    expanded_feats = features.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, l)
    cand_feats = torch.gather(expanded_feats, 1, topM_idx.unsqueeze(-1).expand(-1, -1, l))
    
    # Gather Coords
    coords_3d = get_3d_grid(grid_size, device)
    coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d
    expanded_coords_norm = coords_norm.unsqueeze(0).expand(b * num_organs, -1, -1)
    cand_coords_norm = torch.gather(expanded_coords_norm, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 3))

    # --- 优化点 3: 同时也 Gather Organ Probs (为 Pooling 做准备) ---
    flat_organ_probs = organ_probs.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, 10)
    cand_organ_probs = torch.gather(flat_organ_probs, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 10))
    
    # --- 准备数据 ---
    # 扩展 Question Embedding 以匹配 (B*Num_Organs)
    flat_question = question_embedding.expand(-1, num_organs, -1).reshape(-1, question_embedding.shape[-1])
    
    # 注意：cand_feats 是原始特征，未归一化，适合传入 fusion_module
    
    # --- 调用 FPS ---
    fps_local_idx = batched_hybrid_fps_s(
        coords=cand_coords_norm, 
        features=cand_feats,        # 传入原始特征
        num_samples=k, 
        question_emb=flat_question, 
        fusion_module=fusion_module, 
        scorer_module=scorer_module, # [NEW] 传入可学习的评分器
        alpha=alpha
    )
    
    # --- Step 3: Local Pooling (Visual + Coord + Organ) ---
    # 选中的中心点坐标 (B*NO, K, 3)
    center_coords = torch.gather(cand_coords_norm, 1, fps_local_idx.unsqueeze(-1).expand(-1, -1, 3))
    
    # ============================================================
    # [Analysis Hook] 在这里插入分析代码
    # ============================================================
    if False:
        MY_ORGAN_NAMES = [str(l) for l in range(10)]

        # 这里直接打印表格，并返回数据
        stats_dict, global_indices = analyze_per_organ_stats(
            selected_local_indices=fps_local_idx,
            topM_global_indices=topM_idx,
            organ_logits=organ_logits,
            selected_coords=center_coords,
            organ_names=MY_ORGAN_NAMES # 传入名字列表
        )

        stats = compare_stages_stats(
            topM_indices=topM_idx_reshaped, # Stage 1 Indices
            topM_coords=cand_coords_norm,   # Stage 1 Coords
            fps_local_indices=fps_local_idx,# Stage 2 Indices
            fps_coords=center_coords,       # Stage 2 Coords
            organ_logits=organ_logits,
            organ_names=MY_ORGAN_NAMES,
            total_tokens=total_tokens
        )


    # 计算 Top-M 候选点 到 K 个中心点的距离
    # cand_coords_norm: (B*NO, M, 3)
    # center_coords:    (B*NO, K, 3)
    diff = torch.abs(center_coords.unsqueeze(2) - cand_coords_norm.unsqueeze(1)) 
    dist_linf = torch.max(diff, dim=-1).values # (B*NO, K, M)
    
    mask = dist_linf <= (local_radius / (grid_size - 1)) # 注意归一化后的半径
    
    # 这里的 attention score 应该是原始 TopM 的值
    masked_scores = topM_vals.unsqueeze(1).expand(-1, k, -1).clone() # (B*NO, K, M)
    masked_scores = masked_scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_scores, dim=-1) # (B*NO, K, M)
    weights = torch.nan_to_num(weights, nan=0.0)

    # 统一 Pooling
    pooled_feats = torch.bmm(weights, cand_feats)       # (B*NO, K, L)
    pooled_coords = torch.bmm(weights, cand_coords_norm)# (B*NO, K, 3)
    pooled_organ_probs = torch.bmm(weights, cand_organ_probs) # (B*NO, K, 10) - 修正了边界错位问题

    # --- Step 4: Output ---
    final_tokens = output_embedder(pooled_feats, pooled_coords, pooled_organ_probs)
    output_tokens = final_tokens.view(b, num_organs, k, -1)
    
    return output_tokens

def clean_spatial_outliers(organ_probs, coords, sigma_threshold=2.5):
    """
    基于空间统计学去除离群噪声点。
    假设器官是单连通区域，去除距离加权重心过远的点。
    
    Args:
        organ_probs: (B, N, Num_Organs) Softmax 后的概率
        coords: (N, 3) 或 (B, N, 3) 归一化坐标
        sigma_threshold: 保留重心周围多少倍标准差范围内的点
    
    Returns:
        cleaned_probs: (B, N, Num_Organs) 去噪后的概率
    """
    B, N, K = organ_probs.shape
    device = organ_probs.device
    
    # 确保 coords 是 (B, N, 3)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0).expand(B, -1, -1)
        
    # 1. 计算每个器官的总质量 (B, K)
    # 加上 eps 防止除零
    mass = organ_probs.sum(dim=1) + 1e-6 
    
    # 2. 计算加权重心 (Center of Mass) -> (B, K, 3)
    # (B, N, K) permute -> (B, K, N) @ (B, N, 3) -> (B, K, 3)
    weighted_coords = torch.bmm(organ_probs.transpose(1, 2), coords)
    centers = weighted_coords / mass.unsqueeze(-1)
    
    # 3. 计算每个点到重心的距离 (B, N, K)
    # coords: (B, N, 1, 3)
    # centers: (B, 1, K, 3)
    # dists: (B, N, K)
    dists = torch.norm(coords.unsqueeze(2) - centers.unsqueeze(1), dim=-1)
    
    # 4. 计算每个器官的空间标准差 (近似半径) (B, K)
    # 加权方差: sum(p * (x-u)^2) / sum(p)
    weighted_var = (organ_probs * (dists ** 2)).sum(dim=1) / mass
    std_dev = torch.sqrt(weighted_var + 1e-6) # (B, K)
    
    # 5. 生成掩码：保留距离小于 sigma_threshold * std_dev 的点
    # 动态阈值: (B, 1, K)
    dynamic_radius = (std_dev * sigma_threshold).unsqueeze(1)
    
    # 另外设置一个绝对下限半径，防止器官太小导致所有点都被过滤
    # 假设归一化坐标下，至少保留 0.1 (约10%的图像尺寸) 的范围
    min_radius = 0.1 
    threshold = torch.max(dynamic_radius, torch.tensor(min_radius, device=device))
    
    mask = (dists < threshold).float()
    
    # 6. 应用掩码
    cleaned_probs = organ_probs * mask
    
    # 重新归一化 (可选，但建议做，保持概率和为1或接近原始分布)
    # 这里我们不做严格Softmax，只是把噪声清零
    
    return cleaned_probs

def spatially_aware_feature_pooling_optimized_v3(
    features,           
    attention_map,      
    organ_logits,       
    question_embedding, 
    fusion_module,      
    output_embedder,
    scorer_module,      
    k=10,               
    M=200,              
    local_radius=1,     
    alpha=0.5,          
    normalize_coords=True,
    device=None,
    return_stats=False,
    organ_names=None,
    input_img_size=128
):
    if device is None: device = features.device
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    grid_size = round(total_tokens ** (1/3))
    
    # 准备坐标
    coords_3d = get_3d_grid(grid_size, device)
    coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d
    
    # --- [NEW] Step 0: 概率去噪 ---
    organ_probs = F.softmax(organ_logits, dim=-1) # (B, Total, NO)
    
    # 调用去噪函数
    # sigma_threshold=2.5 意味着保留 98% 的高斯分布区域，切掉极远处的点
    clean_probs = clean_spatial_outliers(organ_probs, coords_norm, sigma_threshold=2.5)
    
    # --- 1. Top-M Selection (使用去噪后的概率) ---
    # 利用去噪后的 Organ Prior 净化 Attention
    # 这样，远离器官重心的 Attention 即使很高，也会被 clean_probs 压为 0
    refined_attn = attention_map * clean_probs.transpose(1, 2)
    flat_attn = refined_attn.view(-1, total_tokens) 
    
    # Top-M
    topM_vals, topM_idx = torch.topk(flat_attn, M, dim=-1)
    
    # ... (后续代码保持不变) ...
    # 注意：后续 gather organ_probs 时，建议也使用 clean_probs
    # 恢复形状用于后续 gather
    topM_idx_reshaped = topM_idx.view(b, num_organs, M)
    
    # Gather Visual Features
    expanded_feats = features.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, l)
    cand_feats = torch.gather(expanded_feats, 1, topM_idx.unsqueeze(-1).expand(-1, -1, l))
    
    # Gather Coords
    coords_3d = get_3d_grid(grid_size, device)
    coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d
    expanded_coords_norm = coords_norm.unsqueeze(0).expand(b * num_organs, -1, -1)
    cand_coords_norm = torch.gather(expanded_coords_norm, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 3))

    # --- 优化点 3: 同时也 Gather Organ Probs (为 Pooling 做准备) ---
    #flat_organ_probs = organ_probs.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, 10)
    #cand_organ_probs = torch.gather(flat_organ_probs, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 10))
    flat_organ_probs = clean_probs.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, num_organs)
    cand_organ_probs = torch.gather(flat_organ_probs, 1, topM_idx.unsqueeze(-1).expand(-1, -1, num_organs))
    
    # --- 准备数据 ---
    # 扩展 Question Embedding 以匹配 (B*Num_Organs)
    flat_question = question_embedding.expand(-1, num_organs, -1).reshape(-1, question_embedding.shape[-1])
    
    # 注意：cand_feats 是原始特征，未归一化，适合传入 fusion_module
    
    # --- 调用 FPS ---
    fps_local_idx = batched_hybrid_fps_s(
        coords=cand_coords_norm, 
        features=cand_feats,        # 传入原始特征
        num_samples=k, 
        question_emb=flat_question, 
        fusion_module=fusion_module, 
        scorer_module=scorer_module, # [NEW] 传入可学习的评分器
        alpha=alpha
    )
    
    # --- Step 3: Local Pooling (Visual + Coord + Organ) ---
    # 选中的中心点坐标 (B*NO, K, 3)
    center_coords = torch.gather(cand_coords_norm, 1, fps_local_idx.unsqueeze(-1).expand(-1, -1, 3))
    
    # ============================================================
    # [Analysis Hook] 在这里插入分析代码
    # ============================================================
    if False:
        MY_ORGAN_NAMES = [str(l) for l in range(10)]

        # 这里直接打印表格，并返回数据
        stats_dict, global_indices = analyze_per_organ_stats(
            selected_local_indices=fps_local_idx,
            topM_global_indices=topM_idx,
            organ_logits=organ_logits,
            selected_coords=center_coords,
            organ_names=MY_ORGAN_NAMES # 传入名字列表
        )

        stats = compare_stages_stats(
            topM_indices=topM_idx_reshaped, # Stage 1 Indices
            topM_coords=cand_coords_norm,   # Stage 1 Coords
            fps_local_indices=fps_local_idx,# Stage 2 Indices
            fps_coords=center_coords,       # Stage 2 Coords
            organ_logits=organ_logits,
            organ_names=MY_ORGAN_NAMES,
            total_tokens=total_tokens
        )


    # 计算 Top-M 候选点 到 K 个中心点的距离
    # cand_coords_norm: (B*NO, M, 3)
    # center_coords:    (B*NO, K, 3)
    diff = torch.abs(center_coords.unsqueeze(2) - cand_coords_norm.unsqueeze(1)) 
    dist_linf = torch.max(diff, dim=-1).values # (B*NO, K, M)
    
    mask = dist_linf <= (local_radius / (grid_size - 1)) # 注意归一化后的半径
    
    # 这里的 attention score 应该是原始 TopM 的值
    masked_scores = topM_vals.unsqueeze(1).expand(-1, k, -1).clone() # (B*NO, K, M)
    masked_scores = masked_scores.masked_fill(~mask, float('-inf'))
    weights = F.softmax(masked_scores, dim=-1) # (B*NO, K, M)
    weights = torch.nan_to_num(weights, nan=0.0)

    # 统一 Pooling
    pooled_feats = torch.bmm(weights, cand_feats)       # (B*NO, K, L)
    pooled_coords = torch.bmm(weights, cand_coords_norm)# (B*NO, K, 3)
    pooled_organ_probs = torch.bmm(weights, cand_organ_probs) # (B*NO, K, 10) - 修正了边界错位问题

    # --- Step 4: Output ---
    final_tokens = output_embedder(pooled_feats, pooled_coords, pooled_organ_probs)
    output_tokens = final_tokens.view(b, num_organs, k, -1)
    
    return output_tokens
    # ...
    # Gather Organ Probs (使用 clean_probs)
    #flat_organ_probs = clean_probs.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, num_organs)
    #cand_organ_probs = torch.gather(flat_organ_probs, 1, topM_idx.unsqueeze(-1).expand(-1, -1, num_organs))
    
    # ... (FPS 和后续逻辑不变) ...

def compare_stages_stats(
    topM_indices,           # (B, NO, M)
    topM_coords,            # (B*NO, M, 3)
    fps_local_indices,      # (B*NO, K)
    fps_coords,             # (B*NO, K, 3)
    organ_logits,           # (B, Total, NO)
    organ_names=None,
    total_tokens=4096,
    input_img_size=128
):
    """
    对比 Top-M 和 FPS-K 的统计指标。
    新增指标: Precision (Hit Rate) - 选中的点中有多少比例确实是该器官。
    """
    B_times_NO, M, _ = topM_coords.shape 
    _, K, _ = fps_coords.shape
    num_organs = organ_logits.shape[-1]
    B = B_times_NO // num_organs
    device = topM_coords.device

    grid_size = round(total_tokens ** (1/3))
    patch_size = input_img_size / grid_size

    if organ_names is None:
        organ_names = [f"Org_{i}" for i in range(num_organs)]

    # --- Step 1: 构建 Pseudo-GT ---
    token_pred = torch.argmax(organ_logits, dim=-1) # (B, Total)
    pseudo_gt_mask = torch.zeros(B, total_tokens, num_organs, dtype=torch.bool, device=device)
    pseudo_gt_mask.scatter_(2, token_pred.unsqueeze(-1), True)
    pseudo_gt_flat = pseudo_gt_mask.permute(0, 2, 1).reshape(B * num_organs, total_tokens)

    # --- Step 2: 计算 Recall 和 Precision ---
    def calc_metrics(indices_global, num_selected):
        """
        返回: (Recall, Precision)
        """
        selected_mask = torch.zeros(B * num_organs, total_tokens, dtype=torch.bool, device=device)
        selected_mask.scatter_(1, indices_global, True)

        # Intersection: 选中的点中，有多少是 True Positive
        intersection = (selected_mask & pseudo_gt_flat).sum(dim=1).float()  # (B*NO,)
        
        # GT Total: 该器官总共有多少点
        gt_total = pseudo_gt_flat.sum(dim=1).float()  # (B*NO,)

        # 1. Recall = Intersection / GT_Total
        recall = torch.where(gt_total > 0, intersection / gt_total, torch.ones_like(gt_total))
        
        # 2. Precision = Intersection / K (or M)
        # 注意：如果 num_selected > gt_total (小器官)，Precision 也不可能达到 1.0，这是正常的
        precision = intersection / float(num_selected)

        return recall.view(B, num_organs).mean(dim=0), precision.view(B, num_organs).mean(dim=0)

    # Stage 1: Top-M
    flat_topM_indices = topM_indices.view(B * num_organs, M)
    recall_m, prec_m = calc_metrics(flat_topM_indices, M)

    # Stage 2: FPS-K
    fps_global_indices = torch.gather(flat_topM_indices, 1, fps_local_indices)
    recall_k, prec_k = calc_metrics(fps_global_indices, K)

    # --- Step 3: Dispersion ---
    def calc_dispersion(coords_norm, num_points):
        dist_matrix = torch.cdist(coords_norm, coords_norm)
        if num_points > 1:
            # sum includes diagonal (0), so divide by N*(N-1) is correct for average off-diagonal distance
            avg_d = dist_matrix.sum(dim=(1,2)) / (num_points * (num_points - 1))
        else:
            avg_d = torch.zeros(B_times_NO, device=device)
        return (avg_d * (grid_size - 1)).view(B, num_organs).mean(dim=0)

    disp_m = calc_dispersion(topM_coords, M)
    disp_k = calc_dispersion(fps_coords, K)

    # --- Step 4: 打印表格 ---
    print("\n" + "="*115)
    print(f"STAGE COMPARISON: Top-{M} -> FPS-{K}")
    print(f"Metrics: Rcl=Recall (Coverage), Prc=Precision (Hit Rate), Dist=Avg Pairwise Dist (px)")
    print("-" * 115)
    # 调整表头，增加 Precision
    print(f"{'Organ':<10} | {'Rcl(M)':<6} {'Prc(M)':<6} -> {'Rcl(K)':<6} {'Prc(K)':<6} | {'Dist(M)':<7} -> {'Dist(K)':<7} | {'Status'}")
    print("-" * 115)

    for i, name in enumerate(organ_names):
        rm, pm = recall_m[i].item(), prec_m[i].item()
        rk, pk = recall_k[i].item(), prec_k[i].item()
        dm, dk = disp_m[i].item(), disp_k[i].item()
        
        status = []
        
        # 1. 覆盖率检查 (Top-M 阶段)
        if rm < 0.1 and pm > 0.5: 
            # Precision高但Recall低，说明器官太大了，M不够用，但这通常不是错误
            pass 
        elif pm < 0.5:
            status.append("M-Noisy") # Top-M 选了很多背景

        # 2. FPS 质量检查
        # 如果 FPS 的 Precision 显著低于 Top-M 的 Precision，说明 FPS 倾向于选背景点
        if pk < pm - 0.2: 
            status.append("FPS-Drift") 
        
        # 3. 坍塌检查
        if dk < 1.0: status.append("Collapse")
        
        status_str = ", ".join(status) if status else "OK"
        
        print(f"{name:<10} | {rm:.3f}  {pm:.3f}  -> {rk:.3f}  {pk:.3f}  | {dm:.2f}    -> {dk:.2f}    | {status_str}")

    print("-" * 115)
    print("="*115 + "\n")
    
    return {
        "recall_m": recall_m, "prec_m": prec_m,
        "recall_k": recall_k, "prec_k": prec_k
    }
def analyze_per_organ_stats(
    selected_local_indices, # (B*NO, K) 来自 FPS 输出的局部索引
    topM_global_indices,    # (B, NO, M) 来自 Top-M 输出的全局索引
    organ_logits,           # (B, Total_Tokens, Num_Organs) 原始预测 Logits
    selected_coords,        # (B*NO, K, 3) 选中的归一化坐标 [0,1]
    organ_names=None,       # (List[str]) 器官名称列表
    total_tokens=4096,      # 特征图的总 Token 数 (例如 16*16*16)
    input_img_size=128      # 原始输入图像的分辨率 (用于换算实际距离)
):
    """
    分析 FPS 采样的质量，按器官分别统计。
    
    Returns:
        stats_dict (dict): 包含每个器官详细数据的字典
        selected_global_indices (Tensor): (B*NO, K) 选中点在全图中的绝对索引
    """
    # 1. 获取维度信息
    B_times_NO, K = selected_local_indices.shape
    num_organs = organ_logits.shape[-1]
    B = B_times_NO // num_organs
    device = selected_local_indices.device

    # 2. 计算空间尺度参数
    # Grid Size: 特征图的边长 (例如 4096^(1/3) ≈ 16)
    grid_size = round(total_tokens ** (1/3))
    # Patch Size: 每个 Token 代表的实际体素大小 (例如 128 / 16 = 8)
    patch_size = input_img_size / grid_size

    # 3. 处理器官名称
    if organ_names is None:
        organ_names = [f"Organ_{i}" for i in range(num_organs)]
    assert len(organ_names) == num_organs, "organ_names 长度必须与 num_organs 一致"

    # =======================================================
    # Part A: 索引映射 (Local -> Global)
    # =======================================================
    # topM_global_indices: (B, NO, M) -> (B*NO, M)
    flat_topM_indices = topM_global_indices.view(B * num_organs, -1)
    
    # 使用 gather 将 FPS 选出的局部索引 (0~M-1) 映射回 全局索引 (0~TotalTokens-1)
    # selected_global_indices: (B*NO, K)
    selected_global_indices = torch.gather(flat_topM_indices, 1, selected_local_indices)

    # =======================================================
    # Part B: 计算 Hit Rate (语义准确度)
    # =======================================================
    # 1. 获取全图每个 Token 的预测类别: (B, Total_Tokens)
    token_pred_labels = torch.argmax(organ_logits, dim=-1)
    
    # 2. 扩展以匹配 (B*NO) 维度: (B, Total) -> (B*NO, Total)
    # 使用 repeat_interleave 保证顺序是 [B0, B0... B1, B1...]
    token_pred_labels_expanded = token_pred_labels.repeat_interleave(num_organs, dim=0)
    
    # 3. 获取选中点的预测类别: (B*NO, K)
    selected_pred_labels = torch.gather(token_pred_labels_expanded, 1, selected_global_indices)
    
    # 4. 生成目标标签
    # 目标是: [0, 1, 2... NO-1, 0, 1...] 重复 B 次
    target_labels = torch.arange(num_organs, device=device).repeat(B).unsqueeze(1).expand(-1, K)
    
    # 5. 计算命中: (B*NO, K)
    hits = (selected_pred_labels == target_labels).float()
    
    # 6. 按器官聚合: (B, NO, K) -> 对 Batch 和 K 求平均 -> (NO,)
    hits_per_organ = hits.view(B, num_organs, K).mean(dim=(0, 2))

    # =======================================================
    # Part C: 计算 Dispersion (空间离散度)
    # =======================================================
    # selected_coords: (B*NO, K, 3)
    # 计算成对欧氏距离矩阵: (B*NO, K, K)
    dist_matrix = torch.cdist(selected_coords, selected_coords, p=2)
    
    # 计算平均距离 (排除对角线上的 0)
    # 矩阵求和后除以 K*(K-1)
    if K > 1:
        sum_dists = dist_matrix.sum(dim=(1, 2)) # (B*NO,)
        avg_dists = sum_dists / (K * (K - 1))
    else:
        avg_dists = torch.zeros(B * num_organs, device=device)
        
    # 按器官聚合: (B, NO) -> 对 Batch 求平均 -> (NO,)
    dispersion_per_organ = avg_dists.view(B, num_organs).mean(dim=0)

    # =======================================================
    # Part D: 格式化输出与距离换算
    # =======================================================
    print("\n" + "="*90)
    print(f"FPS SAMPLING ANALYSIS (Grid: {grid_size}x{grid_size}x{grid_size}, Input: {input_img_size}^3)")
    print("-" * 90)
    # 表头
    header = f"{'Organ Name':<15} | {'Hit Rate':<10} | {'Norm Disp':<10} | {'Grid Dist':<12} | {'Real Dist':<12}"
    print(header)
    print("-" * 90)
    
    stats_dict = {}
    
    # 辅助函数：将归一化距离 (0~1.73) 转换为 像素/体素 距离
    def convert_dist(norm_disp):
        # Grid Distance: 在特征图上有多少个格子的距离
        g_dist = norm_disp * (grid_size - 1)
        # Real Distance: 在原始图像上有多少个体素的距离
        r_dist = g_dist * patch_size
        return g_dist, r_dist

    for i, name in enumerate(organ_names):
        hr = hits_per_organ[i].item()
        disp = dispersion_per_organ[i].item()
        
        g_dist, r_dist = convert_dist(disp)
        
        # 打印行
        row_str = f"{name:<15} | {hr:.4f}     | {disp:.4f}     | {g_dist:.2f} px       | {r_dist:.2f} vox"
        print(row_str)
        
        stats_dict[name] = {
            "hit_rate": hr,
            "norm_dispersion": disp,
            "grid_distance": g_dist,
            "real_distance": r_dist
        }
    
    # 计算全局平均值
    avg_hr = hits_per_organ.mean().item()
    avg_disp = dispersion_per_organ.mean().item()
    avg_g, avg_r = convert_dist(avg_disp)
    
    print("-" * 90)
    print(f"{'AVERAGE':<15} | {avg_hr:.4f}     | {avg_disp:.4f}     | {avg_g:.2f} px       | {avg_r:.2f} vox")
    print("="*90 + "\n")

    return stats_dict, selected_global_indices
    
def analyze_selection_stats(
    selected_local_indices, # (B*NO, K) 来自 FPS 的输出
    topM_global_indices,    # (B, NO, M) 来自 Top-M 的输出
    organ_logits,           # (B, Total_Tokens, Num_Organs) 原始的器官预测
    selected_coords,        # (B*NO, K, 3) 选中的点坐标
    num_organs
):
    """
    分析 FPS 选点的语义准确性和空间分布情况。
    """
    B_times_NO, K = selected_local_indices.shape
    B = B_times_NO // num_organs
    device = selected_local_indices.device

    # ==========================================
    # 1. 索引映射：从 FPS 局部索引 -> 全局 Token 索引
    # ==========================================
    # 调整 topM 形状以匹配 gather: (B*NO, M)
    flat_topM_indices = topM_global_indices.view(B * num_organs, -1)
    
    # 获取选中的点在原始图片中的全局索引 (B*NO, K)
    # 这里的 indices 范围是 [0, Total_Tokens-1]
    selected_global_indices = torch.gather(flat_topM_indices, 1, selected_local_indices)

    # ==========================================
    # 2. 语义分析：Organ Hit Rate
    # ==========================================
    # 获取每个 Token 的预测类别: (B, Total_Tokens)
    token_pred_labels = torch.argmax(organ_logits, dim=-1)
    
    # 扩展预测标签以进行 gather: (B, Total_Tokens) -> (B*NO, Total_Tokens)
    # 注意：这里需要小心，因为每个 batch 的预测是独立的
    # 我们先把它展平处理
    token_pred_labels_expanded = token_pred_labels.repeat_interleave(num_organs, dim=0) # (B*NO, Total_Tokens)
    
    # 获取选中点的预测类别: (B*NO, K)
    selected_pred_labels = torch.gather(token_pred_labels_expanded, 1, selected_global_indices)
    
    # 生成目标标签: 0, 1, 2... NO-1, 0, 1...
    target_labels = torch.arange(num_organs, device=device).repeat(B).unsqueeze(1).expand(-1, K) # (B*NO, K)
    
    # 计算命中率：预测类别 == 目标器官类别
    hits = (selected_pred_labels == target_labels).float()
    organ_hit_rate = hits.mean().item() # 全局平均命中率
    
    # 如果需要看每个器官的详细命中率：
    hits_per_organ = hits.view(B, num_organs, K).mean(dim=(0, 2)) # (Num_Organs,)

    # ==========================================
    # 3. 空间分析：Spatial Dispersion (平均成对距离)
    # ==========================================
    # selected_coords: (B*NO, K, 3)
    # 计算成对距离矩阵: (B*NO, K, K)
    dist_matrix = torch.cdist(selected_coords, selected_coords, p=2)
    
    # 计算平均距离 (排除对角线的0)
    # Sum / (K * (K-1))
    if K > 1:
        avg_pairwise_dist = dist_matrix.sum() / (B_times_NO * K * (K - 1))
    else:
        avg_pairwise_dist = torch.tensor(0.0)

    return {
        "organ_hit_rate": organ_hit_rate,          # float: 总体命中率 (0~1)
        "hits_per_organ": hits_per_organ,          # tensor: 每个器官的命中率
        "spatial_dispersion": avg_pairwise_dist.item(), # float: 平均点间距 (越大越好)
        "selected_global_indices": selected_global_indices # tensor: 记录具体选了哪些点用于可视化
    }



def clean_mask_by_distance_statistics(organ_mask, coords_norm, keep_ratio=0.95):
    """
    GPU 并行版去噪：剔除距离器官几何中心过远的离群点。
    
    Args:
        organ_mask: (B, NO, Total) bool
        coords_norm: (Total, 3) or (B, Total, 3)
    """
    B, NO, Total = organ_mask.shape
    device = organ_mask.device
    
    # 确保 coords 是 (B, 1, Total, 3) 以便广播
    if coords_norm.dim() == 2:
        coords = coords_norm.unsqueeze(0).unsqueeze(0).expand(B, NO, -1, -1)
    else:
        coords = coords_norm.unsqueeze(1).expand(-1, NO, -1, -1)
        
    # 1. 计算每个器官的几何中心
    # mask: (B, NO, Total) -> (B, NO, Total, 1)
    mask_float = organ_mask.float().unsqueeze(-1)
    valid_counts = mask_float.sum(dim=2).clamp(min=1.0) # (B, NO, 1)
    
    # centroids: (B, NO, 3)
    centroids = (coords * mask_float).sum(dim=2) / valid_counts
    centroids = centroids.unsqueeze(2) # (B, NO, 1, 3)
    
    # 2. 计算所有点到中心的距离
    # dists: (B, NO, Total)
    dists = torch.norm(coords - centroids, dim=-1)
    
    # 3. 确定阈值 (这里使用简单的分位数，保留最近的 95%)
    # 注意：只在 mask 为 True 的点中计算阈值比较麻烦，
    # 这里简单处理：把非 mask 点的距离设为无穷大，然后取 topk (smallest)
    
    # 将非器官点的距离设为 +inf
    dists_masked = dists.masked_fill(~organ_mask, float('inf'))
    
    # 计算每个器官实际有多少点
    num_points = organ_mask.sum(dim=-1) # (B, NO)
    
    # 计算要保留多少点 (keep_ratio)
    k_keep = (num_points.float() * keep_ratio).long().clamp(min=1)
    
    # 这里的逻辑稍微复杂，因为每个 batch/organ 的 k 不同
    # 简单策略：计算 mean + std
    # 为了并行，我们计算 masked mean/std
    
    # sum_d = dists.masked_fill(~organ_mask, 0).sum(dim=-1)
    # mean_d = sum_d / valid_counts.squeeze(-1) # (B, NO)
    
    # 更加鲁棒的方法：直接 mask 掉距离特别远的点
    # 比如：如果一个点距离中心 > 2 * 平均距离，则剔除
    
    cleaned_mask = organ_mask.clone()
    
    # 迭代处理避免复杂的 gather (虽然慢一点点，但比 CPU 快)
    # 或者使用简单的全局阈值策略
    
    # 这里给出一个简单的基于 Mean + 2*Std 的实现
    for b in range(B):
        for o in range(NO):
            if num_points[b, o] < 5: continue # 点太少不处理
            
            valid_dists = dists[b, o][organ_mask[b, o]]
            mean_d = valid_dists.mean()
            std_d = valid_dists.std()
            
            threshold = mean_d + 2.5 * std_d # 2.5 sigma 覆盖绝大多数正态分布
            
            # 标记离群点
            outliers = (dists[b, o] > threshold) & organ_mask[b, o]
            cleaned_mask[b, o, outliers] = False
            
    return cleaned_mask

def batched_hybrid_fps_mask(
    coords, 
    features, 
    num_samples, 
    question_emb=None, 
    fusion_module=None, 
    scorer_module=None,
    alpha=0.0,
    valid_mask=None
):
    """
    Optimized FPS with valid_mask support.
    """
    B, N, D = coords.shape
    device = coords.device

    if valid_mask is None:
        valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    
    valid_counts = valid_mask.sum(dim=1)
    if (valid_counts < num_samples).any():
        # 打印具体的错误信息以便调试
        raise ValueError(f"num_samples ({num_samples}) exceeds valid point count for some batches. Min valid: {valid_counts.min()}")

    # 1. Feature Adaptation
    if question_emb is not None and fusion_module is not None:
        metric_features = fusion_module(features, question_emb)
        metric_features_norm = F.normalize(metric_features, dim=-1)
    else:
        metric_features = features
        metric_features_norm = F.normalize(features, dim=-1)

    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float('inf'), device=device)

    # 2. Start Point Selection
    if scorer_module is not None:
        start_scores = scorer_module(metric_features)
        start_scores = start_scores.masked_fill(~valid_mask, float('-inf')) # 使用 float('-inf') 更安全
        farthest = torch.argmax(start_scores, dim=-1)
    else:
        rand_scores = torch.rand(B, N, device=device).masked_fill(~valid_mask, float('-inf'))
        farthest = torch.argmax(rand_scores, dim=-1)

    batch_indices = torch.arange(B, device=device)

    # 3. FPS Loop
    for i in range(num_samples):
        centroids[:, i] = farthest
        
        # 优化：如果是最后一次迭代，不需要计算下一个最远点
        if i == num_samples - 1:
            break

        curr_coord = coords[batch_indices, farthest, :].unsqueeze(1)
        curr_feat = metric_features_norm[batch_indices, farthest, :].unsqueeze(1)

        dist_spatial = torch.norm(coords - curr_coord, dim=-1)
        dot_prod = torch.sum(metric_features_norm * curr_feat, dim=-1)
        dist_semantic = 1.0 - dot_prod

        dist_combined = alpha * dist_spatial + (1 - alpha) * dist_semantic

        # Update distances
        min_dists = torch.min(min_dists, dist_combined)

        # Mask out invalid points
        masked_dists = min_dists.masked_fill(~valid_mask, float('-inf')) # 距离越远越好，无效点设为负无穷
        farthest = torch.argmax(masked_dists, dim=-1)

    return centroids


def spatially_aware_feature_pooling_optimized_4(
    features,           
    attention_map,      
    organ_logits,       
    question_embedding, 
    fusion_module,      
    output_embedder,
    scorer_module,
    k=10,               
    M=200,              
    local_radius=1,     
    alpha=0.5,          
    normalize_coords=True,
    device=None
):
    if device is None:
        device = features.device
    b, total_tokens, l = features.shape
    num_organs = attention_map.shape[1]
    grid_size = round(total_tokens ** (1/3))
    
    # 假设 get_3d_grid 已经在外部定义
    # coords_3d = get_3d_grid(grid_size, device) 
    
    # --- Step 1: Build pseudo-GT mask ---
    pred_class = torch.argmax(organ_logits, dim=-1)
    organ_mask = torch.zeros(b, num_organs, total_tokens, dtype=torch.bool, device=device)
    organ_mask.scatter_(1, pred_class.unsqueeze(1), True)
    N_o = organ_mask.sum(dim=-1)

    if True:
        coords_3d = get_3d_grid(grid_size, device) 
        coords_norm = coords_3d / (grid_size - 1) if normalize_coords else coords_3d
        # organ_mask = clean_mask_by_distance_statistics(organ_mask, coords_norm)


    # --- Step 2: Compute M_use ---
    M_use_matrix = torch.where(
        N_o >= k,
        torch.clamp(N_o, max=M),
        torch.full_like(N_o, k, dtype=torch.long)
    )
    M_use_flat = M_use_matrix.view(-1)
    organ_mask_flat = organ_mask.view(-1, total_tokens)
    attention_flat = attention_map.view(-1, total_tokens)

    # --- Step 3: Construct final score ---
    LARGE_OFFSET = 1e6
    final_score = attention_flat.clone()
    final_score += LARGE_OFFSET * organ_mask_flat.float()

    need_suppress = (N_o >= k).view(-1)
    if need_suppress.any():
        non_organ = ~organ_mask_flat
        suppress_rows = need_suppress.unsqueeze(1).expand(-1, total_tokens)
        final_score[non_organ & suppress_rows] = float('-inf') # 使用 -inf 更彻底

    # --- Step 4: Select Top-M_use ---
    max_M_use = M_use_flat.max().item()
    
    # 🔥 CRITICAL FIX: sorted=True
    # 必须排序，因为后面 valid_in_cand 假设前 M_use 个是有效的
    _, top_idx_all = torch.topk(final_score, max_M_use, dim=-1, sorted=True)

    M_final = max_M_use
    topM_idx = top_idx_all[:, :M_final]

    # --- Gather data ---
    expanded_feats = features.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, l)
    cand_feats = torch.gather(expanded_feats, 1, topM_idx.unsqueeze(-1).expand(-1, -1, l))
    
    # 重新生成坐标 (假设函数存在)
    
    expanded_coords_norm = coords_norm.unsqueeze(0).expand(b * num_organs, -1, -1)
    cand_coords_norm = torch.gather(expanded_coords_norm, 1, topM_idx.unsqueeze(-1).expand(-1, -1, 3))

    flat_organ_logits = organ_logits.unsqueeze(1).expand(-1, num_organs, -1, -1).reshape(-1, total_tokens, num_organs)
    cand_organ_probs = torch.gather(flat_organ_logits, 1, topM_idx.unsqueeze(-1).expand(-1, -1, num_organs))
    
    flat_question = question_embedding.expand(-1, num_organs, -1).reshape(-1, question_embedding.shape[-1])

    # Build valid_mask: first M_use_flat[i] tokens are valid
    # 因为 topk 已经 sorted=True，所以前 M_use 个确实是分数最高的候选点
    valid_in_cand = torch.arange(M_final, device=device).unsqueeze(0) < M_use_flat.unsqueeze(1)

    # --- Call FPS ---
    fps_local_idx = batched_hybrid_fps_mask(
        coords=cand_coords_norm,
        features=cand_feats,
        num_samples=k,
        question_emb=flat_question,
        fusion_module=fusion_module,
        scorer_module=scorer_module,
        alpha=alpha,
        valid_mask=valid_in_cand
    )

    # --- Pooling ---
    center_coords = torch.gather(cand_coords_norm, 1, fps_local_idx.unsqueeze(-1).expand(-1, -1, 3))

    # 移除或注释掉调试代码，除非你确定 compare_stages_stats 已定义
    if False:
        # 恢复形状用于后续 gather
        topM_idx_reshaped = topM_idx.view(b, num_organs, M)

        MY_ORGAN_NAMES = [str(i) for i in range(num_organs)]
        stats = compare_stages_stats(
            topM_indices=topM_idx_reshaped,
            topM_coords=cand_coords_norm,
            fps_local_indices=fps_local_idx,
            fps_coords=center_coords,
            organ_logits=organ_logits,
            organ_names=MY_ORGAN_NAMES,
            total_tokens=total_tokens
        )

    diff = torch.abs(center_coords.unsqueeze(2) - cand_coords_norm.unsqueeze(1))
    dist_linf = torch.max(diff, dim=-1).values
    mask = dist_linf <= (local_radius / (grid_size - 1))

    orig_attn_vals = torch.gather(attention_flat, 1, topM_idx)
    masked_scores = orig_attn_vals.unsqueeze(1).expand(-1, k, -1).clone()
    masked_scores = masked_scores.masked_fill(~mask, float('-inf'))
    
    weights = F.softmax(masked_scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0) # 处理孤立点

    pooled_feats = torch.bmm(weights, cand_feats)
    pooled_coords = torch.bmm(weights, cand_coords_norm)
    pooled_organ_probs = torch.bmm(weights, cand_organ_probs)

    final_tokens = output_embedder(pooled_feats, pooled_coords, pooled_organ_probs)
    output_tokens = final_tokens.view(b, num_organs, k, -1)
    
    return output_tokens




def batched_hybrid_fps_mask_v2(
    coords, 
    features, 
    num_samples, 
    valid_mask, 
    question_emb=None, 
    fusion_module=None, 
    scorer_module=None,
    alpha=0.5,
    candidate_rate=0.5  # [新增] 候选比率，比如只在前 50% 的高分点里采样
):
    B, N, D = coords.shape
    device = coords.device

    # --- 1. 基础 Mask 处理 ---
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()
    
    # --- 2. 特征准备 ---
    if question_emb is not None and fusion_module is not None:
        metric_features = fusion_module(features, question_emb)
        metric_features_norm = F.normalize(metric_features, dim=-1)
    else:
        metric_features = features
        metric_features_norm = F.normalize(features, dim=-1)

    # --- 3. Scorer 打分与候选区筛选 (关键修改) ---
    if scorer_module is not None:
        # 计算分数 (B, N)
        start_scores = scorer_module(metric_features)
        
        # 处理维度问题 (复用之前的修复逻辑)
        if start_scores.dim() == 3:
             if start_scores.shape[2] == N: start_scores = start_scores.transpose(1, 2)
             if start_scores.shape[-1] > 1: start_scores = start_scores.max(dim=-1).values
             else: start_scores = start_scores.squeeze(-1)
        
        # 先把无效区域的分数设为极小
        start_scores = start_scores.masked_fill(~valid_mask, float('-inf'))

        # [核心逻辑]：动态更新 valid_mask，只保留 Top-K 高分点
        # 计算每个样本有多少个有效点
        num_valid = valid_mask.sum(dim=1) # (B,)
        
        # 决定保留多少个候选点 (至少要比 num_samples 大，否则 FPS 没意义)
        # 这里取 max(num_samples, total_valid * rate)
        k_candidates = (num_valid.float() * candidate_rate).long()
        k_candidates = torch.clamp(k_candidates, min=num_samples)

        # 生成 Top-K Mask
        # 我们需要为每个 batch 生成一个 mask
        candidate_mask = torch.zeros_like(valid_mask)
        
        for b in range(B):
            k = k_candidates[b].item()
            # 选出分数最高的 k 个点的索引
            _, topk_indices = torch.topk(start_scores[b], k)
            candidate_mask[b, topk_indices] = True
        
        # 更新 valid_mask：既要是原始有效的，又要是高分的
        valid_mask = valid_mask & candidate_mask

        # 第一个点选分数最高的
        farthest = torch.argmax(start_scores, dim=-1)
        
    else:
        # 如果没有 scorer，就随机选
        rand_scores = torch.rand(B, N, device=device).masked_fill(~valid_mask, float('-inf'))
        farthest = torch.argmax(rand_scores, dim=-1)

    # --- 4. FPS 循环 (逻辑不变，但现在的 valid_mask 已经过滤掉了无关背景) ---
    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float('inf'), device=device)
    batch_indices = torch.arange(B, device=device)

    for i in range(num_samples):
        centroids[:, i] = farthest
        
        if i == num_samples - 1:
            break

        curr_coord = coords[batch_indices, farthest, :].unsqueeze(1)
        curr_feat = metric_features_norm[batch_indices, farthest, :].unsqueeze(1)

        dist_spatial = torch.norm(coords - curr_coord, dim=-1)
        dot_prod = torch.sum(metric_features_norm * curr_feat, dim=-1)
        dist_semantic = 1.0 - dot_prod
        
        dist_combined = alpha * dist_spatial + (1 - alpha) * dist_semantic

        min_dists = torch.min(min_dists, dist_combined)

        # 这里的 valid_mask 已经是 "Top-K Mask" 了
        # 所以 FPS 只会跳到其他高分点上
        masked_dists = min_dists.masked_fill(~valid_mask, float('-inf'))
        farthest = torch.argmax(masked_dists, dim=-1)

    return centroids

def simple_masked_fps_selection_v2(
    features,           # (B, N, D)
    mask,               # (B, N)
    k=10,               # 采样点数
    question_emb=None,  
    fusion_module=None, 
    scorer_module=None, 
    alpha=0.5,
    normalize_coords=True
):
    """
    主入口函数。
    自动生成坐标 -> 根据 Mask 进行 FPS -> 返回结果。
    """
    B, N, D = features.shape
    device = features.device
    
    # --- 1. 自动生成坐标 ---
    # 假设 N 是立方数 (例如 4096 = 16^3)
    grid_size = round(N ** (1/3))
    
    # 生成坐标 (N, 3) -> 扩展为 (B, N, 3)
    coords_3d = get_3d_grid(grid_size, device)
    
    if normalize_coords:
        # 归一化到 [0, 1] 区间，这对 FPS 的 alpha 权重平衡很重要
        coords_norm = coords_3d / (grid_size - 1)
    else:
        coords_norm = coords_3d
        
    coords_batch = coords_norm.unsqueeze(0).expand(B, -1, -1)

    # --- 2. 执行 FPS ---
    fps_indices = batched_hybrid_fps_mask_v2(
        coords=coords_batch,
        features=features,
        num_samples=k,
        valid_mask=mask,
        question_emb=question_emb,
        fusion_module=fusion_module,
        scorer_module=scorer_module,
        alpha=alpha
    )

    # --- 3. Gather 结果 ---
    # fps_indices: (B, k)
    
    # Gather Features
    idx_feat = fps_indices.unsqueeze(-1).expand(-1, -1, D)
    selected_features = torch.gather(features, 1, idx_feat) # (B, k, D)

    # Gather Coords (返回归一化后的坐标)
    idx_coord = fps_indices.unsqueeze(-1).expand(-1, -1, 3)
    selected_coords = torch.gather(coords_batch, 1, idx_coord) # (B, k, 3)

    return selected_features

def batched_hybrid_fps_mask_(
    coords, 
    features, 
    num_samples, 
    valid_mask, 
    question_emb=None, 
    fusion_module=None, 
    scorer_module=None,
    alpha=0.5
):
    """
    核心 FPS 函数。
    特性：
    1. 只在 valid_mask=True 的点中采样。
    2. 如果有效点数量 < num_samples，会自动重复采样已选点，不会报错。
    """
    B, N, D = coords.shape
    device = coords.device

    # --- 1. Mask 预处理与安全检查 ---
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()
    
    valid_counts = valid_mask.sum(dim=1)
    
    # [防崩溃] 如果某样本全是 False (0个有效点)，强制全开，避免 argmax(-inf) 报错
    if (valid_counts == 0).any():
        # print("Warning: Found batch with 0 valid points. Fallback to global FPS.")
        valid_mask[valid_counts == 0] = True

    # [关键] 这里不再检查 valid_counts < num_samples，允许重复采样
    # print(features.shape)
    # --- 2. 特征准备 ---
    if question_emb is not None and fusion_module is not None:
        metric_features = fusion_module(features, question_emb)
        # print(metric_features.shape)
        metric_features_norm = F.normalize(metric_features, dim=-1)
    else:
        metric_features = features
        metric_features_norm = F.normalize(features, dim=-1)

    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float('inf'), device=device)
    
    # --- 3. 起始点选择 ---
    if scorer_module is not None:
        start_scores = scorer_module(metric_features)
        # print(start_scores.shape, valid_mask.shape)
        start_scores = start_scores.masked_fill(~valid_mask, float('-inf'))
        farthest = torch.argmax(start_scores, dim=-1)
        # print(farthest.shape)
    else:
        # 随机选择，但必须在 mask 范围内
        rand_scores = torch.rand(B, N, device=device).masked_fill(~valid_mask, float('-inf'))
        farthest = torch.argmax(rand_scores, dim=-1)

    batch_indices = torch.arange(B, device=device)

    # --- 4. FPS 循环 ---
    for i in range(num_samples):
        centroids[:, i] = farthest
        
        if i == num_samples - 1:
            break

        curr_coord = coords[batch_indices, farthest, :].unsqueeze(1)
        curr_feat = metric_features_norm[batch_indices, farthest, :].unsqueeze(1)

        # 空间距离
        dist_spatial = torch.norm(coords - curr_coord, dim=-1)
        # 语义距离
        dot_prod = torch.sum(metric_features_norm * curr_feat, dim=-1)
        dist_semantic = 1.0 - dot_prod
        
        # 混合
        dist_combined = alpha * dist_spatial + (1 - alpha) * dist_semantic

        # 更新最小距离
        min_dists = torch.min(min_dists, dist_combined)

        # [核心逻辑]
        # Mask=0 的点距离设为 -inf。
        # Mask=1 且已被选中的点，min_dists 为 0。
        # Mask=1 且未被选中的点，min_dists > 0。
        # 当所有有效点都被选中后，max(min_dists) 将在 0 中产生，从而实现重复采样。
        masked_dists = min_dists.masked_fill(~valid_mask, float('-inf'))
        farthest = torch.argmax(masked_dists, dim=-1)

    return centroids

def simple_masked_fps_selection(
    features,           # (B, N, D)
    mask,               # (B, N)
    k=10,               # 采样点数
    question_emb=None,  
    fusion_module=None, 
    scorer_module=None, 
    alpha=0.5,
    normalize_coords=True
):
    """
    主入口函数。
    自动生成坐标 -> 根据 Mask 进行 FPS -> 返回结果。
    """
    B, N, D = features.shape
    device = features.device
    
    # --- 1. 自动生成坐标 ---
    # 假设 N 是立方数 (例如 4096 = 16^3)
    grid_size = round(N ** (1/3))
    
    # 生成坐标 (N, 3) -> 扩展为 (B, N, 3)
    coords_3d = get_3d_grid(grid_size, device)
    
    if normalize_coords:
        # 归一化到 [0, 1] 区间，这对 FPS 的 alpha 权重平衡很重要
        coords_norm = coords_3d / (grid_size - 1)
    else:
        coords_norm = coords_3d
        
    coords_batch = coords_norm.unsqueeze(0).expand(B, -1, -1)

    # --- 2. 执行 FPS ---
    fps_indices = batched_hybrid_fps_mask_(
        coords=coords_batch,
        features=features,
        num_samples=k,
        valid_mask=mask,
        question_emb=question_emb,
        fusion_module=fusion_module,
        scorer_module=scorer_module,
        alpha=alpha
    )

    # --- 3. Gather 结果 ---
    # fps_indices: (B, k)
    
    # Gather Features
    idx_feat = fps_indices.unsqueeze(-1).expand(-1, -1, D)
    selected_features = torch.gather(features, 1, idx_feat) # (B, k, D)

    # Gather Coords (返回归一化后的坐标)
    idx_coord = fps_indices.unsqueeze(-1).expand(-1, -1, 3)
    selected_coords = torch.gather(coords_batch, 1, idx_coord) # (B, k, 3)

    return selected_features, fps_indices

def simple_masked_random_selection_nofps(
    features,           # (B, N, D)
    mask,               # (B, N)
    k=10,               # 采样点数
    **kwargs            # 吞掉所有不需要的参数
):
    """
    Random Baseline: 纯随机采样，替代 FPS。
    
    逻辑：
    1. 不再生成 3D 坐标。
    2. 仅根据 Mask 找出有效点。
    3. 在有效点中随机抽取 k 个 (如果点不够则有放回采样，够则无放回)。
    
    Returns:
        random_indices: (B, k) - 选中的 token 在全局 N 中的索引
        selected_features: (B, k, D) - 选中的 token 对应的特征
    """
    B, N, D = features.shape
    device = features.device
    
    # 确保 mask 是 bool 类型
    mask_bool = mask.bool()
    
    # 用于存储每个 batch 选出的索引
    batch_indices_list = []

    for b in range(B):
        # 1. 获取当前 batch 中 mask 为 True 的所有索引
        # valid_indices: (Num_Valid, )
        valid_indices = torch.nonzero(mask_bool[b]).squeeze(-1)
        num_valid = valid_indices.numel()
        
        if num_valid == 0:
            # 极端情况兜底：如果该样本 mask 全为 0，则在全图中随机选
            chosen_indices = torch.randint(0, N, (k,), device=device)
        elif num_valid >= k:
            # 情况 A: 有效点足够 -> 无放回随机采样 (Random without replacement)
            perm = torch.randperm(num_valid, device=device)[:k]
            chosen_indices = valid_indices[perm]
        else:
            # 情况 B: 有效点不足 k 个 -> 有放回随机采样 (Random with replacement)
            rand_idx = torch.randint(0, num_valid, (k,), device=device)
            chosen_indices = valid_indices[rand_idx]
            
        batch_indices_list.append(chosen_indices)

    # 2. 堆叠索引：(B, k)
    random_indices = torch.stack(batch_indices_list)

    # 3. Gather 特征
    idx_feat = random_indices.unsqueeze(-1).expand(-1, -1, D)
    selected_features = torch.gather(features, 1, idx_feat)  # (B, k, D)

    # 4. 同时返回索引和特征
    return random_indices, selected_features


def select_features_network_moe_structural(
    features, 
    attn_map,                # [B, S, P], S=10 structures
    query,                   # [B, 1, qd] — BERT [CLS]
    selector_model,          # f(q, cand_feats, cls_prior) -> logits
    structural_feature,      # [B, S, L], S=10
    gate_query_proj,         # nn.Linear(qd, d_gate)
    gate_struct_proj,        # nn.Linear(L, d_gate)
    M=100,                   # top-M candidates per structure
    tau=0.002,
    min_token_num=1,
    training=False,
    topK_struct=4,           # number of structures to activate (e.g., 2)
    k_cur=10,         # features per selected structure
    return_details=False,
    device=None,
    # --- MMR args (kept for compatibility, unused) ---
    method='attn_mass',
    alpha=0.9,
    min_k=1,
    max_k=30,
    enable_soft_mmr=False,
    beta_mmr=0.2,
    mmr_exclude_self=True,
    mmr_normalize_redundancy=True,
):
    """
    MOE-style structure-aware feature selection for CT VQA.
    
    Input:
        features: [B, P, d]
        attn_map: [B, S, P] — attention from each structure (S=10) to patches
        query: [B, 1, qd] — BERT [CLS] as question representation
        structural_feature: [B, S, L] — per-structure CLS tokens
    
    Output:
        selected: list of [K * k_per_struct, d] per sample
        (or with details dict if return_details=True)
    """
    k_per_struct = k_cur
    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    assert structural_feature is not None
    assert query.shape[1] == 1, "query must be [B, 1, qd] (CLS token)"
    
    if device is None:
        device = features.device

    bs, P, d = features.shape
    S = attn_map.shape[1]  # should be 10
    assert S == structural_feature.shape[1] == attn_map.shape[1]
    assert attn_map.shape[2] == P

    k_total = topK_struct * k_per_struct  # e.g., 2 * 10 = 20

    selected = []
    details = None
    if return_details:
        details = {
            'struct_logits': [],          # [B, S]
            'selected_structs': [],       # [B, K]
            'per_struct_p': [],           # list of [M_actual] per struct
            'per_struct_indices': [],     # list of global patch indices
        }

    for bi in range(bs):
        feats_b = features[bi]          # [P, d]
        attn_b = attn_map[bi]           # [S, P]
        struct_feat_b = structural_feature[bi]  # [S, L]
        q_repr = query[bi, 0]           # [qd]

        # === Step 1: Learnable gating over S=10 structures ===
        q_gate = gate_query_proj(q_repr.unsqueeze(0))      # [1, d_gate]
        s_gate = gate_struct_proj(struct_feat_b)           # [S, d_gate]
        struct_logits = torch.mm(s_gate, q_gate.t()).squeeze(-1)  # [S]
        topk_vals, topk_struct_idx = torch.topk(struct_logits, topK_struct, dim=0)  # [K]

        all_feats_from_selected = []

        per_struct_p_list = []
        per_struct_idx_list = []

        # === Step 2: For each selected structure, run your original selector logic ===
        for s_idx in topk_struct_idx:
            scores = attn_b[s_idx]  # [P] — attention from structure s_idx to all patches

            # --- Candidate pool: top-M by attention ---
            order = torch.argsort(scores, descending=True, stable=True)
            idx_candidates = order[:M]
            M_actual = idx_candidates.numel()

            if M_actual == 0:
                # No valid candidates: pad with zeros
                selected_feats = torch.zeros(k_per_struct, d, device=device, dtype=features.dtype)
                all_feats_from_selected.append(selected_feats)
                if return_details:
                    per_struct_p_list.append(torch.zeros(0, device=device))
                    per_struct_idx_list.append(torch.zeros(0, device=device, dtype=torch.long))
                continue

            cand_feats = feats_b[idx_candidates]   # [M_actual, d]
            cls_prior = scores[idx_candidates]     # [M_actual]

            # --- Selector scoring (your original logic) ---
            logits_raw = selector_model(q_repr, cand_feats, cls_prior=cls_prior)  # [M_actual]
            p = torch.softmax(logits_raw, dim=0)  # [M_actual]

            # --- Select up to k_per_struct ---
            k_forward = min(k_per_struct, M_actual)
            topk_pos = torch.topk(p, k_forward, dim=0).indices  # [k_f]

            # Ensure min_token_num
            if topk_pos.numel() < min_token_num:
                kept_mask = torch.zeros(M_actual, dtype=torch.bool, device=device)
                kept_mask[topk_pos] = True
                remaining = (~kept_mask).nonzero(as_tuple=True)[0]
                if remaining.numel() > 0:
                    add_pos = remaining[:min_token_num - topk_pos.numel()]
                    topk_pos = torch.cat([topk_pos, add_pos], dim=0)

            # Ensure exactly k_per_struct by repeating
            if topk_pos.numel() < k_per_struct:
                if topk_pos.numel() == 0:
                    final_local = torch.zeros(k_per_struct, dtype=torch.long, device=device)
                else:
                    need = k_per_struct - topk_pos.numel()
                    repeat_idx = topk_pos
                    reps = (need + repeat_idx.numel() - 1) // repeat_idx.numel()
                    pad_seq = repeat_idx.repeat(reps)[:need]
                    final_local = torch.cat([topk_pos, pad_seq], dim=0)
            else:
                final_local = topk_pos[:k_per_struct]

            selected_cand_feats = cand_feats[final_local]  # [k_per_struct, d]
            global_indices = idx_candidates[final_local]   # [k_per_struct]

            # --- Straight-Through trick ---
            if training:
                p_sel = p[final_local]  # [k_per_struct]
                hard_mask = torch.ones_like(p_sel)
                st_mask = hard_mask - p_sel.detach() + p_sel
                output_feats = st_mask.unsqueeze(-1) * selected_cand_feats
            else:
                output_feats = selected_cand_feats

            all_feats_from_selected.append(output_feats)

            if return_details:
                full_p = torch.zeros(M_actual, device=device)
                full_p[final_local] = p_sel if training else 1.0
                per_struct_p_list.append(full_p)
                per_struct_idx_list.append(global_indices)

        # Concatenate features from all selected structures
        final_feats = torch.cat(all_feats_from_selected, dim=0)  # [k_total, d]
        selected.append(final_feats)

        if return_details:
            details['struct_logits'].append(struct_logits.detach())
            details['selected_structs'].append(topk_struct_idx.detach())
            details['per_struct_p'].append(per_struct_p_list)
            details['per_struct_indices'].append(per_struct_idx_list)

    if return_details:
        return selected, details
    return selected

def select_features_network_mmr_variable_k_gpu_purefeature_finetune(
    features, attn_map, query, selector_model, structural_feature=None, M=100,
    tau=0.002, method='attn_mass', alpha=0.9, min_k=1, max_k=30,
    min_token_num=1, training=False, k_cur=10,
    return_details=False, device=None,
    # --- soft MMR specific (kept disabled as requested) ---
    enable_soft_mmr=False,
    beta_mmr=0.2, mmr_exclude_self=True, mmr_normalize_redundancy=True,
):
    """
    Returns exactly k_cur features per token (shape [k_cur, d]) to avoid exploding memory,
    while preserving gradient flow to selector_model via a Straight-Through (ST) top-k trick.

    Behavior:
    - training=True: forward returns hard-selected features (exactly k_cur per token),
      but gradients flow through the soft probabilities p for the selected indices using ST trick.
    - training=False: returns hard-selected features (exactly k_cur per token) with no gradient
      to selector_model (p detached).
    - If M_actual < k_cur, the top candidates are repeated to make exactly k_cur outputs (so the
      output tensor shape is always consistent). This keeps memory bounded and avoids returning
      M*T features.
    - Uses token-specific query q_b[ti] when calling selector_model.
    """

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs, P, d] (kept if needed)
    selected = []
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P, d]
        attn_b = attn_map[bi]          # [T, P]
        q_b = query[bi]                # [T, qd]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # --- Step 1: Candidate pool (top-M by attention) ---
            order = torch.argsort(scores, descending=True, stable=True)
            idx_candidates = order[:M]            # global indices of candidates (M_actual <= M)
            M_actual = idx_candidates.numel()

            if M_actual == 0:
                # No candidates: return k_cur zero rows to keep shape consistent
                batch_list.append(torch.zeros(k_cur, d, device=device, dtype=features.dtype))
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, device=device, dtype=torch.long)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            # --- Step 2: Candidate features and priors ---
            cand_feats = feats_b[idx_candidates]   # [M_actual, d]
            cls_prior = scores[idx_candidates]      # [M_actual]

            # --- Step 3: Selector scoring (use token-specific query) ---
            logits_raw = selector_model(q_b[0], cand_feats, cls_prior=cls_prior)  # [M_actual]
            logits_mmr = logits_raw  # placeholder if MMR would be applied

            # --- Step 4: Soft probabilities ---
            p = torch.softmax(logits_mmr, dim=0)  # [M_actual]

            # choose up to k_cur positions (forward). We will ensure final length == k_cur by padding/repeating.
            k_forward = min(k_cur, M_actual)
            topk_vals, topk_pos = torch.topk(p, k_forward, dim=0)  # local indices into [0..M_actual-1]
            kept_local_pos = topk_pos  # [k_forward]

            # Ensure at least min_token_num if requested (we will still produce exactly k_cur in the end)
            if kept_local_pos.numel() < min_token_num:
                need = min_token_num - kept_local_pos.numel()
                kept_mask = torch.zeros(M_actual, dtype=torch.bool, device=device)
                kept_mask[kept_local_pos] = True
                remaining = (~kept_mask).nonzero(as_tuple=True)[0]
                if remaining.numel() > 0:
                    add_pos = remaining[:need]
                    kept_local_pos = torch.cat([kept_local_pos, add_pos], dim=0)

            # Now ensure exactly k_cur positions by repeating top entries if necessary
            k_selected = kept_local_pos.numel()
            if k_selected < k_cur:
                # repeat from the top of kept_local_pos (or from topk if kept_local_pos shorter)
                if k_selected == 0:
                    # shouldn't happen because M_actual>0, but protect anyway
                    pad_pos = torch.zeros(k_cur, dtype=torch.long, device=device)
                    final_kept_local = pad_pos
                else:
                    # how many to add
                    need = k_cur - k_selected
                    # create repeats by cycling through kept_local_pos (or topk_pos)
                    repeat_idx = kept_local_pos
                    # If kept_local_pos < need, tile
                    reps = (need + repeat_idx.numel() - 1) // repeat_idx.numel()
                    pad_seq = repeat_idx.repeat(reps)[:need]
                    final_kept_local = torch.cat([kept_local_pos, pad_seq], dim=0)  # length k_cur
            else:
                final_kept_local = kept_local_pos[:k_cur]  # ensure exactly k_cur if more

            # final_kept_local: local indices into cand_feats, length == k_cur
            # For gradients we will only use p at these positions.

            # --- Step 5: Straight-Through on selected positions only (avoid M_actual*d intermediate) ---
            # Build selected probability vector and ST mask only for the selected positions
            p_selected = p[final_kept_local]  # [k_cur]
            if training:
                # hard selected mask (forward = ones)
                hard_selected = torch.ones_like(p_selected)
                # ST trick: forward is hard_selected, backward flows through p_selected
                mask_st_selected = hard_selected - p_selected.detach() + p_selected  # [k_cur]
                # build output features: only k_cur x d allocated
                selected_feats = cand_feats[final_kept_local]  # [k_cur, d]
                output_feats = (mask_st_selected.unsqueeze(-1) * selected_feats)  # [k_cur, d]
                # for details, build surrogate_weights over full M_actual for diagnostics if requested
                if return_details:
                    y_for_details = torch.zeros_like(p)
                    y_for_details[final_kept_local] = mask_st_selected
            else:
                # eval: hard selection, no gradient to selector (p detached)
                selected_feats = cand_feats[final_kept_local]  # [k_cur, d]
                output_feats = selected_feats  # [k_cur, d]
                if return_details:
                    y_for_details = torch.zeros_like(p)
                    y_for_details[final_kept_local] = 1.0

            # Append per-token tensor with shape exactly [k_cur, d]
            batch_list.append(output_feats)

            # Fill details if requested
            if return_details:
                # soft_agg: weighted aggregation using surrogate weights over candidates
                soft_agg = torch.matmul(y_for_details, cand_feats)  # [d]
                details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = idx_candidates[final_kept_local]  # global indices
                details['surrogate_weights'][bi][ti] = y_for_details if training else y_for_details.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    return selected

def select_features_network_mmr_variable_k_gpu(
    features,
    attn_map,
    query,
    M=100,
    lambda_mmr=1.0,
    tau=0.002,
    method='attn_mass',
    alpha=0.9,
    min_k=1,
    max_k=30,
    min_token_num=1,
    selector_model=None,
    training=False,
    return_details=False,
    device=None,
    # --- soft MMR specific ---
    enable_soft_mmr=True,
    beta_mmr=0.8,
    mmr_exclude_self=True,
    mmr_normalize_redundancy=True,
    ):
    """
    MMR + variable k per token (GPU-friendly) + integrated selector + 可微软化 MMR.
    新增 soft-MMR 参数:
    enable_soft_mmr: bool, 是否启用可微软化 MMR（默认 True）
    beta_mmr: float, redundancy 惩罚系数（越大越鼓励多样性）
    mmr_exclude_self: bool, 是否在 S 中置零对角（避免 self contribute）
    mmr_normalize_redundancy: bool, 是否把 redundancy 标准化（/sum(p0) 或 /max）以更稳定调参

    返回与之前一致；若 return_details=True，会在 details 中额外返回 'logits_raw' 和 'logits_mmr' 供分析。
    """
    import torch
    import torch.nn.functional as F

    assert features.dim() == 3 and attn_map.dim() == 3 and query.dim() == 3
    if device is None:
        device = features.device

    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    # 如果没有外部 selector，则用默认（调用者如果要训练 selector，应在外部实例化并传入该实例）
    if selector_model is None:
        # 轻量默认 selector（与之前示例一致）
        class DefaultSelector(torch.nn.Module):
            def __init__(self, d_feat, d_query=None, proj_dim=256, use_prior=True, tau=0.07):
                super().__init__()
                if d_query is None:
                    d_query = d_feat
                self.wp = torch.nn.Linear(d_feat, proj_dim)
                self.wt = torch.nn.Linear(d_query, proj_dim)
                self.mlp = torch.nn.Sequential(
                    torch.nn.LayerNorm(proj_dim * 2 + (1 if use_prior else 0)),
                    torch.nn.Linear(proj_dim * 2 + (1 if use_prior else 0), proj_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_dim, 1)
                )
                self.use_prior = use_prior
                self.tau = tau

            def forward(self, query, cand_feats, cls_prior=None):
                Mloc = cand_feats.shape[0]
                p = F.normalize(self.wp(cand_feats), dim=-1)
                q = F.normalize(self.wt(query.unsqueeze(0) if query.dim()==1 else query), dim=-1)
                q = q.squeeze(0)
                dot = torch.matmul(p, q) / self.tau
                q_exp = q.unsqueeze(0).expand(Mloc, -1)
                if cls_prior is None:
                    prior = torch.zeros(Mloc, 1, device=cand_feats.device)
                else:
                    prior = cls_prior.unsqueeze(1)
                mlp_in = torch.cat([p, q_exp, prior], dim=-1)
                mlp_logits = self.mlp(mlp_in).squeeze(-1)
                logits = dot + mlp_logits
                return logits

        selector_model = DefaultSelector(d_feat=d, d_query=query.shape[2], proj_dim=min(256, d), use_prior=True, tau=0.07)
        selector_model.to(device)

    selector_model.eval() if not training else selector_model.train()

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    details = None
    if return_details:
        details = {
            'p': [[None for _ in range(T)] for _ in range(bs)],
            'indices': [[None for _ in range(T)] for _ in range(bs)],
            'surrogate_weights': [[None for _ in range(T)] for _ in range(bs)],
            'soft_agg': [[None for _ in range(T)] for _ in range(bs)],
            'logits_raw': [[None for _ in range(T)] for _ in range(bs)],
            'logits_mmr': [[None for _ in range(T)] for _ in range(bs)],
        }

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d_q]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            if tau > 0.0:
                mask = scores >= tau
                idx_candidates = mask.nonzero(as_tuple=True)[0]
            else:
                idx_candidates = torch.arange(P, device=device)

            if idx_candidates.numel() < min(M, P):
                take = min(M, P)
                idx_candidates = scores.topk(take).indices.to(device)
            else:
                cand_scores = scores[idx_candidates]
                order = torch.argsort(cand_scores, descending=True)
                idx_candidates = idx_candidates[order][:M]

            M_actual = idx_candidates.numel()
            if M_actual == 0:
                # 没有候选，返回仅 query
                batch_list.append(q_b[ti].unsqueeze(0))
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            cand_scores = scores[idx_candidates]  # descending by construction

            if False:
                # 决定 k_cur
                if method == 'cumulative':
                    cum = torch.cumsum(cand_scores, dim=0)
                    pos = (cum >= alpha).nonzero(as_tuple=True)[0]
                    if pos.numel() == 0:
                        n_req = M_actual
                    else:
                        n_req = int(pos[0].item()) + 1
                    k_cur = max(min_k, min(n_req, max_k, M_actual))
                elif method == 'perplexity':
                    p_dist = cand_scores / (cand_scores.sum() + eps)
                    entropy = -(p_dist * (p_dist + eps).log()).sum()
                    perp = torch.exp(entropy).item()
                    k_cur = int(round(perp))
                    k_cur = max(min_k, min(k_cur, max_k, M_actual))
                elif method == 'attn_mass':
                    mass = cand_scores.sum().item()
                    k_cur = int(round(min_k + (max_k - min_k) * mass))
                    k_cur = max(min_k, min(k_cur, max_k, M_actual))
                else:
                    raise ValueError("Unknown method")

                k_cur = max(k_cur, min_token_num)
                k_cur = min(k_cur, M_actual)
            else:
                k_cur = 10

            if k_cur == 0:
                batch_list.append(q_b[ti].unsqueeze(0))
                if return_details:
                    details['p'][bi][ti] = torch.zeros(0, device=device)
                    details['indices'][bi][ti] = torch.zeros(0, dtype=torch.long, device=device)
                    details['surrogate_weights'][bi][ti] = torch.zeros(0, device=device)
                    details['soft_agg'][bi][ti] = torch.zeros(d, device=device)
                    details['logits_raw'][bi][ti] = torch.zeros(0, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(0, device=device)
                continue

            # 若候选数 <= k_cur，直接全部按 attention 保留（不需 selector 排序）
            if M_actual <= k_cur:
                kept_idx = idx_candidates
                if return_details:
                    details['p'][bi][ti] = torch.ones(M_actual, device=device)
                    details['indices'][bi][ti] = kept_idx
                    details['surrogate_weights'][bi][ti] = torch.ones(M_actual, device=device)
                    details['soft_agg'][bi][ti] = torch.matmul(details['surrogate_weights'][bi][ti], feats_b[kept_idx])
                    details['logits_raw'][bi][ti] = torch.zeros(M_actual, device=device)
                    details['logits_mmr'][bi][ti] = torch.zeros(M_actual, device=device)
                token_patches = feats_b[kept_idx]  # [k_cur, d]
                token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
                batch_list.append(token_feats)
                continue

            # ########################
            # 用 selector 在候选池上打分（可训练 head）
            # ########################
            cand_feats = feats_b[idx_candidates]  # [M_actual, d]
            cand_feats_norm = feats_norm_b[idx_candidates]  # [M_actual, d] (已归一化)
            cls_prior = cand_scores  # prior

            logits_raw = selector_model(q_b[ti], cand_feats, cls_prior=cls_prior)  # [M_actual]
            logits_mmr = logits_raw

            if enable_soft_mmr and M_actual > 1:
                # 计算相似矩阵 S (cosine), 排除对角可选
                S = torch.matmul(cand_feats_norm, cand_feats_norm.t())  # [M_actual, M_actual]
                if mmr_exclude_self:
                    # 把对角置0，避免 self contribute
                    try:
                        S = S.clone()
                        S.fill_diagonal_(0.0)
                    except Exception:
                        # 兼容旧 pytorch: 手动减去对角
                        diag = torch.diag(torch.diag(S))
                        S = S - diag

                # base selection probability
                p0 = torch.sigmoid(logits_raw)  # [M_actual]
                # redundancy: S @ p0
                redundancy = torch.matmul(S, p0)  # [M_actual]

                if mmr_normalize_redundancy:
                    # 标准化 redundancy，避免 scale 问题：常见选择是 / (p0.sum()+eps) 或 / redundancy.max()
                    denom = p0.sum().clamp(min=eps)
                    redundancy = redundancy / denom
                    # 也可用 max normalization： redundancy = redundancy / (redundancy.max()+eps)

                # 调整 logits
                logits_mmr = logits_raw - beta_mmr * redundancy

            # logits -> soft scores
            p = torch.sigmoid(logits_mmr)  # [M_actual]

            # top-k on p
            topk_vals, topk_pos = torch.topk(p, k_cur, dim=0)  # positions in candidate pool
            kept_idx = idx_candidates[topk_pos]  # original global indices

            # hard mask & STE surrogate
            hard = torch.zeros_like(p)
            hard[topk_pos] = 1.0
            if training:
                y = (hard - p).detach() + p
            else:
                y = hard

            # soft aggregation
            soft_agg = torch.matmul(y, cand_feats)  # [d]

            # 保证至少 min_token_num
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 构建返回 token feats（query 在前）
            token_patches = feats_b[kept_idx]  # [k_cur, d]
            token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
            batch_list.append(token_feats)

            # fill details
            if return_details:
                details['p'][bi][ti] = p if training else p.detach()
                details['indices'][bi][ti] = kept_idx
                details['surrogate_weights'][bi][ti] = y if training else y.detach()
                details['soft_agg'][bi][ti] = soft_agg if training else soft_agg.detach()
                details['logits_raw'][bi][ti] = logits_raw.detach()
                details['logits_mmr'][bi][ti] = logits_mmr.detach()

        selected.append(batch_list)

    if return_details:
        return selected, details
    else:
        return selected

def select_features_mmr_variable_k_gpu(features,
    attn_map,
    query,
    M=100,
    lambda_mmr=0.99,
    tau=0.002,
    method='attn_mass',
    alpha=0.9,
    min_k=1,
    max_k=30,
    min_token_num=1):
    """
    MMR + variable k per token (GPU-friendly).
    features: [bs, P, d] (torch.Tensor on device)
    attn_map: [bs, T, P]
    query:    [bs, T, d]
    M: candidate pool size (top-M by attention)
    method: 'cumulative' | 'perplexity' | 'attn_mass'
    - cumulative: use alpha (e.g., 0.9) on cumulative attention
    - perplexity: use entropy->perplexity
    - attn_mass: map candidate mass to k
    min_k/max_k: bounds for k per token
    min_token_num: ensure at least this many patches (fallback)
    返回:
    list(bs) -> list(T) -> Tensor [1 + k_cur, d] (query 前置)
    """
    import torch
    import torch.nn.functional as F
    device = features.device
    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            if tau > 0.0:
                mask = scores >= tau
                # print(mask.sum())
                idx_candidates = mask.nonzero(as_tuple=True)[0]
            else:
                idx_candidates = torch.arange(P, device=device)

            if idx_candidates.numel() < min(M, P):
                take = min(M, P)
                idx_candidates = scores.topk(take).indices.to(device)
            else:
                cand_scores = scores[idx_candidates]
                order = torch.argsort(cand_scores, descending=True)
                idx_candidates = idx_candidates[order][:M]

            M_actual = idx_candidates.numel()
            #if M_actual == 0:
            #    batch_list.append(q_b[ti].unsqueeze(0))
            #    continue

            # print(idx_candidates)

            cand_scores = scores[idx_candidates]  # descending
            # 决定当前 token 要选多少个 k_cur（不含 query）
            if method == 'cumulative':
                # 找最小 n s.t. cum_sum >= alpha
                cum = torch.cumsum(cand_scores, dim=0)
                pos = (cum >= alpha).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    n_req = M_actual
                else:
                    n_req = int(pos[0].item()) + 1
                k_cur = max(min_k, min(n_req, max_k, M_actual))
            elif method == 'perplexity':
                p = cand_scores / (cand_scores.sum() + eps)
                entropy = -(p * (p + eps).log()).sum()
                perp = torch.exp(entropy).item()
                k_cur = int(round(perp))
                k_cur = max(min_k, min(k_cur, max_k, M_actual))
            elif method == 'attn_mass':
                # 候选池内的 attention mass（相对于全部P）
                mass = cand_scores.sum().item()  # in (0,1]
                # 线性映射 mass in [0,1] -> k in [min_k, max_k]
                k_cur = int(round(min_k + (max_k - min_k) * mass))
                k_cur = max(min_k, min(k_cur, max_k, M_actual))
            else:
                raise ValueError("Unknown method")

            # 保护：至少保留 min_token_num
            k_cur = max(k_cur, min_token_num)
            k_cur = min(k_cur, M_actual)

            # if lambda_mmr == 0:
            #     kept_idx = idx_candidates[-k_cur:]

            # 若 k_cur == 0（理论上不会），直接只返回 query
            if k_cur == 0:
                batch_list.append(q_b[ti].unsqueeze(0))
                continue

            # 若候选数 <= k_cur，直接全部按 attention 保留
            if M_actual <= k_cur:
                kept_idx = idx_candidates
            else:
                # MMR on cand_feats
                cand_feats = feats_norm_b[idx_candidates]  # [M_actual,d]
                sim_mat = torch.matmul(cand_feats, cand_feats.T)  # [M_actual,M_actual]
                selected_positions = []
                selected_mask = torch.zeros(M_actual, dtype=torch.bool, device=device)
                for step in range(k_cur):
                    if step == 0:
                        _, pick = torch.max(cand_scores, dim=0)
                        pick = int(pick)
                    else:
                        if selected_mask.any():
                            sel_cols = sim_mat[:, selected_mask]  # [M_actual,S]
                            novelty, _ = sel_cols.max(dim=1)
                        else:
                            novelty = torch.zeros(M_actual, device=device)
                        mmr_score = lambda_mmr * cand_scores - (1.0 - lambda_mmr) * (novelty + 1) * 0.5
                        mmr_score = mmr_score.masked_fill(selected_mask, float('-inf'))
                        _, pick = torch.max(mmr_score, dim=0)
                        pick = int(pick)
                    selected_positions.append(pick)
                    selected_mask[pick] = True
                    if selected_mask.all():
                        break
                kept_idx = idx_candidates[torch.tensor(selected_positions, dtype=torch.long, device=device)]

            # print(kept_idx)
            in_ = [l in idx_candidates[:10] for l in kept_idx]
            # print(np.sum(in_))
            # 若仍少于 min_token_num，从余下按 attention 补足
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 拼接 query 在最前
            token_patches = feats_b[kept_idx]  # [k_cur,d]
            token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
            batch_list.append(token_feats)

        selected.append(batch_list)

    return selected

def analyze_image_attention(attention_maps, image_token_start, image_token_end):
    """
    分析每个生成 token 对输入图像 token 的 attention 比重。

    参数:
        attention_maps (list): 每一层的 attention map 列表，每个 attention map 的形状为 [1, 32, 1, k]。
        image_token_start (int): 图像 token 的起始位置。
        image_token_end (int): 图像 token 的结束位置。

    返回:
        dict: 每个生成 token 对图像 token 的 attention 比重，格式为 {token_index: attention_weights}。
    """
    results = {}

    for token_index, layer_attention_maps in enumerate(attention_maps[1:]):
        # 初始化存储每个 token 对图像 token 的 attention 比重
        image_attention_weights = []

        for layer_idx, attention_map in enumerate(layer_attention_maps[:]):
            # attention_map 的形状为 [1, 32, 1, k]
            # 取最后一个维度（k）中对应图像 token 的部分
            image_attention = attention_map[:, :, :, image_token_start:image_token_end].sum(-1) / attention_map[:, :, :, :].sum(-1)

            # 计算每个注意力头对图像 token 的 attention 均值
            # head_attention_weights, _ = image_attention.max(dim=-1)  # 形状为 [1, 32, 1]
            
            head_attention_weights = image_attention
            image_attention_weights.append(head_attention_weights)

        # 将所有层的 attention 权重拼接并计算均值
        image_attention_weights = torch.cat(image_attention_weights, dim=1)  # 形状为 [1, 32 * num_layers, 1]
        image_attention_weights = image_attention_weights.mean(dim=1).squeeze()  # 形状为 [1]

        # 存储结果
        results[token_index] = image_attention_weights.mean().cpu()

    return results

def select_features_mmr_variable_k_gpu(features,
    attn_map,
    query,
    M=50,
    lambda_mmr=0.9,
    tau=0.002,
    method='attn_mass',
    alpha=0.9,
    min_k=1,
    max_k=30,
    min_token_num=1):
    """
    MMR + variable k per token (GPU-friendly).
    features: [bs, P, d] (torch.Tensor on device)
    attn_map: [bs, T, P]
    query:    [bs, T, d]
    M: candidate pool size (top-M by attention)
    method: 'cumulative' | 'perplexity' | 'attn_mass'
    - cumulative: use alpha (e.g., 0.9) on cumulative attention
    - perplexity: use entropy->perplexity
    - attn_mass: map candidate mass to k
    min_k/max_k: bounds for k per token
    min_token_num: ensure at least this many patches (fallback)
    返回:
    list(bs) -> list(T) -> Tensor [1 + k_cur, d] (query 前置)
    """
    
    device = features.device
    bs, P, d = features.shape
    _, T, P2 = attn_map.shape
    assert P == P2
    assert attn_map.device == device and query.device == device

    feats_norm = F.normalize(features, dim=2)  # [bs,P,d]
    selected = []
    eps = 1e-12

    for bi in range(bs):
        feats_b = features[bi]         # [P,d]
        feats_norm_b = feats_norm[bi] # [P,d]
        attn_b = attn_map[bi]         # [T,P]
        q_b = query[bi]               # [T,d]
        batch_list = []

        for ti in range(T):
            scores = attn_b[ti]  # [P]

            # 初步候选：tau 过滤再 top-M 补足
            if tau > 0.0:
                mask = scores >= tau
                # print(mask.sum())
                idx_candidates = mask.nonzero(as_tuple=True)[0]
            else:
                idx_candidates = torch.arange(P, device=device)

            if idx_candidates.numel() < min(M, P):
                take = min(M, P)
                idx_candidates = scores.topk(take).indices.to(device)
            else:
                cand_scores = scores[idx_candidates]
                order = torch.argsort(cand_scores, descending=True)
                idx_candidates = idx_candidates[order][:M]

            M_actual = idx_candidates.numel()


            cand_scores = scores[idx_candidates]  # descending
            # 决定当前 token 要选多少个 k_cur（不含 query）
            if method == 'cumulative':
                # 找最小 n s.t. cum_sum >= alpha
                cum = torch.cumsum(cand_scores, dim=0)
                pos = (cum >= alpha).nonzero(as_tuple=True)[0]
                if pos.numel() == 0:
                    n_req = M_actual
                else:
                    n_req = int(pos[0].item()) + 1
                k_cur = max(min_k, min(n_req, max_k, M_actual))
            elif method == 'perplexity':
                p = cand_scores / (cand_scores.sum() + eps)
                entropy = -(p * (p + eps).log()).sum()
                perp = torch.exp(entropy).item()
                k_cur = int(round(perp))
                k_cur = max(min_k, min(k_cur, max_k, M_actual))
            elif method == 'attn_mass':
                # 候选池内的 attention mass（相对于全部P）
                mass = cand_scores.sum().item()  # in (0,1]
                # 线性映射 mass in [0,1] -> k in [min_k, max_k]
                k_cur = int(round(min_k + (max_k - min_k) * mass))
                k_cur = max(min_k, min(k_cur, max_k, M_actual))
            else:
                raise ValueError("Unknown method")

            # 保护：至少保留 min_token_num
            k_cur = max(k_cur, min_token_num)
            k_cur = min(k_cur, M_actual)

            # if lambda_mmr == 0:
            #     kept_idx = idx_candidates[-k_cur:]

            # 若 k_cur == 0（理论上不会），直接只返回 query
            if k_cur == 0:
                batch_list.append(q_b[ti].unsqueeze(0))
                continue

            # 若候选数 <= k_cur，直接全部按 attention 保留
            if M_actual <= k_cur:
                kept_idx = idx_candidates
            else:
                # MMR on cand_feats
                cand_feats = feats_norm_b[idx_candidates]  # [M_actual,d]
                sim_mat = torch.matmul(cand_feats, cand_feats.T)  # [M_actual,M_actual]
                selected_positions = []
                selected_mask = torch.zeros(M_actual, dtype=torch.bool, device=device)
                for step in range(k_cur):
                    if step == 0:
                        _, pick = torch.max(cand_scores, dim=0)
                        pick = int(pick)
                    else:
                        if selected_mask.any():
                            sel_cols = sim_mat[:, selected_mask]  # [M_actual,S]
                            novelty, _ = sel_cols.max(dim=1)
                        else:
                            novelty = torch.zeros(M_actual, device=device)
                        mmr_score = lambda_mmr * cand_scores - (1.0 - lambda_mmr) * (novelty + 1) * 0.5
                        mmr_score = mmr_score.masked_fill(selected_mask, float('-inf'))
                        _, pick = torch.max(mmr_score, dim=0)
                        pick = int(pick)
                    selected_positions.append(pick)
                    selected_mask[pick] = True
                    if selected_mask.all():
                        break
                kept_idx = idx_candidates[torch.tensor(selected_positions, dtype=torch.long, device=device)]

            # 若仍少于 min_token_num，从余下按 attention 补足
            if kept_idx.numel() < min_token_num:
                need = min_token_num - kept_idx.numel()
                kept_flag = torch.zeros(M_actual, dtype=torch.bool, device=device)
                if kept_idx.numel() > 0:
                    eq = (idx_candidates.unsqueeze(1) == kept_idx.unsqueeze(0))
                    kept_flag = eq.any(dim=1)
                remain_pos = (~kept_flag).nonzero(as_tuple=True)[0]
                if remain_pos.numel() > 0:
                    add_pos = remain_pos[:need]
                    add_idx = idx_candidates[add_pos]
                    if kept_idx.numel() > 0:
                        kept_idx = torch.cat([kept_idx, add_idx], dim=0)
                    else:
                        kept_idx = add_idx

            # 拼接 query 在最前
            token_patches = feats_b[kept_idx]  # [k_cur,d]
            token_feats = torch.cat([q_b[ti].unsqueeze(0), token_patches], dim=0)
            batch_list.append(token_feats)

        selected.append(batch_list)

    return selected

def get_retrieval():
    centers = np.load("/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/text_embedding_organ_centers_train.npy", allow_pickle=True)

    # 你关心的anatomy顺序（合并mediastinum和heart）
    anatomy_list = [
        "trachea and bronchie", "mediastinum+heart", "lung", "esophagus",
        "pleura", "bone", "thyroid", "breast", "abdomen", "others"
    ]

    # 结果字典
    result = {}

    with open("/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_region_report.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vol = row['Volumename']
            anatomy = row['Anatomy'].strip().lower()
            sentence = row['Sentence']
            if anatomy in ["mediastinum", "heart"]:
                if vol not in result:
                    # 初始化为10个空字符串
                    result[vol] = [''] * len(anatomy_list)
                # 先把mediastinum和heart都放到同一个位置
                idx = 1  # anatomy_list中"mediastinum+heart"的位置
                if result[vol][idx]:
                    # 已有内容，拼接
                    result[vol][idx] += ' ' + sentence
                else:
                    result[vol][idx] = sentence
            elif anatomy in ["trachea and bronchie", "lung", "esophagus", "pleura", "bone", "thyroid", "breast", "abdomen", "others"]:
                if vol not in result:
                    result[vol] = [''] * len(anatomy_list)
                idx = anatomy_list.index(anatomy if anatomy != "mediastinum" and anatomy != "heart" else "mediastinum+heart")
                result[vol][idx] = sentence

    key_reports = [{}]*10
    for idx in range(10):
        for key in list(centers[idx].keys()):
            key_reports[idx][key] = result[key][idx]
    
    return centers, key_reports

def select_features(features, attn_map, b=10, r=0.6):
    """
    features: [bs, 4096, 768]
    attn_map: [bs, 10, 4096]
    b: 每个token选择的特征数
    r: 余弦相似度阈值
    """
    bs, num_tokens, num_feats = attn_map.shape
    feat_dim = features.shape[-1]
    selected_features = []

    for batch_idx in range(bs):
        batch_selected = []
        feats = features[batch_idx]  # [4096, 768]
        attn = attn_map[batch_idx]   # [10, 4096]

        for token_idx in range(num_tokens):
            attn_scores = attn[token_idx]  # [4096]
            # 按分数从大到小排序
            sorted_scores, sorted_indices = torch.sort(attn_scores, descending=True)
            token_selected = []
            for idx in sorted_indices:
                candidate_feat = feats[idx]  # [768]
                # 如果还没选满b个，且与已选的都小于r
                if len(token_selected) == 0:
                    token_selected.append(candidate_feat)
                else:
                    # 计算与已选的余弦相似度
                    selected_feats = torch.stack(token_selected)  # [n, 768]
                    candidate_feat_norm = F.normalize(candidate_feat.unsqueeze(0), dim=1)  # [1, 768]
                    selected_feats_norm = F.normalize(selected_feats, dim=1)  # [n, 768]
                    # 计算与所有已选的最大相似度
                    sim = torch.matmul(selected_feats_norm, candidate_feat_norm.T).squeeze(1)  # [n]
                    max_sim = sim.abs().max()  # 绝对值，防止负相关
                    if max_sim < r:
                        token_selected.append(candidate_feat)
                if len(token_selected) >= b:
                    break
            # 补零
            if len(token_selected) < b:
                num_pad = b - len(token_selected)
                pad_feats = [torch.zeros(feat_dim, device=features.device, dtype=features.dtype) for _ in range(num_pad)]
                token_selected.extend(pad_feats)
            # 选出的b个特征
            token_selected = torch.stack(token_selected)  # [b, 768]
            batch_selected.append(token_selected)
        # 10组特征cat
        batch_selected = torch.cat(batch_selected, dim=0)  # [10*b, 768]
        selected_features.append(batch_selected)
    # [bs, 10*b, 768]
    selected_features = torch.stack(selected_features, dim=0)
    return selected_features

def select_features_minpos(features, attn_map, b=10):
    """
    features: [bs, 4096, 768]
    attn_map: [bs, 10, 4096]
    b: 每个 token 要选出的特征数
    return:   [bs, 10*b, 768]
    """
    bs, num_tokens, _ = attn_map.shape
    feat_dim = features.shape[-1]
    device   = features.device
    out = []

    for bi in range(bs):
        feats = features[bi]          # [4096, 768]
        attn  = attn_map[bi]          # [10, 4096]
        batch_tokens = []

        for ti in range(num_tokens):
            score = attn[ti]          # [4096]

            # 1) 只保留正分
            pos_mask = score > 0
            if pos_mask.sum() == 0:   # 无正分
                batch_tokens.append(torch.zeros(b, feat_dim, device=device))
                continue

            pos_idx   = torch.where(pos_mask)[0]
            pos_score = score[pos_idx]

            # 2) 按正分升序取前 b 个
            _, order = torch.sort(pos_score)
            chosen_idx = pos_idx[order[:b]]        # 最多 b 个
            token_feats = feats[chosen_idx]        # [min(b,N), 768]

            # 3) 不足 b 时零填充
            pad_len = b - token_feats.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, feat_dim, device=device)
                token_feats = torch.cat([token_feats, pad], dim=0)

            batch_tokens.append(token_feats)

        out.append(torch.cat(batch_tokens, dim=0))   # [10*b, 768]

    return torch.stack(out, dim=0)    

def select_features_segment(features, attn_map, b=10, eps=1e-8):
    """
    features  : [bs, 4096, 768]
    attn_map  : [bs, 10, 4096]
    b         : 每个 token 要选出的特征数
    eps       : 防止除以 0
    return    : [bs, 10*b, 768]   与原函数完全一致
    """
    bs, num_tokens, _ = attn_map.shape
    feat_dim = features.shape[-1]
    selected = []

    for bi in range(bs):
        feats = features[bi]          # [4096, 768]
        attn  = attn_map[bi]          # [10, 4096]
        batch_tokens = []

        for ti in range(num_tokens):
            score = attn[ti]          # [4096]

            # 1) 只保留正分
            pos_mask = score > 0
            if pos_mask.sum() == 0:     # 无正分，直接全零
                batch_tokens.append(torch.zeros(b, feat_dim, device=features.device))
                continue

            pos_idx   = torch.where(pos_mask)[0]
            pos_score = score[pos_idx]  # [N_pos]

            # 2) 把正分区间切成 b 段
            min_s, max_s = pos_score.min(), pos_score.max()
            if max_s - min_s < eps:     # 所有正分相同，直接选前 b 个
                chosen_idx = pos_idx[:b]
            else:
                seg_width = (max_s - min_s) / b
                chosen_idx = []
                for k in range(b):
                    lower = min_s + k * seg_width
                    upper = min_s + (k + 1) * seg_width
                    seg_mask = (pos_score >= lower) & (pos_score < upper)
                    # 段内取最高
                    if seg_mask.any():
                        seg_idx = pos_idx[seg_mask][pos_score[seg_mask].argmax()]
                    else:
                        seg_idx = pos_idx[0]        # fallback：选全局第一个
                    chosen_idx.append(seg_idx)

            chosen_idx = torch.tensor(chosen_idx, device=features.device)
            token_feats = feats[chosen_idx]          # [b, 768]

            # 3) 若不足 b 个，零填充
            if token_feats.shape[0] < b:
                pad = torch.zeros(b - token_feats.shape[0], feat_dim,
                                  device=features.device)
                token_feats = torch.cat([token_feats, pad], dim=0)

            batch_tokens.append(token_feats)

        # 10 组拼接
        selected.append(torch.cat(batch_tokens, dim=0))   # [10*b, 768]

    return torch.stack(selected, dim=0)      

def select_features_top100_segment(features, attn_map, b=10, r=0.95):
    """
    features: [bs, 4096, 768]
    attn_map: [bs, 10, 4096]
    b: 每个 token 要选出的特征数
    r: 余弦相似度阈值（去重用）
    return:   [bs, 10*b, 768]
    """
    bs, num_tokens, _ = attn_map.shape
    feat_dim = features.shape[-1]
    device = features.device
    out = []

    for bi in range(bs):
        feats = features[bi]          # [4096, 768]
        attn  = attn_map[bi]          # [10, 4096]
        batch_tokens = []

        for ti in range(num_tokens):
            score = attn[ti]          # [4096]

            # 1) 取前 100 高分
            top_scores, top_idx = torch.topk(score, k=100, largest=True)
            top_feats = feats[top_idx]            # [100, 768]



            # 2) 将 100 个分数切成 b 段（等宽）
            if b == 1:        # 避免除 0
                seg_width = 1.0
            else:
                seg_width = (top_scores.max() - top_scores.min()) / b

            chosen = []
            for k in range(b):
                low  = top_scores.min() + k * seg_width
                high = top_scores.min() + (k + 1) * seg_width
                # 段内候选
                mask = (top_scores >= low) & (top_scores < high)
                if not mask.any():                  # 空段
                    mask = (top_scores == top_scores.min())  # fallback 到最低分
                seg_idx = top_idx[mask][top_scores[mask].argmax()]   # 段内最高分
                seg_feat = feats[seg_idx]

                # 3) 余弦去重
                if len(chosen) == 0:
                    chosen.append(seg_feat)
                else:
                    cand_norm = F.normalize(seg_feat.unsqueeze(0), dim=1)
                    sel_norm  = F.normalize(torch.stack(chosen), dim=1)
                    max_sim   = torch.matmul(sel_norm, cand_norm.T).abs().max()
                    if max_sim < r:
                        chosen.append(seg_feat)

                if len(chosen) >= b:
                    break

            # 4) 补齐 b 个
            chosen = torch.stack(chosen) if chosen else torch.empty(0, feat_dim, device=device)
            pad_len = b - chosen.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, feat_dim, device=device)
                chosen = torch.cat([chosen, pad], dim=0)

            batch_tokens.append(chosen)             # [b, 768]

        out.append(torch.cat(batch_tokens, dim=0))  # [10*b, 768]

    return torch.stack(out, dim=0)         

def select_features_weighted_cover(features, attn_map, b=10):
    """
    features : [bs, 4096, 768]
    attn_map : [bs, 10, 4096]  每个 token 的 CLS-token attention 分数
    b        : 每个 token 要选出的特征数
    return   : [bs, 10*b, 768]
    """
    bs, num_tokens, _ = attn_map.shape
    feat_dim = features.shape[-1]
    device   = features.device
    out = []

    # 预先归一化特征，方便后续余弦距离
    feats_norm = F.normalize(features, dim=2)  # [bs, 4096, 768]

    for bi in range(bs):
        feats = features[bi]          # [4096, 768]
        scores = attn_map[bi]         # [10, 4096]

        batch_tokens = []
        for ti in range(num_tokens):
            score = scores[ti]        # [4096]
            token_feats = feats       # [4096, 768]
            token_norm  = feats_norm[bi]

            # 归一化 attention → 权重
            weight = score / (score.sum() + 1e-8)   # [4096]

            # 贪心加权覆盖
            selected_idx = []
            mask = torch.ones_like(weight, dtype=torch.bool)

            # 预先计算余弦距离矩阵（4096×4096，GPU 可承受）
            dist = 1 - torch.matmul(token_norm, token_norm.T)  # [4096, 4096]

            for _ in range(b):
                if not mask.any():
                    break
                if not selected_idx:                       # 第一轮：权重最大
                    idx = torch.argmax(weight * mask.float())
                else:
                    sel = torch.tensor(selected_idx, device=device)
                    # 到已选集合的最小距离
                    min_dist = dist[:, sel].min(dim=1)[0]
                    # 加权后最远
                    idx = torch.argmax(weight * mask.float() * min_dist)
                selected_idx.append(idx)
                mask[idx] = False

            # 若不足 b，用零填充
            if len(selected_idx) < b:
                pad = torch.zeros(b - len(selected_idx), feat_dim, device=device)
                token_out = torch.cat([token_feats[selected_idx], pad], dim=0)
            else:
                idx_tensor = torch.tensor(selected_idx, device=device)[:b]
                if idx_tensor.numel() == 0:
                    token_out = torch.zeros(b, feat_dim, device=device)
                else:
                    token_out = token_feats[idx_tensor]

            batch_tokens.append(token_out)  # [b, 768]

        out.append(torch.cat(batch_tokens, dim=0))  # [10*b, 768]

    return torch.stack(out, dim=0)      # [bs, 10*b, 768]

def select_features_adaptive(features,
                             attn_map,
                             query,
                             r=0.6,
                             tau=0.000,
                             max_per_token=30,
                             min_token_num=1):
    """
    静态、非端到端地根据 attention 分布为每个 token 选 patch。

    新增:
      query: [bs, num_tokens, 768] —— 每个 token 的 query 向量，会插到对应 list 最前面
      min_token_num: 每个 token 最终最少保留的 patch 数（不含 query）

    返回
      list[list[Tensor]]
        外层长度 = batch_size
        内层长度 = num_tokens
        最里层 Tensor 形状 = [1 + k_i, 768]，其中第 0 维是 query, k_i ≥ min_token_num
    """
    bs, num_tokens, _ = attn_map.shape
    selected = []

    for bi in range(bs):
        feats      = features[bi]                # [4096, 768]
        feats_norm = F.normalize(feats, dim=1)
        attn       = attn_map[bi]                # [num_tokens, 4096]
        q_batch    = query[bi]                   # [num_tokens, 768]

        batch_list = []
        for ti in range(num_tokens):
            q_vec   = q_batch[ti]                # [768]
            scores  = attn[ti]                   # [4096]

            # 1. 绝对阈值过滤
            valid_mask = scores >= tau
            idx_above  = valid_mask.nonzero(as_tuple=False).squeeze(-1)

            # print(idx_above)
            # 若阈值后不足 min_token_num，则取全局 top-min_token_num
            if idx_above.numel() < min_token_num:
                idx_above = scores.topk(min_token_num).indices

            # 2. 按 attention 排序 + 硬上限
            scores_above = scores[idx_above]
            _, order = torch.sort(scores_above, descending=True)
            idx_sorted = idx_above[order][:max_per_token]
            # 3. 贪婪去重
            keep_idx, keep_norm = [], []
            for idx in idx_sorted:
                cand_norm = feats_norm[idx]
                if keep_norm:
                    sim = torch.matmul(torch.stack(keep_norm), cand_norm).max()
                    if sim >= r:
                        continue
                keep_norm.append(cand_norm)
                keep_idx.append(idx)

            # 若仍不足 min_token_num，再从剩余高分里补
            if len(keep_idx) < min_token_num:
                need = min_token_num - len(keep_idx)
                remain = [i for i in idx_sorted if i not in keep_idx]
                # 按 attention 从高到低补
                remain_sorted = sorted(remain, key=lambda i: scores[i].item(), reverse=True)
                keep_idx.extend(remain_sorted[:need])

            token_feats = torch.cat([feats[ii:ii+1] for ii in keep_idx], 0)           # [k, 768]
            # print(token_feats.shape, keep_idx)
            token_feats = torch.cat([q_vec.unsqueeze(0), token_feats], dim=0)  # [1+k, 768]
            batch_list.append(token_feats)

        selected.append(batch_list)
    return selected


class QFormer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(QFormer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Define linear projection layers for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Multihead attention layers for cross-attention
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward_(self, query, key, value, add_residual=True):
        # Apply linear projections
        query = self.query_proj(query).permute(1, 0, 2)
        key = self.key_proj(key).permute(1, 0, 2)
        value = self.value_proj(value).permute(1, 0, 2)
        
        # Compute first cross-attention
        attn_output1, _ = self.cross_attn1(query, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output1 = attn_output1 + query
        
        # Apply normalization and dropout
        attn_output1 = self.norm1(attn_output1)
        attn_output1 = self.dropout1(attn_output1)
        
        # Compute second cross-attention
        attn_output2, _ = self.cross_attn2(attn_output1, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output2 = attn_output2 + attn_output1
        
        # Apply normalization and dropout
        attn_output2 = self.norm2(attn_output2)
        attn_output2 = self.dropout2(attn_output2)
        
        return attn_output2.permute(1, 0, 2), attn_output2.permute(1, 0, 2)

    def forward__(self, text, img_fea, add_residual=True):
        # Apply linear projections
        query = self.query_proj(text).permute(1, 0, 2)
        key = self.key_proj(img_fea).permute(1, 0, 2)
        value = self.value_proj(img_fea).permute(1, 0, 2)
        
        # Compute first cross-attention
        attn_output1, _ = self.cross_attn1(query, key, value)
        
        # Optionally add residual connection
        if add_residual:
            attn_output1 = attn_output1 + query
        
        # Apply normalization and dropout
        attn_output1 = self.norm1(attn_output1)
        attn_output1 = self.dropout1(attn_output1)
        
        # Compute second cross-attention and get attention weights
        attn_output2, attn_weights = self.cross_attn2(attn_output1, key, value, need_weights=True)
        
        # Optionally add residual connection
        if add_residual:
            attn_output2 = attn_output2 + attn_output1
        
        # Apply normalization and dropout
        attn_output2 = self.norm2(attn_output2)
        attn_output2 = self.dropout2(attn_output2)
        
        # Average attention weights across heads
        avg_attn_weights = attn_weights.mean(dim=0)  # Shape: (query_len, key_len)
        
        # Select top 10 keys for each query
        top_k_values = []
        for i in range(avg_attn_weights.size(0)):  # Iterate over each query
            top_k_indices = avg_attn_weights[i].topk(10, dim=-1).indices
            selected_values = value[top_k_indices, :, :]  # Select top 10 values
            top_k_values.append(selected_values)
        
        # Concatenate selected values
        top_k_values = torch.cat(top_k_values, dim=0)  # Shape: (10 * query_len, batch_size, embed_dim)
        
        # Permute top_k_values to match the output shape
        top_k_values = top_k_values.permute(1, 0, 2)
        
        # Concatenate the first output with the selected top_k_values
        combined_output = torch.cat((attn_output2.permute(1, 0, 2), top_k_values), dim=1)
        
        return attn_output2.permute(1, 0, 2), combined_output

    def forward(self, query, img_fea, add_residual=True,
            dominant_num=10, contextual_num=0):

        """
        query   : [B, Lq, D]   text tokens
        img_fea : [B, Li, D]   image tokens
        """
        """
        query   : [B, Lq, D]   text tokens
        img_fea : [B, Li, D]   image tokens
        """
        B, Lq, D = query.shape
        _, Li, _ = img_fea.shape

        # 1. 原两层 cross-attention，拿 attn_weights
        q = self.query_proj(query).permute(1, 0, 2)      # [Lq, B, D]
        k = self.key_proj(img_fea).permute(1, 0, 2)      # [Li, B, D]
        v = self.value_proj(img_fea).permute(1, 0, 2)    # [Li, B, D]

        attn_out1, _ = self.cross_attn1(q, k, v)
        if add_residual:
            attn_out1 = attn_out1 + q
        attn_out1 = self.norm1(attn_out1)
        attn_out1 = self.dropout1(attn_out1)

        attn_out2, attn_weights = self.cross_attn2(
            attn_out1, k, v, need_weights=True)          # attn_weights: [B, Lq, Li]

        if add_residual:
            attn_out2 = attn_out2 + attn_out1
        attn_out2 = self.norm2(attn_out2)
        attn_out2 = self.dropout2(attn_out2)

        text_out = attn_out2.permute(1, 0, 2)            # [B, Lq, D]  保持原接口

        # 2. 每个 image token 的归属：被哪个 text token 最大关注
        owner = attn_weights.argmax(dim=1)               # [B, Li] 取值 0..Lq-1

        # 3. 逐 text token 做 VisionZIP
        #    先准备好 key/value（key 直接当 metric）
        k = k.permute(1, 0, 2)    # [B, Li, D]
        v = v.permute(1, 0, 2)    # [B, Li, D]

        # 用于收集最终结果
        zipped_list = []

        for b in range(B):
            # 取出当前 batch 数据
            k_b = k[b]             # [Li, D]
            v_b = v[b]             # [Li, D]
            owner_b = owner[b]     # [Li]

            zipped_per_text = []
            for t in range(Lq):
                # 属于该 text token 的 image token 索引
                idx = torch.where(owner_b == t)[0]
                if idx.numel() == 0:          # 没有归属的 img token，用空 tensor
                    # 可以给一个可学习的 dummy token，或跳过
                    # 这里简单用 0 填充
                    comp = torch.zeros(dominant_num + contextual_num, D,
                                    device=k.device, dtype=k.dtype)
                    zipped_per_text.append(comp)
                    continue

                group_k = k_b[idx]           # [N_group, D]
                group_v = v_b[idx]

                N_group = group_k.size(0)
                if N_group <= dominant_num + contextual_num:
                    # token 太少，全部保留再 pad
                    pad = dominant_num + contextual_num - N_group
                    comp = torch.cat([group_v,
                                    torch.zeros(pad, D, device=k.device)], dim=0)
                    zipped_per_text.append(comp)
                    continue

                # 计算每个 img token 对该 text token 的注意力权重
                w = attn_weights[b, t, idx]  # [N_group]

                # 3.1 dominant
                _, dom_idx = torch.topk(w, dominant_num)
                dom_k = group_k[dom_idx]     # [dominant_num, D]
                dom_v = group_v[dom_idx]

                if contextual_num == 0:
                    comp = dom_v          # 只保留 dominant
                else:
                    # 3.2 contextual
                    mask = torch.ones(N_group, dtype=torch.bool, device=k.device)
                    mask[dom_idx] = False
                    remain_k = group_k[mask]
                    remain_v = group_v[mask]

                    step = max(1, remain_k.size(0) // contextual_num)
                    tgt_idx = torch.arange(0, remain_k.size(0), step)[:contextual_num].to(k.device)
                    tgt_k = remain_k[tgt_idx]
                    tgt_v = remain_v[tgt_idx]
                    # print(remain_k.device, tgt_idx.device)
                    merge_k = remain_k[~torch.isin(torch.arange(remain_k.size(0), device=k.device), tgt_idx)]
                    merge_v = remain_v[~torch.isin(torch.arange(remain_v.size(0), device=k.device), tgt_idx)]

                    sim = torch.mm(merge_k, tgt_k.t())                   # [N_merge, contextual_num]
                    assign = sim.argmax(dim=1)
                    one_hot = torch.zeros_like(sim).scatter_(1, assign.unsqueeze(1), 1)
                    counts = one_hot.sum(dim=0, keepdim=True).clamp(min=1)

                    agg_v = torch.mm(one_hot.t(), merge_v) / counts.t()  # [contextual_num, D]
                    ctx_v = tgt_v + agg_v

                    comp = torch.cat([dom_v, ctx_v], dim=0)              # [N, D]
                zipped_per_text.append(comp)

            # 把 Lq 个结果叠成 [Lq, N, D] 再转置回 [B, Lq, N, D]
            zipped_tensor = torch.stack(zipped_per_text, dim=0)      # [Lq, N, D]
            zipped_list.append(zipped_tensor)

        # 最终形状 [B, Lq, N, D] ；若下游只关心图级别，可做 pooling
        zipped_img = torch.stack(zipped_list, dim=0)               # [B, Lq, N, D]

        return text_out, torch.cat([text_out, zipped_img.reshape((text_out.shape[0], -1, zipped_img.shape[-1]))], 1)
