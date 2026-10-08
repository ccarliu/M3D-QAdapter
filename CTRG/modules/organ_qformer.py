import torch
import torch.nn as nn
import torch.nn.functional as F

class OrganFeatureRefiner(nn.Module):
    def __init__(
        self, 
        num_organs=10, 
        feat_dim=768, 
        num_tokens=32, 
        dict_atoms=256, 
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.dict_atoms = dict_atoms
        self.use_global_token = use_global_token

        # 原有参数
        self.all_organ_queries = nn.Parameter(torch.randn(num_organs, num_tokens, feat_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=nhead, batch_first=True, norm_first=True
        )
        self.shared_qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.all_organ_dicts = nn.Parameter(torch.randn(num_organs, dict_atoms, feat_dim))
        self.dict_scale_logit = nn.Parameter(torch.zeros(1))
        self.temperature_logit = nn.Parameter(torch.ones(1) * (-2.0))
        
        nn.init.orthogonal_(self.all_organ_dicts)
        nn.init.orthogonal_(self.all_organ_queries)
        
        self.fusion_layer = nn.Linear(feat_dim * 2, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

        # 全局 Token 相关组件
        if self.use_global_token:
            self.global_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
            
            self.global_token_refiner = nn.TransformerEncoderLayer(
                d_model=feat_dim, 
                nhead=nhead, 
                batch_first=True,
                norm_first=True
            )
            
            self.organ_global_tokens = nn.Parameter(
                torch.randn(num_organs, feat_dim)
            )
            nn.init.normal_(self.organ_global_tokens, std=0.02)

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        norm = x.norm(p=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    def process_single_organ(self, ifeat, mask, organ_idx):
        """
        处理单个器官
        Args:
            ifeat: [B, N, L] - 输入特征
            mask: [B, N] - 单个器官的 mask
            organ_idx: int - 器官索引 (0-9)
        Returns:
            final_feat: [B, num_tokens, feat_dim]
            global_token: [B, feat_dim] 或 None
            aux_loss: scalar
            metrics: dict
        """
        B, N, L = ifeat.shape
        
        # Mask 处理
        mask_sum = mask.sum(dim=1)
        safe_mask = mask.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        # Q-Former
        batch_queries = self.all_organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1)
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat,
            memory_key_padding_mask=key_padding_mask
        )
        
        # Query 多样性 Loss
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=ifeat.device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        # 字典分解
        batch_dicts = self.all_organ_dicts[organ_idx].unsqueeze(0).expand(B, -1, -1)
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat - L_rec

        # Loss 计算
        loss_reconstruction = F.mse_loss(L_rec, q_feat)
        loss_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=ifeat.device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_sparsity +
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div
        )

        # 输出特征
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        # 生成全局 Token
        global_token = None
        attn_weights = None
        if self.use_global_token:
            global_token, attn_weights = self.aggregate_global_token(
                final_feat, torch.tensor([organ_idx] * B, device=ifeat.device)
            )

        # 诊断指标
        # metrics = self.get_diagnostic_metrics(
        #     q_feat, S_res, alpha, dict_scale, temperature, attn_weights
        # )

        metrics = None

        return final_feat, global_token, aux_loss, metrics

    def process_all_organs(self, ifeat, masks):
        """
        同时处理所有器官
        Args:
            ifeat: [B, N, L] - 输入特征
            masks: [B, num_organs, N] - 所有器官的 masks
        Returns:
            all_final_feats: [B, num_organs, num_tokens, feat_dim]
            all_global_tokens: [B, num_organs, feat_dim]
            total_aux_loss: scalar
            all_metrics: dict (包含每个器官的指标)
        """
        B, num_organs, N = masks.shape
        device = ifeat.device
        
        assert num_organs == self.num_organs, \
            f"Expected {self.num_organs} organs, got {num_organs}"
        
        all_final_feats = []
        all_global_tokens = []
        all_aux_losses = []
        all_metrics = {f'organ_{i}': {} for i in range(num_organs)}
        
        # 🔧 方式 1: 循环处理每个器官（简单但效率较低）
        for organ_idx in range(num_organs):
            organ_mask = masks[:, organ_idx, :]  # [B, N]
            
            final_feat, global_token, aux_loss, metrics = self.process_single_organ(
                ifeat, organ_mask, organ_idx
            )
            
            all_final_feats.append(final_feat)
            all_global_tokens.append(global_token)
            all_aux_losses.append(aux_loss)
            all_metrics[f'organ_{organ_idx}'] = metrics
        
        # 堆叠结果
        all_final_feats = torch.stack(all_final_feats, dim=1)  # [B, num_organs, num_tokens, feat_dim]
        
        if self.use_global_token:
            all_global_tokens = torch.stack(all_global_tokens, dim=1)  # [B, num_organs, feat_dim]
        else:
            all_global_tokens = None
        
        # 平均所有器官的 aux_loss
        total_aux_loss = torch.stack(all_aux_losses).mean()
        
        # 计算跨器官的平均指标
        avg_metrics = self._aggregate_metrics(all_metrics)
        
        return all_final_feats, all_global_tokens, total_aux_loss, all_metrics, avg_metrics

    def process_all_organs_batched(self, ifeat, masks):
        """
        🚀 批量处理所有器官（高效版本）
        Args:
            ifeat: [B, N, L] - 输入特征
            masks: [B, num_organs, N] - 所有器官的 masks
        Returns:
            all_final_feats: [B, num_organs, num_tokens, feat_dim]
            all_global_tokens: [B, num_organs, feat_dim]
            total_aux_loss: scalar
            all_metrics: dict
        """
        B, num_organs, N = masks.shape
        device = ifeat.device
        L = ifeat.size(-1)
        
        # 🔧 关键：将 Batch 和 Organ 维度合并
        # ifeat: [B, N, L] -> [B*num_organs, N, L]
        ifeat_expanded = ifeat.unsqueeze(1).expand(-1, num_organs, -1, -1)
        ifeat_flat = ifeat_expanded.reshape(B * num_organs, N, L)
        
        # masks: [B, num_organs, N] -> [B*num_organs, N]
        masks_flat = masks.reshape(B * num_organs, N)
        
        # 处理 Mask
        mask_sum = masks_flat.sum(dim=1)
        safe_mask = masks_flat.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)
        
        # 🔧 准备 Queries：每个样本对应其器官的 query
        # [num_organs, num_tokens, feat_dim] -> [B*num_organs, num_tokens, feat_dim]
        organ_indices = torch.arange(num_organs, device=device).repeat(B)  # [0,1,2,...,9, 0,1,2,...,9, ...]
        batch_queries = self.all_organ_queries[organ_indices]  # [B*num_organs, num_tokens, feat_dim]
        
        # Q-Former
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat_flat,
            memory_key_padding_mask=key_padding_mask
        )  # [B*num_organs, num_tokens, feat_dim]
        
        # Query 多样性 Loss
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B * num_organs, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)
        
        # 字典分解
        batch_dicts = self.all_organ_dicts[organ_indices]  # [B*num_organs, dict_atoms, feat_dim]
        d_norm = self.safe_normalize(batch_dicts, dim=-1)
        
        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)
        
        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat - L_rec
        
        # Loss 计算
        loss_reconstruction = F.mse_loss(L_rec, q_feat)
        loss_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)
        
        total_aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_sparsity +
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div
        )
        
        # metrics = self.get_diagnostic_metrics(q_feat, L_rec, S_res, alpha, dict_scale, temperature)
        # print(metrics)

        # 输出特征
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)
        
        # 🔧 重塑回 [B, num_organs, num_tokens, feat_dim]
        all_final_feats = final_feat.view(B, num_organs, self.num_tokens, self.feat_dim)
        
        # 生成全局 Tokens
        all_global_tokens = None
        attn_weights = None
        if self.use_global_token:
            global_token, attn_weights = self.aggregate_global_token(
                final_feat, organ_indices
            )  # [B*num_organs, feat_dim]
            all_global_tokens = global_token.view(B, num_organs, self.feat_dim)
        
        return all_final_feats, all_global_tokens, total_aux_loss, None

    def aggregate_global_token(self, final_feat, organ_indices):
        """
        从 final_feat 聚合出全局 token
        Args:
            final_feat: [B, num_tokens, feat_dim]
            organ_indices: [B] - 器官索引
        Returns:
            global_token: [B, feat_dim]
            attn_weights: [B, num_tokens]
        """
        B = final_feat.size(0)
        
        # Attention-based Pooling
        attn_scores = self.global_token_aggregator(final_feat)  # [B, num_tokens, 1]
        attn_weights = F.softmax(attn_scores, dim=1)             # [B, num_tokens, 1]
        global_token = torch.sum(final_feat * attn_weights, dim=1)  # [B, feat_dim]
        
        # 结合器官先验
        organ_prior = self.organ_global_tokens[organ_indices]  # [B, feat_dim]
        global_token = global_token + 0.1 * organ_prior
        
        # L2 归一化
        global_token = self.safe_normalize(global_token, dim=-1)
        
        return global_token, attn_weights.squeeze(-1)
        

    def forward(self, ifeat, mask=None, idx=None, multi_organ_mode=False):
        """
        统一的前向传播接口
        
        Args:
            ifeat: [B, N, L] - 输入特征
            mask: [B, N] 或 [B, num_organs, N] - Mask
            idx: [B] 或 None - 器官索引（单器官模式需要）
            multi_organ_mode: bool - 是否使用多器官模式
        
        Returns:
            模式 1（单器官）:
                final_feat: [B, num_tokens, feat_dim]
                global_token: [B, feat_dim]
                aux_loss: scalar
                metrics: dict
            
            模式 2（多器官）:
                all_final_feats: [B, num_organs, num_tokens, feat_dim]
                all_global_tokens: [B, num_organs, feat_dim]
                total_aux_loss: scalar
                metrics: dict
        """
        # 🔧 自动检测模式
        if multi_organ_mode or (mask is not None and mask.dim() == 3):
            # 多器官模式
            assert mask is not None and mask.dim() == 3, \
                "Multi-organ mode requires mask with shape [B, num_organs, N]"
            return self.process_all_organs_batched(ifeat, mask)
        
        else:
            # 单器官模式
            assert idx is not None, "Single organ mode requires idx parameter"
            assert mask is not None and mask.dim() == 2, \
                "Single organ mode requires mask with shape [B, N]"
            
            B = ifeat.size(0)
            organ_indices = idx.view(B).long()
            
            # 如果 idx 中所有样本都是同一个器官，直接调用 process_single_organ
            if torch.all(organ_indices == organ_indices[0]):
                return self.process_single_organ(ifeat, mask, organ_indices[0].item())
            
            # 否则需要分别处理（batch 中有不同器官）
            else:
                return self._process_mixed_batch(ifeat, mask, organ_indices)

    def _process_mixed_batch(self, ifeat, mask, organ_indices):
        """
        处理 batch 中包含不同器官的情况
        Args:
            ifeat: [B, N, L]
            mask: [B, N]
            organ_indices: [B]
        """
        B = ifeat.size(0)
        device = ifeat.device
        
        all_final_feats = []
        all_global_tokens = []
        all_aux_losses = []
        
        # 按器官分组处理
        unique_organs = torch.unique(organ_indices)
        
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            organ_ifeat = ifeat[organ_mask_bool]
            organ_mask = mask[organ_mask_bool]
            
            final_feat, global_token, aux_loss, _ = self.process_single_organ(
                organ_ifeat, organ_mask, organ_idx.item()
            )
            
            all_final_feats.append(final_feat)
            all_global_tokens.append(global_token)
            all_aux_losses.append(aux_loss)
        
        # 重新组合成原始 batch 顺序
        final_feat_full = torch.zeros(
            B, self.num_tokens, self.feat_dim, device=device
        )
        global_token_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        
        idx_counter = 0
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            count = organ_mask_bool.sum().item()
            
            final_feat_full[organ_mask_bool] = all_final_feats[idx_counter]
            if self.use_global_token:
                global_token_full[organ_mask_bool] = all_global_tokens[idx_counter]
            
            idx_counter += 1
        
        total_aux_loss = torch.stack(all_aux_losses).mean()
        
        metrics = {}  # 简化版，不返回详细指标
        
        return final_feat_full, global_token_full, total_aux_loss, metrics

    def _aggregate_metrics(self, all_metrics):
        """聚合所有器官的指标"""
        avg_metrics = {}
        
        # 提取第一个器官的所有 key
        first_organ_metrics = all_metrics['organ_0']
        
        for key in first_organ_metrics.keys():
            values = [all_metrics[f'organ_{i}'][key] for i in range(self.num_organs)]
            avg_metrics[f'avg_{key}'] = sum(values) / len(values)
            avg_metrics[f'std_{key}'] = torch.tensor(values).std().item() if len(values) > 1 else 0.0
        
        return avg_metrics

    def get_diagnostic_metrics(self, q_feat, L_rec, S_res, alpha, dict_scale, temperature, attn_weights=None, masks=None):


        """
        📊 全面的诊断指标监控
        Args:
            q_feat: [B_total, Tokens, Dim] - 输入特征
            L_rec:  [B_total, Tokens, Dim] - 字典重构部分 (Low Rank)
            S_res:  [B_total, Tokens, Dim] - 残差部分 (Sparse/Anomaly)
            alpha:  [B_total, Tokens, Atoms] - 字典系数
            dict_scale: scalar
            temperature: scalar
            attn_weights: [B_total, Tokens, N] - (可选) Q-Former 注意力权重
            masks: [B_total, N] - (可选) 对应的器官 Mask
        """
        with torch.no_grad():
            metrics = {}
            
            # -------------------------------------------------------
            # 1. 分解能量分析 (Decomposition Energy)
            # -------------------------------------------------------
            # 目的：监控 L 和 S 的比例。
            # - 如果 S_ratio 接近 0：模型忽略了异常，只重构了共性 (Over-smoothing)。
            # - 如果 S_ratio 接近 1：字典没起作用，模型退化为恒等映射。
            # - 理想情况：S_ratio 应该在一个较小的范围 (e.g., 0.1 ~ 0.4)，保留必要的细节。
            q_energy = torch.norm(q_feat, p=2, dim=-1).mean()
            l_energy = torch.norm(L_rec, p=2, dim=-1).mean()
            s_energy = torch.norm(S_res, p=2, dim=-1).mean()
            
            metrics["energy/input"] = q_energy.item()
            metrics["energy/low_rank"] = l_energy.item()
            metrics["energy/residual"] = s_energy.item()
            metrics["energy/res_ratio"] = (s_energy / (q_energy + 1e-6)).item()

            # -------------------------------------------------------
            # 2. 字典稀疏性与利用率 (Dictionary Health)
            # -------------------------------------------------------
            # alpha_sparsity: 平均每个 Token 用了多少个原子 (Top-K 之外是否真的被抑制了)
            # dead_atoms: 当前 Batch 中有多少原子完全没被用到 (防止字典坍塌)
            
            # 统计 > 0.01 的系数个数
            active_elements = (alpha > 0.01).float().sum(dim=-1).mean()
            
            # 统计当前 Batch 内完全没被激活的原子数量 (Sum over Batch & Tokens)
            atom_usage_counts = alpha.sum(dim=(0, 1)) 
            dead_atoms_batch = (atom_usage_counts < 1e-5).float().sum()
            
            # 熵：衡量选择原子的确定性。越低越好，说明模型很确定用哪几个原子。
            alpha_entropy = -(alpha * torch.log(alpha + 1e-9)).sum(dim=-1).mean()

            metrics["dict/active_atoms_per_token"] = active_elements.item()
            metrics["dict/dead_atoms_in_batch"] = dead_atoms_batch.item()
            metrics["dict/entropy"] = alpha_entropy.item()
            metrics["dict/scale_factor"] = dict_scale.item()
            metrics["dict/temperature"] = temperature.item()

            # -------------------------------------------------------
            # 3. Attention 质量分析 (Q-Former Health)
            # -------------------------------------------------------
            # 只有提供了 attn_weights 和 masks 才能计算
            if attn_weights is not None and masks is not None:
                # attn_weights: [B, Tokens, N]
                # masks: [B, N]
                
                # 扩展 mask: [B, 1, N]
                mask_expanded = masks.unsqueeze(1)
                
                # 计算 Mask 内部的平均注意力强度 vs 外部的平均注意力强度
                # 理想情况：Inside 很高，Outside 接近 0
                attn_inside = (attn_weights * mask_expanded).sum() / (mask_expanded.sum() + 1e-6)
                attn_outside = (attn_weights * (1 - mask_expanded)).sum() / ((1 - mask_expanded).sum() + 1e-6)
                
                # 覆盖率：Mask 内部有多少像素被至少一个 Query 关注到了 (Max > 0.1)
                max_attn_per_pixel, _ = attn_weights.max(dim=1) # [B, N]
                covered_pixels = ((max_attn_per_pixel > 0.05).float() * masks).sum()
                total_mask_pixels = masks.sum() + 1e-6
                coverage_ratio = covered_pixels / total_mask_pixels

                metrics["attn/intensity_inside"] = attn_inside.item()
                metrics["attn/intensity_outside"] = attn_outside.item()
                metrics["attn/mask_coverage"] = coverage_ratio.item()

        return metrics


import torch
import torch.nn as nn
import torch.nn.functional as F

class OrganFeatureRefiner_v2_LRMR(nn.Module):
    def __init__(
        self, 
        num_organs=10, 
        feat_dim=768, 
        num_tokens=32, 
        dict_atoms=256, 
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.dict_atoms = dict_atoms
        self.use_global_token = use_global_token

        # 原有参数
        self.all_organ_queries = nn.Parameter(torch.randn(num_organs, num_tokens, feat_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=nhead, batch_first=True, norm_first=True
        )
        self.shared_qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.all_organ_dicts = nn.Parameter(torch.randn(num_organs, dict_atoms, feat_dim))
        self.dict_scale_logit = nn.Parameter(torch.zeros(1))
        self.temperature_logit = nn.Parameter(torch.ones(1) * (-2.0))
        
        nn.init.orthogonal_(self.all_organ_dicts)
        nn.init.orthogonal_(self.all_organ_queries)
        
        self.fusion_layer = nn.Linear(feat_dim * 2, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

        # 全局 Token 相关组件
        if self.use_global_token:
            self.global_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
            
            # 注意：虽然定义了但未被使用，为了保持结构一致性保留在此
            self.global_token_refiner = nn.TransformerEncoderLayer(
                d_model=feat_dim, 
                nhead=nhead, 
                batch_first=True,
                norm_first=True
            )
            
            self.organ_global_tokens = nn.Parameter(
                torch.randn(num_organs, feat_dim)
            )
            nn.init.normal_(self.organ_global_tokens, std=0.02)

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        norm = x.norm(p=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    def process_single_organ(self, ifeat, mask, organ_idx):
        """
        处理单个器官 (包含 LRMR 修改)
        """
        B, N, L = ifeat.shape
        
        # Mask 处理
        mask_sum = mask.sum(dim=1)
        safe_mask = mask.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        # Q-Former
        batch_queries = self.all_organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1)
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat,
            memory_key_padding_mask=key_padding_mask
        )
        
        # Query 多样性 Loss
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=ifeat.device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        # 字典分解
        batch_dicts = self.all_organ_dicts[organ_idx].unsqueeze(0).expand(B, -1, -1)
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat - L_rec

        # --- Loss 计算 (修改部分) ---
        
        # [修改 1] 使用 L1 Loss 替代 MSE
        # L1 Loss 对异常值不敏感，允许 S_res 中存在大幅度的稀疏噪声（即病灶）
        loss_reconstruction = F.l1_loss(L_rec, q_feat)
        
        # [修改 2] 增加残差稀疏性 Loss
        # 强制 S_res 大部分为 0，只保留真正的异常
        loss_residual_sparsity = torch.mean(torch.abs(S_res))

        # 字典系数的稀疏性 (保持不变，这是为了保证 L_rec 是低秩的)
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=ifeat.device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_coeff_sparsity +  # 字典系数稀疏
            0.1 * loss_residual_sparsity + # [新增] 残差稀疏
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div
        )

        # 输出特征
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        # 生成全局 Token
        global_token = None
        if self.use_global_token:
            global_token, _ = self.aggregate_global_token(
                final_feat, torch.tensor([organ_idx] * B, device=ifeat.device)
            )

        return final_feat, global_token, aux_loss, None


    def process_all_organs_batched(self, ifeat, masks):
        """
        批量处理所有器官 (包含 LRMR 修改)
        """
        B, num_organs, N = masks.shape
        device = ifeat.device
        L = ifeat.size(-1)
        
        ifeat_expanded = ifeat.unsqueeze(1).expand(-1, num_organs, -1, -1)
        ifeat_flat = ifeat_expanded.reshape(B * num_organs, N, L)
        masks_flat = masks.reshape(B * num_organs, N)
        
        mask_sum = masks_flat.sum(dim=1)
        safe_mask = masks_flat.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)
        
        organ_indices = torch.arange(num_organs, device=device).repeat(B)
        batch_queries = self.all_organ_queries[organ_indices]
        
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat_flat,
            memory_key_padding_mask=key_padding_mask
        )
        
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B * num_organs, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)
        
        batch_dicts = self.all_organ_dicts[organ_indices]
        d_norm = self.safe_normalize(batch_dicts, dim=-1)
        
        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)
        
        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat - L_rec
        
        # --- Loss 计算 (修改部分) ---
        
        # [修改 1] L1 Loss
        loss_reconstruction = F.l1_loss(L_rec, q_feat)
        
        # [修改 2] 残差稀疏 Loss
        loss_residual_sparsity = torch.mean(torch.abs(S_res))

        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)
        
        total_aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_coeff_sparsity +
            0.1 * loss_residual_sparsity + # [新增]
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div
        )
        
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)
        
        all_final_feats = final_feat.view(B, num_organs, self.num_tokens, self.feat_dim)
        
        all_global_tokens = None
        if self.use_global_token:
            global_token, _ = self.aggregate_global_token(
                final_feat, organ_indices
            )
            all_global_tokens = global_token.view(B, num_organs, self.feat_dim)
        
        return all_final_feats, all_global_tokens, total_aux_loss, None

    def aggregate_global_token(self, final_feat, organ_indices):
        B = final_feat.size(0)
        attn_scores = self.global_token_aggregator(final_feat)
        attn_weights = F.softmax(attn_scores, dim=1)
        global_token = torch.sum(final_feat * attn_weights, dim=1)
        organ_prior = self.organ_global_tokens[organ_indices]
        global_token = global_token + 0.1 * organ_prior
        global_token = self.safe_normalize(global_token, dim=-1)
        return global_token, attn_weights.squeeze(-1)

    def forward(self, ifeat, mask=None, idx=None, multi_organ_mode=False):
        if multi_organ_mode or (mask is not None and mask.dim() == 3):
            assert mask is not None and mask.dim() == 3
            return self.process_all_organs_batched(ifeat, mask)
        else:
            assert idx is not None
            assert mask is not None and mask.dim() == 2
            B = ifeat.size(0)
            organ_indices = idx.view(B).long()
            if torch.all(organ_indices == organ_indices[0]):
                return self.process_single_organ(ifeat, mask, organ_indices[0].item())
            else:
                return self._process_mixed_batch(ifeat, mask, organ_indices)

    def _process_mixed_batch(self, ifeat, mask, organ_indices):
        B = ifeat.size(0)
        device = ifeat.device
        all_final_feats = []
        all_global_tokens = []
        all_aux_losses = []
        unique_organs = torch.unique(organ_indices)
        
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            organ_ifeat = ifeat[organ_mask_bool]
            organ_mask = mask[organ_mask_bool]
            final_feat, global_token, aux_loss, _ = self.process_single_organ(
                organ_ifeat, organ_mask, organ_idx.item()
            )
            all_final_feats.append(final_feat)
            all_global_tokens.append(global_token)
            all_aux_losses.append(aux_loss)
        
        final_feat_full = torch.zeros(B, self.num_tokens, self.feat_dim, device=device)
        global_token_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        
        idx_counter = 0
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            final_feat_full[organ_mask_bool] = all_final_feats[idx_counter]
            if self.use_global_token:
                global_token_full[organ_mask_bool] = all_global_tokens[idx_counter]
            idx_counter += 1
        
        total_aux_loss = torch.stack(all_aux_losses).mean()
        return final_feat_full, global_token_full, total_aux_loss, {}

class OrganFeatureRefiner_v2_LRMR_dual(nn.Module):
    def __init__(
        self, 
        num_organs=10, 
        feat_dim=768, 
        num_tokens=32, 
        dict_atoms=256, 
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.dict_atoms = dict_atoms
        self.use_global_token = use_global_token

        # --- 核心组件 ---
        self.all_organ_queries = nn.Parameter(torch.randn(num_organs, num_tokens, feat_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=nhead, batch_first=True, norm_first=True
        )
        self.shared_qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 字典学习组件
        self.all_organ_dicts = nn.Parameter(torch.randn(num_organs, dict_atoms, feat_dim))
        self.dict_scale_logit = nn.Parameter(torch.zeros(1))
        self.temperature_logit = nn.Parameter(torch.ones(1) * (-2.0))
        
        # 初始化
        nn.init.orthogonal_(self.all_organ_dicts)
        nn.init.orthogonal_(self.all_organ_queries)
        
        # 融合层
        self.fusion_layer = nn.Linear(feat_dim * 2, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

        # --- Token 聚合器 (Dual Stream) ---
        if self.use_global_token:
            # 1. 全局聚合器：用于 final_feat (结构+异常)
            self.global_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
            
            # 2. [新增] 残差聚合器：专门用于 S_res (仅异常)
            # 使用独立的权重，防止语义混淆
            self.residual_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
            
            # 器官先验 Embedding (只加给 global token)
            self.organ_global_tokens = nn.Parameter(
                torch.randn(num_organs, feat_dim)
            )
            nn.init.normal_(self.organ_global_tokens, std=0.02)

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        norm = x.norm(p=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    def aggregate_dual_tokens(self, final_feat, S_res, organ_indices):
        """
        同时聚合 Global Token 和 Residual Token
        """
        # --- 1. Global Token (来自 final_feat) ---
        # 目的：对齐完整报告 (Organ + Abnormality)
        attn_scores_g = self.global_token_aggregator(final_feat) # [B, N, 1]
        attn_weights_g = F.softmax(attn_scores_g, dim=1)
        token_global = torch.sum(final_feat * attn_weights_g, dim=1) # [B, D]
        
        # 加上器官先验 (因为 final_feat 包含器官身份)
        organ_prior = self.organ_global_tokens[organ_indices]
        token_global = token_global + 0.1 * organ_prior
        token_global = self.safe_normalize(token_global, dim=-1)

        # --- 2. Residual Token (来自 S_res) ---
        # 目的：对齐异常文本 (Abnormality + Location)
        attn_scores_r = self.residual_token_aggregator(S_res) # [B, N, 1]
        attn_weights_r = F.softmax(attn_scores_r, dim=1)
        token_residual = torch.sum(S_res * attn_weights_r, dim=1) # [B, D]
        
        # 注意：残差 Token 不加 organ_prior，保持纯净的异常语义
        token_residual = self.safe_normalize(token_residual, dim=-1)

        return token_global, token_residual

    def process_single_organ(self, ifeat, mask, organ_idx):
        B, N, L = ifeat.shape
        
        # Mask 处理
        mask_sum = mask.sum(dim=1)
        safe_mask = mask.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        # Q-Former
        batch_queries = self.all_organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1)
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat,
            memory_key_padding_mask=key_padding_mask
        )
        
        # Query 多样性 Loss
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=ifeat.device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        # 字典分解
        batch_dicts = self.all_organ_dicts[organ_idx].unsqueeze(0).expand(B, -1, -1)
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat - L_rec

        # --- Loss 计算 (LRMR 策略) ---
        # 1. L1 重构 Loss (允许稀疏异常)
        loss_reconstruction = F.l1_loss(L_rec, q_feat)
        
        # 2. 残差稀疏 Loss (迫使 S_res 仅保留异常)
        loss_residual_sparsity = torch.mean(torch.abs(S_res))

        # 3. 字典系数稀疏性
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        # 4. 字典正交性
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=ifeat.device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_coeff_sparsity +
            0.1 * loss_residual_sparsity + 
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div
        )

        # metrics = self.get_diagnostic_metrics(q_feat, L_rec, S_res, alpha, dict_scale, temperature)
        # print(metrics)

        # 输出特征融合
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        # --- 生成双流 Token ---
        token_global = None
        token_residual = None
        if self.use_global_token:
            indices = torch.tensor([organ_idx] * B, device=ifeat.device)
            token_global, token_residual = self.aggregate_dual_tokens(
                final_feat, S_res, indices
            )

        return final_feat, token_global, token_residual, aux_loss, None

    def calculate_label_guided_loss(self, S_res, organ_labels):
        """
        S_res: [B * 10, 32, 768] (已经 flatten 过的 batch)
        organ_labels: [B, 10]    (原始标签)
        """
        # 1. 维度对齐：把标签展平以匹配 S_res 的第一维
        # [B, 10] -> [B * 10]
        flat_labels = organ_labels.view(-1).float()
        
        # 2. 制作掩码
        # is_normal: [B*10, 1, 1] -> 对应正常样本 (Label=0)
        is_normal = (flat_labels == 0).view(-1, 1, 1)
        
        # is_abnormal: [B*10, 1, 1] -> 对应异常样本 (Label=1)
        is_abnormal = (flat_labels == 1).view(-1, 1, 1)
        
        # 3. 计算残差能量 (L1 Norm)
        res_energy = torch.abs(S_res)
        
        # =========================================================
        # 核心逻辑：双重标准
        # =========================================================
        
        # 【正常样本】：零容忍
        # 只要是正常器官，S_res 必须是 0。权重给极大 (20.0)。
        # 这会强迫 L_rec 完美拟合 q_feat，实现“只用字典重建”。
        loss_normal_zero = (res_energy * is_normal).mean() / (is_normal.sum() + 1e-6) * 20.0
        
        # 【异常样本】：宽容
        # 允许存在残差，只给一个极小的稀疏惩罚 (0.01) 防止数值爆炸即可。
        # 这样模型就会把所有“字典解释不了的差异”都堆到这里。
        loss_abnormal_sparse = (res_energy * is_abnormal).sum() / (is_abnormal.sum() + 1e-6) * 0.01
        
        return loss_normal_zero + loss_abnormal_sparse

    def process_all_organs_batched(self, ifeat, masks, organ_labels = None):
        B, num_organs, N = masks.shape
        device = ifeat.device
        L = ifeat.size(-1)
        # print(ifeat.shape, masks.shape)
        ifeat_expanded = ifeat.unsqueeze(1).expand(-1, num_organs, -1, -1) # B 10 4096 768
        ifeat_flat = ifeat_expanded.reshape(B * num_organs, N, L) # 20 4096 768
        masks_flat = masks.reshape(B * num_organs, N) # 20 4096
        
        mask_sum = masks_flat.sum(dim=1)
        safe_mask = masks_flat.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)
        
        organ_indices = torch.arange(num_organs, device=device).repeat(B)
        batch_queries = self.all_organ_queries[organ_indices]
        
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat_flat,
            memory_key_padding_mask=key_padding_mask
        )
        
        # Query Diversity
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B * num_organs, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)
        
        # Dictionary Learning
        batch_dicts = self.all_organ_dicts[organ_indices]
        d_norm = self.safe_normalize(batch_dicts, dim=-1)
        
        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)
        
        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat - L_rec
        
        # metrics = self.get_diagnostic_metrics(q_feat, L_rec, S_res, alpha, dict_scale, temperature)
        # print(metrics)
        
        # --- Loss Calculation ---
        loss_reconstruction = F.l1_loss(L_rec, q_feat)
        # loss_residual_sparsity = torch.mean(torch.abs(S_res))
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        ###########################
        if organ_labels is not None:
            loss_s = self.calculate_label_guided_loss(S_res, organ_labels)
        else:
            loss_s = torch.zeros_like(loss_dict_ortho)
        ###########################
        
        total_aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_coeff_sparsity +
            #0.1 * loss_residual_sparsity +
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div + loss_s
        )
        

        # metrics = self.get_diagnostic_metrics(q_feat, L_rec, S_res, alpha, dict_scale, temperature)
        # print(metrics)

        # Fusion
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)
        
        # Reshape back to [B, Num_Organs, Tokens, Dim]
        all_final_feats = final_feat.view(B, num_organs, self.num_tokens, self.feat_dim)
        
        # --- Dual Token Aggregation ---
        all_global_tokens = None
        all_residual_tokens = None
        
        if self.use_global_token:
            token_global, token_residual = self.aggregate_dual_tokens(
                final_feat, S_res, organ_indices
            )
            all_global_tokens = token_global.view(B, num_organs, self.feat_dim)
            all_residual_tokens = token_residual.view(B, num_organs, self.feat_dim)
        
        return all_final_feats, all_global_tokens, all_residual_tokens, total_aux_loss, None

    def forward(self, ifeat, mask=None, idx=None, abnorm_label = None, multi_organ_mode=False):
        """
        Returns:
            final_feat: [B, N, D] or [B, Num_Organs, N, D]
            token_global: [B, D] or [B, Num_Organs, D] (用于对齐完整报告)
            token_residual: [B, D] or [B, Num_Organs, D] (用于对齐异常文本)
            aux_loss: scalar
            extra_info: dict
        """
        if multi_organ_mode or (mask is not None and mask.dim() == 3):
            assert mask is not None and mask.dim() == 3
            return self.process_all_organs_batched(ifeat, mask, abnorm_label)
        else:
            assert idx is not None
            assert mask is not None and mask.dim() == 2
            B = ifeat.size(0)
            organ_indices = idx.view(B).long()
            if torch.all(organ_indices == organ_indices[0]):
                return self.process_single_organ(ifeat, mask, organ_indices[0].item())
            else:
                return self._process_mixed_batch(ifeat, mask, organ_indices)

    def _process_mixed_batch(self, ifeat, mask, organ_indices):
        B = ifeat.size(0)
        device = ifeat.device
        all_final_feats = []
        all_global_tokens = []
        all_residual_tokens = []
        all_aux_losses = []
        unique_organs = torch.unique(organ_indices)
        
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            organ_ifeat = ifeat[organ_mask_bool]
            organ_mask = mask[organ_mask_bool]
            
            final_feat, t_glob, t_res, aux_loss, _ = self.process_single_organ(
                organ_ifeat, organ_mask, organ_idx.item()
            )
            
            all_final_feats.append(final_feat)
            all_global_tokens.append(t_glob)
            all_residual_tokens.append(t_res)
            all_aux_losses.append(aux_loss)
        
        # Reassemble batch
        final_feat_full = torch.zeros(B, self.num_tokens, self.feat_dim, device=device)
        token_global_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        token_residual_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        
        idx_counter = 0
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            final_feat_full[organ_mask_bool] = all_final_feats[idx_counter]
            if self.use_global_token:
                token_global_full[organ_mask_bool] = all_global_tokens[idx_counter]
                token_residual_full[organ_mask_bool] = all_residual_tokens[idx_counter]
            idx_counter += 1
        
        total_aux_loss = torch.stack(all_aux_losses).mean()
        return final_feat_full, token_global_full, token_residual_full, total_aux_loss, {}

    def _aggregate_metrics(self, all_metrics):
        """聚合所有器官的指标"""
        avg_metrics = {}
        
        # 提取第一个器官的所有 key
        first_organ_metrics = all_metrics['organ_0']
        
        for key in first_organ_metrics.keys():
            values = [all_metrics[f'organ_{i}'][key] for i in range(self.num_organs)]
            avg_metrics[f'avg_{key}'] = sum(values) / len(values)
            avg_metrics[f'std_{key}'] = torch.tensor(values).std().item() if len(values) > 1 else 0.0
        
        return avg_metrics

    def get_diagnostic_metrics(self, q_feat, L_rec, S_res, alpha, dict_scale, temperature, attn_weights=None, masks=None):


        """
        📊 全面的诊断指标监控
        Args:
            q_feat: [B_total, Tokens, Dim] - 输入特征
            L_rec:  [B_total, Tokens, Dim] - 字典重构部分 (Low Rank)
            S_res:  [B_total, Tokens, Dim] - 残差部分 (Sparse/Anomaly)
            alpha:  [B_total, Tokens, Atoms] - 字典系数
            dict_scale: scalar
            temperature: scalar
            attn_weights: [B_total, Tokens, N] - (可选) Q-Former 注意力权重
            masks: [B_total, N] - (可选) 对应的器官 Mask
        """
        with torch.no_grad():
            metrics = {}
            
            # -------------------------------------------------------
            # 1. 分解能量分析 (Decomposition Energy)
            # -------------------------------------------------------
            # 目的：监控 L 和 S 的比例。
            # - 如果 S_ratio 接近 0：模型忽略了异常，只重构了共性 (Over-smoothing)。
            # - 如果 S_ratio 接近 1：字典没起作用，模型退化为恒等映射。
            # - 理想情况：S_ratio 应该在一个较小的范围 (e.g., 0.1 ~ 0.4)，保留必要的细节。
            q_energy = torch.norm(q_feat, p=2, dim=-1).mean()
            l_energy = torch.norm(L_rec, p=2, dim=-1).mean()
            s_energy = torch.norm(S_res, p=2, dim=-1).mean()
            
            metrics["energy/input"] = q_energy.item()
            metrics["energy/low_rank"] = l_energy.item()
            metrics["energy/residual"] = s_energy.item()
            metrics["energy/res_ratio"] = (s_energy / (q_energy + 1e-6)).item()

            # -------------------------------------------------------
            # 2. 字典稀疏性与利用率 (Dictionary Health)
            # -------------------------------------------------------
            # alpha_sparsity: 平均每个 Token 用了多少个原子 (Top-K 之外是否真的被抑制了)
            # dead_atoms: 当前 Batch 中有多少原子完全没被用到 (防止字典坍塌)
            
            # 统计 > 0.01 的系数个数
            active_elements = (alpha > 0.01).float().sum(dim=-1).mean()
            
            # 统计当前 Batch 内完全没被激活的原子数量 (Sum over Batch & Tokens)
            atom_usage_counts = alpha.sum(dim=(0, 1)) 
            dead_atoms_batch = (atom_usage_counts < 1e-5).float().sum()
            
            # 熵：衡量选择原子的确定性。越低越好，说明模型很确定用哪几个原子。
            alpha_entropy = -(alpha * torch.log(alpha + 1e-9)).sum(dim=-1).mean()

            metrics["dict/active_atoms_per_token"] = active_elements.item()
            metrics["dict/dead_atoms_in_batch"] = dead_atoms_batch.item()
            metrics["dict/entropy"] = alpha_entropy.item()
            metrics["dict/scale_factor"] = dict_scale.item()
            metrics["dict/temperature"] = temperature.item()

            # -------------------------------------------------------
            # 3. Attention 质量分析 (Q-Former Health)
            # -------------------------------------------------------
            # 只有提供了 attn_weights 和 masks 才能计算
            if attn_weights is not None and masks is not None:
                # attn_weights: [B, Tokens, N]
                # masks: [B, N]
                
                # 扩展 mask: [B, 1, N]
                mask_expanded = masks.unsqueeze(1)
                
                # 计算 Mask 内部的平均注意力强度 vs 外部的平均注意力强度
                # 理想情况：Inside 很高，Outside 接近 0
                attn_inside = (attn_weights * mask_expanded).sum() / (mask_expanded.sum() + 1e-6)
                attn_outside = (attn_weights * (1 - mask_expanded)).sum() / ((1 - mask_expanded).sum() + 1e-6)
                
                # 覆盖率：Mask 内部有多少像素被至少一个 Query 关注到了 (Max > 0.1)
                max_attn_per_pixel, _ = attn_weights.max(dim=1) # [B, N]
                covered_pixels = ((max_attn_per_pixel > 0.05).float() * masks).sum()
                total_mask_pixels = masks.sum() + 1e-6
                coverage_ratio = covered_pixels / total_mask_pixels

                metrics["attn/intensity_inside"] = attn_inside.item()
                metrics["attn/intensity_outside"] = attn_outside.item()
                metrics["attn/mask_coverage"] = coverage_ratio.item()

        return metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

class OrganFeatureRefiner_v2_LRMR_dual_Independent(nn.Module):
    def __init__(
        self, 
        num_organs=10, 
        feat_dim=768, 
        num_tokens=32, 
        dict_atoms=256, 
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.dict_atoms = dict_atoms
        self.use_global_token = use_global_token

        # --- 核心组件 ---
        # 依然保留 Learnable Queries，作为每个 Q-Former 的输入
        self.all_organ_queries = nn.Parameter(torch.randn(num_organs, num_tokens, feat_dim))
        
        # 【改动点】：不再共享 Q-Former，而是为每个器官创建一个独立的 Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=nhead, batch_first=True, norm_first=True
        )
        # 使用 ModuleList 存储 num_organs 个独立的 TransformerDecoder
        # 注意：TransformerDecoder 初始化时会深拷贝 decoder_layer，所以权重是独立的
        self.organ_qformers = nn.ModuleList([
            nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            for _ in range(num_organs)
        ])
        
        # 字典学习组件 (保持不变)
        self.all_organ_dicts = nn.Parameter(torch.randn(num_organs, dict_atoms, feat_dim))
        self.dict_scale_logit = nn.Parameter(torch.zeros(1))
        self.temperature_logit = nn.Parameter(torch.ones(1) * (-2.0))
        
        # 初始化
        nn.init.orthogonal_(self.all_organ_dicts)
        nn.init.orthogonal_(self.all_organ_queries)
        
        # 融合层
        self.fusion_layer = nn.Linear(feat_dim * 2, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

        # --- Token 聚合器 (Dual Stream) ---
        if self.use_global_token:
            # 1. 全局聚合器
            self.global_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
            
            # 2. 残差聚合器
            self.residual_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
            
            # 器官先验 Embedding
            self.organ_global_tokens = nn.Parameter(
                torch.randn(num_organs, feat_dim)
            )
            nn.init.normal_(self.organ_global_tokens, std=0.02)

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        norm = x.norm(p=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    def aggregate_dual_tokens(self, final_feat, S_res, organ_indices):
        """
        同时聚合 Global Token 和 Residual Token
        """
        # --- 1. Global Token (来自 final_feat) ---
        attn_scores_g = self.global_token_aggregator(final_feat) # [B, N, 1]
        attn_weights_g = F.softmax(attn_scores_g, dim=1)
        token_global = torch.sum(final_feat * attn_weights_g, dim=1) # [B, D]
        
        # 加上器官先验
        organ_prior = self.organ_global_tokens[organ_indices]
        token_global = token_global + 0.1 * organ_prior
        token_global = self.safe_normalize(token_global, dim=-1)

        # --- 2. Residual Token (来自 S_res) ---
        attn_scores_r = self.residual_token_aggregator(S_res) # [B, N, 1]
        attn_weights_r = F.softmax(attn_scores_r, dim=1)
        token_residual = torch.sum(S_res * attn_weights_r, dim=1) # [B, D]
        
        token_residual = self.safe_normalize(token_residual, dim=-1)

        return token_global, token_residual

    def process_single_organ(self, ifeat, mask, organ_idx):
        B, N, L = ifeat.shape
        
        # Mask 处理
        mask_sum = mask.sum(dim=1)
        safe_mask = mask.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        # Q-Former 【改动点】：调用特定的 organ_qformers[organ_idx]
        batch_queries = self.all_organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1)
        
        # 使用独立的 Q-Former
        q_feat = self.organ_qformers[organ_idx](
            tgt=batch_queries,
            memory=ifeat,
            memory_key_padding_mask=key_padding_mask
        )
        
        # --- 以下逻辑保持不变 (LRMR & Loss) ---
        
        # Query 多样性 Loss
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=ifeat.device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        # 字典分解
        batch_dicts = self.all_organ_dicts[organ_idx].unsqueeze(0).expand(B, -1, -1)
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat - L_rec

        # Loss 计算
        loss_reconstruction = F.l1_loss(L_rec, q_feat)
        loss_residual_sparsity = torch.mean(torch.abs(S_res))
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=ifeat.device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_coeff_sparsity +
            0.1 * loss_residual_sparsity + 
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div
        )

        # 输出特征融合
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        token_global = None
        token_residual = None
        if self.use_global_token:
            indices = torch.tensor([organ_idx] * B, device=ifeat.device)
            token_global, token_residual = self.aggregate_dual_tokens(
                final_feat, S_res, indices
            )

        return final_feat, token_global, token_residual, aux_loss, None

    def calculate_label_guided_loss(self, S_res, organ_labels):
        flat_labels = organ_labels.view(-1).float()
        is_normal = (flat_labels == 0).view(-1, 1, 1)
        is_abnormal = (flat_labels == 1).view(-1, 1, 1)
        
        res_energy = torch.abs(S_res)
        
        loss_normal_zero = (res_energy * is_normal).mean() / (is_normal.sum() + 1e-6) * 20.0
        loss_abnormal_sparse = (res_energy * is_abnormal).sum() / (is_abnormal.sum() + 1e-6) * 0.01
        
        return loss_normal_zero + loss_abnormal_sparse

    def process_all_organs_batched(self, ifeat, masks, organ_labels = None):
        """
        处理所有器官。
        由于 Q-Former 权重独立，必须循环调用 Q-Former，
        但后续的字典计算和 Loss 可以 Batch 化以提高效率。
        """
        B, num_organs, N = masks.shape
        device = ifeat.device
        
        # 准备 Mask
        # masks: [B, Num_Organs, N] -> Flatten -> [B*Num_Organs, N]
        masks_flat = masks.reshape(B * num_organs, N)
        mask_sum = masks_flat.sum(dim=1)
        safe_mask = masks_flat.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask_flat = (safe_mask == 0) # [B*Num_Organs, N]
        
        # 准备 ifeat
        # ifeat: [B, N, D] -> Expand -> [B, Num_Organs, N, D] -> Flatten -> [B*Num_Organs, N, D]
        ifeat_expanded = ifeat.unsqueeze(1).expand(-1, num_organs, -1, -1)
        ifeat_flat = ifeat_expanded.reshape(B * num_organs, N, -1)

        # --- 步骤 1: 独立 Q-Former 特征提取 (必须循环) ---
        q_feats_list = []
        
        for i in range(self.num_organs):
            # 提取当前器官的 Batch 数据
            # ifeat: [B, N, D] (所有器官共享同一张图的特征，直接用 ifeat 即可)
            # mask: 需要切片
            
            # 注意：这里为了利用 Transformer 的并行性，我们还是得把 mask 传进去
            # 为了简单起见，我们这里直接取 ifeat，mask 取对应器官的 mask
            
            curr_mask = masks[:, i, :] # [B, N]
            curr_mask_sum = curr_mask.sum(dim=1)
            curr_safe_mask = curr_mask.clone()
            curr_safe_mask[curr_mask_sum == 0, 0] = 1.0
            curr_key_padding_mask = (curr_safe_mask == 0)
            
            # Query
            curr_query = self.all_organ_queries[i].unsqueeze(0).expand(B, -1, -1) # [B, Tokens, D]
            
            # Forward 独立的 Q-Former
            q_out = self.organ_qformers[i](
                tgt=curr_query,
                memory=ifeat,
                memory_key_padding_mask=curr_key_padding_mask
            ) # [B, Tokens, D]
            
            q_feats_list.append(q_out)
            
        # --- 步骤 2: 堆叠并 Flatten，进入统一的 LRMR 流程 ---
        # Stack: [B, Num_Organs, Tokens, D]
        q_feat_stacked = torch.stack(q_feats_list, dim=1)
        
        # Flatten: [B * Num_Organs, Tokens, D]
        # 这样我们可以复用之前的高效矩阵运算代码
        q_feat = q_feat_stacked.view(B * num_organs, self.num_tokens, self.feat_dim)
        
        # --- 以下逻辑与原版完全一致 (LRMR 是 Batch 化的) ---
        
        organ_indices = torch.arange(num_organs, device=device).repeat(B) # [B*Num_Organs]
        
        # Query Diversity
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B * num_organs, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)
        
        # Dictionary Learning
        batch_dicts = self.all_organ_dicts[organ_indices] # [B*Num_Organs, Atoms, D]
        d_norm = self.safe_normalize(batch_dicts, dim=-1)
        
        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)
        
        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat - L_rec
        
        # --- Loss Calculation ---
        loss_reconstruction = F.l1_loss(L_rec, q_feat)
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        if organ_labels is not None:
            loss_s = self.calculate_label_guided_loss(S_res, organ_labels)
        else:
            loss_s = torch.zeros_like(loss_dict_ortho)
        
        total_aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_coeff_sparsity +
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div + loss_s
        )

        # Fusion
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)
        
        # Reshape back to [B, Num_Organs, Tokens, Dim]
        all_final_feats = final_feat.view(B, num_organs, self.num_tokens, self.feat_dim)
        
        # --- Dual Token Aggregation ---
        all_global_tokens = None
        all_residual_tokens = None
        
        if self.use_global_token:
            token_global, token_residual = self.aggregate_dual_tokens(
                final_feat, S_res, organ_indices
            )
            all_global_tokens = token_global.view(B, num_organs, self.feat_dim)
            all_residual_tokens = token_residual.view(B, num_organs, self.feat_dim)
        
        return all_final_feats, all_global_tokens, all_residual_tokens, total_aux_loss, None

    def forward(self, ifeat, mask=None, idx=None, abnorm_label = None, multi_organ_mode=False):
        if multi_organ_mode or (mask is not None and mask.dim() == 3):
            assert mask is not None and mask.dim() == 3
            return self.process_all_organs_batched(ifeat, mask, abnorm_label)
        else:
            assert idx is not None
            assert mask is not None and mask.dim() == 2
            B = ifeat.size(0)
            organ_indices = idx.view(B).long()
            if torch.all(organ_indices == organ_indices[0]):
                return self.process_single_organ(ifeat, mask, organ_indices[0].item())
            else:
                return self._process_mixed_batch(ifeat, mask, organ_indices)

    def _process_mixed_batch(self, ifeat, mask, organ_indices):
        B = ifeat.size(0)
        device = ifeat.device
        all_final_feats = []
        all_global_tokens = []
        all_residual_tokens = []
        all_aux_losses = []
        unique_organs = torch.unique(organ_indices)
        
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            organ_ifeat = ifeat[organ_mask_bool]
            organ_mask = mask[organ_mask_bool]
            
            final_feat, t_glob, t_res, aux_loss, _ = self.process_single_organ(
                organ_ifeat, organ_mask, organ_idx.item()
            )
            
            all_final_feats.append(final_feat)
            all_global_tokens.append(t_glob)
            all_residual_tokens.append(t_res)
            all_aux_losses.append(aux_loss)
        
        # Reassemble batch
        final_feat_full = torch.zeros(B, self.num_tokens, self.feat_dim, device=device)
        token_global_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        token_residual_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        
        idx_counter = 0
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            final_feat_full[organ_mask_bool] = all_final_feats[idx_counter]
            if self.use_global_token:
                token_global_full[organ_mask_bool] = all_global_tokens[idx_counter]
                token_residual_full[organ_mask_bool] = all_residual_tokens[idx_counter]
            idx_counter += 1
        
        total_aux_loss = torch.stack(all_aux_losses).mean()
        return final_feat_full, token_global_full, token_residual_full, total_aux_loss, {}

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionAggregator(nn.Module):
    """
    [新增组件] 基于 Query 的聚合器
    使用 Cross-Attention，以 Organ Prior 为 Query，Image Features 为 Key/Value。
    这能让 Global Token 更加符合解剖学的标准定义。
    """
    def __init__(self, feat_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim)
        )

    def forward(self, query, kv):
        """
        Args:
            query: [B, 1, D] (Organ Prior)
            kv:    [B, N, D] (Final Image Features)
        Returns:
            out:   [B, D]
        """
        # Cross Attention: Query(Prior) 找 KV(Image) 中的相关信息
        attn_out, _ = self.mha(query=query, key=kv, value=kv)
        
        # Residual Connection (保留先验信息) + Norm
        x = self.norm1(query + attn_out)
        
        # FFN + Norm
        x = self.norm2(x + self.ffn(x))
        
        return x.squeeze(1) # [B, D]


class OrganFeatureRefiner_v2_LRMR_dual_v2(nn.Module):
    def __init__(
        self, 
        num_organs=10, 
        feat_dim=768, 
        num_tokens=32, 
        dict_atoms=256, 
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.dict_atoms = dict_atoms
        self.use_global_token = use_global_token

        # --- 1. 核心组件: Q-Former & Dictionary ---
        self.all_organ_queries = nn.Parameter(torch.randn(num_organs, num_tokens, feat_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=nhead, batch_first=True, norm_first=True
        )
        self.shared_qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 字典学习组件
        self.all_organ_dicts = nn.Parameter(torch.randn(num_organs, dict_atoms, feat_dim))
        self.dict_scale_logit = nn.Parameter(torch.zeros(1))
        self.temperature_logit = nn.Parameter(torch.ones(1) * (-2.0))
        
        # 初始化
        nn.init.orthogonal_(self.all_organ_dicts)
        nn.init.orthogonal_(self.all_organ_queries)
        
        # 融合层
        self.fusion_layer = nn.Linear(feat_dim * 2, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

        # --- 2. Token 聚合器 (改进部分) ---
        if self.use_global_token:
            # [改进 A] Global Token: 使用 Cross-Attention
            # 这里的 nhead 可以设为 4 或 8
            self.global_token_aggregator = CrossAttentionAggregator(feat_dim, num_heads=8)
            
            # [改进 B] Residual Token: 混合 Avg + Max Pooling
            # Attention Net 用于 Avg Pooling
            self.residual_attn_net = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.Tanh(), # Tanh 比 ReLU 在 Attention 权重计算中更平滑
                nn.Linear(feat_dim // 4, 1)
            )
            # Fusion Layer 用于融合 Avg 和 Max 的结果
            self.residual_fusion = nn.Linear(feat_dim * 2, feat_dim)
            
            # 器官先验 Embedding (作为 Cross-Attention 的 Query)
            self.organ_global_tokens = nn.Parameter(
                torch.randn(num_organs, feat_dim)
            )
            nn.init.normal_(self.organ_global_tokens, std=0.02)

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        norm = x.norm(p=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    

    def get_diagnostic_metrics(self, q_feat, L_rec, S_res, alpha, dict_scale, temperature, attn_weights=None, masks=None):


        """
        📊 全面的诊断指标监控
        Args:
            q_feat: [B_total, Tokens, Dim] - 输入特征
            L_rec:  [B_total, Tokens, Dim] - 字典重构部分 (Low Rank)
            S_res:  [B_total, Tokens, Dim] - 残差部分 (Sparse/Anomaly)
            alpha:  [B_total, Tokens, Atoms] - 字典系数
            dict_scale: scalar
            temperature: scalar
            attn_weights: [B_total, Tokens, N] - (可选) Q-Former 注意力权重
            masks: [B_total, N] - (可选) 对应的器官 Mask
        """
        with torch.no_grad():
            metrics = {}
            
            # -------------------------------------------------------
            # 1. 分解能量分析 (Decomposition Energy)
            # -------------------------------------------------------
            # 目的：监控 L 和 S 的比例。
            # - 如果 S_ratio 接近 0：模型忽略了异常，只重构了共性 (Over-smoothing)。
            # - 如果 S_ratio 接近 1：字典没起作用，模型退化为恒等映射。
            # - 理想情况：S_ratio 应该在一个较小的范围 (e.g., 0.1 ~ 0.4)，保留必要的细节。
            q_energy = torch.norm(q_feat, p=2, dim=-1).mean()
            l_energy = torch.norm(L_rec, p=2, dim=-1).mean()
            s_energy = torch.norm(S_res, p=2, dim=-1).mean()
            
            metrics["energy/input"] = q_energy.item()
            metrics["energy/low_rank"] = l_energy.item()
            metrics["energy/residual"] = s_energy.item()
            metrics["energy/res_ratio"] = (s_energy / (q_energy + 1e-6)).item()

            # -------------------------------------------------------
            # 2. 字典稀疏性与利用率 (Dictionary Health)
            # -------------------------------------------------------
            # alpha_sparsity: 平均每个 Token 用了多少个原子 (Top-K 之外是否真的被抑制了)
            # dead_atoms: 当前 Batch 中有多少原子完全没被用到 (防止字典坍塌)
            
            # 统计 > 0.01 的系数个数
            active_elements = (alpha > 0.01).float().sum(dim=-1).mean()
            
            # 统计当前 Batch 内完全没被激活的原子数量 (Sum over Batch & Tokens)
            atom_usage_counts = alpha.sum(dim=(0, 1)) 
            dead_atoms_batch = (atom_usage_counts < 1e-5).float().sum()
            
            # 熵：衡量选择原子的确定性。越低越好，说明模型很确定用哪几个原子。
            alpha_entropy = -(alpha * torch.log(alpha + 1e-9)).sum(dim=-1).mean()

            metrics["dict/active_atoms_per_token"] = active_elements.item()
            metrics["dict/dead_atoms_in_batch"] = dead_atoms_batch.item()
            metrics["dict/entropy"] = alpha_entropy.item()
            metrics["dict/scale_factor"] = dict_scale.item()
            metrics["dict/temperature"] = temperature.item()

            # -------------------------------------------------------
            # 3. Attention 质量分析 (Q-Former Health)
            # -------------------------------------------------------
            # 只有提供了 attn_weights 和 masks 才能计算
            if attn_weights is not None and masks is not None:
                # attn_weights: [B, Tokens, N]
                # masks: [B, N]
                
                # 扩展 mask: [B, 1, N]
                mask_expanded = masks.unsqueeze(1)
                
                # 计算 Mask 内部的平均注意力强度 vs 外部的平均注意力强度
                # 理想情况：Inside 很高，Outside 接近 0
                attn_inside = (attn_weights * mask_expanded).sum() / (mask_expanded.sum() + 1e-6)
                attn_outside = (attn_weights * (1 - mask_expanded)).sum() / ((1 - mask_expanded).sum() + 1e-6)
                
                # 覆盖率：Mask 内部有多少像素被至少一个 Query 关注到了 (Max > 0.1)
                max_attn_per_pixel, _ = attn_weights.max(dim=1) # [B, N]
                covered_pixels = ((max_attn_per_pixel > 0.05).float() * masks).sum()
                total_mask_pixels = masks.sum() + 1e-6
                coverage_ratio = covered_pixels / total_mask_pixels

                metrics["attn/intensity_inside"] = attn_inside.item()
                metrics["attn/intensity_outside"] = attn_outside.item()
                metrics["attn/mask_coverage"] = coverage_ratio.item()

        return metrics

    def aggregate_dual_tokens(self, final_feat, S_res, organ_indices):
        """
        [核心修改] 同时聚合 Global Token 和 Residual Token
        """
        # --- 1. Global Token (改进方案二：Query-Based) ---
        # 目的：利用器官先验 (Query) 主动从 final_feat (Key/Value) 中提取结构化信息
        
        # 获取当前 batch 对应的器官先验，并调整形状为 [B, 1, D]
        organ_prior_query = self.organ_global_tokens[organ_indices].unsqueeze(1)
        
        # 通过 Cross-Attention 聚合
        # 输出已经是 [B, D]，且包含了 organ_prior 的残差连接
        token_global = self.global_token_aggregator(query=organ_prior_query, kv=final_feat)
        token_global = self.safe_normalize(token_global, dim=-1)

        # --- 2. Residual Token (增强版：Avg + Max) ---
        # 目的：捕捉稀疏的异常信号。Max Pooling 对捕捉微小病灶至关重要。
        
        # 分支 A: Attention Pooling (关注整体异常分布)
        attn_scores_r = self.residual_attn_net(S_res) # [B, N, 1]
        attn_weights_r = F.softmax(attn_scores_r, dim=1)
        token_res_avg = torch.sum(S_res * attn_weights_r, dim=1) # [B, D]
        
        # 分支 B: Max Pooling (关注最显著的异常点)
        # S_res 是稀疏的，Max 能避免平均操作稀释掉强烈的局部病变信号
        token_res_max, _ = torch.max(S_res, dim=1) # [B, D]
        
        # 融合两个分支
        combined_res = torch.cat([token_res_avg, token_res_max], dim=-1)
        token_residual = self.residual_fusion(combined_res)
        token_residual = self.safe_normalize(token_residual, dim=-1)

        return token_global, token_residual

    def calculate_label_guided_loss(self, S_res, organ_labels):
        """
        S_res: [B * 10, 32, 768] (已经 flatten 过的 batch)
        organ_labels: [B, 10]    (原始标签)
        """
        flat_labels = organ_labels.view(-1).float()
        is_normal = (flat_labels == 0).view(-1, 1, 1)
        is_abnormal = (flat_labels == 1).view(-1, 1, 1)
        
        res_energy = torch.abs(S_res)
        
        # 正常样本：零容忍
        loss_normal_zero = (res_energy * is_normal).mean() / (is_normal.sum() + 1e-6) * 0.1
        
        # 异常样本：宽容 (仅微弱稀疏惩罚)
        loss_abnormal_sparse = (res_energy * is_abnormal).mean() / (is_abnormal.sum() + 1e-6) * 0.01
        
        # print(loss_normal_zero, loss_abnormal_sparse)

        return loss_normal_zero + loss_abnormal_sparse

    def process_single_organ(self, ifeat, mask, organ_idx):
        B, N, L = ifeat.shape
        
        # Mask 处理
        mask_sum = mask.sum(dim=1)
        safe_mask = mask.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        # Q-Former
        batch_queries = self.all_organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1)
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat,
            memory_key_padding_mask=key_padding_mask
        )
        
        # Query 多样性 Loss
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=ifeat.device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        # 字典分解
        batch_dicts = self.all_organ_dicts[organ_idx].unsqueeze(0).expand(B, -1, -1)
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat.detach() - L_rec

        # --- Loss 计算 ---
        loss_reconstruction = F.l1_loss(L_rec, q_feat.detach())
        loss_residual_sparsity = torch.mean(torch.abs(S_res))
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=ifeat.device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_coeff_sparsity +
            0.1 * loss_residual_sparsity + 
            0.1 * loss_dict_ortho +
            0.3 * loss_query_div
        )

        # metrics = self.get_diagnostic_metrics(q_feat, L_rec, S_res, alpha, dict_scale, temperature)
        # print(metrics)

        # 输出特征融合
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        # --- 生成双流 Token ---
        token_global = None
        token_residual = None
        if self.use_global_token:
            indices = torch.tensor([organ_idx] * B, device=ifeat.device)
            token_global, token_residual = self.aggregate_dual_tokens(
                final_feat, S_res, indices
            )

        return q_feat, token_global, token_residual, aux_loss, None

    def process_all_organs_batched(self, ifeat, masks, organ_labels = None):
        B, num_organs, N = masks.shape
        device = ifeat.device
        L = ifeat.size(-1)
        
        ifeat_expanded = ifeat.unsqueeze(1).expand(-1, num_organs, -1, -1)
        ifeat_flat = ifeat_expanded.reshape(B * num_organs, N, L)
        masks_flat = masks.reshape(B * num_organs, N)
        
        mask_sum = masks_flat.sum(dim=1)
        safe_mask = masks_flat.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)
        
        organ_indices = torch.arange(num_organs, device=device).repeat(B)
        batch_queries = self.all_organ_queries[organ_indices]
        
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat_flat,
            memory_key_padding_mask=key_padding_mask
        )
        
        # Query Diversity
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        # q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        # mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        # off_diagonal = q_sim_matrix[:, mask_off_diag].view(B * num_organs, self.num_tokens, -1)
        # loss_query_div = torch.mean(off_diagonal ** 2)
        
        # Dictionary Learning
        batch_dicts = self.all_organ_dicts[organ_indices]
        d_norm = self.safe_normalize(batch_dicts, dim=-1)
        
        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature
        
        topk_val, topk_idx = torch.topk(similarity, self.top_k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()
        
        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        
        alpha = F.softmax(sparse_mask, dim=-1)
        
        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat.detach() - L_rec
        
        # metrics = self.get_diagnostic_metrics(q_feat, L_rec, S_res, alpha, dict_scale, temperature)
        # print(metrics)
        
        # --- Loss Calculation ---
        loss_reconstruction = F.l1_loss(L_rec, q_feat.detach(), reduction='mean')
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        # Label Guided Loss
        if organ_labels is not None:
            loss_s = self.calculate_label_guided_loss(S_res, organ_labels)
        else:
            loss_s = torch.zeros_like(loss_dict_ortho)
        
        total_aux_loss = (
            1.0 * loss_reconstruction +
            0.1 * loss_coeff_sparsity +
            0.1 * loss_dict_ortho +
            # 0.3 * loss_query_div + 
            loss_s
        )
        # print(loss_reconstruction, loss_coeff_sparsity, loss_dict_ortho, loss_s)
        # Fusion
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)
        
        # Reshape back to [B, Num_Organs, Tokens, Dim]
        all_final_feats = final_feat.view(B, num_organs, self.num_tokens, self.feat_dim)
        
        # --- Dual Token Aggregation ---
        all_global_tokens = None
        all_residual_tokens = None
        
        if self.use_global_token:
            token_global, token_residual = self.aggregate_dual_tokens(
                final_feat, S_res, organ_indices
            )
            all_global_tokens = token_global.view(B, num_organs, self.feat_dim)
            all_residual_tokens = token_residual.view(B, num_organs, self.feat_dim)
        
        return all_final_feats, all_global_tokens, all_residual_tokens, total_aux_loss, None

    def _process_mixed_batch(self, ifeat, mask, organ_indices):
        B = ifeat.size(0)
        device = ifeat.device
        all_final_feats = []
        all_global_tokens = []
        all_residual_tokens = []
        all_aux_losses = []
        unique_organs = torch.unique(organ_indices)
        
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            organ_ifeat = ifeat[organ_mask_bool]
            organ_mask = mask[organ_mask_bool]
            
            final_feat, t_glob, t_res, aux_loss, _ = self.process_single_organ(
                organ_ifeat, organ_mask, organ_idx.item()
            )
            
            all_final_feats.append(final_feat)
            all_global_tokens.append(t_glob)
            all_residual_tokens.append(t_res)
            all_aux_losses.append(aux_loss)
        
        # Reassemble batch
        final_feat_full = torch.zeros(B, self.num_tokens, self.feat_dim, device=device)
        token_global_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        token_residual_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        
        idx_counter = 0
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            final_feat_full[organ_mask_bool] = all_final_feats[idx_counter]
            if self.use_global_token:
                token_global_full[organ_mask_bool] = all_global_tokens[idx_counter]
                token_residual_full[organ_mask_bool] = all_residual_tokens[idx_counter]
            idx_counter += 1
        
        total_aux_loss = torch.stack(all_aux_losses).mean()
        return final_feat_full, token_global_full, token_residual_full, total_aux_loss, {}

    def forward(self, ifeat, mask=None, idx=None, abnorm_label = None, multi_organ_mode=False):
        """
        Returns:
            final_feat: [B, N, D] or [B, Num_Organs, N, D]
            token_global: [B, D] or [B, Num_Organs, D]
            token_residual: [B, D] or [B, Num_Organs, D]
            aux_loss: scalar
            extra_info: dict
        """
        B, N_seq = ifeat.size(0), ifeat.size(1)

        # Handle mask=None: create an all-ones mask
        if mask is None:
            if multi_organ_mode:
                # [B, num_organs, N]
                mask = torch.ones(B, self.num_organs, N_seq, device=ifeat.device, dtype=ifeat.dtype)
            else:
                # [B, N]
                mask = torch.ones(B, N_seq, device=ifeat.device, dtype=ifeat.dtype)

        if multi_organ_mode or (mask.dim() == 3):
            assert mask.dim() == 3
            return self.process_all_organs_batched(ifeat, mask, abnorm_label)
        else:
            assert idx is not None
            assert mask.dim() == 2
            B = ifeat.size(0)
            organ_indices = idx.view(B).long()
            if torch.all(organ_indices == organ_indices[0]):
                return self.process_single_organ(ifeat, mask, organ_indices[0].item())
            else:
                return self._process_mixed_batch(ifeat, mask, organ_indices)


class OrganFeatureRefiner_v2_LRMR_dual_v2_new(OrganFeatureRefiner_v2_LRMR_dual_v2):
    """
    Improved variant of OrganFeatureRefiner_v2_LRMR_dual_v2 with
    stronger label-guided supervision, cross-organ dictionary
    decorrelation, and query diversity in multi-organ mode.
    Original class is kept untouched.
    """

    def __init__(
        self,
        num_organs=10,
        feat_dim=768,
        num_tokens=32,
        dict_atoms=256,
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True,
        label_margin=0.02,
        label_weight_normal=0.5,
        label_weight_abnormal=0.1,
        dict_within_weight=0.05,
        dict_cross_weight=0.02,
        query_div_weight=0.1,
    ):
        super().__init__(
            num_organs=num_organs,
            feat_dim=feat_dim,
            num_tokens=num_tokens,
            dict_atoms=dict_atoms,
            top_k=top_k,
            nhead=nhead,
            num_layers=num_layers,
            use_global_token=use_global_token,
        )
        self.label_margin = label_margin
        self.label_weight_normal = label_weight_normal
        self.label_weight_abnormal = label_weight_abnormal
        self.dict_within_weight = dict_within_weight
        self.dict_cross_weight = dict_cross_weight
        self.query_div_weight = query_div_weight

    def calculate_label_guided_loss(self, S_res, organ_labels):
        """
        Label-guided residual loss with stable mean-based
        normalization.

        S_res: [B * num_organs, T, D]
        organ_labels: [B, num_organs] with {0: normal, 1: abnormal}.
        """
        flat_labels = organ_labels.view(-1).float()
        is_normal = (flat_labels == 0).view(-1, 1, 1)
        is_abnormal = (flat_labels == 1).view(-1, 1, 1)

        res_energy = torch.abs(S_res)

        # Normal organs: push residual energy toward zero.
        if is_normal.any():
            loss_normal = (res_energy * is_normal).mean()
        else:
            loss_normal = torch.zeros_like(res_energy.mean())

        # Abnormal organs: encourage non-trivial residuals via margin.
        if is_abnormal.any():
            mean_abnormal = (res_energy * is_abnormal).mean()
            loss_abnormal = F.relu(self.label_margin - mean_abnormal)
        else:
            loss_abnormal = torch.zeros_like(res_energy.mean())

        return (
            self.label_weight_normal * loss_normal
            + self.label_weight_abnormal * loss_abnormal
        )

    def _dictionary_regularization(self, device):
        """
        Compute within-organ atom orthogonality and cross-organ
        prototype decorrelation losses.
        """
        # [num_organs, dict_atoms, D]
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)

        # Within-organ: each organ's atoms approximately orthogonal.
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_within = torch.mean((gram_matrix - identity) ** 2)

        # Cross-organ: organ prototypes decorrelated.
        organ_proto = all_d_norm.mean(dim=1)  # [num_organs, D]
        gram_org = torch.matmul(organ_proto, organ_proto.t())
        identity_org = torch.eye(self.num_organs, device=device)
        loss_cross = torch.mean((gram_org - identity_org) ** 2)

        return loss_within, loss_cross

    def process_all_organs_batched(self, ifeat, masks, organ_labels=None):
        """
        Multi-organ processing with:
        - consistent query diversity regularization,
        - LRMR dictionary decomposition with top-k guarding,
        - stronger label-guided residual supervision,
        - within- and cross-organ dictionary regularization.
        """
        B, num_organs, N = masks.shape
        device = ifeat.device
        L = ifeat.size(-1)

        ifeat_expanded = ifeat.unsqueeze(1).expand(-1, num_organs, -1, -1)
        ifeat_flat = ifeat_expanded.reshape(B * num_organs, N, L)
        masks_flat = masks.reshape(B * num_organs, N)

        mask_sum = masks_flat.sum(dim=1)
        safe_mask = masks_flat.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        organ_indices = torch.arange(num_organs, device=device).repeat(B)
        batch_queries = self.all_organ_queries[organ_indices]

        # Q-Former
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat_flat,
            memory_key_padding_mask=key_padding_mask,
        )

        # Query diversity regularization (per organ and sample).
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(
            self.num_tokens, dtype=torch.bool, device=device
        )
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(
            B * num_organs, self.num_tokens, -1
        )
        loss_query_div = torch.mean(off_diagonal ** 2)

        # Dictionary learning with top-k guarding.
        batch_dicts = self.all_organ_dicts[organ_indices]
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature

        k = min(self.top_k, self.dict_atoms)
        topk_val, topk_idx = torch.topk(similarity, k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()

        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)

        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat.detach() - L_rec

        # Reconstruction and sparsity losses.
        loss_reconstruction = F.l1_loss(L_rec, q_feat.detach(), reduction="mean")
        loss_coeff_sparsity = (
            torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms
        )

        loss_dict_within, loss_dict_cross = self._dictionary_regularization(device)

        # Label-guided residual loss.
        if organ_labels is not None:
            loss_s = self.calculate_label_guided_loss(S_res, organ_labels)
        else:
            loss_s = torch.zeros_like(loss_dict_within)

        total_aux_loss = (
            1.0 * loss_reconstruction
            + 0.1 * loss_coeff_sparsity
            # + self.dict_within_weight * loss_dict_within
            # + self.dict_cross_weight * loss_dict_cross
            # + self.query_div_weight * loss_query_div
            + loss_s
        )

        # Fusion.
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        # Reshape back to [B, Num_Organs, Tokens, Dim]
        all_final_feats = q_feat.view(
            B, num_organs, self.num_tokens, self.feat_dim
        )

        # Dual token aggregation (reuse parent implementation).
        all_global_tokens = None
        all_residual_tokens = None
        if self.use_global_token:
            token_global, token_residual = self.aggregate_dual_tokens(
                q_feat, S_res, organ_indices
            )
            all_global_tokens = token_global.view(B, num_organs, self.feat_dim)
            all_residual_tokens = token_residual.view(
                B, num_organs, self.feat_dim
            )

        return all_final_feats, all_global_tokens, all_residual_tokens, total_aux_loss, None


class OrganFeatureRefiner_v2_LRMR_dual_v3(OrganFeatureRefiner_v2_LRMR_dual_v2_new):
    """
    v3: keep v2_new architecture, but replace label-guided residual loss
    with rank-margin objective to avoid the conflict of suppressing
    abnormal residuals directly.
    """

    def __init__(
        self,
        num_organs=10,
        feat_dim=768,
        num_tokens=32,
        dict_atoms=256,
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True,
        label_margin=0.02,
        label_weight_normal=0.5,
        label_weight_abnormal=0.1,
        dict_within_weight=0.05,
        dict_cross_weight=0.02,
        query_div_weight=0.1,
    ):
        super().__init__(
            num_organs=num_organs,
            feat_dim=feat_dim,
            num_tokens=num_tokens,
            dict_atoms=dict_atoms,
            top_k=top_k,
            nhead=nhead,
            num_layers=num_layers,
            use_global_token=use_global_token,
            label_margin=label_margin,
            label_weight_normal=label_weight_normal,
            label_weight_abnormal=label_weight_abnormal,
            dict_within_weight=dict_within_weight,
            dict_cross_weight=dict_cross_weight,
            query_div_weight=query_div_weight,
        )

    

    def calculate_label_guided_loss(self, S_res, organ_labels):
        """
        Rank-margin label-guided residual loss.

        Goal:
        - normal organs: keep residual energy small
        - abnormal organs: residual energy should be higher than normal by a margin

        S_res: [B * num_organs, T, D]
        organ_labels: [B, num_organs] with {0: normal, 1: abnormal}
        """
        flat_labels = organ_labels.view(-1).float()
        sample_energy = torch.mean(torch.abs(S_res), dim=(1, 2))  # [B * num_organs]

        normal_mask = (flat_labels == 0)
        abnormal_mask = (flat_labels == 1)

        zero = torch.zeros((), device=S_res.device, dtype=S_res.dtype)

        if normal_mask.any():
            mean_normal = sample_energy[normal_mask].mean()
            loss_normal = mean_normal
        else:
            mean_normal = zero
            loss_normal = zero

        if abnormal_mask.any():
            mean_abnormal = sample_energy[abnormal_mask].mean()
        else:
            mean_abnormal = zero

        if normal_mask.any() and abnormal_mask.any():
            loss_rank_margin = F.relu(self.label_margin - (mean_abnormal - mean_normal))
        else:
            loss_rank_margin = zero

        return (
            self.label_weight_normal * loss_normal
            + self.label_weight_abnormal * loss_rank_margin
        )


class OrganFeatureRefiner_v2_LRMR_dual_v4(nn.Module):
    """
    v4 independent implementation.

    Key points:
    1) No inheritance from v2/v3 classes.
    2) Keep rank-margin label-guided residual supervision.
    3) Capture real Q-Former cross-attention maps for query->memory projection.
    4) Keep output signatures compatible with v3 usage.
    """

    def __init__(
        self,
        num_organs=10,
        feat_dim=768,
        num_tokens=32,
        dict_atoms=256,
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True,
        label_margin=0.02,
        label_weight_normal=0.5,
        label_weight_abnormal=0.1,
        dict_within_weight=0.05,
        dict_cross_weight=0.02,
        query_div_weight=0.1,
        anomaly_threshold=1.5,
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.dict_atoms = dict_atoms
        self.use_global_token = use_global_token

        self.label_margin = label_margin
        self.label_weight_normal = label_weight_normal
        self.label_weight_abnormal = label_weight_abnormal
        self.dict_within_weight = dict_within_weight
        self.dict_cross_weight = dict_cross_weight
        self.query_div_weight = query_div_weight
        self.anomaly_threshold = anomaly_threshold

        self.all_organ_queries = nn.Parameter(torch.randn(num_organs, num_tokens, feat_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
        )
        self.shared_qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.all_organ_dicts = nn.Parameter(torch.randn(num_organs, dict_atoms, feat_dim))
        self.dict_scale_logit = nn.Parameter(torch.zeros(1))
        self.temperature_logit = nn.Parameter(torch.ones(1) * (-2.0))

        nn.init.orthogonal_(self.all_organ_dicts)
        nn.init.orthogonal_(self.all_organ_queries)

        self.fusion_layer = nn.Linear(feat_dim * 2, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

        if self.use_global_token:
            self.global_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1),
            )
            self.residual_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1),
            )
            self.organ_global_tokens = nn.Parameter(torch.randn(num_organs, feat_dim))
            nn.init.normal_(self.organ_global_tokens, std=0.02)

        self._cross_attn_capture_enabled = False
        self._enable_cross_attn_capture()

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        norm = x.norm(p=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    def aggregate_dual_tokens(self, final_feat, S_res, organ_indices):
        attn_scores_g = self.global_token_aggregator(final_feat)
        attn_weights_g = F.softmax(attn_scores_g, dim=1)
        token_global = torch.sum(final_feat * attn_weights_g, dim=1)

        organ_prior = self.organ_global_tokens[organ_indices]
        token_global = token_global + 0.1 * organ_prior
        token_global = self.safe_normalize(token_global, dim=-1)

        attn_scores_r = self.residual_token_aggregator(S_res)
        attn_weights_r = F.softmax(attn_scores_r, dim=1)
        token_residual = torch.sum(S_res * attn_weights_r, dim=1)
        token_residual = self.safe_normalize(token_residual, dim=-1)

        return token_global, token_residual

    def _enable_cross_attn_capture(self):
        if self._cross_attn_capture_enabled:
            return

        for layer in self.shared_qformer.layers:
            layer.last_self_attn = None
            layer.last_cross_attn = None

            def _patched_sa_block(this, x, attn_mask, key_padding_mask, *args, **kwargs):
                attn_out, attn_w = this.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                this.last_self_attn = attn_w
                return this.dropout1(attn_out)

            def _patched_mha_block(this, x, mem, attn_mask, key_padding_mask, *args, **kwargs):
                attn_out, attn_w = this.multihead_attn(
                    x,
                    mem,
                    mem,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                this.last_cross_attn = attn_w
                return this.dropout2(attn_out)

            layer._sa_block = _patched_sa_block.__get__(layer, layer.__class__)
            layer._mha_block = _patched_mha_block.__get__(layer, layer.__class__)

        self._cross_attn_capture_enabled = True

    def _collect_cross_attn_maps(self, B, T, N, safe_mask):
        dtype = self.all_organ_queries.dtype
        device = safe_mask.device
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        valid = safe_mask.unsqueeze(1).to(dtype)

        q2m_attn_last = None
        rollout_per_layer = []
        query_rollout = eye

        for layer in self.shared_qformer.layers:
            self_w = getattr(layer, "last_self_attn", None)
            cross_w = getattr(layer, "last_cross_attn", None)

            if self_w is not None:
                if self_w.dim() == 3:
                    self_w = self_w.unsqueeze(1)
                if self_w.dim() == 4 and self_w.size(0) == B and self_w.size(-2) == T and self_w.size(-1) == T:
                    self_attn = self_w.mean(dim=1)
                    self_attn = self_attn + eye
                    self_attn = self_attn / self_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    query_rollout = torch.bmm(self_attn, query_rollout)

            if cross_w is None:
                continue
            if cross_w.dim() == 3:
                cross_w = cross_w.unsqueeze(1)
            if not (cross_w.dim() == 4 and cross_w.size(0) == B and cross_w.size(-2) == T and cross_w.size(-1) == N):
                continue

            cross_attn = cross_w.mean(dim=1)
            cross_attn = cross_attn * valid
            cross_attn = cross_attn / cross_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            q2m_attn_last = cross_attn
            rollout_per_layer.append(torch.bmm(query_rollout, cross_attn))

        if q2m_attn_last is None:
            return None, None

        q2m_attn_rollout = torch.stack(rollout_per_layer, dim=0).mean(dim=0)
        q2m_attn_rollout = q2m_attn_rollout * valid
        q2m_attn_rollout = q2m_attn_rollout / q2m_attn_rollout.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return q2m_attn_last, q2m_attn_rollout

    def _collect_cross_attn_maps_(self, B, T, N, safe_mask):
        attn_layers = []
        for layer in self.shared_qformer.layers:
            w = getattr(layer, "last_cross_attn", None)
            if w is None:
                continue
            if w.dim() == 3:
                w = w.unsqueeze(1)
            if w.dim() == 4 and w.size(0) == B and w.size(-2) == T and w.size(-1) == N:
                attn_layers.append(w)

        if len(attn_layers) == 0:
            return None, None

        q2m_attn_last = attn_layers[-1].mean(dim=1)
        stack = torch.stack(attn_layers, dim=0)
        q2m_attn_rollout = stack.mean(dim=0).mean(dim=1)

        valid = safe_mask.unsqueeze(1)
        q2m_attn_last = q2m_attn_last * valid
        q2m_attn_last = q2m_attn_last / q2m_attn_last.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        q2m_attn_rollout = q2m_attn_rollout * valid
        q2m_attn_rollout = q2m_attn_rollout / q2m_attn_rollout.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return q2m_attn_last, q2m_attn_rollout

    def _dictionary_regularization(self, device):
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)

        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_within = torch.mean((gram_matrix - identity) ** 2)

        organ_proto = all_d_norm.mean(dim=1)
        gram_org = torch.matmul(organ_proto, organ_proto.t())
        identity_org = torch.eye(self.num_organs, device=device)
        loss_cross = torch.mean((gram_org - identity_org) ** 2)

        return loss_within, loss_cross

    def calculate_label_guided_loss(self, S_res, organ_labels):
        flat_labels = organ_labels.view(-1).float()
        sample_energy = torch.mean(torch.abs(S_res), dim=(1, 2))

        normal_mask = (flat_labels == 0)
        abnormal_mask = (flat_labels == 1)

        zero = torch.zeros((), device=S_res.device, dtype=S_res.dtype)

        if normal_mask.any():
            mean_normal = sample_energy[normal_mask].mean()
            loss_normal = mean_normal
        else:
            mean_normal = zero
            loss_normal = zero

        if abnormal_mask.any():
            mean_abnormal = sample_energy[abnormal_mask].mean()
        else:
            mean_abnormal = zero

        if normal_mask.any() and abnormal_mask.any():
            loss_rank_margin = F.relu(self.label_margin - (mean_abnormal - mean_normal))
        else:
            loss_rank_margin = zero

        return self.label_weight_normal * loss_normal + self.label_weight_abnormal * loss_rank_margin

    def process_single_organ(self, ifeat, mask, organ_idx):
        # ifeat: [B, N, D], mask: [B, N], organ_idx: int
        # B: batch size, N: number of memory/source tokens, D: feature dim
        B, N, _ = ifeat.shape
        device = ifeat.device

        # Make sure cross-attention capture is enabled for Q-Former decoder layers.
        self._enable_cross_attn_capture()

        # Build a safe token mask. If a sample has no valid token, force index 0 as valid
        # to avoid fully-masked attention instability.
        mask_sum = mask.sum(dim=1)
        safe_mask = mask.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        # 1) Q-Former forward: organ-specific learnable queries attend to memory tokens.
        batch_queries = self.all_organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1)
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat,
            memory_key_padding_mask=key_padding_mask,
        )

        # 2) Query diversity diagnostic loss (off-diagonal similarity among query tokens).
        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        # 3) Dictionary sparse reconstruction in query space.
        batch_dicts = self.all_organ_dicts[organ_idx].unsqueeze(0).expand(B, -1, -1)
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature

        # Top-k sparse coding over organ dictionary atoms.
        k = min(self.top_k, self.dict_atoms)
        topk_val, topk_idx = torch.topk(similarity, k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()

        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale

        # Residual in query space: larger residual usually indicates harder-to-reconstruct patterns.
        S_res = q_feat.detach() - L_rec

        # 4) Query-level anomaly scoring using robust normalization (median + MAD).
        token_residual_energy = torch.mean(torch.abs(S_res), dim=-1)
        median_energy = token_residual_energy.median(dim=1, keepdim=True)[0]
        mad = (token_residual_energy - median_energy).abs().median(dim=1, keepdim=True)[0]
        mad = mad.clamp_min(1e-6)
        token_anomaly_score = (token_residual_energy - median_energy) / mad
        token_anomaly_prob = torch.sigmoid(token_anomaly_score)
        token_anomaly_pred = (token_anomaly_score > self.anomaly_threshold).float()

        # Top suspicious query tokens.
        k_anom = min(4, self.num_tokens)
        _, topk_anomaly_idx = torch.topk(token_anomaly_score, k=k_anom, dim=1)

        # 5) Prefer real Q-Former cross-attention maps (query->memory).
        q2m_attn_last, q2m_attn_rollout = self._collect_cross_attn_maps(
            B=B,
            T=self.num_tokens,
            N=N,
            safe_mask=safe_mask,
        )

        # Fallback to similarity-based proxy attention if real maps are unavailable.
        if q2m_attn_rollout is None:
            ifeat_norm = self.safe_normalize(ifeat, dim=-1)
            q2m_logits = torch.bmm(q_feat_norm, ifeat_norm.transpose(1, 2)) / temperature
            q2m_logits = q2m_logits.masked_fill((safe_mask == 0).unsqueeze(1), -1e9)
            q2m_attn_rollout = F.softmax(q2m_logits, dim=-1)
            q2m_attn_last = q2m_attn_rollout

        # 6) Project query anomaly to original memory tokens via attention weighting.
        query_anom_weight = F.softmax(token_anomaly_score, dim=1)
        memory_anomaly_score = torch.bmm(query_anom_weight.unsqueeze(1), q2m_attn_rollout).squeeze(1)
        memory_anomaly_score = memory_anomaly_score * safe_mask

        # Normalize memory-level score and derive prob / binary prediction.
        valid_cnt = safe_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mem_mean = memory_anomaly_score.sum(dim=1, keepdim=True) / valid_cnt
        mem_var = (((memory_anomaly_score - mem_mean) * safe_mask) ** 2).sum(dim=1, keepdim=True) / valid_cnt
        mem_std = torch.sqrt(mem_var + 1e-6)
        memory_anomaly_z = (memory_anomaly_score - mem_mean) / mem_std
        memory_anomaly_prob = torch.sigmoid(memory_anomaly_z)
        memory_anomaly_pred = ((memory_anomaly_z > self.anomaly_threshold) & (safe_mask > 0)).float()

        # Top suspicious original memory tokens.
        k_mem = min(16, N)
        memory_anomaly_z_masked = memory_anomaly_z.masked_fill(safe_mask == 0, -1e9)
        topk_memory_anomaly_score, topk_memory_anomaly_idx = torch.topk(memory_anomaly_z_masked, k=k_mem, dim=1)

        # 7) Auxiliary losses for reconstruction quality and dictionary regularity.
        loss_reconstruction = F.l1_loss(L_rec, q_feat.detach(), reduction="mean")
        loss_residual_sparsity = torch.mean(torch.abs(S_res))
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms

        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        aux_loss = (
            1.0 * loss_reconstruction
            + 0.1 * loss_coeff_sparsity
            # + 0.1 * loss_residual_sparsity
            # + 0.1 * loss_dict_ortho
            # + 0.0 * loss_query_div
        )

        # 8) Fuse reconstructed component and residual, then output refined query features.
        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        # Optional dual token aggregation (global/residual summary per sample).
        token_global = None
        token_residual = None
        if self.use_global_token:
            indices = torch.tensor([organ_idx] * B, device=device)
            token_global, token_residual = self.aggregate_dual_tokens(final_feat, S_res, indices)

        # Rich diagnostics for downstream visualization/debugging.
        extra_info = {
            "token_residual_energy": token_residual_energy.detach(),
            "token_anomaly_score": token_anomaly_score.detach(),
            "token_anomaly_prob": token_anomaly_prob.detach(),
            "token_anomaly_pred": token_anomaly_pred.detach(),
            "topk_anomaly_idx": topk_anomaly_idx.detach(),
            "memory_anomaly_score": memory_anomaly_score.detach(),
            "memory_anomaly_z": memory_anomaly_z.detach(),
            "memory_anomaly_prob": memory_anomaly_prob.detach(),
            "memory_anomaly_pred": memory_anomaly_pred.detach(),
            "topk_memory_anomaly_idx": topk_memory_anomaly_idx.detach(),
            "topk_memory_anomaly_score": topk_memory_anomaly_score.detach(),
            "qformer_cross_attn_last": q2m_attn_last.detach(),
            "qformer_cross_attn_rollout": q2m_attn_rollout.detach(),
        }

        return final_feat, token_global, token_residual, aux_loss, extra_info

    def process_all_organs_batched(self, ifeat, masks, organ_labels=None):
        B, num_organs, N = masks.shape
        device = ifeat.device
        L = ifeat.size(-1)

        ifeat_expanded = ifeat.unsqueeze(1).expand(-1, num_organs, -1, -1)
        ifeat_flat = ifeat_expanded.reshape(B * num_organs, N, L)
        masks_flat = masks.reshape(B * num_organs, N)

        mask_sum = masks_flat.sum(dim=1)
        safe_mask = masks_flat.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        organ_indices = torch.arange(num_organs, device=device).repeat(B)
        batch_queries = self.all_organ_queries[organ_indices]

        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat_flat,
            memory_key_padding_mask=key_padding_mask,
        )

        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B * num_organs, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        batch_dicts = self.all_organ_dicts[organ_indices]
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature

        k = min(self.top_k, self.dict_atoms)
        topk_val, topk_idx = torch.topk(similarity, k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()

        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale
        S_res = q_feat.detach() - L_rec

        loss_reconstruction = F.l1_loss(L_rec, q_feat.detach(), reduction="mean")
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms

        loss_dict_within, loss_dict_cross = self._dictionary_regularization(device)

        if organ_labels is not None:
            loss_s = self.calculate_label_guided_loss(S_res, organ_labels)
        else:
            loss_s = torch.zeros_like(loss_dict_within)

        total_aux_loss = (
            1.0 * loss_reconstruction
            + 0.1 * loss_coeff_sparsity
            # + self.dict_within_weight * loss_dict_within
            # + self.dict_cross_weight * loss_dict_cross
            # + self.query_div_weight * loss_query_div
            + loss_s
        )

        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        all_final_feats = final_feat.view(B, num_organs, self.num_tokens, self.feat_dim)

        all_global_tokens = None
        all_residual_tokens = None
        if self.use_global_token:
            token_global, token_residual = self.aggregate_dual_tokens(final_feat, S_res, organ_indices)
            all_global_tokens = token_global.view(B, num_organs, self.feat_dim)
            all_residual_tokens = token_residual.view(B, num_organs, self.feat_dim)

        return all_final_feats, all_global_tokens, all_residual_tokens, total_aux_loss, {}

    def _process_mixed_batch(self, ifeat, mask, organ_indices):
        B = ifeat.size(0)
        device = ifeat.device

        all_final_feats = []
        all_global_tokens = []
        all_residual_tokens = []
        all_aux_losses = []
        extra_infos = []
        unique_organs = torch.unique(organ_indices)

        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            organ_ifeat = ifeat[organ_mask_bool]
            organ_mask = mask[organ_mask_bool]

            final_feat, t_glob, t_res, aux_loss, extra_info = self.process_single_organ(
                organ_ifeat,
                organ_mask,
                organ_idx.item(),
            )

            all_final_feats.append(final_feat)
            all_global_tokens.append(t_glob)
            all_residual_tokens.append(t_res)
            all_aux_losses.append(aux_loss)
            extra_infos.append(extra_info)

        final_feat_full = torch.zeros(B, self.num_tokens, self.feat_dim, device=device)
        token_global_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        token_residual_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None

        idx_counter = 0
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            final_feat_full[organ_mask_bool] = all_final_feats[idx_counter]
            if self.use_global_token:
                token_global_full[organ_mask_bool] = all_global_tokens[idx_counter]
                token_residual_full[organ_mask_bool] = all_residual_tokens[idx_counter]
            idx_counter += 1

        total_aux_loss = torch.stack(all_aux_losses).mean()
        return final_feat_full, token_global_full, token_residual_full, total_aux_loss, extra_infos

    def forward(self, ifeat, mask=None, idx=None, abnorm_label=None, multi_organ_mode=False):
        if multi_organ_mode or (mask is not None and mask.dim() == 3):
            assert mask is not None and mask.dim() == 3
            return self.process_all_organs_batched(ifeat, mask, abnorm_label)

        assert idx is not None
        assert mask is not None and mask.dim() == 2

        B = ifeat.size(0)
        organ_indices = idx.view(B).long()

        if torch.all(organ_indices == organ_indices[0]):
            return self.process_single_organ(ifeat, mask, organ_indices[0].item())

        return self._process_mixed_batch(ifeat, mask, organ_indices)


# ======================================================================
# Gating-based normal/abnormal decomposition (replaces dictionary)
# ======================================================================
class OrganFeatureRefiner_v2_Gating(nn.Module):
    """
    Replaces dictionary sparse coding with a learned per-token gate.

        q_feat  = QFormer(image_feats)
        gate    = sigmoid(gate_mlp(q_feat))       # [B*O, T, 1],  1→abnormal
        S_res   = q_feat * gate                    # abnormal signal
        L_norm  = q_feat * (1 - gate)              # normal signal
        final   = fusion([L_norm, S_res]) + q_feat
    """

    def __init__(
        self,
        num_organs=10,
        feat_dim=768,
        num_tokens=32,
        nhead=8,
        num_layers=2,
        use_global_token=True,
        label_margin=0.3,
        label_weight_normal=1.0,
        label_weight_abnormal=0.5,
        gate_sparsity_weight=0.1,
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.use_global_token = use_global_token

        self.label_margin = label_margin
        self.label_weight_normal = label_weight_normal
        self.label_weight_abnormal = label_weight_abnormal
        self.gate_sparsity_weight = gate_sparsity_weight

        self.all_organ_queries = nn.Parameter(
            torch.randn(num_organs, num_tokens, feat_dim)
        )
        nn.init.orthogonal_(self.all_organ_queries)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim, nhead=nhead,
            batch_first=True, norm_first=True,
        )
        self.shared_qformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, 1),
        )
        nn.init.zeros_(self.gate_mlp[-1].bias)

        self.fusion_layer = nn.Linear(feat_dim * 2, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

        if self.use_global_token:
            self.global_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1),
            )
            self.residual_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1),
            )
            self.organ_global_tokens = nn.Parameter(
                torch.randn(num_organs, feat_dim)
            )
            nn.init.normal_(self.organ_global_tokens, std=0.02)

    # ------------------------------------------------------------------
    def safe_normalize(self, x, dim=-1, eps=1e-6):
        return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

    def aggregate_dual_tokens(self, q_feat, S_res, organ_indices):
        w_g = F.softmax(self.global_token_aggregator(q_feat), dim=1)
        token_global = torch.sum(q_feat * w_g, dim=1)
        token_global = token_global + 0.1 * self.organ_global_tokens[organ_indices]
        token_global = self.safe_normalize(token_global, dim=-1)

        w_r = F.softmax(self.residual_token_aggregator(S_res), dim=1)
        token_residual = torch.sum(S_res * w_r, dim=1)
        token_residual = self.safe_normalize(token_residual, dim=-1)
        return token_global, token_residual

    # ------------------------------------------------------------------
    def calculate_label_guided_loss(self, gate_mean, organ_labels):
        """Push gate→0 for normal organs; maintain margin for abnormal."""
        flat = organ_labels.view(-1).float()
        nm = flat == 0
        am = flat == 1
        zero = torch.zeros((), device=gate_mean.device, dtype=gate_mean.dtype)

        m_n = gate_mean[nm].mean() if nm.any() else zero
        loss_normal = m_n if nm.any() else zero
        m_a = gate_mean[am].mean() if am.any() else zero

        loss_margin = (
            F.relu(self.label_margin - (m_a - m_n))
            if (nm.any() and am.any()) else zero
        )
        return (self.label_weight_normal * loss_normal
                + self.label_weight_abnormal * loss_margin)

    # ------------------------------------------------------------------
    def _gate_and_fuse(self, q_feat):
        """Core: gate -> S_res / L_norm -> fused feature."""
        gate_logit = self.gate_mlp(q_feat)              # [*, T, 1]
        gate = torch.sigmoid(gate_logit)                 # [*, T, 1]
        S_res = q_feat * gate
        L_norm = q_feat * (1 - gate)
        combined = torch.cat([L_norm, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)
        return final_feat, S_res, gate_logit, gate

    # ------------------------------------------------------------------
    def process_all_organs_batched(self, ifeat, masks, organ_labels=None):
        B, num_organs, N = masks.shape
        device = ifeat.device
        L = ifeat.size(-1)

        ifeat_flat = (
            ifeat.unsqueeze(1)
            .expand(-1, num_organs, -1, -1)
            .reshape(B * num_organs, N, L)
        )
        masks_flat = masks.reshape(B * num_organs, N)
        safe_mask = masks_flat.clone()
        safe_mask[masks_flat.sum(dim=1) == 0, 0] = 1.0

        organ_indices = torch.arange(num_organs, device=device).repeat(B)
        q_feat = self.shared_qformer(
            tgt=self.all_organ_queries[organ_indices],
            memory=ifeat_flat,
            memory_key_padding_mask=(safe_mask == 0),
        )

        final_feat, S_res, gate_logit, gate = self._gate_and_fuse(q_feat)

        # ── aux losses ──
        gate_mean = gate.squeeze(-1).mean(dim=1)          # [B*O]
        loss_s = (
            self.calculate_label_guided_loss(gate_mean, organ_labels)
            if organ_labels is not None
            else torch.tensor(0.0, device=device)
        )
        total_aux_loss = loss_s + self.gate_sparsity_weight * gate.mean()

        # ── reshape ──
        all_final = final_feat.view(B, num_organs, self.num_tokens, self.feat_dim)
        all_res = S_res.view(B, num_organs, self.num_tokens, self.feat_dim)

        all_global = None
        if self.use_global_token:
            tg, _ = self.aggregate_dual_tokens(q_feat, S_res, organ_indices)
            all_global = tg.view(B, num_organs, self.feat_dim)

        metrics = {
            "gate": gate.squeeze(-1).view(B, num_organs, self.num_tokens),
            "gate_logit": gate_logit.squeeze(-1).view(B, num_organs, self.num_tokens),
        }
        return all_final, all_global, all_res, total_aux_loss, metrics

    # ------------------------------------------------------------------
    def process_single_organ(self, ifeat, mask, organ_idx):
        B, N, _ = ifeat.shape
        device = ifeat.device

        safe_mask = mask.clone()
        safe_mask[mask.sum(dim=1) == 0, 0] = 1.0

        q_feat = self.shared_qformer(
            tgt=self.all_organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1),
            memory=ifeat,
            memory_key_padding_mask=(safe_mask == 0),
        )

        final_feat, S_res, gate_logit, gate = self._gate_and_fuse(q_feat)
        aux_loss = self.gate_sparsity_weight * gate.mean()

        token_global = token_residual = None
        if self.use_global_token:
            idx_t = torch.full((B,), organ_idx, device=device, dtype=torch.long)
            token_global, token_residual = self.aggregate_dual_tokens(
                q_feat, S_res, idx_t,
            )

        extra = {
            "gate": gate.squeeze(-1).detach(),
            "gate_logit": gate_logit.squeeze(-1).detach(),
        }
        return final_feat, token_global, token_residual, aux_loss, extra

    # ------------------------------------------------------------------
    def _process_mixed_batch(self, ifeat, mask, organ_indices):
        B = ifeat.size(0)
        device = ifeat.device
        results = {}
        for oi in torch.unique(organ_indices).tolist():
            sel = organ_indices == oi
            results[oi] = self.process_single_organ(ifeat[sel], mask[sel], oi)

        T, D = self.num_tokens, self.feat_dim
        all_ff = torch.zeros(B, T, D, device=device)
        all_tg = torch.zeros(B, D, device=device) if self.use_global_token else None
        all_tr = torch.zeros(B, D, device=device) if self.use_global_token else None
        total_loss = torch.tensor(0.0, device=device)

        for oi, (ff, tg, tr, al, _) in results.items():
            sel = organ_indices == oi
            all_ff[sel] = ff
            if self.use_global_token and tg is not None:
                all_tg[sel] = tg
                all_tr[sel] = tr
            total_loss = total_loss + al * sel.sum()
        return all_ff, all_tg, all_tr, total_loss / max(B, 1), {}

    # ------------------------------------------------------------------
    def forward(self, ifeat, mask=None, idx=None, abnorm_label=None,
                multi_organ_mode=False):
        if multi_organ_mode or (mask is not None and mask.dim() == 3):
            assert mask is not None and mask.dim() == 3
            return self.process_all_organs_batched(ifeat, mask, abnorm_label)

        assert idx is not None and mask is not None and mask.dim() == 2
        B = ifeat.size(0)
        organ_indices = idx.view(B).long()
        if torch.all(organ_indices == organ_indices[0]):
            return self.process_single_organ(ifeat, mask, organ_indices[0].item())
        return self._process_mixed_batch(ifeat, mask, organ_indices)


class OrganFeatureRefiner_v2_LRMR_dual_v5(nn.Module):
    """
    v5 independent implementation.

    Key points:
    1) No inheritance from previous classes.
    2) Keep rank-margin label-guided residual supervision.
    3) Keep real Q-Former cross-attention capture for query->memory projection.
    4) In batched mode with organ labels, reconstruction loss is computed only on normal organs.
    5) Keep output signatures compatible with v4 usage.
    """

    def __init__(
        self,
        num_organs=10,
        feat_dim=768,
        num_tokens=32,
        dict_atoms=256,
        top_k=64,
        nhead=8,
        num_layers=2,
        use_global_token=True,
        label_margin=0.02,
        label_weight_normal=0.5,
        label_weight_abnormal=0.1,
        dict_within_weight=0.05,
        dict_cross_weight=0.02,
        query_div_weight=0.1,
        anomaly_threshold=1.5,
    ):
        super().__init__()
        self.num_organs = num_organs
        self.num_tokens = num_tokens
        self.feat_dim = feat_dim
        self.top_k = top_k
        self.dict_atoms = dict_atoms
        self.use_global_token = use_global_token

        self.label_margin = label_margin
        self.label_weight_normal = label_weight_normal
        self.label_weight_abnormal = label_weight_abnormal
        self.dict_within_weight = dict_within_weight
        self.dict_cross_weight = dict_cross_weight
        self.query_div_weight = query_div_weight
        self.anomaly_threshold = anomaly_threshold

        self.all_organ_queries = nn.Parameter(torch.randn(num_organs, num_tokens, feat_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
        )
        self.shared_qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.all_organ_dicts = nn.Parameter(torch.randn(num_organs, dict_atoms, feat_dim))
        self.dict_scale_logit = nn.Parameter(torch.zeros(1))
        self.temperature_logit = nn.Parameter(torch.ones(1) * (-2.0))

        nn.init.orthogonal_(self.all_organ_dicts)
        nn.init.orthogonal_(self.all_organ_queries)

        self.fusion_layer = nn.Linear(feat_dim * 2, feat_dim)
        self.layer_norm = nn.LayerNorm(feat_dim)

        if self.use_global_token:
            self.global_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1),
            )
            self.residual_token_aggregator = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1),
            )
            self.organ_global_tokens = nn.Parameter(torch.randn(num_organs, feat_dim))
            nn.init.normal_(self.organ_global_tokens, std=0.02)

        self._cross_attn_capture_enabled = False
        self._enable_cross_attn_capture()

    def safe_normalize(self, x, dim=-1, eps=1e-6):
        norm = x.norm(p=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    def aggregate_dual_tokens(self, final_feat, S_res, organ_indices):
        attn_scores_g = self.global_token_aggregator(final_feat)
        attn_weights_g = F.softmax(attn_scores_g, dim=1)
        token_global = torch.sum(final_feat * attn_weights_g, dim=1)

        organ_prior = self.organ_global_tokens[organ_indices]
        token_global = token_global + 0.1 * organ_prior
        token_global = self.safe_normalize(token_global, dim=-1)

        attn_scores_r = self.residual_token_aggregator(S_res)
        attn_weights_r = F.softmax(attn_scores_r, dim=1)
        token_residual = torch.sum(S_res * attn_weights_r, dim=1)
        token_residual = self.safe_normalize(token_residual, dim=-1)

        return token_global, token_residual

    def _enable_cross_attn_capture(self):
        if self._cross_attn_capture_enabled:
            return

        for layer in self.shared_qformer.layers:
            layer.last_self_attn = None
            layer.last_cross_attn = None

            def _patched_sa_block(this, x, attn_mask, key_padding_mask, *args, **kwargs):
                attn_out, attn_w = this.self_attn(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                this.last_self_attn = attn_w
                return this.dropout1(attn_out)

            def _patched_mha_block(this, x, mem, attn_mask, key_padding_mask, *args, **kwargs):
                attn_out, attn_w = this.multihead_attn(
                    x,
                    mem,
                    mem,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                this.last_cross_attn = attn_w
                return this.dropout2(attn_out)

            layer._sa_block = _patched_sa_block.__get__(layer, layer.__class__)
            layer._mha_block = _patched_mha_block.__get__(layer, layer.__class__)

        self._cross_attn_capture_enabled = True

    def _collect_cross_attn_maps(self, B, T, N, safe_mask):
        dtype = self.all_organ_queries.dtype
        device = safe_mask.device
        eye = torch.eye(T, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        valid = safe_mask.unsqueeze(1).to(dtype)

        q2m_attn_last = None
        rollout_per_layer = []
        query_rollout = eye

        for layer in self.shared_qformer.layers:
            self_w = getattr(layer, "last_self_attn", None)
            cross_w = getattr(layer, "last_cross_attn", None)

            if self_w is not None:
                if self_w.dim() == 3:
                    self_w = self_w.unsqueeze(1)
                if self_w.dim() == 4 and self_w.size(0) == B and self_w.size(-2) == T and self_w.size(-1) == T:
                    self_attn = self_w.mean(dim=1)
                    self_attn = self_attn + eye
                    self_attn = self_attn / self_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    query_rollout = torch.bmm(self_attn, query_rollout)

            if cross_w is None:
                continue
            if cross_w.dim() == 3:
                cross_w = cross_w.unsqueeze(1)
            if not (cross_w.dim() == 4 and cross_w.size(0) == B and cross_w.size(-2) == T and cross_w.size(-1) == N):
                continue

            cross_attn = cross_w.mean(dim=1)
            cross_attn = cross_attn * valid
            cross_attn = cross_attn / cross_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            q2m_attn_last = cross_attn
            rollout_per_layer.append(torch.bmm(query_rollout, cross_attn))

        if q2m_attn_last is None:
            return None, None

        q2m_attn_rollout = torch.stack(rollout_per_layer, dim=0).mean(dim=0)
        q2m_attn_rollout = q2m_attn_rollout * valid
        q2m_attn_rollout = q2m_attn_rollout / q2m_attn_rollout.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return q2m_attn_last, q2m_attn_rollout

    def _dictionary_regularization(self, device):
        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)

        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_within = torch.mean((gram_matrix - identity) ** 2)

        organ_proto = all_d_norm.mean(dim=1)
        gram_org = torch.matmul(organ_proto, organ_proto.t())
        identity_org = torch.eye(self.num_organs, device=device)
        loss_cross = torch.mean((gram_org - identity_org) ** 2)

        return loss_within, loss_cross

    def calculate_label_guided_loss(self, S_res, organ_labels):
        flat_labels = organ_labels.view(-1).float()
        sample_energy = torch.mean(torch.abs(S_res), dim=(1, 2))

        normal_mask = (flat_labels == 0)
        abnormal_mask = (flat_labels == 1)

        zero = torch.zeros((), device=S_res.device, dtype=S_res.dtype)

        if normal_mask.any():
            mean_normal = sample_energy[normal_mask].mean()
            loss_normal = mean_normal
        else:
            mean_normal = zero
            loss_normal = zero

        if abnormal_mask.any():
            mean_abnormal = sample_energy[abnormal_mask].mean()
        else:
            mean_abnormal = zero

        if normal_mask.any() and abnormal_mask.any():
            loss_rank_margin = F.relu(self.label_margin - (mean_abnormal - mean_normal))
        else:
            loss_rank_margin = zero

        return self.label_weight_normal * loss_normal + self.label_weight_abnormal * loss_rank_margin

    def process_single_organ(self, ifeat, mask, organ_idx):
        B, N, _ = ifeat.shape
        device = ifeat.device

        self._enable_cross_attn_capture()

        mask_sum = mask.sum(dim=1)
        safe_mask = mask.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        batch_queries = self.all_organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1)
        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat,
            memory_key_padding_mask=key_padding_mask,
        )

        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        batch_dicts = self.all_organ_dicts[organ_idx].unsqueeze(0).expand(B, -1, -1)
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature

        k = min(self.top_k, self.dict_atoms)
        topk_val, topk_idx = torch.topk(similarity, k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()

        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale

        S_res = q_feat - L_rec.detach()

        token_residual_energy = torch.mean(torch.abs(S_res), dim=-1)
        median_energy = token_residual_energy.median(dim=1, keepdim=True)[0]
        mad = (token_residual_energy - median_energy).abs().median(dim=1, keepdim=True)[0]
        mad = mad.clamp_min(1e-6)
        token_anomaly_score = (token_residual_energy - median_energy) / mad
        token_anomaly_prob = torch.sigmoid(token_anomaly_score)
        token_anomaly_pred = (token_anomaly_score > self.anomaly_threshold).float()

        k_anom = min(4, self.num_tokens)
        _, topk_anomaly_idx = torch.topk(token_anomaly_score, k=k_anom, dim=1)

        q2m_attn_last, q2m_attn_rollout = self._collect_cross_attn_maps(
            B=B,
            T=self.num_tokens,
            N=N,
            safe_mask=safe_mask,
        )

        if q2m_attn_rollout is None:
            ifeat_norm = self.safe_normalize(ifeat, dim=-1)
            q2m_logits = torch.bmm(q_feat_norm, ifeat_norm.transpose(1, 2)) / temperature
            q2m_logits = q2m_logits.masked_fill((safe_mask == 0).unsqueeze(1), -1e9)
            q2m_attn_rollout = F.softmax(q2m_logits, dim=-1)
            q2m_attn_last = q2m_attn_rollout

        query_anom_weight = F.softmax(token_anomaly_score, dim=1)
        memory_anomaly_score = torch.bmm(query_anom_weight.unsqueeze(1), q2m_attn_rollout).squeeze(1)
        memory_anomaly_score = memory_anomaly_score * safe_mask

        valid_cnt = safe_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mem_mean = memory_anomaly_score.sum(dim=1, keepdim=True) / valid_cnt
        mem_var = (((memory_anomaly_score - mem_mean) * safe_mask) ** 2).sum(dim=1, keepdim=True) / valid_cnt
        mem_std = torch.sqrt(mem_var + 1e-6)
        memory_anomaly_z = (memory_anomaly_score - mem_mean) / mem_std
        memory_anomaly_prob = torch.sigmoid(memory_anomaly_z)
        memory_anomaly_pred = ((memory_anomaly_z > self.anomaly_threshold) & (safe_mask > 0)).float()

        k_mem = min(16, N)
        memory_anomaly_z_masked = memory_anomaly_z.masked_fill(safe_mask == 0, -1e9)
        topk_memory_anomaly_score, topk_memory_anomaly_idx = torch.topk(memory_anomaly_z_masked, k=k_mem, dim=1)

        loss_reconstruction = F.l1_loss(L_rec, q_feat.detach(), reduction="mean")
        loss_residual_sparsity = torch.mean(torch.abs(S_res))
        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms

        all_d_norm = self.safe_normalize(self.all_organ_dicts, dim=-1)
        gram_matrix = torch.matmul(all_d_norm, all_d_norm.transpose(1, 2))
        identity = torch.eye(self.dict_atoms, device=device).unsqueeze(0)
        loss_dict_ortho = torch.mean((gram_matrix - identity) ** 2)

        aux_loss = (
            1.0 * loss_reconstruction
            + 0.1 * loss_coeff_sparsity
            # + 0.1 * loss_residual_sparsity
            # + 0.1 * loss_dict_ortho
            # + 0.0 * loss_query_div
        )

        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        token_global = None
        token_residual = None
        if self.use_global_token:
            indices = torch.tensor([organ_idx] * B, device=device)
            token_global, token_residual = self.aggregate_dual_tokens(final_feat, S_res, indices)

        extra_info = {
            "token_residual_energy": token_residual_energy.detach(),
            "token_anomaly_score": token_anomaly_score.detach(),
            "token_anomaly_prob": token_anomaly_prob.detach(),
            "token_anomaly_pred": token_anomaly_pred.detach(),
            "topk_anomaly_idx": topk_anomaly_idx.detach(),
            "memory_anomaly_score": memory_anomaly_score.detach(),
            "memory_anomaly_z": memory_anomaly_z.detach(),
            "memory_anomaly_prob": memory_anomaly_prob.detach(),
            "memory_anomaly_pred": memory_anomaly_pred.detach(),
            "topk_memory_anomaly_idx": topk_memory_anomaly_idx.detach(),
            "topk_memory_anomaly_score": topk_memory_anomaly_score.detach(),
            "qformer_cross_attn_last": q2m_attn_last.detach(),
            "qformer_cross_attn_rollout": q2m_attn_rollout.detach(),
        }

        return final_feat, token_global, token_residual, aux_loss, extra_info

    def process_all_organs_batched(self, ifeat, masks, organ_labels=None):
        B, num_organs, N = masks.shape
        device = ifeat.device
        L = ifeat.size(-1)

        ifeat_expanded = ifeat.unsqueeze(1).expand(-1, num_organs, -1, -1)
        ifeat_flat = ifeat_expanded.reshape(B * num_organs, N, L)
        masks_flat = masks.reshape(B * num_organs, N)

        mask_sum = masks_flat.sum(dim=1)
        safe_mask = masks_flat.clone()
        safe_mask[mask_sum == 0, 0] = 1.0
        key_padding_mask = (safe_mask == 0)

        organ_indices = torch.arange(num_organs, device=device).repeat(B)
        batch_queries = self.all_organ_queries[organ_indices]

        q_feat = self.shared_qformer(
            tgt=batch_queries,
            memory=ifeat_flat,
            memory_key_padding_mask=key_padding_mask,
        )

        q_feat_norm = self.safe_normalize(q_feat, dim=-1)
        q_sim_matrix = torch.bmm(q_feat_norm, q_feat_norm.transpose(1, 2))
        mask_off_diag = ~torch.eye(self.num_tokens, dtype=torch.bool, device=device)
        off_diagonal = q_sim_matrix[:, mask_off_diag].view(B * num_organs, self.num_tokens, -1)
        loss_query_div = torch.mean(off_diagonal ** 2)

        batch_dicts = self.all_organ_dicts[organ_indices]
        d_norm = self.safe_normalize(batch_dicts, dim=-1)

        temperature = torch.exp(self.temperature_logit).clamp(0.05, 0.5)
        similarity = torch.bmm(q_feat_norm, d_norm.transpose(1, 2)) / temperature

        k = min(self.top_k, self.dict_atoms)
        topk_val, topk_idx = torch.topk(similarity, k, dim=-1)
        topk_val_stable = topk_val - topk_val.max(dim=-1, keepdim=True)[0].detach()

        fill_value = -1e9 if similarity.dtype != torch.float16 else -10000.0
        sparse_mask = torch.full_like(similarity, fill_value)
        sparse_mask.scatter_(-1, topk_idx, topk_val_stable)
        alpha = F.softmax(sparse_mask, dim=-1)

        dict_scale = torch.sigmoid(self.dict_scale_logit) + 0.5
        L_rec = torch.bmm(alpha, batch_dicts) * dict_scale

        S_res = q_feat - L_rec.detach()
        S_res_dict = q_feat.detach() - L_rec

        if organ_labels is not None:
            flat_labels = organ_labels.view(-1).to(device)
            normal_mask = (flat_labels == 0)
            rec_error = torch.mean(torch.abs(L_rec - q_feat.detach()), dim=(1, 2))
            if normal_mask.any():
                loss_reconstruction = rec_error[normal_mask].mean()
            else:
                loss_reconstruction = torch.zeros((), device=device, dtype=q_feat.dtype)
        else:
            loss_reconstruction = F.l1_loss(L_rec, q_feat.detach(), reduction="mean")

        loss_coeff_sparsity = torch.mean(torch.sum(alpha > 0.01, dim=-1).float()) / self.dict_atoms

        loss_dict_within, loss_dict_cross = self._dictionary_regularization(device)

        if organ_labels is not None:
            loss_s = self.calculate_label_guided_loss(S_res_dict, organ_labels)
        else:
            loss_s = torch.zeros_like(loss_dict_within)

        total_aux_loss = (
            1.0 * loss_reconstruction
            + 0.1 * loss_coeff_sparsity
            # + self.dict_within_weight * loss_dict_within
            # + self.dict_cross_weight * loss_dict_cross
            # + self.query_div_weight * loss_query_div
            + loss_s
        )

        combined = torch.cat([L_rec, S_res], dim=-1)
        final_feat = self.fusion_layer(combined)
        final_feat = self.layer_norm(final_feat + q_feat)

        all_final_feats = final_feat.view(B, num_organs, self.num_tokens, self.feat_dim)
        all_residual_feats = S_res.view(B, num_organs, self.num_tokens, self.feat_dim)

        all_global_tokens = None
        if self.use_global_token:
            token_global, _ = self.aggregate_dual_tokens(q_feat, S_res, organ_indices)
            all_global_tokens = token_global.view(B, num_organs, self.feat_dim)

        return all_final_feats, all_global_tokens, all_residual_feats, total_aux_loss, {}

    def _process_mixed_batch(self, ifeat, mask, organ_indices):
        B = ifeat.size(0)
        device = ifeat.device

        all_final_feats = []
        all_global_tokens = []
        all_residual_tokens = []
        all_aux_losses = []
        extra_infos = []
        unique_organs = torch.unique(organ_indices)

        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            organ_ifeat = ifeat[organ_mask_bool]
            organ_mask = mask[organ_mask_bool]

            final_feat, t_glob, t_res, aux_loss, extra_info = self.process_single_organ(
                organ_ifeat,
                organ_mask,
                organ_idx.item(),
            )

            all_final_feats.append(final_feat)
            all_global_tokens.append(t_glob)
            all_residual_tokens.append(t_res)
            all_aux_losses.append(aux_loss)
            extra_infos.append(extra_info)

        final_feat_full = torch.zeros(B, self.num_tokens, self.feat_dim, device=device)
        token_global_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None
        token_residual_full = torch.zeros(B, self.feat_dim, device=device) if self.use_global_token else None

        idx_counter = 0
        for organ_idx in unique_organs:
            organ_mask_bool = (organ_indices == organ_idx)
            final_feat_full[organ_mask_bool] = all_final_feats[idx_counter]
            if self.use_global_token:
                token_global_full[organ_mask_bool] = all_global_tokens[idx_counter]
                token_residual_full[organ_mask_bool] = all_residual_tokens[idx_counter]
            idx_counter += 1

        total_aux_loss = torch.stack(all_aux_losses).mean()
        return final_feat_full, token_global_full, token_residual_full, total_aux_loss, extra_infos

    def forward(self, ifeat, mask=None, idx=None, abnorm_label=None, multi_organ_mode=False):
        if multi_organ_mode or (mask is not None and mask.dim() == 3):
            assert mask is not None and mask.dim() == 3
            return self.process_all_organs_batched(ifeat, mask, abnorm_label)

        assert idx is not None
        assert mask is not None and mask.dim() == 2

        B = ifeat.size(0)
        organ_indices = idx.view(B).long()

        if torch.all(organ_indices == organ_indices[0]):
            return self.process_single_organ(ifeat, mask, organ_indices[0].item())

        return self._process_mixed_batch(ifeat, mask, organ_indices)