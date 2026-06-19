import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class DSERegularizerLoss(nn.Module):
    """GT-adaptive-k feature-DSE regularizer for dense ViT patch tokens.

    GT masks are used only to choose the number of feature clusters. Cluster
    assignment itself follows the original DSE style and is obtained from
    k-means on feature tokens.
    """

    def __init__(self,
                 loss_weight=1.0,
                 normalize=True,
                 patch_size=None,
                 purity_thr=0.6,
                 min_tokens_per_class=4,
                 max_tokens_per_class=None,
                 use_intra_image=True,
                 min_loss=None,
                 max_loss=None,
                 kmeans_iters=20,
                 kmeans_tol=1e-4,
                 ignore_index=255,
                 num_classes=19,
                 **kwargs):
        super().__init__()
        del kwargs
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.patch_size = patch_size
        self.purity_thr = purity_thr
        self.min_tokens_per_class = min_tokens_per_class
        self.max_tokens_per_class = max_tokens_per_class
        self.use_intra_image = use_intra_image
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.kmeans_iters = kmeans_iters
        self.kmeans_tol = kmeans_tol
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def estimator(self, representations, method_type):
        assert len(representations.shape) == 2, 'Input should be a 2D tensor'
        num_samples, feat_dim = representations.shape
        if num_samples == 0 or feat_dim == 0:
            return torch.tensor(0.0, device=representations.device)

        if method_type == 'effective_rank':
            singular_values = torch.linalg.svdvals(representations)
            probs = singular_values / (singular_values.sum() + 1e-12)
            probs = probs + 1e-12
            entropy = -torch.sum(probs * torch.log(probs))
            denom = min(num_samples, feat_dim)
            if denom > 0:
                rank = torch.exp(entropy) / denom
            else:
                rank = torch.tensor(0.0, device=representations.device)
        elif method_type == 'centered_singular_sum':
            centered = representations - representations.mean(dim=0, keepdim=True)
            singular_values = torch.linalg.svdvals(centered)
            max_dim = max(num_samples - 1, feat_dim)
            denom = math.sqrt(max_dim) if max_dim > 0 else 1.0
            rank = singular_values.sum() / denom
        else:
            raise ValueError(f'Unsupported estimator type: {method_type}')

        return rank

    def _validate_gt(self, gt_semantic_seg):
        if gt_semantic_seg is None:
            raise ValueError(
                'DSERegularizerLoss is GT-mask-guided and requires '
                '`gt_semantic_seg` in forward().'
            )
        if gt_semantic_seg.dim() == 4:
            if gt_semantic_seg.shape[1] != 1:
                raise ValueError(
                    'gt_semantic_seg with 4 dims must have shape [B, 1, H, W], '
                    f'got {tuple(gt_semantic_seg.shape)}'
                )
            gt_semantic_seg = gt_semantic_seg[:, 0]
        elif gt_semantic_seg.dim() != 3:
            raise ValueError(
                'gt_semantic_seg must have shape [B, 1, H, W] or [B, H, W], '
                f'got {tuple(gt_semantic_seg.shape)}'
            )
        return gt_semantic_seg.long()

    def _build_gt_cluster_stats(self, gt_semantic_seg, expected_tokens):
        if self.patch_size is None:
            raise ValueError(
                'DSERegularizerLoss requires an explicit `patch_size` so GT '
                'patchify matches the backbone token grid.'
            )
        patch_size = int(self.patch_size)
        if patch_size <= 0:
            raise ValueError(f'patch_size must be positive, got {self.patch_size}')

        gt_semantic_seg = self._validate_gt(gt_semantic_seg)
        batch_size, height, width = gt_semantic_seg.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        num_tokens = grid_h * grid_w
        if num_tokens != expected_tokens:
            raise ValueError(
                'Token count does not match GT patch grid: '
                f'N={expected_tokens}, grid={grid_h}x{grid_w}, '
                f'patch_size={patch_size}, gt_shape={(height, width)}'
            )

        covered_h = grid_h * patch_size
        covered_w = grid_w * patch_size
        gt_semantic_seg = gt_semantic_seg[:, :covered_h, :covered_w]
        patches = F.unfold(
            gt_semantic_seg.unsqueeze(1).float(),
            kernel_size=patch_size,
            stride=patch_size,
        ).long()
        patches = patches.reshape(batch_size, patch_size * patch_size, num_tokens)

        valid_pixel_mask = (
            (patches != self.ignore_index)
            & (patches >= 0)
            & (patches < self.num_classes)
        )
        valid_counts = valid_pixel_mask.sum(dim=1)
        token_valid = valid_counts > 0

        local_ks = []
        for batch_idx in range(batch_size):
            valid_pixels = valid_pixel_mask[batch_idx]
            valid_token_count = int(token_valid[batch_idx].sum().item())
            if valid_token_count == 0:
                local_ks.append(0)
                continue

            gt_classes = patches[batch_idx][valid_pixels].unique()
            class_count = int(gt_classes.numel())
            local_ks.append(min(class_count, valid_token_count))

        return token_valid, local_ks

    def _kmeans_plus_plus_init(self, inputs, num_clusters):
        num_samples = inputs.shape[0]
        first_idx = torch.randint(
            0, num_samples, (1,), device=inputs.device)
        centroids = inputs[first_idx]

        for _ in range(num_clusters - 1):
            dist_to_centroids = torch.cdist(inputs, centroids)
            min_dist = dist_to_centroids.min(dim=1).values
            probs = min_dist / min_dist.sum().clamp_min(1e-12)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                probs = torch.ones_like(probs) / probs.numel()
            chosen_idx = torch.multinomial(probs, 1)
            centroids = torch.cat([centroids, inputs[chosen_idx]], dim=0)

        return centroids

    def _kmeans(self, inputs, num_clusters):
        num_samples = inputs.shape[0]
        if num_samples == 0 or num_clusters <= 0:
            return None, None

        num_clusters = min(int(num_clusters), num_samples)
        with torch.no_grad():
            cluster_inputs = inputs.detach().float()
            centroids = self._kmeans_plus_plus_init(
                cluster_inputs, num_clusters)

            for _ in range(int(self.kmeans_iters)):
                distances = torch.cdist(cluster_inputs, centroids)
                labels = distances.argmin(dim=1)

                new_centroids = []
                for cluster_idx in range(num_clusters):
                    cluster_mask = labels == cluster_idx
                    if cluster_mask.any():
                        new_centroids.append(
                            cluster_inputs[cluster_mask].mean(dim=0))
                    else:
                        fallback_idx = torch.randint(
                            0, num_samples, (1,), device=inputs.device)
                        new_centroids.append(
                            cluster_inputs[fallback_idx].squeeze(0))
                new_centroids = torch.stack(new_centroids, dim=0)

                shift = (new_centroids - centroids).pow(2).sum().sqrt()
                centroids = new_centroids
                if shift.item() < self.kmeans_tol:
                    break

            distances = torch.cdist(cluster_inputs, centroids)
            labels = distances.argmin(dim=1)

        return centroids.detach(), labels.detach()

    def _zero_loss(self, representations):
        return representations.sum() * 0.0

    def _sample_class_tokens(self, class_repr):
        if self.max_tokens_per_class is None:
            return class_repr

        max_tokens = int(self.max_tokens_per_class)
        if max_tokens <= 0 or class_repr.shape[0] <= max_tokens:
            return class_repr

        perm = torch.randperm(class_repr.shape[0], device=class_repr.device)
        return class_repr[perm[:max_tokens]]

    def compute_dse_loss(self, representations, gt_semantic_seg):
        batch_size, num_tokens, feat_dim = representations.shape
        token_valid, local_ks = self._build_gt_cluster_stats(
            gt_semantic_seg,
            expected_tokens=num_tokens,
        )
        if token_valid.shape[0] != batch_size:
            raise ValueError(
                'Batch size mismatch between representations and gt_semantic_seg: '
                f'{batch_size} vs {token_valid.shape[0]}'
            )
        token_valid = token_valid.to(representations.device)

        if self.normalize:
            representations = F.normalize(representations, dim=-1)

        valid_repr = representations[token_valid]
        if valid_repr.shape[0] == 0:
            return self._zero_loss(representations)

        avg_l2 = torch.norm(
            valid_repr - valid_repr.mean(dim=0, keepdim=True),
            dim=-1,
            p=2,
        ).mean()
        denom = avg_l2 + 1e-12
        device = representations.device

        if self.use_intra_image:
            intra_image_terms = []
            for batch_idx in range(batch_size):
                image_repr = representations[batch_idx][token_valid[batch_idx]]
                local_k = local_ks[batch_idx]
                centroids, cluster_labels = self._kmeans(image_repr, local_k)
                if cluster_labels is None:
                    continue

                image_terms = []
                for cluster_idx in range(int(centroids.shape[0])):
                    cluster_repr = image_repr[cluster_labels == cluster_idx]
                    if cluster_repr.shape[0] > 1:
                        image_terms.append(self.estimator(
                            cluster_repr, 'centered_singular_sum'))

                if image_terms:
                    intra_image_terms.append(
                        torch.mean(torch.stack(image_terms)))

            if intra_image_terms:
                m_intra_image = torch.mean(torch.stack(intra_image_terms)) / denom
            else:
                m_intra_image = torch.tensor(0.0, device=device)

        global_k = min(int(sum(local_ks)), int(valid_repr.shape[0]))
        centroids, global_labels = self._kmeans(valid_repr, global_k)

        intra_batch_terms = []
        if global_labels is not None:
            for cluster_idx in range(int(centroids.shape[0])):
                cluster_repr = valid_repr[global_labels == cluster_idx]
                if cluster_repr.shape[0] > 1:
                    intra_batch_terms.append(
                        self.estimator(cluster_repr, 'centered_singular_sum')
                    )

        if intra_batch_terms:
            m_intra_batch = torch.mean(torch.stack(intra_batch_terms)) / denom
        else:
            m_intra_batch = torch.tensor(0.0, device=device)

        if centroids is not None and centroids.shape[0] > 1:
            dist_to_centroids = torch.cdist(valid_repr, centroids)
            own_cluster_mask = torch.zeros_like(
                dist_to_centroids, dtype=torch.bool, device=device)
            own_cluster_mask[
                torch.arange(valid_repr.shape[0], device=device),
                global_labels,
            ] = True
            min_dist_to_other = torch.min(
                dist_to_centroids.masked_fill(own_cluster_mask, float('inf')),
                dim=1,
            ).values
            m_inter = min_dist_to_other.mean() / denom
        else:
            m_inter = torch.tensor(0.0, device=device)

        if valid_repr.shape[0] > 1:
            shuffled = valid_repr[
                torch.randperm(valid_repr.size(0), device=device)
            ]
            m_dim = self.estimator(shuffled, 'effective_rank')
        else:
            m_dim = torch.tensor(0.0, device=device)

        if self.use_intra_image:
            return m_inter + m_dim - 0.5 * (m_intra_batch + m_intra_image)
        return m_inter + m_dim - 0.5 * m_intra_batch

    def forward(self, representations, gt_semantic_seg=None):
        if representations.dim() == 4:
            layer_losses = [
                self.compute_dse_loss(layer_repr.float(), gt_semantic_seg)
                for layer_repr in representations
            ]
            if not layer_losses:
                dse_value = self._zero_loss(representations)
            else:
                dse_value = torch.stack(layer_losses).mean()
        elif representations.dim() == 3:
            dse_value = self.compute_dse_loss(
                representations.float(),
                gt_semantic_seg,
            )
        else:
            raise ValueError(
                'DSERegularizerLoss expects [B, N, D] or [L, B, N, D], '
                f'got {tuple(representations.shape)}'
            )
        loss = -self.loss_weight * dse_value
        if self.min_loss is not None or self.max_loss is not None:
            loss = loss.clamp(min=self.min_loss, max=self.max_loss)
        return loss
