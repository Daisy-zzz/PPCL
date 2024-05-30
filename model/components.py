import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Sampler
import time
class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout1(x)
        x = self.act(self.fc2(x))
        x = self.dropout2(x)
        return x
    
class ProductLayer(nn.Module):
    def __init__(self, cat_dim, num_pair):
        super(ProductLayer, self).__init__()
        self.num_pair = num_pair
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(cat_dim, 1)), requires_grad=True)
             for i in range(self.num_pair)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(cat_dim, 1)), requires_grad=True)
             for i in range(self.num_pair)])
        
    def forward(self, inputs):
        embed_list = inputs
        num_inputs = len(embed_list)
        pair_list = []
        cross_output = []
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                pair_list.append([embed_list[i], embed_list[j]])
        for idx, pair in enumerate(pair_list):
            x_0, x_1 = pair[0].unsqueeze(2), pair[1].unsqueeze(2)
            x_2 = torch.tensordot(x_1, self.kernels[idx], dims=([1], [0]))
            x_2 = torch.matmul(x_0, x_2)
            x_2 = x_2 + self.bias[idx]
            cross_output.append(x_2.squeeze(2))
        cross_output = torch.cat(cross_output, dim=-1)
        return cross_output
    
class CrossNet(nn.Module):
    def __init__(self, cat_dim, input_dim, cross_dim, num_pair, num_layers=4):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(input_dim, 1)), requires_grad=True)
             for i in range(self.num_layers)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(input_dim, 1)), requires_grad=True)
             for i in range(self.num_layers)])
        self.product = ProductLayer(cat_dim, num_pair)
        # define cross for num_feat and embed
        self.mlp_kernel = nn.Parameter(nn.init.xavier_normal_(torch.empty(cross_dim, 1)),
                                           requires_grad=True)
        self.mlp_bias = nn.Parameter(nn.init.zeros_(torch.empty(cross_dim, 1)), requires_grad=True)

    def forward(self, embedding_list, num_feat):
        cat_embed = torch.cat(embedding_list, dim=-1)
        product = self.product(embedding_list)
        cat_embed = torch.cat([cat_embed, product], dim=-1)
        # cross part for num_feat
        cross_input = num_feat
        x_0 = cross_input.unsqueeze(2)
        cross_output = cross_input
        x_1 = x_0
        for i in range(self.num_layers):
            x_2 = torch.tensordot(x_1, self.kernels[i], dims=([1], [0]))
            x_2 = torch.matmul(x_0, x_2)
            x_2 = x_2 + self.bias[i]
            cross_output = torch.cat((x_2.squeeze(2), cross_output), dim=-1)
            x_1 = x_2
        # Concatenate processed features
        mlp_input = torch.cat([cat_embed, cross_output], dim=-1)

        # cross
        x_0 = mlp_input.unsqueeze(2)
        x_1 = torch.tensordot(x_0, self.mlp_kernel, dims=([1], [0]))
        x_1 = torch.matmul(x_0, x_1)
        x_1 = x_1 + self.mlp_bias
        mlp_input = torch.cat((x_1.squeeze(2), mlp_input), dim=-1)

        return mlp_input

'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import torch
import torch.nn as nn


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmc'):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        device = features.device
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))
        for l in range(1,labels.shape[1]):
            mask[:, labels.shape[1]-l:] = 0
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)
            layer_loss = self.sup_con_loss(features, mask=mask_labels)
            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                  1/(l)).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1/l).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            _, unique_indices = unique(layer_labels, dim=0)
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]
        return cumulative_loss / labels.shape[1]


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(anchor_feature, contrast_feature.T),
        #     self.temperature)
        # # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = torch.cosine_similarity(anchor_feature.unsqueeze(1), contrast_feature.unsqueeze(0), dim=-1) / self.temperature
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class sampler(Sampler):
    def __init__(self, data, batch_size, cat, subcat, concept):
        self.data = data
        self.batch_size = batch_size
        self.num_samples = len(self.data)
        self.cat = np.array(cat, dtype=np.int16)
        self.subcat = np.array(subcat, dtype=np.int16)
        self.concept = np.array(concept, dtype=np.int16)
        
    def __len__(self):
        return self.num_samples

    def __iter__(self):
        print('Start sampling......')
        sample_st = time.time()
        idxs = np.random.permutation(self.num_samples)
        first_batch = idxs[:self.batch_size]
        first_cat = self.cat[first_batch]
        first_subcat = self.subcat[first_batch]
        first_concept = self.concept[first_batch]
        rest_batch = idxs[self.batch_size:]
        rest_cat = self.cat[rest_batch]
        rest_subcat = self.subcat[rest_batch]
        rest_concept = self.concept[rest_batch]
        return_idx = []
        while len(rest_batch) >= self.batch_size * 4:
            cat_idx = []
            subcat_idx = []
            concept_idx = []
            for cat, subcat, concept in zip(first_cat, first_subcat, first_concept):
                cat_mask = (rest_cat == cat) & (rest_subcat != subcat)
                if np.sum(cat_mask) > 0:
                    cat_mask = np.where(cat_mask)[0]
                    cat_idx.append(rest_batch[np.random.choice(cat_mask)])
                subcat_mask = (rest_subcat == subcat) & (rest_concept != concept)
                if np.sum(subcat_mask) > 0:
                    subcat_mask = np.where(subcat_mask)[0]
                    subcat_idx.append(rest_batch[np.random.choice(subcat_mask)])
                concept_mask = (rest_concept == concept)
                if np.sum(concept_mask) > 0:
                    concept_mask = np.where(concept_mask)[0]
                    concept_idx.append(rest_batch[np.random.choice(concept_mask)])
            #combine three idx
            cat_idx = np.array(cat_idx)
            subcat_idx = np.array(subcat_idx)
            concept_idx = np.array(concept_idx)
            # caculate union of three idx
            union_idx = np.unique(np.concatenate([first_batch, cat_idx, subcat_idx, concept_idx]))
            # choose all elements in rest_batch that are not in union_idx to form a new rest_batch
            rest_batch = np.setdiff1d(rest_batch, union_idx)
            if len(union_idx) < self.batch_size * 4:
                deleted_idx = rest_batch[: self.batch_size * 4 - len(union_idx)]
                rest_batch = rest_batch[self.batch_size * 4 - len(union_idx):]
                union_idx = np.concatenate([union_idx, deleted_idx])
            return_idx.append(union_idx)
            # update first_batch and rest_batch
            first_batch = rest_batch[:self.batch_size]
            first_cat = self.cat[first_batch]
            first_subcat = self.subcat[first_batch]                                  
            first_concept = self.concept[first_batch]
            rest_batch = rest_batch[self.batch_size:]
            rest_cat = self.cat[rest_batch]
            rest_subcat = self.subcat[rest_batch]
            rest_concept = self.concept[rest_batch]

        if len(rest_batch) > 0:
            return_idx.append(rest_batch)

        return_idx = list(np.concatenate(return_idx).astype(np.int32))
        sample_ed = time.time()
        print('End sampling......, sampling time: ', sample_ed - sample_st)
        
        return iter(return_idx)
        

class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'cos':
            return - F.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='cos'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss
