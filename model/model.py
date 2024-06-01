import torch
import torch.nn as nn
from model.components import MLP, CrossNet, SupConLoss, HMLC, RnCLoss

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # Define embedding layer
        self.uid_embed = nn.Embedding(config.uid_dim, config.cat_dim)
        self.ispro_embed = nn.Embedding(config.ispro_dim, config.cat_dim)
        self.canbuy_pro_embed = nn.Embedding(config.canbuy_pro_dim, config.cat_dim)
        self.ispublic_embed = nn.Embedding(config.ispublic_dim, config.cat_dim)
        self.tz_id_embed = nn.Embedding(config.tz_id_dim, config.cat_dim)
        self.tz_offset_embed = nn.Embedding(config.tz_offset_dim, config.cat_dim)
        self.post_hour_embed = nn.Embedding(config.post_hour_dim, config.cat_dim)
        self.post_day_embed = nn.Embedding(config.post_day_dim, config.cat_dim)
        self.post_month_embed = nn.Embedding(config.post_month_dim, config.cat_dim)
        self.geoacc_embed = nn.Embedding(config.geoacc_dim, config.cat_dim)
        self.hid_dim = config.hid_dim
        self.cross_num = config.cross_num
        self.num_pairs = config.cat_num * (config.cat_num - 1) // 2
        self.align_dim = 512
        # Define MLP layers with dropout
        self.cross_dim = config.cat_dim * config.cat_num + self.num_pairs + config.num_dim * (self.cross_num + 1)
        self.cross_net = CrossNet(config.cat_dim, config.num_dim, self.cross_dim, self.num_pairs, self.cross_num)
        self.user_layer = nn.Linear(self.cross_dim * 2, self.align_dim // 2)
        self.user_mlp = MLP(config.cat_dim * config.cat_num + config.num_dim, self.align_dim // 2, self.align_dim // 2)
        
        self.visual_mlp = MLP(config.visual_input, self.align_dim // config.neck, self.align_dim)
        self.text_mlp = MLP(config.text_input, self.align_dim // config.neck, self.align_dim)
        self.v_t_mlp = MLP(self.align_dim * 2, self.align_dim, self.align_dim)
        self.predictor = MLP(self.align_dim * 2, self.align_dim, self.align_dim)
        self.output_layer = nn.Linear(self.align_dim, 1)
        self.contra_post = HMLC(temperature=config.temp, base_temperature=config.temp)
        self.contra_user = SupConLoss(temperature=config.temp, base_temperature=config.temp)
        self.contra_pop = SupConLoss(temperature=config.temp, base_temperature=config.temp)
        self.contra_rnc = RnCLoss(temperature=config.temp)
        self.task = config.task
    def user_encoder(self, cat_embed, num_feat, user_feat_cross):
        user_feat_mlp = self.user_mlp(torch.cat([cat_embed, num_feat], dim=-1))
        user_feat = torch.cat([user_feat_cross, user_feat_mlp], dim=-1)
    
        return user_feat
    
    def post_encoder(self, visual_feat, text_feat):
        visual_feat = self.visual_mlp(visual_feat)
        visual_feat = visual_feat / torch.norm(visual_feat, dim=-1, keepdim=True)
        text_feat = self.text_mlp(text_feat)
        text_feat = text_feat / torch.norm(text_feat, dim=-1, keepdim=True)
        v_t_feat = torch.cat([visual_feat, text_feat], dim=-1)
        v_t_feat = self.v_t_mlp(v_t_feat)
        return v_t_feat

    def forward(self, cat_feat, num_feat, visual_feat, text_feat, dlabel, label):
        # Embed categorical features
        uid_embed = self.uid_embed(cat_feat[:, 0])
        ispro_embed = self.ispro_embed(cat_feat[:, 1])
        canbuy_pro_embed = self.canbuy_pro_embed(cat_feat[:, 2])
        ispublic_embed = self.ispublic_embed(cat_feat[:, 3])
        tz_id_embed = self.tz_id_embed(cat_feat[:, 4])
        tz_offset_embed = self.tz_offset_embed(cat_feat[:, 5])
        post_hour_embed = self.post_hour_embed(cat_feat[:, 6])
        post_day_embed = self.post_day_embed(cat_feat[:, 7])
        post_month_embed = self.post_month_embed(cat_feat[:, 8])
        # product embedding
        embedding_list = [uid_embed, ispro_embed, canbuy_pro_embed, ispublic_embed, tz_id_embed, tz_offset_embed, post_hour_embed, post_day_embed, post_month_embed]
        cat_embed = torch.cat(embedding_list, dim=-1)
        user_feat_cross = self.cross_net(embedding_list, num_feat)
        user_feat_cross = self.user_layer(user_feat_cross)
        # user encoder loss
        user_feat = self.user_encoder(cat_embed, num_feat, user_feat_cross)
        user_feat_arg = self.user_encoder(cat_embed, num_feat, user_feat_cross)
        user_feat_arg = torch.stack([user_feat_arg, user_feat], dim=1)
        

        # MLP for visual features
        post_feat = self.post_encoder(visual_feat, text_feat)
        post_feat_arg = self.post_encoder(visual_feat, text_feat)
        post_feat_arg = torch.stack([post_feat_arg, post_feat], dim=1)
        # post encoder loss
        post_label = torch.cat([cat_feat[:, [9]], cat_feat[:, [10]], cat_feat[:, [11]]], dim=1)
        
        # Concatenate user and visual/text features
        combine_feat = torch.cat([user_feat, post_feat], dim=-1)
        combine_feat_detach = combine_feat.detach()
        # MLP for prediction
        pop_feat = self.predictor(combine_feat)
        pop_feat_arg_1 = self.predictor(combine_feat_detach)
        pop_feat_arg_2 = self.predictor(combine_feat_detach)
        pop_feat_arg = torch.stack([pop_feat_arg_1, pop_feat_arg_2], dim=1)
        # pop encoder loss
        pop_label = cat_feat[:, 0].unsqueeze(1)
        
        # Output layer
        output = self.output_layer(pop_feat)
        if self.training:
            loss_user = self.contra_user(user_feat_arg, dlabel)
            loss_post = self.contra_post(post_feat_arg, post_label)
            loss_pop = self.contra_pop(pop_feat_arg, pop_label)

        else:
            loss_user = 0
            loss_post = 0
            loss_pop = 0

        return output, (loss_post, loss_user, loss_pop)
