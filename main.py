import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import wandb
from model.model import Model
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from model.components import sampler
import math
import time
wandb.init(project='PPCL')
wandb.watch_called = False 
batch_reduction = 0
config = wandb.config
config.visual_input = 512
config.text_input = 512  
config.task = 'all'
config.dataset = '300k'     
config.batch_size = 256 // (2 ** batch_reduction)
config.test_batch_size = 2048
config.epochs = 50            
config.lr = 2e-3 / math.sqrt(2) ** batch_reduction
config.weight_decay = 1e-4          
config.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu') 
config.seed = 0  
config.cat_dim = 32
config.cat_num = 9
config.hid_dim = 128
config.num_dim = 11 * 3
config.lamda = 0.9 
config.neck = 4 
config.cross_num = 4 
config.temp = 0.1 
config.clusters = 40
config.w_post, config.w_user, config.w_pop = 1, 3, 0.1



class SMPDataset(torch.utils.data.Dataset):
    def __init__(self, cat_data, num_data, visual_feat, text_feat, label, dlabel):
        self.cat_data = cat_data
        self.num_data = num_data
        self.visual_feat = visual_feat
        self.text_feat = text_feat
        self.label = label
        self.dlabel = dlabel

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # choose category data to form category_features
        category_data = self.cat_data[index]
        # choose continuous data to form continuous_features
        continuous_data = self.num_data[index]
        # choose text data to form text_features
        category_data = category_data.tolist()
        continuous_data = continuous_data.tolist()
        cat_feat = torch.tensor(category_data).int().to(config.device)
        num_feat = torch.tensor(continuous_data).float().to(config.device)
        num_feat = torch.cat([num_feat, torch.sqrt(num_feat), torch.square(num_feat)], dim=-1)
        visual_feat = torch.tensor(self.visual_feat[index]).float().to(config.device)
        text_feat = torch.tensor(self.text_feat[index]).float().to(config.device)
        label = torch.tensor(self.label[index]).float().to(config.device)
        dlabel = torch.tensor(self.dlabel[index]).int().to(config.device)
        return cat_feat, num_feat, visual_feat, text_feat, label, dlabel

        
def train(model, train_loader, optimizer, criterion, scaler):
    model.train()
    running_loss, steps = 0.0, 0
    post_l, user_l, pop_l = 0.0, 0.0, 0.0
    # use tqdm to show the progress bar
    for i, (cat_feat, num_feat, visual_feat, text_feat, labels, dlabel) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
        # Zero the parameter gradients
        optimizer.zero_grad()
        labels = labels.unsqueeze(1)
        # print(inputs.shape, attenion_mask.shape, labels.shape)
        with autocast():
            outputs, (post_loss, user_loss, pop_loss) = model(cat_feat, num_feat, visual_feat, text_feat, dlabel, labels)
            reg_loss = criterion(outputs, labels)
        loss = (1 - config.lamda) * (config.w_post * post_loss + config.w_user * user_loss + config.w_pop * pop_loss) + config.lamda * reg_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        post_l += post_loss.item()
        user_l += user_loss.item()
        pop_l += pop_loss.item()
        steps += 1

    return running_loss / steps, post_l / steps, user_l / steps, pop_l / steps


def validate(model, test_loader):
    model.eval()
    with torch.no_grad():
        test_outputs = []
        test_labels = []
        for _, (cat_feat, num_feat, visual_feat, text_feat, labels, dlabel) in enumerate(tqdm(test_loader, desc='Validating', leave=False)):
            outputs, (_,_,_) = model(cat_feat, num_feat, visual_feat, text_feat, dlabel, labels)
            test_outputs.append(outputs.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
        test_outputs = np.concatenate(test_outputs)
        # check if test_outputs contains nan or inf
        test_labels = np.concatenate(test_labels)
        test_loss = mean_squared_error(test_labels, test_outputs)
        test_mae = mean_absolute_error(test_labels, test_outputs)
        test_src, _ = stats.spearmanr(test_labels, test_outputs, axis=None)
    return test_loss, test_mae, test_src, test_outputs


def main(config):
    torch.manual_seed(config.seed) # pytorch random seed
    np.random.seed(config.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load the data
    data = pd.read_csv('./dataset/train_data_preprocessed_{}.csv'.format(config.dataset))
    #data['id'] = data.index
    visual_feat = np.load('./dataset/image_clip.npy')
    text_feat = np.load('./dataset/text_clip.npy')
    visual_feat = visual_feat[data['id']]
    text_feat = text_feat[data['id']]
    # use k-means to cluster the data['followercount'], user cluster_id to make a new column
    data['cluster_id'] = KMeans(n_clusters=config.clusters, random_state=config.seed).fit_predict(data['followerCount'].values.reshape(-1, 1))
    # compute the number of col 'vuid''s unique values
    label_encoder = LabelEncoder()
    data['user_id'] = label_encoder.fit_transform(data['user_id'])
    data['ispro'] = label_encoder.fit_transform(data['ispro'])
    data['canbuy_pro'] = label_encoder.fit_transform(data['canbuy_pro'])
    data['ispublic'] = label_encoder.fit_transform(data['ispublic'])
    data['tz_id'] = label_encoder.fit_transform(data['tz_id'])
    data['tz_offset'] = label_encoder.fit_transform(data['tz_offset'])
    data['post_hour'] = label_encoder.fit_transform(data['post_hour'])
    data['post_day'] = label_encoder.fit_transform(data['post_day'])
    data['post_month'] = label_encoder.fit_transform(data['post_month'])
    data['geoacc'] = label_encoder.fit_transform(data['geoacc'])
    data['category_id'] = label_encoder.fit_transform(data['category_id'])
    data['subcategory_id'] = label_encoder.fit_transform(data['subcategory_id'])
    data['concept_id'] = label_encoder.fit_transform(data['concept_id'])
    config.uid_dim = data['user_id'].nunique()
    config.ispro_dim = data['ispro'].nunique()
    config.canbuy_pro_dim = data['canbuy_pro'].nunique()
    config.ispublic_dim = data['ispublic'].nunique()
    config.tz_id_dim = data['tz_id'].nunique()
    config.tz_offset_dim = data['tz_offset'].nunique()
    config.post_hour_dim = data['post_hour'].nunique()
    config.post_day_dim = data['post_day'].nunique()
    config.post_month_dim = data['post_month'].nunique()
    config.geoacc_dim = data['geoacc'].nunique()
    continuous_cols = ['totalViews','totalTags','totalGeotagged','totalFaves','totalInGroup','photoCount','followerCount','followingCount','pfirst_year','pfirst_taken_year', 'geoacc']
    categorical_cols = ['user_id','ispro','canbuy_pro','ispublic','tz_id','tz_offset','post_hour','post_day','post_month', 'category_id', 'subcategory_id', 'concept_id']
    num_data = data[continuous_cols].values
    cat_data = data[categorical_cols].values

    train_num_data = num_data[:int(len(data) * 0.8)]
    val_num_data = num_data[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_num_data = num_data[int(len(data) * 0.9):]
    train_cat_data = cat_data[:int(len(data) * 0.8)]
    val_cat_data = cat_data[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_cat_data = cat_data[int(len(data) * 0.9):]
    train_visual_feat = visual_feat[:int(len(data) * 0.8)]
    val_visual_feat = visual_feat[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_visual_feat = visual_feat[int(len(data) * 0.9):]
    train_text_feat = text_feat[:int(len(data) * 0.8)]
    val_text_feat = text_feat[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_text_feat = text_feat[int(len(data) * 0.9):]
    train_label = data['label'].values[:int(len(data) * 0.8)]
    val_label = data['label'].values[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_label = data['label'].values[int(len(data) * 0.9):]
    train_dlabel = data['cluster_id'].values[:int(len(data) * 0.8)]
    val_dlable = data['cluster_id'].values[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_dlabel = data['cluster_id'].values[int(len(data) * 0.9):]
    
    min_max_scaler = preprocessing.MinMaxScaler()
    train_num_data = min_max_scaler.fit_transform(train_num_data)
    val_num_data = min_max_scaler.transform(val_num_data)
    test_num_data = min_max_scaler.transform(test_num_data)
    # Define the data loaders
    train_dataset = SMPDataset(train_cat_data, train_num_data, train_visual_feat, train_text_feat, train_label, train_dlabel)
    val_dataset = SMPDataset(val_cat_data, val_num_data, val_visual_feat, val_text_feat, val_label, val_dlable)
    test_dataset = SMPDataset(test_cat_data, test_num_data, test_visual_feat, test_text_feat, test_label, test_dlabel)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, #shuffle=True)
                              sampler=sampler(train_dataset, config.batch_size // 4, train_cat_data[:, 9], train_cat_data[:, 10], train_cat_data[:, 11]))
    val_loader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
    model = Model(config).to(config.device)
    # compute params num
    total_para = 0
    for param in model.parameters():
        total_para += np.prod(param.size())
    print('total parameter for the model: ', total_para)
    # Define the loss function and optimizer
    criterion = nn.MSELoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Train the model
    best_mae = float('inf')  # Initialize best MAE with a large value
    best_mse = float('inf')
    best_src = float('-inf')
    best_outputs = None
    patience = 5  # Number of epochs to wait for improvement
    counter = 0  # Counter to keep track of epochs without improvement
    best_model = None
    scaler = GradScaler()
    #wandb.watch(model, log="all")
    for epoch in range(config.epochs):
        train_st = time.time()
        train_loss, post_loss, user_loss, pop_loss = train(model, train_loader, optimizer, criterion, scaler)
        scheduler.step()
        train_ed = time.time()
        print('train time: ', train_ed - train_st)
        val_loss, val_mae, val_src, _ = validate(model, val_loader)
        test_st = time.time()
        test_loss, test_mae, test_src, outputs = validate(model, test_loader)
        test_ed = time.time()
        print('test time: ', test_ed - test_st)

        if best_mae > test_mae:
            best_mae = round(test_mae, 3)
            best_mse = round(test_loss, 3)
            best_src = round(test_src, 3)
            best_outputs = outputs
            counter = 0
        else:
            counter += 1
        print('train_loss: {}, val_loss: {}, test loss: {}, test_mae: {}, test_src: {} \n'.format(round(train_loss, 3), round(val_loss, 3), round(test_loss, 3), round(test_mae, 3), round(test_src, 3)))

        if counter >= patience:
            print(f"Early stopping: No improvement for {patience} epochs.")
            break
    # save best_mae to log.txt
    with open('log.txt', 'a') as f:
        print('best_mae: {}, best_mse: {}, best_src: {} \n'.format(best_mae, best_mse, best_src))
        f.write('task: {}, dataset: {}, mae: {}, mse: {}, src: {} \n'.format(config.task, config.dataset, best_mae, best_mse, best_src))
    # np.save('./dataset/test_outputs.npy', best_outputs)
    # torch.save(best_model, "model.h5")
    # wandb.save('model.h5')

if __name__ == '__main__':
    main(config)
