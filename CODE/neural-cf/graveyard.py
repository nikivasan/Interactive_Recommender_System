# import torch
# from torch.autograd import Variable
# from tqdm import tqdm
# from tensorboardX import SummaryWriter
# from utils import save_checkpoint, use_optimizer
# from metrics import MetronAtK
# import gc
# import psutil
# import pandas as pd
# import numpy as np
# import os 


# class Engine(object):
#     """Meta Engine for training & evaluating NCF model

#     Note: Subclass should implement self.model !
#     """

#     def __init__(self, config):
#         self.config = config  # model configuration
#         self._metron = MetronAtK(top_k=10)
#         self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
#         self._writer.add_text('config', str(config), 0)
#         self.opt = use_optimizer(self.model, config)
#         # choose loss function based on feedback type
#         if self.config.get('is_explicit', False):
#             self.crit = torch.nn.MSELoss()
#         else:
#             self.crit = torch.nn.BCELoss()
#         self.user_seen_items = None
#         self.all_items = None
#         self.ratings_df = None

#         print("Engine initialized with config:", self.config)

    
#     def log_memory(self, epoch):
#         ram = psutil.virtual_memory().used / 1024**3
#         gpu = torch.cuda.memory_allocated() / 1024**2
#         print(f"[Epoch {epoch}] RAM: {ram:.2f} GB | GPU: {gpu:.2f} MB")
#         gc.collect()
#         torch.cuda.empty_cache()

#     def train_single_batch(self, users, items, ratings):
#         assert hasattr(self, 'model'), 'Please specify the exact model !'
#         if self.config['use_cuda'] is True:
#             users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
#         self.opt.zero_grad()
#         ratings_pred = self.model(users, items)

#         # print("Predictions:", ratings_pred[:10].detach().cpu().numpy())
#         # print("Targets:", ratings[:10].detach().cpu().numpy())
#         # print("Ratings min/max:", ratings.min().item(), ratings.max().item())

#         loss = self.crit(ratings_pred.view(-1), ratings)
#         loss.backward()
#         self.opt.step()
#         loss = loss.item()
#         return loss

#     def train_an_epoch(self, train_loader, epoch_id):
#         assert hasattr(self, 'model'), 'Please specify the exact model !'
#         self.model.train()
#         total_loss = 0
#         for batch_id, batch in enumerate(train_loader):
#             assert isinstance(batch[0], torch.LongTensor)
#             user, item, rating = batch[0], batch[1], batch[2]
#             rating = rating.float()
#             loss = self.train_single_batch(user, item, rating)
#             print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
#             total_loss += loss
#         avg_loss = total_loss / len(train_loader)
#         self._writer.add_scalar('model/loss', avg_loss, epoch_id)
#         self.log_memory(epoch_id)
        
#         if self.config['use_cuda']:
#             torch.cuda.empty_cache()
#             gc.collect()
        
#         return avg_loss
        

#     def evaluate(self, evaluate_data, epoch_id):
#         assert hasattr(self, 'model'), 'Please specify the exact model !'
#         self.model.eval()
#         with torch.no_grad():
#             # explicit feedback case 
#             if self.config.get('is_explicit', False):
#                 test_users, test_items, true_ratings = evaluate_data
#                 print(f"[Eval] Evaluating on {len(set(test_users))} unique users")
#                 if self.config['use_cuda']:
#                     test_users = test_users.cuda()
#                     test_items = test_items.cuda()
#                     true_ratings = true_ratings.cuda()

#                 preds = self.model(test_users, test_items).view(-1).cpu().detach().numpy()
#                 preds = preds * self.config['max_rating']

#                 # For Metron
#                 self._metron.set_explicit_subjects(test_users.cpu().tolist(), test_items.cpu().tolist(), true_ratings.cpu().numpy() * self.config['max_rating'])
#                 self._metron.set_predictions(preds)
#                 mae = self._metron.cal_mae()
#                 rmse = self._metron.cal_rmse()

#                 self._writer.add_scalar('performance/RMSE', rmse, epoch_id)
#                 self._writer.add_scalar('performance/MAE', mae, epoch_id)
#                 print(f"[Evaluating Epoch {epoch_id}] RMSE = {rmse:.4f} | MAE = {mae:.4f}")
                
#                 del test_users, test_items, true_ratings, preds
        
#             # implicit feedback case
#             else: 
#                 test_users, test_items = evaluate_data[0], evaluate_data[1]
#                 negative_users, negative_items = evaluate_data[2], evaluate_data[3]
                
#                 if self.config['use_cuda'] is True:
#                     test_users = test_users.cuda()
#                     test_items = test_items.cuda()
#                     negative_users = negative_users.cuda()
#                     negative_items = negative_items.cuda()

#                 if self.config['use_bachify_eval'] == False:    
#                     test_scores = self.model(test_users, test_items)
#                     negative_scores = self.model(negative_users, negative_items)
#                 else:
#                     test_scores = []
#                     negative_scores = []
#                     bs = self.config['batch_size']
#                     for start_idx in range(0, len(test_users), bs):
#                         end_idx = min(start_idx + bs, len(test_users))
#                         batch_test_users = test_users[start_idx:end_idx]
#                         batch_test_items = test_items[start_idx:end_idx]
#                         test_scores.append(self.model(batch_test_users, batch_test_items))
#                     for start_idx in tqdm(range(0, len(negative_users), bs)):
#                         end_idx = min(start_idx + bs, len(negative_users))
#                         batch_negative_users = negative_users[start_idx:end_idx]
#                         batch_negative_items = negative_items[start_idx:end_idx]
#                         negative_scores.append(self.model(batch_negative_users, batch_negative_items))
#                     test_scores = torch.concatenate(test_scores, dim=0)
#                     negative_scores = torch.concatenate(negative_scores, dim=0)


#                     if self.config['use_cuda'] is True:
#                         test_users = test_users.cpu()
#                         test_items = test_items.cpu()
#                         test_scores = test_scores.cpu()
#                         negative_users = negative_users.cpu()
#                         negative_items = negative_items.cpu()
#                         negative_scores = negative_scores.cpu()
#                     self._metron.subjects = [test_users.data.view(-1).tolist(),
#                                         test_items.data.view(-1).tolist(),
#                                         test_scores.data.view(-1).tolist(),
#                                         negative_users.data.view(-1).tolist(),
#                                         negative_items.data.view(-1).tolist(),
#                                         negative_scores.data.view(-1).tolist()]
#                 hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
#                 self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
#                 self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
#                 print('[Evaluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
                
#                 del test_users, test_items, test_scores
#                 del negative_users, negative_items, negative_scores
            
#             if self.config['use_cuda']:
#                 torch.cuda.empty_cache()
#             gc.collect()

#             # return {'rmse': rmse, 'mae': mae} if self.config.get('is_explicit', False) else {'hit_ratio': hit_ratio, 'ndcg': ndcg}
#             return (rmse, mae) if self.config.get('is_explicit', False) else (hit_ratio, ndcg)

#     def save(self, alias, epoch_id, **kwargs):
#         """
#         Save model with filename adapted based on feedback type.

#         kwargs:
#             - For implicit: pass hit_ratio and ndcg
#             - For explicit: pass rmse and mae
#         """
#         run_number = self.config.get('run_number', 0)

#         if self.config.get('is_explicit', False):
#             rmse = kwargs.get('rmse', 0.0)
#             mae = kwargs.get('mae', 0.0)
#             metric_str = f"RMSE{rmse:.4f}_MAE{mae:.4f}"
#             feedback_type = "explicit"
#         else:
#             hit_ratio = kwargs.get('hit_ratio', 0.0)
#             ndcg = kwargs.get('ndcg', 0.0)
#             metric_str = f"HR{hit_ratio:.4f}_NDCG{ndcg:.4f}"
#             feedback_type = "implicit"

#         # Save the model
#         model_path = self.config['model_dir'].format(
#             feedback_type, 
#             run_number,
#             alias, 
#             epoch_id, 
#             metric_str
#         )
#         save_checkpoint(self.model, model_path)
#         # model_dir = self.config['model_dir'].format(self.config['run_number'],alias, epoch_id, metric_str)
#         # save_checkpoint(self.model, model_dir)
    
#     def generate_recommendations(self, epoch_id=None, top_n=10, user_ids=None, full_ranking=True):
#         if user_ids is None:
#             user_ids = list(self.user_seen_items.keys()) 
        
#         if full_ranking:
#             self.generate_full_rankings_for_users(user_ids, epoch_id)
#         else:
#             self.generate_top_n(top_n=top_n, epoch_id=epoch_id)

#     def generate_top_n(self, top_n=10, epoch_id=None):
#         self.model.eval()
#         rows = []
#         movies = pd.read_csv('data/ml-latest-small/movies.csv')
#         for user in range(self.config['num_users']):
#             seen = self.user_seen_items.get(user, set())
#             unseen = np.array(list(set(self.all_items) - seen))
#             if len(unseen) == 0:
#                 continue
#             users = torch.tensor([user] * len(unseen))
#             items = torch.tensor(unseen)
#             if self.config['use_cuda']:
#                 users = users.cuda()
#                 items = items.cuda()
#             with torch.no_grad():
#                 scores = self.model(users, items).cpu().numpy().flatten()

#             top_indices = np.argsort(scores)[-top_n:][::-1]
#             for idx in top_indices:
#                 rows.append((user, unseen[idx], float(scores[idx])))
        
#         recs_df = pd.DataFrame(rows, columns=['user', 'itemId', 'score'])
#         recs_df.drop_duplicates(subset=['user', 'itemId'], inplace=True)
#         recs_df['itemId'] = recs_df['itemId'].astype(int)
#         movies['itemId'] = movies['itemId'].astype(int)
#         recs_df = recs_df.merge(movies[['itemId', 'title', 'genre']], on='itemId', how='left')
        
#         filename_epoch = f"recommendations-out/run{self.config['run_number']}/topn_epoch_{epoch_id}.csv"
#         filename_final = f"recommendations-out/run{self.config['run_number']}/final_topn.csv"

#         recs_df.to_csv(filename_epoch, index=False)

#         if epoch_id == self.config['num_epoch'] - 1:
#             recs_df.to_csv(filename_final, index=False)
#             print(f"Saved Top-{top_n} recommendations to {filename_final}")

#         print(f"Saved Top-{top_n} recommendations to {filename_epoch}")
    
#     def generate_full_rankings_for_users(self, user_ids=None, epoch_id=None):
#         """
#         Generate predictions for all unseen items for a subset of users.
#         Save full ranked lists, not just top-N.
#         """
#         self.model.eval()
#         rows = []
#         movies = pd.read_csv('data/ml-latest-small/movies.csv')
#         movies['movieId'] = movies['movieId'].astype(int)
#         movies.rename(columns={'movieId': 'itemId'}, inplace=True)

#         if user_ids is None:
#             user_ids = list(self.user_seen_items.keys())

#         for user in user_ids:
#             seen = self.user_seen_items.get(user, set())
#             unseen = np.array(list(set(self.all_items) - seen))
#             if len(unseen) == 0:
#                 continue

#             users = torch.tensor([user] * len(unseen))
#             items = torch.tensor(unseen)

#             if self.config['use_cuda']:
#                 users = users.cuda()
#                 items = items.cuda()

#             with torch.no_grad():
#                 scores = self.model(users, items).cpu().numpy().flatten()
#                 if self.config.get('is_explicit', False):
#                     scores = scores * self.config['max_rating']

#             for item, score in zip(unseen, scores):
#                 rows.append((user, item, float(score)))

#         recs_df = pd.DataFrame(rows, columns=['user', 'itemId', 'score'])
#         recs_df['itemId'] = recs_df['itemId'].astype(int)

#         recs_df = recs_df.merge(movies[['itemId', 'title', 'genres']], on='itemId', how='left')
        
#         if hasattr(self, 'ratings_df'):
#             recs_df = recs_df.merge(
#                 self.ratings_df.rename(columns={'userId': 'user'}),
#                 how='left',
#                 on=['user', 'itemId']
#             )
#             recs_df.rename(columns={'rating': 'ground_truth'}, inplace=True)

#         # Sort by score
#         recs_df = recs_df.sort_values(by=['user', 'score'], ascending=[True, False])

#         # Save
#         directory = f"recommendations-out/run{self.config['run_number']}"
#         os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

#         # Build the filename inside that directory
#         if epoch_id is not None:
#             filename = f"{directory}/full_rankings_users_{epoch_id}.csv"
#         else:
#             filename = f"{directory}/final_full_rankings.csv"
#         recs_df.to_csv(filename, index=False)
#         print(f"Saved full ranked recommendations for users to {filename}")
    



import pandas as pd
# import numpy as np
# from gmf import GMFEngine
# from mlp import MLPEngine
# from neumf import NeuMFEngine
# from data import SampleGenerator
# import matplotlib.pyplot as plt
# import time 


# def print_parameter_summary(model):
#         """
#         Prints the number of trainable parameters in the given model, 
#         broken down by layer, and shows the total.
#         """
#         print("\nModel Parameter Summary:")
#         total_params = 0
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 num_params = param.numel()
#                 print(f"  Layer '{name}': {num_params} params")
#                 total_params += num_params
#         print(f"Total Trainable Parameters: {total_params}\n")

# gmf_config = {'alias': 'gmf_factor8neg4-implict',
#               'num_epoch': 25, # 200
#               'batch_size': 1024,
#               # 'optimizer': 'sgd',
#               # 'sgd_lr': 1e-3,
#               # 'sgd_momentum': 0.9,
#               # 'optimizer': 'rmsprop',
#               # 'rmsprop_lr': 1e-3,
#               # 'rmsprop_alpha': 0.99,
#               # 'rmsprop_momentum': 0,
#               'optimizer': 'adam',
#               'adam_lr': 1e-3,
#               'num_users': 6040,
#               'num_items': 3706,
#               'latent_dim': 8,
#               'num_negative': 4,
#               'l2_regularization': 0,  # 0.01
#               'weight_init_gaussian': True,
#               'use_cuda': True,
#               'use_bachify_eval': False,
#               'device_id': 0,
#               'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

# mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
#               'num_epoch': 25, # 200
#               'batch_size': 256,  # 1024,
#               'optimizer': 'adam',
#               'adam_lr': 1e-3,
#               'num_users': 6040,
#               'num_items': 3706,
#               'latent_dim': 8,
#               'num_negative': 4,
#               'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
#               'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
#               'weight_init_gaussian': True,
#               'use_cuda': False,
#               'use_bachify_eval': False,
#               'device_id': 0,
#               'pretrain': False,
#               'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
#               'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}
# ## changes made:
# # hyperparams: 
# # increased learning rate
# # deep layers [16, 64, 32, 16, 8] -> [64, 128, 64, 32, 16]
# # deeper latent dim 8 -> 32
# # larger regularization param 1e-6 -> 1e-4
# # architectural changes:
# # added eval on subset of users capability
# # added explicit rating capabilities (RMSE/MAE vs NDCG/HitRatio)
# # added generate full recommendations for each user or only top n capability
# # added run number to automatically store epoch-level recs and final recs + plots 


# neumf_config = {'alias': 'neumf_factor8neg4',
#                 'num_epoch': 8, 
#                 'batch_size': 1024,
#                 'optimizer': 'adam',
#                 'adam_lr': 1e-3,
#                 'num_users': None,
#                 'num_items': None,
#                 'latent_dim_mf': 32,
#                 'latent_dim_mlp': 32,
#                 'num_negative': 4,
#                 'layers': [64, 128, 64, 32, 16],  # layers[0] is the concat of latent user vector & latent item vector [16, 64, 32, 16, 8]
#                 'l2_regularization': 0.00001,
#                 'weight_init_gaussian': True,
#                 'use_cuda': True,
#                 'use_bachify_eval': True,
#                 'device_id': 0,
#                 'pretrain': False,
#                 #### added params
#                 'generate_full_recs': True,  
#                 'eval_user_subset': None, 
#                 'is_explicit': True,
#                 'run_number' : '8',
#                 #################
#                 'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
#                 'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
#                 'model_dir': 'checkpoints/{}/run{}/{}_Epoch{}_{}.model'
#                 }

# # Load Data
# # ml1m_dir = 'data/ml-latest-small/ratings.dat'
# # ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
# ml1m_rating = pd.read_csv('data/ml-latest-small/ratings.csv')
# ml1m_rating.rename(columns={'userId': 'uid', 'movieId': 'mid'}, inplace=True)

# # Reindex
# user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
# user_id['userId'] = np.arange(len(user_id))
# ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')

# item_id = ml1m_rating[['mid']].drop_duplicates()
# item_id['itemId'] = np.arange(len(item_id))
# ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')

# # Set num_users and num_items dynamically
# num_users = ml1m_rating['userId'].nunique()
# num_items = ml1m_rating['itemId'].nunique()
# neumf_config['num_users'] = num_users
# neumf_config['num_items'] = num_items

# print('Number of users: ', num_users)
# print('Number of items: ', num_items)

# ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

# # DataLoader for training
# sample_generator = SampleGenerator(ratings=ml1m_rating)
# neumf_config['max_rating'] = sample_generator.max_rating
# evaluate_data = sample_generator.evaluate_data
# user_seen_items = ml1m_rating.groupby('userId')['itemId'].apply(set).to_dict()
# all_items = list(ml1m_rating['itemId'].unique())

# # Initialize Model 
# config = neumf_config
# engine = NeuMFEngine(config)
# engine.ratings_df = ml1m_rating[['userId', 'itemId', 'rating']]
# engine.user_seen_items = user_seen_items
# engine.all_items = all_items

# print_parameter_summary(engine.model)

# # Training Loop
# losses = []
# rmses = []
# maes = []

# for epoch in range(config['num_epoch']):
#     print('Epoch {} starts !'.format(epoch))
#     print('-' * 80)
    
#     train_loader = sample_generator.instance_a_train_loader(
#         config['num_negative'], config['batch_size']
#         )
    
#     loss = engine.train_an_epoch(train_loader, epoch_id=epoch)
#     losses.append(loss)
#     metrics = engine.evaluate(evaluate_data, epoch_id=epoch)

#     engine.generate_recommendations(
#         epoch_id=epoch,
#         top_n=10,
#         user_ids=config.get('eval_user_subset'),
#         full_ranking=config.get('generate_full_recs', False)
#     )

#     if config['is_explicit']:
#         rmse, mae = metrics
#         rmses.append(rmse)
#         maes.append(mae)
#         print(f"[Epoch {epoch}] RMSE = {rmse:.4f} | MAE = {mae:.4f}")
#         engine.save(neumf_config['alias'], epoch, rmse=rmse, mae=mae)
#     else:
#         hit_ratio, ndcg = metrics
#         print(f"[Epoch {epoch}] HR@10 = {hit_ratio:.4f} | NDCG@10 = {ndcg:.4f}")
#         engine.save(neumf_config['alias'], epoch, hit_ratio=hit_ratio, ndcg=ndcg)

# # Final recommendations
# engine.generate_recommendations(
#     epoch_id=neumf_config['num_epoch'] - 1,
#     top_n=10,
#     user_ids=neumf_config.get('eval_user_subset'),
#     full_ranking=neumf_config.get('generate_full_recs', False)
# )

# print("\nTraining complete.")


# # Generate Plots
# print("\nGenerating plots...")
# epochs = list(range(1, config['num_epoch'] + 1))

# plt.figure()
# plt.plot(epochs, losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Average Loss')
# plt.title('Average Training Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'recommendations-out/run{config["run_number"]}/loss_plot.png')
# plt.close()

# plt.figure()
# plt.plot(epochs, rmses, label='RMSE')
# plt.plot(epochs, maes, label='MAE')
# plt.xlabel('Epoch')
# plt.ylabel('Score')
# plt.title('RMSE and MAE Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'recommendations-out/run{config["run_number"]}/metrics_plot.png')
# plt.close()

# print("Saved plots to output directory.")

