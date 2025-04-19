import torch
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK
import gc
import psutil
import pandas as pd
import numpy as np
import os 
from diversity import (
    calculate_item_popularity, 
    generate_diverse_recommendations_by_popularity,
    calculate_diversity_metrics,
    analyze_recommendation_overlap,
    force_diversity_recommendations,
    enhanced_diverse_recommendations
)

from xQuAD import generate_smooth_xquad_recommendations


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # choose loss function based on feedback type
        if self.config.get('is_explicit', False):
            self.crit = torch.nn.MSELoss()
        else:
            self.crit = torch.nn.BCELoss()
        self.user_seen_items = None
        self.all_items = None
        self.ratings_df = None
        self.item_popularity = None

        print("Engine initialized with config:", self.config)

    def init_diversity_components(self):
        """Initialize diversity components if not already set"""
        if self.ratings_df is not None and self.item_popularity is None:
            print("Initializing diversity components...")
            # Calculate item popularity
            self.item_popularity = calculate_item_popularity(self.ratings_df)
            print(f"Calculated popularity for {len(self.item_popularity)} items")
    
    def log_memory(self, epoch):
        ram = psutil.virtual_memory().used / 1024**3
        gpu = torch.cuda.memory_allocated() / 1024**2
        print(f"[Epoch {epoch}] RAM: {ram:.2f} GB | GPU: {gpu:.2f} MB")
        gc.collect()
        torch.cuda.empty_cache()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()

        # Forward pass: model predictions
        ratings_pred = self.model(users, items)

        # Standard loss calculation
        loss = self.crit(ratings_pred.view(-1), ratings)
        
        # Apply popularity regularization if enabled
        if self.config.get('use_popularity_reg', False) and self.item_popularity is not None:
            pop_reg_strength = self.config.get('pop_reg_strength', 0.1)
            
            # Get popularity values for the current batch of items
            item_pop_values = torch.tensor(
                [self.item_popularity.get(item.item(), 0.0) for item in items],
                device=ratings_pred.device
            )
            
            # Squared regularization penalty: higher predicted ratings incur more penalty
            pop_penalty = pop_reg_strength * torch.mean(torch.pow(ratings_pred.view(-1), 2) * item_pop_values)
            loss += pop_penalty
            
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        
        # Initialize diversity components if not already done
        if self.config.get('use_popularity_reg', False):
            self.init_diversity_components()
            
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        self._writer.add_scalar('model/loss', avg_loss, epoch_id)
        self.log_memory(epoch_id)
        
        if self.config['use_cuda']:
            torch.cuda.empty_cache()
            gc.collect()
        
        return avg_loss
        
    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            # explicit feedback case 
            if self.config.get('is_explicit', False):
                test_users, test_items, true_ratings = evaluate_data
                print(f"[Eval] Evaluating on {len(set(test_users))} unique users")
                if self.config['use_cuda']:
                    test_users = test_users.cuda()
                    test_items = test_items.cuda()
                    true_ratings = true_ratings.cuda()

                preds = self.model(test_users, test_items).view(-1).cpu().detach().numpy()
                preds = preds * self.config['max_rating']

                # For Metron
                self._metron.set_explicit_subjects(test_users.cpu().tolist(), test_items.cpu().tolist(), true_ratings.cpu().numpy() * self.config['max_rating'])
                self._metron.set_predictions(preds)
                mae = self._metron.cal_mae()
                rmse = self._metron.cal_rmse()

                self._writer.add_scalar('performance/RMSE', rmse, epoch_id)
                self._writer.add_scalar('performance/MAE', mae, epoch_id)
                print(f"[Evaluating Epoch {epoch_id}] RMSE = {rmse:.4f} | MAE = {mae:.4f}")
                
                del test_users, test_items, true_ratings, preds
        
            # implicit feedback case
            else: 
                test_users, test_items = evaluate_data[0], evaluate_data[1]
                negative_users, negative_items = evaluate_data[2], evaluate_data[3]
                
                if self.config['use_cuda'] is True:
                    test_users = test_users.cuda()
                    test_items = test_items.cuda()
                    negative_users = negative_users.cuda()
                    negative_items = negative_items.cuda()

                if self.config['use_bachify_eval'] == False:    
                    test_scores = self.model(test_users, test_items)
                    negative_scores = self.model(negative_users, negative_items)
                else:
                    test_scores = []
                    negative_scores = []
                    bs = self.config['batch_size']
                    for start_idx in range(0, len(test_users), bs):
                        end_idx = min(start_idx + bs, len(test_users))
                        batch_test_users = test_users[start_idx:end_idx]
                        batch_test_items = test_items[start_idx:end_idx]
                        test_scores.append(self.model(batch_test_users, batch_test_items))
                    for start_idx in tqdm(range(0, len(negative_users), bs)):
                        end_idx = min(start_idx + bs, len(negative_users))
                        batch_negative_users = negative_users[start_idx:end_idx]
                        batch_negative_items = negative_items[start_idx:end_idx]
                        negative_scores.append(self.model(batch_negative_users, batch_negative_items))
                    test_scores = torch.concatenate(test_scores, dim=0)
                    negative_scores = torch.concatenate(negative_scores, dim=0)


                    if self.config['use_cuda'] is True:
                        test_users = test_users.cpu()
                        test_items = test_items.cpu()
                        test_scores = test_scores.cpu()
                        negative_users = negative_users.cpu()
                        negative_items = negative_items.cpu()
                        negative_scores = negative_scores.cpu()
                    self._metron.subjects = [test_users.data.view(-1).tolist(),
                                        test_items.data.view(-1).tolist(),
                                        test_scores.data.view(-1).tolist(),
                                        negative_users.data.view(-1).tolist(),
                                        negative_items.data.view(-1).tolist(),
                                        negative_scores.data.view(-1).tolist()]
                hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
                self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
                self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
                print('[Evaluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
                
                del test_users, test_items, test_scores
                del negative_users, negative_items, negative_scores
            
            if self.config['use_cuda']:
                torch.cuda.empty_cache()
            gc.collect()

            return (rmse, mae) if self.config.get('is_explicit', False) else (hit_ratio, ndcg)

    def save(self, alias, epoch_id, **kwargs):
        """
        Save model with filename adapted based on feedback type.

        kwargs:
            - For implicit: pass hit_ratio and ndcg
            - For explicit: pass rmse and mae
        """
        run_number = self.config.get('run_number', 0)

        if self.config.get('is_explicit', False):
            rmse = kwargs.get('rmse', 0.0)
            mae = kwargs.get('mae', 0.0)
            metric_str = f"RMSE{rmse:.4f}_MAE{mae:.4f}"
            feedback_type = "explicit"
        else:
            hit_ratio = kwargs.get('hit_ratio', 0.0)
            ndcg = kwargs.get('ndcg', 0.0)
            metric_str = f"HR{hit_ratio:.4f}_NDCG{ndcg:.4f}"
            feedback_type = "implicit"

        # Save the model
        model_path = self.config['model_dir'].format(
            feedback_type, 
            run_number,
            alias, 
            epoch_id, 
            metric_str
        )
        save_checkpoint(self.model, model_path)
    
    def generate_recommendations(self, epoch_id=None, top_n=10, user_ids=None, full_ranking=True, silent=True):
        """Generate recommendations using either standard or diverse method"""
        # Initialize diversity components if using diversity and components not already initialized
        if self.config.get('use_popularity_reg', False):
            self.init_diversity_components()
            
        if user_ids is None:
            user_ids = list(self.user_seen_items.keys()) 
        
        # Use diverse recommendations if enabled
        if self.config.get('use_popularity_reg', False) and self.item_popularity:
            # Choose diversity method based on configuration
            diversity_method = self.config.get('diversity_method', 'popularity')
            
            if diversity_method == 'popularity':
                return self.generate_diverse_recommendations_by_popularity(user_ids, epoch_id, top_n)
            elif diversity_method == 'xquad_smooth':
                return self.generate_smooth_xquad_recommendations(user_ids, epoch_id, top_n, silent)
            elif diversity_method == 'enhanced':
                recs_df = enhanced_diverse_recommendations(
                    self.model,
                    user_ids,
                    self.item_popularity,
                    self.user_seen_items,
                    self.all_items,
                    self.config,
                    pop_reg_strength=self.config.get('pop_reg_strength', 0.7),
                    long_tail_boost=self.config.get('long_tail_boost', 0.2),
                    top_k=top_n
                )
                
                # Calculate and save diversity metrics
                diversity_metrics = calculate_diversity_metrics(recs_df)
                overlap_metrics = analyze_recommendation_overlap(recs_df)
                
                directory = f"recommendations-out/run{self.config['run_number']}"
                os.makedirs(directory, exist_ok=True)
                
                filename = f"{directory}/enhanced_recs_epoch_{epoch_id}.csv" if epoch_id is not None else f"{directory}/final_enhanced_recs.csv"
                recs_df.to_csv(filename, index=False)
                
                metrics_df = pd.DataFrame({
                    **diversity_metrics,
                    **overlap_metrics,
                    'epoch': epoch_id if epoch_id is not None else 'final'
                }, index=[0])
                
                metrics_filename = f"{directory}/diversity_metrics_epoch_{epoch_id}.csv" if epoch_id is not None else f"{directory}/final_diversity_metrics.csv"
                metrics_df.to_csv(metrics_filename, index=False)
                
                return recs_df
            else:
                # Default to the existing method
                return self.generate_diverse_recommendations_by_popularity(user_ids, epoch_id, top_n)
        elif full_ranking:
            return self.generate_full_rankings_for_users(user_ids, epoch_id)
        else:
            return self.generate_top_n(top_n=top_n, epoch_id=epoch_id)


    def generate_smooth_xquad_recommendations(self, user_ids, epoch_id, top_n, silent):
        """Wrapper method that calls the xQuAD implementation"""
        from xQuAD import generate_smooth_xquad_recommendations
        return generate_smooth_xquad_recommendations(self, user_ids, epoch_id, top_n, silent)
    
    def generate_diverse_recommendations_by_popularity(self, user_ids=None, epoch_id=None, top_n=10):
        """Generate diverse recommendations for users using popularity regularization"""
        print("Generating popularity-regularized recommendations...")
        if user_ids is None:
            user_ids = list(self.user_seen_items.keys()) 
            
        # Generate diverse recommendations
        pop_reg_strength = self.config.get('pop_reg_strength', 0.2)
        recs_df = force_diversity_recommendations(
            self.model,
            user_ids,
            self.item_popularity,
            self.user_seen_items,
            self.all_items,
            self.config,
            pop_reg_strength=pop_reg_strength,
            top_k=top_n
        )
        
        # Add movie information if available
        try:
            movies_df = pd.read_csv('data/ml-latest-small/movies.csv')
            movies_df['movieId'] = movies_df['movieId'].astype(int)
            movies_df.rename(columns={'movieId': 'itemId'}, inplace=True)
            
            recs_df = recs_df.merge(
                movies_df[['itemId', 'title', 'genres']],
                on='itemId', 
                how='left'
            )
        except Exception as e:
            print(f"Warning: Could not load movie information: {e}")
            
        # Add ground truth if available
        if hasattr(self, 'ratings_df'):
            recs_df = recs_df.merge(
                self.ratings_df.rename(columns={'userId': 'user'}),
                how='left',
                on=['user', 'itemId']
            )
            recs_df.rename(columns={'rating': 'ground_truth'}, inplace=True)
            
        # Calculate diversity metrics
        diversity_metrics = calculate_diversity_metrics(recs_df)
        print("Diversity metrics:", diversity_metrics)
        
        # Calculate overlap metrics
        overlap_metrics = analyze_recommendation_overlap(recs_df)
        print("Overlap metrics:", overlap_metrics)
        
        # Log diversity metrics
        if epoch_id is not None:
            self._writer.add_scalar('diversity/user_coverage', diversity_metrics['user_coverage'], epoch_id)
            self._writer.add_scalar('diversity/item_coverage', diversity_metrics['item_coverage'], epoch_id)
            self._writer.add_scalar('diversity/entropy', diversity_metrics['normalized_entropy'], epoch_id)
            self._writer.add_scalar('diversity/gini', diversity_metrics['gini_coefficient'], epoch_id)
            self._writer.add_scalar('diversity/overlap', overlap_metrics['avg_overlap_ratio'], epoch_id)
            self._writer.add_scalar('diversity/concentration', overlap_metrics['item_concentration'], epoch_id)
        
        # Save recommendations
        directory = f"recommendations-out/run{self.config['run_number']}"
        os.makedirs(directory, exist_ok=True)
        
        if epoch_id is not None:
            filename = f"{directory}/diverse_recs_epoch_{epoch_id}.csv"
        else:
            filename = f"{directory}/final_diverse_recs.csv"
            
        recs_df.to_csv(filename, index=False)
        print(f"Saved diverse recommendations to {filename}")
        
        # Save metrics
        metrics_df = pd.DataFrame({
            **diversity_metrics,
            **overlap_metrics,
            'epoch': epoch_id if epoch_id is not None else 'final'
        }, index=[0])
        
        metrics_filename = f"{directory}/diversity_metrics_epoch_{epoch_id}.csv" if epoch_id is not None else f"{directory}/final_diversity_metrics.csv"
        metrics_df.to_csv(metrics_filename, index=False)
        
        # Return the recommendations dataframe
        return recs_df

    def generate_top_n(self, top_n=10, epoch_id=None):
        self.model.eval()
        rows = []
        try:
            movies = pd.read_csv('data/ml-latest-small/movies.csv')
        except:
            movies = None
            
        for user in range(self.config['num_users']):
            seen = self.user_seen_items.get(user, set())
            unseen = np.array(list(set(self.all_items) - seen))
            if len(unseen) == 0:
                continue
            users = torch.tensor([user] * len(unseen))
            items = torch.tensor(unseen)
            if self.config['use_cuda']:
                users = users.cuda()
                items = items.cuda()
            with torch.no_grad():
                scores = self.model(users, items).cpu().numpy().flatten()

            top_indices = np.argsort(scores)[-top_n:][::-1]
            for idx in top_indices:
                rows.append((user, unseen[idx], float(scores[idx])))
        
        recs_df = pd.DataFrame(rows, columns=['user', 'itemId', 'score'])
        recs_df.drop_duplicates(subset=['user', 'itemId'], inplace=True)
        recs_df['itemId'] = recs_df['itemId'].astype(int)
        
        if movies is not None:
            movies['itemId'] = movies['movieId'].astype(int)
            recs_df = recs_df.merge(movies[['itemId', 'title', 'genres']], on='itemId', how='left')
        
        directory = f"recommendations-out/run{self.config['run_number']}"
        os.makedirs(directory, exist_ok=True)
        
        filename_epoch = f"{directory}/topn_epoch_{epoch_id}.csv"
        filename_final = f"{directory}/final_topn.csv"

        recs_df.to_csv(filename_epoch, index=False)

        if epoch_id == self.config['num_epoch'] - 1:
            recs_df.to_csv(filename_final, index=False)
            print(f"Saved Top-{top_n} recommendations to {filename_final}")

        print(f"Saved Top-{top_n} recommendations to {filename_epoch}")
        return recs_df
    
    def generate_full_rankings_for_users(self, user_ids=None, epoch_id=None):
        """
        Generate predictions for all unseen items for a subset of users.
        Save full ranked lists, not just top-N.
        """
        self.model.eval()
        rows = []
        
        try:
            movies = pd.read_csv('data/ml-latest-small/movies.csv')
            movies['movieId'] = movies['movieId'].astype(int)
            movies.rename(columns={'movieId': 'itemId'}, inplace=True)
        except:
            movies = None

        if user_ids is None:
            user_ids = list(self.user_seen_items.keys())

        for user in user_ids:
            seen = self.user_seen_items.get(user, set())
            unseen = np.array(list(set(self.all_items) - seen))
            if len(unseen) == 0:
                continue

            users = torch.tensor([user] * len(unseen))
            items = torch.tensor(unseen)

            if self.config['use_cuda']:
                users = users.cuda()
                items = items.cuda()

            with torch.no_grad():
                scores = self.model(users, items).cpu().numpy().flatten()
                if self.config.get('is_explicit', False):
                    scores = scores * self.config['max_rating']

            for item, score in zip(unseen, scores):
                rows.append((user, item, float(score)))

        recs_df = pd.DataFrame(rows, columns=['user', 'itemId', 'score'])
        recs_df['itemId'] = recs_df['itemId'].astype(int)

        if movies is not None:
            recs_df = recs_df.merge(movies[['itemId', 'title', 'genres']], on='itemId', how='left')
        
        if hasattr(self, 'ratings_df'):
            recs_df = recs_df.merge(
                self.ratings_df.rename(columns={'userId': 'user'}),
                how='left',
                on=['user', 'itemId']
            )
            recs_df.rename(columns={'rating': 'ground_truth'}, inplace=True)

        # Sort by score
        recs_df = recs_df.sort_values(by=['user', 'score'], ascending=[True, False])

        # Save
        directory = f"recommendations-out/run{self.config['run_number']}"
        os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist

        # Build the filename inside that directory
        if epoch_id is not None:
            filename = f"{directory}/full_rankings_users_{epoch_id}.csv"
        else:
            filename = f"{directory}/final_full_rankings.csv"
        recs_df.to_csv(filename, index=False)
        print(f"Saved full ranked recommendations for users to {filename}")
        
        return recs_df