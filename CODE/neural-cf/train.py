import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import matplotlib.pyplot as plt
import time
import os

def plot_training_loss(epochs, losses, plots_dir):
    # Training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Average Training Loss Over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/loss_plot.png', dpi=300)
    plt.close()

def plot_rmse_mae(epochs, rmses, maes, plots_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rmses, marker='o', linestyle='-', linewidth=2, label='RMSE')
    plt.plot(epochs, maes, marker='s', linestyle='-', linewidth=2, label='MAE')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('RMSE and MAE Over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/metrics_plot.png', dpi=300)
    plt.close()

def plot_diversity_metrics(epochs, diversity_metrics_list, plots_dir):
    if diversity_metrics_list:
        plt.figure(figsize=(14, 10))
        div_df = pd.DataFrame(diversity_metrics_list)
        metrics_to_plot = [
            ('user_coverage', 'User Coverage'),
            ('item_coverage', 'Item Coverage'), 
            ('normalized_entropy', 'Normalized Entropy'),
            ('gini_coefficient', 'Gini Coefficient'), 
            ('avg_overlap_ratio', 'Average Overlap Between Users'),
            ('item_concentration', 'Item Concentration')
        ]
        for i, (metric, title) in enumerate(metrics_to_plot):
            if metric in div_df.columns:
                plt.subplot(3, 2, i+1)
                plt.plot(epochs, div_df[metric], marker='o', linestyle='-', linewidth=2)
                plt.title(title, fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 1.0 if metric != 'gini_coefficient' else None)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/diversity_metrics.png', dpi=300)
        plt.close()
        div_df['epoch'] = epochs
        div_df.to_csv(f'{output_dir}/diversity_metrics_all_epochs.csv', index=False)
    try:
        # Analyze the top recommended movies across users
        if os.path.exists(f"{output_dir}/diverse_recs_epoch_{config['num_epoch']-1}.csv"):
            diverse_recs = pd.read_csv(f"{output_dir}/diverse_recs_epoch_{config['num_epoch']-1}.csv")
            
            # Get movie metadata
            movies = pd.read_csv('data/ml-latest-small/movies.csv')
            movies.rename(columns={'movieId': 'itemId'}, inplace=True)
            
            # Count item frequency
            item_counts = diverse_recs['itemId'].value_counts().reset_index()
            item_counts.columns = ['itemId', 'count']
            item_counts = item_counts.merge(movies[['itemId', 'title']], on='itemId', how='left')
            
            # Calculate percentage of users
            total_users = diverse_recs['user'].nunique()
            item_counts['percentage'] = item_counts['count'] / total_users * 100
            
            # Handle NaN values in title column by using itemId as a backup title
            item_counts['title'] = item_counts.apply(
                lambda row: f"Item {row['itemId']}" if pd.isna(row['title']) else row['title'], 
                axis=1
            )
            
            # Save top 50 most recommended movies
            top_items = item_counts.sort_values('count', ascending=False).head(50)
            top_items.to_csv(f"{output_dir}/top_recommended_movies.csv", index=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            plt.barh(top_items['title'].str[:30].head(20), top_items['percentage'].head(20))
            plt.xlabel('Percentage of Users (%)')
            plt.ylabel('Movie Title')
            plt.title('Top 20 Most Recommended Movies')
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/top_recommendations_distribution.png', dpi=300)
            plt.close()
            
            print(f"Top 5 most recommended movies:")
            for _, row in top_items.head(5).iterrows():
                print(f"  - {row['title']} ({row['percentage']:.1f}% of users)")
            
    except Exception as e:
        print(f"Error analyzing recommendation distribution: {e}")

def run_train_loop(config, engine, sample_generator, evaluate_data):
    # Training Loop
    losses = []
    rmses = []
    maes = []
    diversity_metrics_list = []

    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(
            config['num_negative'], config['batch_size']
            )
        
        loss = engine.train_an_epoch(train_loader, epoch_id=epoch)
        losses.append(loss)
        metrics = engine.evaluate(evaluate_data, epoch_id=epoch)

        # Generate diverse recommendations (5 per user)
        recs_df = engine.generate_recommendations(
            epoch_id=epoch,
            top_n=5,  
            user_ids=config.get('eval_user_subset'),
            full_ranking=False,
            silent=True  
        )
        
        # Track metrics
        if config['is_explicit']:
            rmse, mae = metrics
            rmses.append(rmse)
            maes.append(mae)
            print(f"[Epoch {epoch}] RMSE = {rmse:.4f} | MAE = {mae:.4f}")
            engine.save(neumf_config['alias'], epoch, rmse=rmse, mae=mae)
        else:
            hit_ratio, ndcg = metrics
            print(f"[Epoch {epoch}] HR@10 = {hit_ratio:.4f} | NDCG@10 = {ndcg:.4f}")
            engine.save(neumf_config['alias'], epoch, hit_ratio=hit_ratio, ndcg=ndcg)
        
        # Read diversity metrics from file 
        metrics_file = f"{output_dir}/diversity_metrics_epoch_{epoch}.csv"
        if os.path.exists(metrics_file):
            try:
                metrics_df = pd.read_csv(metrics_file)
                metrics_dict = metrics_df.iloc[0].to_dict()
                diversity_metrics_list.append(metrics_dict)
                print(f"[Epoch {epoch}] Diversity metrics:")
                print(f"  - User coverage: {metrics_dict.get('user_coverage', 0):.4f}")
                print(f"  - Item coverage: {metrics_dict.get('item_coverage', 0):.4f}")
                print(f"  - Entropy: {metrics_dict.get('normalized_entropy', 0):.4f}")
                print(f"  - Avg. overlap: {metrics_dict.get('avg_overlap_ratio', 0):.4f}")
            except Exception as e:
                print(f"Error reading diversity metrics: {e}")
        
    return losses, rmses, maes, diversity_metrics_list, recs_df


# Define configuration with diversity options enabled
neumf_config = {'alias': 'neumf_factor8neg4',
                'num_epoch': 8, 
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': None,
                'num_items': None,
                'latent_dim_mf': 32,
                'latent_dim_mlp': 32,
                'num_negative': 4,
                'layers': [64, 128, 64, 32, 16],
                'l2_regularization': 0.00001,
                'weight_init_gaussian': True,
                'use_cuda': True,
                'use_bachify_eval': True,
                'device_id': 0,
                'pretrain': False,
                # Recommender parameters
                'generate_full_recs': False,  
                'is_explicit': True,
                'run_number': '14',
                # Diversity parameters
                'use_popularity_reg': False,  
                'pop_reg_strength': 0.7,
                'diversity_method': 'popularity',  # Options: 'popularity', 'xquad_smooth', 'enhanced'
                'xquad_lambda': 0.7, 
                'popularity_threshold': 0.1,  
                'long_tail_boost': 0.2, 
                # Model directory
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir': 'checkpoints/{}/run{}/{}_Epoch{}_{}.model'
                }

###################################################################################
start = time.time()
# Load Data
ml1m_rating = pd.read_csv('data/ml-latest-small/ratings.csv')
ml1m_rating.rename(columns={'userId': 'uid', 'movieId': 'mid'}, inplace=True)

# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')

item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')

# Set num_users and num_items dynamically
num_users = ml1m_rating['userId'].nunique()
num_items = ml1m_rating['itemId'].nunique()
neumf_config['num_users'] = num_users
neumf_config['num_items'] = num_items


ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]

# Create output directory
output_dir = f"recommendations-out/run{neumf_config['run_number']}"
os.makedirs(output_dir, exist_ok=True)

# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
neumf_config['max_rating'] = sample_generator.max_rating
evaluate_data = sample_generator.evaluate_data
user_seen_items = ml1m_rating.groupby('userId')['itemId'].apply(set).to_dict()
all_items = list(ml1m_rating['itemId'].unique())

# Initialize Model 
config = neumf_config
engine = NeuMFEngine(config)
engine.ratings_df = ml1m_rating[['userId', 'itemId', 'rating']]
engine.user_seen_items = user_seen_items
engine.all_items = all_items

# Initialize diversity components before training starts
if config.get('use_popularity_reg', False):
    engine.init_diversity_components()
    with open(f"{output_dir}/diversity_config.txt", "w") as f:
        f.write(f"Diversity enabled: {config.get('use_popularity_reg', False)}\n")
        f.write(f"Popularity regularization strength: {config.get('pop_reg_strength', 0.2)}\n")
        f.write(f"Number of recommendations per user: 5\n")


losses, rmses, maes, diversity_metrics_list, recs_df = run_train_loop(
                                                                    config,
                                                                    engine,
                                                                    sample_generator,
                                                                    evaluate_data
                                                                )
# Final recommendations
engine.generate_recommendations(
    epoch_id=config['num_epoch'] - 1,
    top_n=5,
    user_ids=config.get('eval_user_subset'),
    full_ranking=False # switch if needed
)

print("\nTraining complete.")

# Generate Plots
print("\nGenerating plots...")
epochs = list(range(1, config['num_epoch'] + 1))

# Create plots directory
plots_dir = f"{output_dir}/plots"
os.makedirs(plots_dir, exist_ok=True)
plot_training_loss(epochs, losses, plots_dir)

# RMSE and MAE plot
plot_rmse_mae(epochs, rmses, maes, plots_dir)

# Diversity metrics plot
plot_diversity_metrics(epochs, diversity_metrics_list, plots_dir)


print("Saved plots and analysis to output directory.")
end = time.time()
runtime_minutes = (end - start) / 60.0
print("Runtime: {:.2f} minutes".format(runtime_minutes))