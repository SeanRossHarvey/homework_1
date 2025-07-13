import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import networkx as nx
from typing import List, Dict, Tuple
import pandas as pd


class Visualiser:
    """Create visualisations for text analysis."""
    
    def __init__(self):
        # set plot style for consistent appearance
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_similarity_networks(self, similarities: np.ndarray, labels: List[str], 
                               thresholds: List[float] = [0.2, 0.4, 0.6, 0.8],
                               titles: List[str] = None) -> plt.Figure:
        """Plot document similarity networks at different thresholds."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        # get unique topics and create colour map
        unique_topics = sorted(set(labels))
        colours = plt.cm.Set3(np.linspace(0, 1, len(unique_topics)))
        topic_colours = {topic: colours[i] for i, topic in enumerate(unique_topics)}
        
        for idx, threshold in enumerate(thresholds):
            ax = axes[idx]
            
            # create network graph
            G = nx.Graph()
            
            # add nodes (one per document)
            for i in range(len(labels)):
                G.add_node(i)
            
            # add edges where similarity exceeds threshold
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    if similarities[i, j] > threshold:
                        G.add_edge(i, j, weight=similarities[i, j])
            
            # create layout for network visualisation
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # draw nodes coloured by topic
            for topic in unique_topics:
                node_list = [i for i, label in enumerate(labels) if label == topic]
                nx.draw_networkx_nodes(G, pos, nodelist=node_list,
                                     node_color=[topic_colours[topic]], 
                                     node_size=100,
                                     label=topic, ax=ax)
            
            # draw edges with low opacity
            nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
            
            ax.set_title(f'Similarity network (threshold = {threshold})', fontsize=14)
            ax.legend(loc='upper right', fontsize=10)
            ax.axis('off')
            
            # add network statistics to plot
            n_edges = G.number_of_edges()
            n_components = nx.number_connected_components(G)
            ax.text(0.02, 0.02, f'Edges: {n_edges}\nComponents: {n_components}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices(self, cm_data: Dict) -> plt.Figure:
        """Plot confusion matrix and its normalised versions."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        matrices = [
            ('Confusion matrix', cm_data['confusion_matrix'], 'd'),
            ('Row normalised', cm_data['row_normalised'], '.2f'),
            ('Column normalised', cm_data['column_normalised'], '.2f')
        ]
        
        for ax, (title, matrix, fmt) in zip(axes, matrices):
            sns.heatmap(matrix, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=cm_data['labels'],
                       yticklabels=cm_data['labels'],
                       ax=ax, cbar_kws={'label': 'Count' if fmt == 'd' else 'Proportion'})
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('True', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_wordclouds(self, cluster_analysis: Dict[int, Dict], 
                         n_cols: int = 3) -> plt.Figure:
        """Create word clouds for each cluster."""
        n_clusters = len(cluster_analysis)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (cluster_id, analysis) in enumerate(cluster_analysis.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Create word frequency dict
            word_freq = {word: freq for word, freq in analysis['top_words']}
            
            # Create word cloud
            wordcloud = WordCloud(width=400, height=400, 
                                background_color='white',
                                colormap='viridis').generate_from_frequencies(word_freq)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            
            # Add title with cluster info
            dominant_topic = analysis['dominant_topic']
            size = analysis['size']
            ax.set_title(f'Cluster {cluster_id}\n{dominant_topic} ({size} docs)', 
                        fontsize=12)
        
        # Hide empty subplots
        for idx in range(n_clusters, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_similarity_distribution(self, similarities: np.ndarray, 
                                   topics: List[str]) -> plt.Figure:
        """Plot distribution of similarities within and between topics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # calculate within and between topic similarities
        unique_topics = sorted(set(topics))
        within_sims = []
        between_sims = []
        
        for i in range(len(topics)):
            for j in range(i+1, len(topics)):
                if topics[i] == topics[j]:
                    within_sims.append(similarities[i, j])
                else:
                    between_sims.append(similarities[i, j])
        
        # plot similarity distributions
        ax1.hist(within_sims, bins=50, alpha=0.7, label='Within-topic', density=True)
        ax1.hist(between_sims, bins=50, alpha=0.7, label='Between-topic', density=True)
        ax1.set_xlabel('Cosine similarity', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Distribution of document similarities', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # prepare data for box plots by topic pair
        topic_pairs = []
        pair_sims = []
        
        # collect within-topic similarities
        for topic in unique_topics:
            topic_indices = [i for i, t in enumerate(topics) if t == topic]
            if len(topic_indices) > 1:
                for i in range(len(topic_indices)):
                    for j in range(i+1, len(topic_indices)):
                        topic_pairs.append(f'{topic}\n(within)')
                        pair_sims.append(similarities[topic_indices[i], topic_indices[j]])
        
        # collect between-topic similarities (sample for readability)
        for i, topic1 in enumerate(unique_topics):
            for topic2 in unique_topics[i+1:]:
                indices1 = [idx for idx, t in enumerate(topics) if t == topic1]
                indices2 = [idx for idx, t in enumerate(topics) if t == topic2]
                
                for idx1 in indices1[:10]:  # Sample to avoid too many points
                    for idx2 in indices2[:10]:
                        topic_pairs.append(f'{topic1[:4]}-{topic2[:4]}')
                        pair_sims.append(similarities[idx1, idx2])
        
        # create dataframe for box plot
        df = pd.DataFrame({'Topic Pair': topic_pairs, 'Similarity': pair_sims})
        
        # create box plot
        unique_pairs = df['Topic Pair'].unique()
        if len(unique_pairs) > 10:
            # Select representative pairs
            within_pairs = [p for p in unique_pairs if '(within)' in p]
            between_pairs = [p for p in unique_pairs if '(within)' not in p][:6]
            selected_pairs = within_pairs + between_pairs
            df_plot = df[df['Topic Pair'].isin(selected_pairs)]
        else:
            df_plot = df
        
        df_plot.boxplot(column='Similarity', by='Topic Pair', ax=ax2)
        ax2.set_xlabel('Topic pair', fontsize=12)
        ax2.set_ylabel('Cosine similarity', fontsize=12)
        ax2.set_title('Similarity by topic pair', fontsize=14)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_topic_characteristics(self, dtm: np.ndarray, features: List[str], 
                                 topics: List[str]) -> plt.Figure:
        """Plot top words for each topic."""
        unique_topics = sorted(set(topics))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, topic in enumerate(unique_topics[:4]):  # Plot up to 4 topics
            ax = axes[idx]
            
            # Get documents for this topic
            topic_mask = [t == topic for t in topics]
            topic_dtm = dtm[topic_mask]
            
            # Calculate average word frequencies
            avg_frequencies = np.mean(topic_dtm.toarray(), axis=0)
            
            # Get top 15 words
            top_indices = np.argsort(avg_frequencies)[-15:][::-1]
            top_words = [features[i] for i in top_indices]
            top_freqs = [avg_frequencies[i] for i in top_indices]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_words))
            ax.barh(y_pos, top_freqs)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_words)
            ax.invert_yaxis()
            ax.set_xlabel('Average TF-IDF score', fontsize=12)
            ax.set_title(f'Top words in {topic}', fontsize=14)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig