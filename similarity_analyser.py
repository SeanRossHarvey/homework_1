import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Tuple
import pandas as pd


class SimilarityAnalyser:
    """Analyse document similarities and perform clustering."""
    
    def __init__(self):
        pass
    
    def calculate_similarities(self, dtm: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarities between documents."""
        similarities = cosine_similarity(dtm)
        return similarities
    
    def cluster_by_threshold(self, similarities: np.ndarray, threshold: float) -> np.ndarray:
        """Cluster documents based on similarity threshold."""
        # convert similarities to distances for clustering
        distances = 1 - similarities
        
        # use hierarchical clustering with distance threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,
            affinity='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(distances)
        return labels
    
    def create_confusion_matrices(self, true_labels: List[str], predicted_labels: np.ndarray) -> Dict:
        """Create confusion matrix and normalised versions."""
        # get unique topics for matrix labels
        unique_topics = sorted(set(true_labels))
        
        # map predicted clusters to best matching topics
        cluster_to_topic = self._map_clusters_to_topics(true_labels, predicted_labels)
        mapped_predictions = [cluster_to_topic.get(label, 'Unknown') for label in predicted_labels]
        
        # create confusion matrix
        cm = confusion_matrix(true_labels, mapped_predictions, labels=unique_topics)
        
        # row normalised (by true labels) - shows recall
        cm_row_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_row_norm = np.nan_to_num(cm_row_norm)
        
        # column normalised (by predicted labels) - shows precision
        cm_col_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        cm_col_norm = np.nan_to_num(cm_col_norm)
        
        return {
            'confusion_matrix': cm,
            'row_normalised': cm_row_norm,
            'column_normalised': cm_col_norm,
            'labels': unique_topics,
            'cluster_mapping': cluster_to_topic
        }
    
    def _map_clusters_to_topics(self, true_labels: List[str], predicted_labels: np.ndarray) -> Dict[int, str]:
        """Map cluster IDs to topic names based on majority vote."""
        cluster_to_topic = {}
        
        for cluster_id in np.unique(predicted_labels):
            # get true labels for this cluster
            cluster_mask = predicted_labels == cluster_id
            cluster_true_labels = [label for i, label in enumerate(true_labels) if cluster_mask[i]]
            
            # find most common topic in this cluster (majority vote)
            if cluster_true_labels:
                topic_counts = pd.Series(cluster_true_labels).value_counts()
                cluster_to_topic[cluster_id] = topic_counts.index[0]
        
        return cluster_to_topic
    
    def analyse_clusters(self, texts: List[str], features: List[str], 
                        dtm: np.ndarray, labels: np.ndarray, 
                        topics: List[str]) -> Dict[int, Dict]:
        """Analyse word frequencies within each cluster."""
        cluster_analysis = {}
        
        for cluster_id in np.unique(labels):
            # get documents in this cluster
            cluster_mask = labels == cluster_id
            cluster_dtm = dtm[cluster_mask]
            cluster_topics = [topics[i] for i, mask in enumerate(cluster_mask) if mask]
            
            # calculate average word frequencies across cluster
            avg_frequencies = np.mean(cluster_dtm.toarray(), axis=0)
            
            # get top 20 words by frequency
            top_indices = np.argsort(avg_frequencies)[-20:][::-1]
            top_words = [(features[i], avg_frequencies[i]) for i in top_indices]
            
            # get topic distribution in cluster
            topic_dist = pd.Series(cluster_topics).value_counts().to_dict()
            
            cluster_analysis[cluster_id] = {
                'size': np.sum(cluster_mask),
                'top_words': top_words,
                'topic_distribution': topic_dist,
                'dominant_topic': max(topic_dist, key=topic_dist.get) if topic_dist else 'Unknown'
            }
        
        return cluster_analysis
    
    def get_similarity_statistics(self, similarities: np.ndarray, topics: List[str]) -> Dict:
        """Calculate statistics about similarities within and between topics."""
        unique_topics = sorted(set(topics))
        stats = {}
        
        for topic in unique_topics:
            # get document indices for this topic
            topic_indices = [i for i, t in enumerate(topics) if t == topic]
            
            if len(topic_indices) > 1:
                # calculate within-topic similarities
                within_sims = []
                for i in range(len(topic_indices)):
                    for j in range(i+1, len(topic_indices)):
                        within_sims.append(similarities[topic_indices[i], topic_indices[j]])
                
                stats[f'{topic}_within'] = {
                    'mean': np.mean(within_sims),
                    'std': np.std(within_sims),
                    'min': np.min(within_sims),
                    'max': np.max(within_sims)
                }
        
        # calculate between-topic similarities
        for i, topic1 in enumerate(unique_topics):
            for topic2 in unique_topics[i+1:]:
                indices1 = [idx for idx, t in enumerate(topics) if t == topic1]
                indices2 = [idx for idx, t in enumerate(topics) if t == topic2]
                
                between_sims = []
                for idx1 in indices1:
                    for idx2 in indices2:
                        between_sims.append(similarities[idx1, idx2])
                
                if between_sims:
                    stats[f'{topic1}_vs_{topic2}'] = {
                        'mean': np.mean(between_sims),
                        'std': np.std(between_sims),
                        'min': np.min(between_sims),
                        'max': np.max(between_sims)
                    }
        
        return stats