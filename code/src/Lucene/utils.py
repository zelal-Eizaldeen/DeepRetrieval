import numpy as np

def dcg_at_k(retrieved, target, k):
    """
    Compute DCG@k (Discounted Cumulative Gain).
    """
    retrieved = retrieved[:k]
    gains = [1.0 if item == target else 0.0 for item in retrieved]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(retrieved, target, k):
    """
    Compute NDCG@k.
    """
    dcg = dcg_at_k(retrieved, target, k)
    ideal_dcg = dcg_at_k([target], target, k)  # Ideal DCG: only the target at top
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def dcg_at_k_rank(rank, k):
    """
    Compute DCG@k (Discounted Cumulative Gain) for a single relevant document at a specific rank.
    
    Args:
        rank (int): The rank of the first occurrence of the answer (1-indexed)
        k (int): Number of documents to consider
        
    Returns:
        float: DCG score
    """
    if rank > k:
        return 0.0
    
    # Create a relevance array with a single 1 at the position of the first answer
    relevance = np.zeros(k)
    if rank <= k:
        relevance[rank-1] = 1.0
    
    # Calculate discounted gains using log2(i+1) for positions i (0-indexed)
    discounts = np.log2(np.arange(2, k + 2))
    return np.sum(relevance / discounts)


def ideal_dcg_at_k(k):
    """
    Compute ideal DCG@k where the answer is at rank 1.
    
    Args:
        k (int): Number of documents to consider
        
    Returns:
        float: Ideal DCG score
    """
    # Ideal case: answer is at rank 1
    return dcg_at_k_rank(1, k)


def ndcg_for_rank(rank, k):
    """
    Compute NDCG@k for a document at a specific rank.
    
    Args:
        rank (int): The rank of the first occurrence of the answer (1-indexed)
        k (int): Number of documents to consider
        
    Returns:
        float: NDCG score between 0 and 1
    """
    dcg = dcg_at_k_rank(rank, k)
    ideal_dcg = ideal_dcg_at_k(k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0