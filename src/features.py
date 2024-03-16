from typing import List, Tuple
import pandas as pd
import torch


def create_feature_map(path: str, glosses: List[str]) -> Tuple[List[List[int]], List[int]]:
    """Generates a feature map from a csv file.

    Args:
        path (str): Path to a CSV file. File must have column "Gloss" and at least one more column.
        glosses (List[str]): The glosses in the dataset. 

    Returns:
        Tuple[List[List[int]], List[int]]: A tuple containing 1) A feature map of length (num_glosses, num_features) where each sublist is the feature list for a gloss, and each item is the int value of a particular feature
        2) A list where each item is the number of possible values for each feature
    """
    features = pd.read_csv(path)
    assert 'Gloss' in features.columns

    def _map_gloss(gloss: str) -> list[int]:
        try:
            return features[features['Gloss'] == gloss].values[0][1:].tolist()
        except:
            print(gloss)
    feature_map = list(map(_map_gloss, glosses))

    # Compute the number of distinct labels in each feature column
    feature_lengths = (features.max(axis=0, numeric_only=True) + 1).values.tolist()
    return feature_map, feature_lengths


def map_labels_to_features(labels: torch.Tensor, feature_map: List[List[int]]) -> torch.LongTensor:
    """Maps a batch of sequences of label ids to a batch of features, each a sequence of ids

    Args:
        labels (torch.Tensor): A tensor `(batch_size, seq_length)` of label ids 
        feature_map (List[List[int]]): A nested list `(num_glosses, num_features)` that maps gloss ids to feature vectors

    Returns:
        torch.LongTensor: A tensor `(batch_size, num_features, seq_length)` of feature ids
    """
    labels_to_features_map = torch.tensor(feature_map, dtype=torch.int64)
    num_features = labels_to_features_map.shape[1]

    # Temporarily remove the -100 mask so we can do a single index operation
    labels_masked = labels.clone().masked_fill_(labels == -100, 1)

    # Index into the features map using the labels tensor to get a tensor of size (batch_size, seq_length, num_features)
    features = labels_to_features_map[labels_masked]

    # For each -100 values in the original labels tensor at (a, b), set all values (a, b, x) to -100 in the features tensor
    mask = labels == -100
    mask = mask.unsqueeze(-1).expand(-1, -1, num_features)
    features[mask] = -100

    features = features.permute(0, 2, 1)
    assert features.shape == (labels.shape[0], labels_to_features_map.shape[1], labels.shape[1])
    return features
