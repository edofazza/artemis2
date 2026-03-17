import pandas as pd
import random
import os


def cross_annotations(k):
    """path = ("/Users/edoardofazzari/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/"
            "PhD/reseaches/thesis/DATASETS/baboonland/")

    df = pd.read_csv(os.path.join(path, 'train.csv'), delimiter=' ')
    indexes = list(range(0, df.shape[0]))
    random.shuffle(indexes)
    for i in range(k):
        consider_indexes = indexes[len(indexes)//k * i: len(indexes)//k * (i+1)]
        rows = df.iloc[consider_indexes]
        rows.to_csv(os.path.join(path, f'{i}_light.csv'), index=False)"""
    # BABOONLAND
    path = ("/Users/edoardofazzari/Library/CloudStorage/OneDrive-ScuolaSuperioreSant'Anna/"
            "PhD/reseaches/thesis/DATASETS/baboonland/")

    df = pd.read_csv(os.path.join(path, 'train.csv'), delimiter=' ')

    # Group by video_id
    grouped = df.groupby('video_id')
    video_groups = list(grouped)

    # Shuffle video groups
    random.shuffle(video_groups)

    # Create k partitions
    partitions = [[] for _ in range(k)]

    for i, (video_id, group) in enumerate(video_groups):
        partitions[i % k].append(group)

    # Save each partition to a CSV file
    for i in range(k):
        partition_df = pd.concat(partitions[i])
        partition_df.to_csv(os.path.join(path, f'{i}_light.csv'), index=False)


if __name__ == '__main__':
    cross_annotations(5)
