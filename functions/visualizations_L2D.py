import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import warnings
from collections import Counter
from itertools import combinations
from collections import defaultdict
import seaborn as sns
pd.set_option('future.no_silent_downcasting', True)
from functions.utils_L2D import load_and_restore_parquet, normalize_width, flatten_and_clean_values

def count_column_values_across_dfs(parquet_paths, columns):
    counts = {col: Counter() for col in columns}

    for path in tqdm(parquet_paths):
        try:
            df = load_and_restore_parquet(path)

            for col in columns:
                if col not in df.columns:
                    continue

                if col == 'time_of_day':
                    values = pd.to_datetime(df[col], errors='coerce').dt.hour.dropna().unique()
                elif col == 'width':
                    values = {normalize_width(v) for v in df[col].dropna()}
                    values.discard(None)  # remove failed parses
                else:
                    values = flatten_and_clean_values(col, df[col])

                for v in values:
                    counts[col][v] += 1

        except: pass

    return counts

def plot_presence_counts(counts_dict):
    for col, counter in counts_dict.items():
        if not counter:
            continue

        items = counter.most_common()
        labels, values = zip(*items)

        plt.figure(figsize=(10, 5))
        plt.bar(labels, values)
        plt.title(f"Number of DataFrames each {col} value appears in")
        plt.xlabel(col)
        plt.ylabel("Number of DataFrames")
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.show()

def compute_cooccurrence_matrix(parquet_paths, target_col):
    pair_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for path in tqdm(parquet_paths):
        try:
            df = load_and_restore_parquet(path)
            if target_col not in df.columns:
                continue

            for item in df[target_col].dropna():
                if isinstance(item, str) and ',' in item:
                    values = [v.strip() for v in item.split(',')]
                elif isinstance(item, list):
                    values = item
                else:
                    values = [item]

                unique_values = sorted(set(values))
                for val in unique_values:
                    label_counts[val] += 1
                for a, b in combinations(unique_values, 2):
                    pair_counts[(a, b)] += 1
                    pair_counts[(b, a)] += 1  # symmetrical

        except: pass

    labels = sorted(label_counts)
    matrix = pd.DataFrame(0, index=labels, columns=labels)

    for (a, b), count in pair_counts.items():
        matrix.at[a, b] = count

    return matrix

def plot_cooccurrence_heatmap(matrix, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, cmap="Blues")
    plt.title(title)
    plt.xlabel("Co-occurring with")
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def normalize_list_like(val):
    """Returns a set of clean values from a list or comma-separated string."""
    if pd.isna(val):
        return set()
    if isinstance(val, list):
        return set(val)
    if isinstance(val, str) and ',' in val:
        return set([v.strip() for v in val.split(',') if v.strip()])
    return {val}

def compute_cross_column_cooccurrence(parquet_paths, col_a, col_b):
    counts = defaultdict(int)
    values_a = set()
    values_b = set()

    for path in tqdm(parquet_paths):
        try:
            df = load_and_restore_parquet(path)
            if col_a not in df.columns or col_b not in df.columns:
                continue

            # Get unique values across the entire df (1 count per df per value)
            vals_a = set()
            vals_b = set()

            for val in df[col_a].dropna():
                vals_a.update(normalize_list_like(val))
            for val in df[col_b].dropna():
                vals_b.update(normalize_list_like(val))

            for a in vals_a:
                for b in vals_b:
                    counts[(a, b)] += 1
                    values_a.add(a)
                    values_b.add(b)

        except: pass

    if not values_a or not values_b:
        print(f"⚠️ No co-occurring values found between {col_a} and {col_b}")
        return pd.DataFrame()

    # Create DataFrame
    matrix = pd.DataFrame(0, index=sorted(values_a), columns=sorted(values_b))
    for (a, b), count in counts.items():
        matrix.loc[a, b] = count

    return matrix

def plot_cross_column_heatmap(matrix, col_a, col_b):
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, cmap="Oranges")
    plt.title(f"Co-occurrence: {col_a} vs {col_b}")
    plt.xlabel(col_b)
    plt.ylabel(col_a)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()