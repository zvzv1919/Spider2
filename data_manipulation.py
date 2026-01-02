#!/usr/bin/env python3

import pandas as pd
import math


def label_back_to_thread_table(issue_df, support_df, save=True, output_path=None):
    categories_by_root_ts = (
        issue_df.groupby('root_ts')['category']
        .apply(lambda x: ', '.join(x.dropna().astype(str)))
        .reset_index()
        .rename(columns={'category': 'C'})
    )
    result_df = support_df.merge(categories_by_root_ts, on='root_ts', how='left')
    result_df['C'] = result_df['C'].fillna('')
    if save:
        result_df.to_csv(output_path, index=False)
    return result_df

def left_join(df1, df2, on_column, save=True, output_path=None):
    result_df = df1.merge(df2, on=on_column, how='left')
    if save:
        result_df.to_csv(output_path, index=False)
    return result_df

def filter_by_column(df, column, values, save=True, output_path=None):
    result_df = df[df[column].isin(values)]
    if save:
        result_df.to_csv(output_path, index=False)
    return result_df

def column_manipulation(df, column, manipulation_function, save=True, output_path=None):
    df[column] = df[column].apply(manipulation_function)
    if save:
        df.to_csv(output_path, index=False)
    return df

def drop_columns(df, columns, save=True, output_path=None):
    df.drop(columns=columns)
    if save:
        df.to_csv(output_path, index=False)
    return df

def pick_columns(df, columns, save=True, output_path=None):
    df = df[columns]
    if save:
        df.to_csv(output_path, index=False)
    return df

def rename_column(df, old_column, new_column, save=True, output_path=None):
    df = df.rename(columns={old_column: new_column})
    if save:
        df.to_csv(output_path, index=False)
    return df

def column_stats(df, column):
    value_counts = df[column].value_counts()
    print(value_counts)

def column_stats_labels(df, column):
    all_labels = df[column].str.split(",").explode().str.strip()
    value_counts = all_labels.value_counts()
    print(value_counts)

def sample_by_mask(df, mask, sample_size, save=False, output_path=None):
    df = df[mask]
    df = df.sample(n=sample_size)
    if save:
        df.to_csv(output_path, index=False)
    return df

def draft_benchmark(df, label_column):
    """
    Create a stratified sample of the dataframe based on the label column.
    Example usage:
        df = pd.read_csv("path/to/data.csv", dtype=str).fillna("")
        df_sampled = draft_benchmark(df, "label_column")
        df_sampled.to_csv("path/to/output_sampled.csv", index=False)
    """
    category_counts = df[label_column].value_counts()
    category_counts = category_counts[category_counts.index != ""]  # Remove empty category
    category_counts = category_counts.head(20)  # Pick top 20
    df_sampled = sample_by_mask(df, pd.Series(True, index=df.index), 0)
    for category, cnt in category_counts.items():
        category_sample = sample_by_mask(df, df["Service Sub Category"] == category, int(math.sqrt(cnt)) // 2 + 5) # sample 5-20 from each category
        df_sampled = pd.concat([df_sampled, category_sample])
    return df_sampled

if __name__ == '__main__':
    # Example usage:
    # df = pd.read_csv("path/to/your_data.csv", dtype=str).fillna("")
    # df = filter_by_column(df, "label", ["category1", "category2"], save=True, output_path="output.csv")
    # column_stats_labels(df, "label")
    pass
