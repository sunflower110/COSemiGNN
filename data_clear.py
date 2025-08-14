import pandas as pd
import argparse

def data_clear(tr_feature_path, tr_class_path, tr_edge_path):
    """
    Read three files, clean and align the data.

    Parameters:
        tr_feature_path: str  Path to feature file
        tr_class_path: str    Path to label file
        tr_edge_path: str     Path to edge file

    Returns:
        tr_edge_df: DataFrame
        tr_feature_withlabel_df: DataFrame
    """
    # Read files
    tr_feature_df = pd.read_csv(tr_feature_path)
    tr_class_df = pd.read_csv(tr_class_path)
    tr_edge_df = pd.read_csv(tr_edge_path)

    # 1. Remove rows with missing values in tr_feature
    tr_feature_clean = tr_feature_df.dropna()

    # 2. Align tr_feature and tr_class by txId
    tr_feature_withlabel_df = pd.merge(
        tr_feature_clean,
        tr_class_df,
        on='txId',
        how='inner'
    )

    return tr_edge_df, tr_feature_withlabel_df


def main():
    parser = argparse.ArgumentParser(description="Clean feature file and align with labels")
    parser.add_argument('--tr_feature', required=False,default="data/elliptic++Dataset/txs_features.csv",help='Path to feature file')
    parser.add_argument('--tr_class', required=False,default="data/elliptic++Dataset/txs_classes.csv", help='Path to label file')
    parser.add_argument('--tr_edge', required=False,default="data/elliptic++Dataset/txs_edgelist.csv", help='Path to edge file')
    parser.add_argument('--out_edge', required=False,default="data/txs_edgelist.csv", help='Output path for cleaned edge data')
    parser.add_argument('--out_feature', required=False,default="data/txs_label.csv", help='Output path for cleaned feature+label data')

    args = parser.parse_args()

    # Call data cleaning
    tr_edge_df, tr_feature_withlabel_df = data_clear(
        args.tr_feature,
        args.tr_class,
        args.tr_edge
    )

    # Save results
    tr_edge_df.to_csv(args.out_edge, index=False)
    tr_feature_withlabel_df.to_csv(args.out_feature, index=False)

    print(f"Cleaned edge data saved to: {args.out_edge}")
    print(f"Cleaned feature+label data saved to: {args.out_feature}")


if __name__ == '__main__':

    tr_feature_df = pd.read_csv("data/txs_label.csv")
    tr_feature_df = tr_feature_df[tr_feature_df["class"].isin([1, 2])]

    main()
