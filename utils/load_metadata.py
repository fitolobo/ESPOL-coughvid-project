import pandas as pd
import numpy as np


def load_metadata(root, directory, metadata_file="metadata_compiled.csv"):

    data_dir = root + directory

    metadata = pd.read_csv(data_dir + metadata_file, sep=",")
    print(metadata.columns)

    # convert strings 'True'/'False' to genuine booleans
    cols_to_boolean = [
        "respiratory_condition",
        "fever_muscle_pain",
        "dyspnea_1",
        "wheezing_1",
        "stridor_1",
        "choking_1",
        "congestion_1",
        "nothing_1",
        "dyspnea_2",
        "wheezing_2",
        "stridor_2",
        "choking_2",
        "congestion_2",
        "nothing_2",
        "dyspnea_3",
        "wheezing_3",
        "stridor_3",
        "choking_3",
        "congestion_3",
        "nothing_3",
        "dyspnea_4",
        "wheezing_4",
        "stridor_4",
        "choking_4",
        "congestion_4",
        "nothing_4",
    ]
    # metadata[cols_to_boolean] = metadata[cols_to_boolean].apply(lambda x: x.astype(bool))
    for c in cols_to_boolean:
        metadata.loc[metadata[c].notnull(), c] = metadata.loc[
            metadata[c].notnull(), c
        ].astype(bool)

    print("NULL or NA records for each column:")
    print(metadata.isnull().sum())

    cols_to_fillna = [
        "gender",
        "status",
        "diagnosis_1",
        "diagnosis_2",
        "diagnosis_3",
        "diagnosis_4",
    ]
    metadata[cols_to_fillna] = metadata[cols_to_fillna].fillna("n/a")

    return metadata


def sample_df_balanced(df, group_col, n, random=42):
    assert isinstance(
        group_col, str
    ), "Input group_col must be a plain string with the column name: {}".format(
        type(group_col)
    )
    # df_count = df[[group_col]].groupby([group_col]).cumcount()+1
    df["N"] = np.zeros(len(df[group_col]))
    df_count = (
        df[[group_col, "N"]].groupby([group_col]).count().reset_index()
    )  # cumcount()+1

    out_df = pd.DataFrame()
    for igroup in df[group_col].unique():

        n_orig = df_count.loc[df_count[group_col] == igroup, "N"].values[0]
        if n_orig < n:  # need to upsample
            delta = max(n - n_orig, 0)
            tmp_df = df.loc[df[group_col] == igroup,]
            delta_df = tmp_df.sample(n=delta,
                                     random_state=random,
                                     replace=False)
            out_df = pd.concat([out_df, tmp_df, delta_df])
        else:  # downsample
            tmp_df = df.loc[df[group_col] == igroup,].sample(
                n=n, random_state=random, replace=False
            )
            out_df = pd.concat([out_df, tmp_df])
    return out_df.drop("N", axis=1, inplace=False)


def sample_all_data(df, group_col, target_size, random_state=42):
    out_df = pd.DataFrame()
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group]
        n_samples = len(group_data)
        
        if n_samples < target_size:
            # Upsample with replacement to reach target size
            additional = group_data.sample(n=target_size-n_samples, replace=True, random_state=random_state)
            group_data = pd.concat([group_data, additional])
        
        out_df = pd.concat([out_df, group_data])
    
    return out_df


