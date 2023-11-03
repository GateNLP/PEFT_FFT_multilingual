"""general utility functions"""

import pandas as pd
import numpy as np
import os
import json
from functools import lru_cache
from sklearn.preprocessing import MultiLabelBinarizer

import torch

labels = [
    "Appeal_to_Authority",
    "Appeal_to_Fear-Prejudice",
    "Appeal_to_Hypocrisy",
    "Appeal_to_Popularity",
    "Appeal_to_Time",
    "Appeal_to_Values",
    "Causal_Oversimplification",
    "Consequential_Oversimplification",
    "Conversation_Killer",
    "Doubt",
    "Exaggeration-Minimisation",
    "False_Dilemma-No_Choice",
    "Flag_Waving",
    "Guilt_by_Association",
    "Loaded_Language",
    "Name_Calling-Labeling",
    "Obfuscation-Vagueness-Confusion",
    "Questioning_the_Reputation",
    "Red_Herring",
    "Repetition",
    "Slogans",
    "Straw_Man",
    "Whataboutism",
]

multibin = MultiLabelBinarizer().fit([labels])

coarse_labels = [
        "Justification",
        "Simplification",
        "Distraction",
        "Call",
        "Manipulative_Wording",
        "Attack_Reputation"
    ]

coarse_multibin = MultiLabelBinarizer().fit([coarse_labels])

def upsample(df):
    """Replicates single-class examples until all classes are balanced."""
    multibin = MultiLabelBinarizer().fit([labels])
    train_labels = multibin.transform(df["labels"].str.split(",").values)
    biggest = train_labels.sum(axis=0).max()
    lookup_counts = {k: v for k, v in zip(multibin.classes_, train_labels.sum(axis=0))}

    dfs = []
    for cls in lookup_counts.keys():
        aux_df = df[df["labels"].str.split(",").apply(lambda x: x == [cls])]
        if len(aux_df) > 0:
            dfs.append(aux_df.sample(biggest - lookup_counts[cls], replace=True))

    dfs.append(df)
    joined_dfs = pd.concat(dfs)

    return joined_dfs


def upsamplev2(df):
    """Replicates examples that have multiple classes if their addition does not explode any other class."""
    multibin = MultiLabelBinarizer().fit([labels])
    train_labels = multibin.transform(df["labels"].str.split(",").values)

    biggest = train_labels.sum(axis=0).max()  # no class should have more than this number of positive examples
    remaining = np.repeat(biggest, len(labels)) - train_labels.sum(axis=0)  # how many to add for each class

    # sorted dict with {example_idx: number of classes}
    classes_per_example = {
        idx: value for idx, value in enumerate(train_labels.sum(axis=1))
    }
    classes_per_example = {
        k: v
        for k, v in sorted(
            classes_per_example.items(), key=lambda item: item[1], reverse=True
        )
    }

    added_examples_idx = []
    for k, v in classes_per_example.items():
        cls = multibin.transform([df.iloc[k, 3].split(",")])[0]
        while not any(remaining - cls < 0) and cls.sum() > 1:  # keep replicating the same example while it's safe
            remaining = remaining - cls
            added_examples_idx.append(k)

    joined_df = pd.concat([df, df.iloc[added_examples_idx]])

    return joined_df


def make_dataframe(input_folder, labels_fn=None, spans_folder=None):
    """More generic way of loading a dataset. Used by the load_dataset function."""
    text = []
    for fil in filter(
        lambda x: x.endswith(".txt") and not x.startswith("."), os.listdir(input_folder)
    ):
        iD = fil[7:].split(".")[0]
        lines = list(
            enumerate(
                open(input_folder + fil, "r", encoding="utf-8").read().splitlines(), 1
            )
        )
        text.extend([(iD,) + line for line in lines])

    df_text = pd.DataFrame(text, columns=["id", "line", "text"])
    df_text.id = df_text.id.apply(int)
    df_text.line = df_text.line.apply(int)
    df_text = df_text.set_index(["id", "line"])

    df = df_text

    if labels_fn and spans_folder is None:
        # MAKE LABEL DATAFRAME
        labels = pd.read_csv(labels_fn, sep="\t", encoding="utf-8", header=None)
        labels = labels.rename(columns={0: "id", 1: "line", 2: "labels"})
        labels = labels.set_index(["id", "line"])
        # labels = labels[labels.labels.notna()].copy()

        # JOIN
        df = labels.join(df_text)[["text", "labels"]]

    if spans_folder is not None:
        grouped_texts = (
            df.reset_index("line").groupby("id")["text"].apply(lambda x: " ".join(x))
        )

        labels_df = pd.DataFrame(columns=["id", "labels", "start", "end"])
        for fil in filter(lambda x: x.endswith(".txt"), os.listdir(spans_folder)):
            iD = fil[7:].split("-")[0]
            try:
                aux_df = pd.read_csv(
                    spans_folder + fil, sep="\t", encoding="utf-8", header=None
                )
                aux_df.columns = ["id", "labels", "start", "end"]

                labels_df = pd.concat([labels_df, aux_df])
            except pd.errors.EmptyDataError:
                pass

        labels_df.id = labels_df.id.apply(int)
        labels_df = labels_df.set_index("id")

        df = labels_df.join(grouped_texts)
        df["span_text"] = df.apply(lambda x: x["text"][x["start"] : x["end"]], axis=1)
        df = df.drop(["text", "start", "end"], axis=1)
        df = df.rename({"span_text": "text"}, axis=1)

    df = df[df.text.str.strip().str.len() > 0].copy()
    return df.reset_index()


def load_dataset(
    languages: list,
    split: str,
    with_spans=False,
    with_translations=False,
    with_unlabelled=True,
    coarse=False,
) -> pd.DataFrame:
    """Loads train/dev/test datasets for one or multiple languages."""
    df = pd.DataFrame()
    for lang in languages:
        if with_translations:
            if split == "train":
                aux_df = pd.read_csv(f"../translated_data/{lang}/train_translated.csv")
                aux_df = aux_df[["id", "line", "text_translated", "labels", "lang"]]
                aux_df = aux_df.rename({"text_translated": "text"}, axis=1)
            elif split == "dev":
                aux_df = pd.read_csv(f"../translated_data/{lang}/dev_translated.csv")
                aux_df = aux_df[["id", "line", "text_translated", "labels", "lang"]]
                aux_df = aux_df.rename({"text_translated": "text"}, axis=1)

        else:
            data_folder = f"data/{lang}/{split}-articles-subtask-3/"
            labels_folder = (
                f"data/{lang}/{split}-labels-subtask-3.txt"
                if split != "test"
                else None
            )
            spans_folder = (
                f"data/{lang}/{split}-labels-subtask-3-spans/"
                if split != "test"
                else None
            )

            aux_df = make_dataframe(
                data_folder,
                labels_folder,
                spans_folder=spans_folder if with_spans else None,
            )
            aux_df["lang"] = lang
            if split != "test":
                aux_df["labels"] = aux_df["labels"].fillna("")

        df = pd.concat([df, aux_df])

    if not with_unlabelled and split != "test":
        df = df[df["labels"] != ""]  # removes sentences that were not assigned any class


    if split != "test":
        if coarse:
            mapping = _load_flipped_hierarchy()
            df["labels"] = df["labels"].str.split(",").apply(lambda x: [mapping[label] for label in x])
            df["labels"] = df["labels"].apply(lambda x: ",".join(x))
            labels = coarse_multibin.transform(df["labels"].str.split(",").values)

        else:
            labels = multibin.transform(df["labels"].str.split(",").values)

        df["labels"] = labels.tolist()

    return df


def train_test_split(df, train_size):
    """Splits a dataset into train/test. "Deprecated": Used prior to having access to the official dev set."""
    indices = df.index
    train_idxs = np.random.choice(
        indices, size=int(len(indices) * train_size), replace=False
    )
    dev_idxs = np.setdiff1d(indices, train_idxs)

    train_df = df[np.in1d(df.index.get_level_values(0), train_idxs)]
    test_df = df[np.in1d(df.index.get_level_values(0), dev_idxs)]

    return train_df, test_df


def save_results(y_pred, dev_df, out_fname):
    """Using when performing inference to generate the submission file."""
    y_pred = list(map(lambda x: ",".join(x), y_pred))
    df = pd.DataFrame(y_pred, dev_df.index)
    df.to_csv(out_fname, sep="\t", header=None)


## Category Hierarchy Functions


@lru_cache
def _load_category_hierarchy():
    """Loads the coarse and fine hierarchy data.

    LRU cached to boost performance of multiple calls.

    Returns:
        dict: Keys are coarse categories, values are sets of fine categories.
    """
    with open("category_map_st3.json") as f:
        category_map = json.load(f)

    for coarse in category_map.keys():
        category_map[coarse] = set(category_map[coarse])

    return category_map


@lru_cache
def _load_flipped_hierarchy():
    """Flips the hierachy to look up coarse from fine category.

    LRU cached to boost performance of multiple calls.

    Returns:
        dict: A fine -> set of co
    """
    flipped = {}

    for coarse, fines in _load_category_hierarchy().items():
        for fine in fines:
            flipped[fine] = coarse

    return flipped


def get_fines_in_coarse(coarse_cat):
    """Get the set of fine categories for a given coarse category.

    Args:
        coarse_cat (str): A coarse category

    Returns:
        set: The fine categories
    """
    return _load_category_hierarchy()[coarse_cat]


def get_coarse_for_fine(fine_cat):
    """Get the coarse category of a fine category.

    Args:
        fine_cat (str): The fine category

    Returns:
        str: The coarse category name
    """
    return _load_flipped_hierarchy()[fine_cat]


def is_fine_in_coarse(fine_cat, coarse_cat):
    """Check if a fine category is in the given coarse category

    Args:
        fine_cat (str): Fine category name
        coarse_cat (str): Coarse category name

    Returns:
        bool: Whether the coarse category is correct for the fine category.
    """
    return _load_flipped_hierarchy()[fine_cat] == coarse_cat
