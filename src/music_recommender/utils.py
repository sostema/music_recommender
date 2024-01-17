import os
import re
from typing import Tuple

import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz


def remove_similar_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove similar tracks from the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing track information.

    Returns:
    - pd.DataFrame: DataFrame with similar tracks removed.
    """
    # Convert track names to lowercase and remove non-alphanumeric characters
    df["artist_name_small"] = df["artist_name"].str.lower()
    df["artist_name_small"] = df["artist_name_small"].apply(
        lambda x: re.sub("[^A-Za-z0-9äöüÄÖÜß]+", "", x)
    )

    df["track_name_small"] = df["track_name"].str.lower()
    df["track_name_small"] = df["track_name_small"].apply(
        lambda x: re.sub("[^A-Za-z0-9äöüÄÖÜß]+", "", x)
    )

    # Count occurrences of each track
    df["count"] = df.groupby(["artist_name", "track_name"]).transform("size")
    # Sort by count in descending order
    df = df.sort_values(by="count", ascending=False)
    df[["artist_name", "track_name"]] = df.groupby(
        ["artist_name_small", "track_name_small"]
    )[["artist_name", "track_name"]].transform("first")
    df = df.sort_index()
    return df


def remove_unpopular_artists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unpopular artists from the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing artist information.

    Returns:
    - pd.DataFrame: DataFrame with unpopular artists removed.
    """
    # Filter out artists with less than 1000 occurrences
    return df[df.groupby("artist_name").transform("size") > 100]


def remove_inactive_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove inactive users from the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing user information.

    Returns:
    - pd.DataFrame: DataFrame with inactive users removed.
    """
    # Filter out users with fewer than 500 unique tracks in their playlists
    return df[df.groupby("user_id")["track_name"].transform("nunique") > 100]


def remove_uniform_playlists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove playlists with uniform content from the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing playlist information.

    Returns:
    - pd.DataFrame: DataFrame with playlists having more than 10 unique artists.
    """
    # Filter out playlists with fewer than 10 unique artists
    return df[df.groupby("playlist_name")["artist_name"].transform("nunique") > 10]


def get_tracklist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a tracklist DataFrame from the input DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing track information.

    Returns:
    - pd.DataFrame: DataFrame containing unique artist-track combinations.
    """
    # Extract unique artist-track combinations, sort by artist_name, and reset index
    tracklist = (
        df[["artist_name", "track_name"]]
        .drop_duplicates()
        .sort_values(by="artist_name")
        .reset_index(drop=True)
    )
    # Concatenate artist_name and track_name to create a full_name column
    tracklist["full_name"] = tracklist["artist_name"] + " - " + tracklist["track_name"]
    return tracklist


def generate_data() -> Tuple[pd.DataFrame, csr_matrix, pd.DataFrame]:
    """
    Generate processed data, user-artist matrix, and tracklist.

    Returns:
    - Tuple[pd.DataFrame, csr_matrix, pd.DataFrame]: Processed data, user-artist matrix, and tracklist.
    """
    # Read the Spotify dataset, drop duplicates and NaN values
    big_data = (
        pd.read_csv(
            "./data/spotify_dataset.csv",
            on_bad_lines="skip",
            names=["user_id", "artist_name", "track_name", "playlist_name"],
            header=None,
            skiprows=[0],
        )
        .dropna()
        .drop_duplicates()
    )
    # Remove similar tracks, unpopular artists, inactive users, and uniform playlists
    data = remove_similar_tracks(big_data)
    data = remove_unpopular_artists(data)
    data = remove_inactive_users(data)
    # data = remove_uniform_playlists(data)
    # Get the tracklist
    tracklist = get_tracklist(data)

    # Extract relevant columns, calculate ratings, and create user-artist matrix
    data = data[["user_id", "artist_name", "playlist_name"]].drop_duplicates()
    data["rating"] = data.groupby(["user_id", "artist_name"])[
        "playlist_name"
    ].transform("count")
    user_artist_matrix = csr_matrix(
        (
            data["rating"],
            (
                pd.Categorical(data["user_id"]).codes,
                pd.Categorical(data["artist_name"]).codes,
            ),
        )
    )
    return data, user_artist_matrix, tracklist


def save_data(
    data: pd.DataFrame, user_artist_matrix: csr_matrix, tracklist: pd.DataFrame
):
    """
    Save processed data, user-artist matrix, and tracklist to files.

    Parameters:
    - data (pd.DataFrame): Processed data DataFrame.
    - user_artist_matrix (csr_matrix): User-artist matrix.
    - tracklist (pd.DataFrame): Tracklist DataFrame.

    Returns:
    - None
    """
    # Save processed data to Parquet format
    data.to_parquet("./data/processed_data.pqt")
    # Save user-artist matrix to NPZ format
    save_npz("./data/uam.npz", user_artist_matrix)
    # Save tracklist to Parquet format
    tracklist.to_parquet("./data/tracklist.pqt")


def get_data() -> Tuple[pd.DataFrame, csr_matrix, pd.DataFrame]:
    """
    Load processed data, user-artist matrix, and tracklist from files.

    Returns:
    - Tuple[pd.DataFrame, csr_matrix, pd.DataFrame]: Processed data, user-artist matrix, and tracklist.
    """
    if os.path.exists("./data/processed_data.pqt"):
        # Read processed data from Parquet format
        df = pd.read_parquet("./data/processed_data.pqt")
        # Load user-artist matrix from NPZ format
        user_artist_matrix = load_npz("./data/uam.npz")
        # Read tracklist from Parquet format
        tracklist = pd.read_parquet("./data/tracklist.pqt")
    else:
        df, user_artist_matrix, tracklist = generate_data()
        save_data(df, user_artist_matrix, tracklist)

    return df, user_artist_matrix, tracklist
