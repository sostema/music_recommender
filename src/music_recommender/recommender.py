from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from music_recommender.utils import get_data


class Recommender:
    def __init__(self):
        """
        Initialize the Recommender class.

        Loads data, user-artist matrix, and tracklist.
        Computes artist similarity using cosine similarity on the user-artist matrix.
        """
        self.df, self.user_artist_matrix, self.tracklist = get_data()
        self.artist_similarity = cosine_similarity(self.user_artist_matrix.T)

    def get_popular_artists(self) -> List[int]:
        """
        Get popular artists based on overall ratings.

        Returns:
        - List[int]: List of artist indices sorted by popularity.
        """
        # Sum along axis=0 to get the total rating for each artist
        artist_popularity = self.user_artist_matrix.sum(axis=0).A1
        popular_artists = np.argsort(artist_popularity)[::-1]
        return popular_artists.tolist()

    def get_popular_artist_recommendations(self) -> List[str]:
        """
        Get popular artist recommendations.

        Assumes a new user who hasn't listened to any artist.

        Returns:
        - List[str]: List of recommended popular artists.
        """
        popular_artists = self.get_popular_artists()

        # If it's a new user, assume they haven't listened to any artist
        user_artists = set()

        recommended_artists = []
        for idx in popular_artists:
            if idx in user_artists:
                continue
            recommended_artists.append(pd.Categorical(self.df["artist_name"]).categories[idx])
            if len(recommended_artists) >= 10:
                break

        return recommended_artists

    def get_item_based_recommendations(
        self,
        selected_artists: Optional[List[str]],
    ) -> List[str]:
        """
        Get item-based artist recommendations.

        Parameters:
        - selected_artists (Optional[List[str]]): List of selected artists by the user.

        Returns:
        - List[int]: List of recommended artist indices.
        """
        if selected_artists:
            selected_artist_indices = [
                pd.Categorical(self.df["artist_name"]).categories.get_loc(artist)
                for artist in selected_artists
            ]
            similar_scores = self.artist_similarity[selected_artist_indices].sum(axis=0)
            similar_artists = list(enumerate(similar_scores))
            sorted_artists = sorted(similar_artists, key=lambda x: x[1], reverse=True)

            recommended_artists = []
            for idx, _ in sorted_artists:
                if idx in selected_artist_indices:
                    continue
                recommended_artists.append(pd.Categorical(self.df["artist_name"]).categories[idx])
                if len(recommended_artists) >= 10:
                    return recommended_artists

            return recommended_artists
        return []

    def get_user_based_recommendations(
        self,
        selected_artists: Optional[List[str]],
    ) -> List[str]:
        """
        Get user-based artist recommendations.

        Parameters:
        - selected_artists (Optional[List[str]]): List of selected artists by the user.

        Returns:
        - List[str]: List of recommended artists based on user behavior.
        """
        if selected_artists:
            selected_artist_indices = [
                pd.Categorical(self.df["artist_name"]).categories.get_loc(artist)
                for artist in selected_artists
            ]
            user_ratings = np.zeros((self.user_artist_matrix.shape[1]))
            for artist_idx in selected_artist_indices:
                user_ratings[artist_idx] += 1
            user_similarity: np.ndarray = cosine_similarity(
                user_ratings.reshape(1, -1), self.user_artist_matrix
            )[0]
            k_nearest_users = np.argsort(user_similarity, axis=0)[:-11:-1]
            average_user_rating = user_ratings.mean()
            # print(average_user_rating)
            # print(user_similarity.shape)
            # print(self.user_artist_matrix[k_nearest_users].shape)
            # print(self.user_artist_matrix[k_nearest_users].sum(axis=1).shape)
            # print((
            #     self.user_artist_matrix[k_nearest_users]
            #     - self.user_artist_matrix[k_nearest_users].mean(axis=0)
            # ).shape)
            # print(user_similarity[k_nearest_users].shape)
            artist_ratings = average_user_rating + (+ np.multiply((
                self.user_artist_matrix[k_nearest_users]
                - self.user_artist_matrix[k_nearest_users].mean(axis=0)
            ), user_similarity[k_nearest_users][:, None]) / self.user_artist_matrix[k_nearest_users].sum(axis=1)).sum(axis=0)
            artist_ratings = np.ravel(artist_ratings)
            sorted_artist_ratings = np.argsort(artist_ratings)[::-1]

            recommended_artists = []
            user_artists = set(selected_artist_indices)
            for artist_idx in sorted_artist_ratings:
                if artist_idx not in user_artists:
                    recommended_artists.append(
                        pd.Categorical(self.df["artist_name"]).categories[artist_idx]
                    )
                    if len(recommended_artists) >= 10:
                        return recommended_artists

            return recommended_artists
        return []
