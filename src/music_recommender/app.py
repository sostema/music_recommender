import streamlit as st

from music_recommender.recommender import Recommender


def main():

    # Streamlit UI
    st.title("Рекомендательная система \"Интернет-магазин музыки\"")

    # Instantiate the Recommender class
    recommender = Recommender()

    # Select artists from the multiselect dropdown
    selected_tracks = st.multiselect(
        "Выберите свои любимые песни:",
        recommender.tracklist["full_name"].unique(),
    )

    # Button to get recommendations
    if st.button("Получить рекомендации по артистам"):
        # Extract selected artists from the tracklist
        selected_artists = recommender.tracklist[
            recommender.tracklist["full_name"].isin(selected_tracks)
        ]["artist_name"].tolist()

        # Get recommendations using the Recommender class methods
        popular_artist_recommendations = (
            recommender.get_popular_artist_recommendations()
        )
        item_based_recommendations = recommender.get_item_based_recommendations(
            selected_artists
        )
        user_based_recommendations = recommender.get_user_based_recommendations(
            selected_artists
        )

        # Display recommendations in Streamlit UI
        st.subheader("Рекомендации:")

        # Create three columns for different types of recommendations
        col1, col2, col3 = st.columns(3)

        with col1:
            # Display popular artist recommendations
            if popular_artist_recommendations:
                st.success("Самые популярные артисты (для новых пользователей)")
                st.write(popular_artist_recommendations)
            else:
                st.warning("No popular artist recommendations available.")

        with col2:
            # Display item-based recommendations
            if item_based_recommendations:
                st.success("Рекомендации на основе ваших предпочтений")
                st.write(item_based_recommendations)
            else:
                st.warning("No item-based recommendations available.")

        with col3:
            # Display user-based recommendations
            if user_based_recommendations:
                st.success("Схожие пользователи также любят следующих артистов")
                st.write(user_based_recommendations)
            else:
                st.warning("No user-based recommendations available.")


if __name__ == "__main__":
    main()
