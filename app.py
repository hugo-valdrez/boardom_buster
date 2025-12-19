import streamlit as st
import polars as pl

from src.ml.recommender import BoardGameRecommender
from src.other.abstract.write_reader_factory import WriterReaderFactory
from src.config import settings


@st.cache_resource
def load_recommender():
    """Load and cache the recommender."""
    recommender = BoardGameRecommender()
    recommender.load_data()
    return recommender


@st.cache_resource
def load_game_names():
    """Load game names for search."""
    data_dir = settings.PATHS["processed_data"]
    writer_reader = WriterReaderFactory.create_from_directory(data_dir)
    df = writer_reader.read()
    return df.select(["id", "name"]).to_dicts()


def search_games(query: str, games: list, max_results: int = 10) -> list:
    """Simple fuzzy search on game names."""
    if not query or len(query) < 2:
        return []
    
    query_lower = query.lower()
    matches = []
    
    for game in games:
        name = game["name"]
        name_lower = name.lower()
        
        # Exact match gets highest priority
        if query_lower == name_lower:
            matches.append((0, game))
        # Starts with query
        elif name_lower.startswith(query_lower):
            matches.append((1, game))
        # Contains query
        elif query_lower in name_lower:
            matches.append((2, game))
    
    # Sort by priority, then alphabetically
    matches.sort(key=lambda x: (x[0], x[1]["name"]))
    return [m[1] for m in matches[:max_results]]


def search_callback(query: str, games: list) -> list[str]:
    """Callback for searchbox - returns list of game names."""
    matches = search_games(query, games)
    return [game["name"] for game in matches]


def main():
    st.set_page_config(
        page_title="Boardom Buster",
        page_icon="🎲",
        layout="centered"
    )
    
    st.title("🎲 Boardom Buster")
    st.caption("Find similar board games to your favorites!")
    
    # Load data
    with st.spinner("Loading recommender..."):
        recommender = load_recommender()
        games = load_game_names()
    
    # Build name -> game lookup
    name_to_game = {game["name"]: game for game in games}
    game_names = sorted([game["name"] for game in games])
    
    # Searchable selectbox - updates live as you type
    selected_name = st.selectbox(
        "Search for a game",
        options=[""] + game_names,
        index=0,
        placeholder="Type to search (e.g., Catan, Ticket to Ride...)",
    )
    
    # Update selected game when selection changes
    if selected_name and selected_name in name_to_game:
        st.session_state.selected_game = name_to_game[selected_name]
    
    # Show recommendations if a game is selected
    if "selected_game" in st.session_state:
        selected = st.session_state.selected_game
        
        st.divider()
        st.subheader(f"Recommendations based on: **{selected['name']}**")
        
        try:
            # Get recommendations
            with st.spinner("Finding similar games..."):
                recommendations = recommender.recommend(selected["id"], top_k=10)
            
            # Display recommendations
            for i, row in enumerate(recommendations.iter_rows(named=True), 1):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{i}. {row['name']}**")
                    
                    # Score breakdown
                    similarity = row.get("cosine_similarity", 0)
                    year_sim = row.get("normalized_year_similarity", 0)
                    time_sim = row.get("normalized_playing_time_similarity", 0)
                    rating = row.get("normalized_avg_rating", 0)
                    popularity = row.get("normalized_popularity", 0)
                    
                    st.caption(
                        f"🎯 Similarity: {similarity:.0%} · "
                        f"📅 Year: {year_sim:.0%} · "
                        f"⏱️ Time: {time_sim:.0%} · "
                        f"⭐ Rating: {rating:.0%} · "
                        f"🔥 Popularity: {popularity:.0%}"
                    )
                
                with col2:
                    score = row.get("final_score", row.get("cosine_similarity", 0))
                    st.metric("Score", f"{score:.2f}")
                    
        except Exception as e:
            st.error(f"Error getting recommendations: {e}")
    
    # Footer
    st.divider()
    st.caption(".")


if __name__ == "__main__":
    main()
