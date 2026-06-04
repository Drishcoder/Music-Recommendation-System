import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🎵 Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a0f2e 50%, #0f1a2e 100%);
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a0f2e 50%, #0f1a2e 100%);
    }
    
    .main {
        background: transparent;
    }
    
    h1 {
        background: linear-gradient(135deg, #ff006e 0%, #fb5607 50%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5em !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        margin-bottom: 5px !important;
    }
    
    h2 {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.8em;
        margin-top: 20px;
    }
    
    h3 {
        color: #f0f0f0;
        font-weight: 600;
        font-size: 1.3em;
    }
    
    /* Premium Cards */
    .premium-card {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.1) 0%, rgba(251, 86, 7, 0.1) 100%);
        border: 1px solid rgba(255, 0, 110, 0.3);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(255, 0, 110, 0.15);
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(255, 0, 110, 0.25);
        border-color: rgba(255, 0, 110, 0.5);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%);
        border-left: 4px solid #ff006e;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 15px;
        box-shadow: 0 8px 24px rgba(255, 0, 110, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 0, 110, 0.2);
    }
    
    .recommendation-card:hover {
        transform: translateX(10px);
        box-shadow: 0 12px 36px rgba(255, 0, 110, 0.3);
        border-color: rgba(255, 0, 110, 0.4);
    }
    
    .recommendation-card h3 {
        color: #ff006e;
        margin: 0 0 10px 0;
        font-size: 1.4em;
    }
    
    .recommendation-card p {
        margin: 8px 0;
        color: #e0e0e0;
        font-size: 1.05em;
    }
    
    .rec-number {
        background: linear-gradient(135deg, #ff006e 0%, #fb5607 100%);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5em;
        font-weight: 800;
        box-shadow: 0 4px 15px rgba(255, 0, 110, 0.4);
    }
    
    .metric-badge {
        background: linear-gradient(135deg, #ff006e 0%, #fb5607 100%);
        padding: 12px 20px;
        border-radius: 25px;
        color: white;
        font-weight: 600;
        display: inline-block;
        margin: 5px 5px 5px 0;
        font-size: 0.95em;
        box-shadow: 0 4px 15px rgba(255, 0, 110, 0.3);
    }
    
    .setting-section {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.08) 0%, rgba(251, 86, 7, 0.08) 100%);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 0, 110, 0.2);
        margin-bottom: 20px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff006e 0%, #fb5607 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 0, 110, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 0, 110, 0.4);
    }
    
    .stSelectbox, .stSlider {
        background: transparent;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.05) 0%, rgba(251, 86, 7, 0.05) 100%) !important;
        border: 1px solid rgba(255, 0, 110, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    .subtitle {
        color: #a0a0c0;
        font-size: 1.1em;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .stat-box {
        background: linear-gradient(135deg, rgba(255, 0, 110, 0.15) 0%, rgba(251, 86, 7, 0.15) 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 0, 110, 0.3);
    }
    
    .stat-box h4 {
        color: #ff006e;
        margin: 0 0 10px 0;
    }
    
    .stat-box p {
        color: #ffffff;
        font-size: 2em;
        font-weight: 700;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# Load and cache data
@st.cache_resource
def load_data():
    df = pd.read_csv("dataset.csv")
    df_clean = df.drop_duplicates(subset=["track_name", "artists"])
    df_clean = df_clean.dropna(subset=["artists", "album_name", "track_name"])

    target_genres = ['classical', 'black-metal', 'dance', 'acoustic', 'hip-hop', 'indian', 'rock', 'pop-film']
    df_filtered = df_clean[df_clean['track_genre'].isin(target_genres)].copy()

    return df_filtered


@st.cache_resource
def build_model(df_filtered):
    audio_features = ["danceability", "energy", "loudness", "speechiness",
                      "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

    X = df_filtered[audio_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    similarity_matrix = cosine_similarity(X_pca)
    df_filtered_reset = df_filtered.reset_index(drop=True)

    return similarity_matrix, df_filtered_reset, X_pca, audio_features


def recommend_song_with_filter(song_title, df_filtered, similarity_matrix, num_recommendation=5):
    """Recommend songs with cultural filtering (Indian vs. Non-Indian)"""
    try:
        input_idx = df_filtered[df_filtered['track_name'].str.lower() == song_title.lower()].index[0]
        input_song = df_filtered.iloc[input_idx]
    except IndexError:
        return None, None

    input_genre = str(input_song['track_genre']).lower()
    sim_scores = list(enumerate(similarity_matrix[input_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    filtered_matches = []
    scores_list = []

    for index, score in sim_scores:
        if index == input_idx:
            continue

        candidate_genre = str(df_filtered.loc[index, 'track_genre']).lower()

        if 'indian' in input_genre and 'indian' in candidate_genre:
            filtered_matches.append(index)
            scores_list.append(score)
        elif 'indian' not in input_genre and 'indian' not in candidate_genre:
            filtered_matches.append(index)
            scores_list.append(score)

        if len(filtered_matches) == num_recommendation:
            break

    if not filtered_matches:
        return None, None

    recommendations = df_filtered.iloc[filtered_matches][['track_name', 'artists', 'track_genre']]
    return recommendations, scores_list


# Load data and build model
df_filtered = load_data()
similarity_matrix, df_filtered_reset, X_pca, audio_features = build_model(df_filtered)

# ── Header ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1], gap="large")
with col1:
    st.markdown("<h1>🎵 Music Recommendation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>✨ Discover your next favorite song using AI-powered similarity analysis</p>",
                unsafe_allow_html=True)

st.markdown("---")

# ── Controls (left panel only) ───────────────────────────────────────────────
col1, col2 = st.columns([1, 2.5], gap="large")

with col1:
    st.markdown("<div class='setting-section'>", unsafe_allow_html=True)
    st.markdown("<h3>⚙️ Recommendation Engine</h3>", unsafe_allow_html=True)

    # Song input
    all_songs = sorted(df_filtered_reset['track_name'].unique())
    selected_song = st.selectbox(
        "🔍 Search for a song",
        all_songs,
        help="Choose a song to get personalized recommendations"
    )

    # Number of recommendations
    num_recommendations = st.slider(
        "📊 Number of songs",
        1, 10, 5,
        help="How many recommendations do you want?"
    )

    # Full-width button — no nested columns
    if st.button("🚀 Get Recommendations", use_container_width=True):
        st.session_state.search_triggered = True

    st.markdown("</div>", unsafe_allow_html=True)

# col2 intentionally left empty; recommendations render full-width below

# ── Recommendations (full-width, below controls) ─────────────────────────────
if st.session_state.get('search_triggered', False):
    recommendations, scores = recommend_song_with_filter(
        selected_song,
        df_filtered_reset,
        similarity_matrix,
        num_recommendations
    )

    if recommendations is not None:
        # Selected song info card
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<h3>🎼 Your Selected Song</h3>", unsafe_allow_html=True)
        input_song_info = df_filtered_reset[
            df_filtered_reset['track_name'].str.lower() == selected_song.lower()
        ].iloc[0]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                f"<p style='color: #a0a0c0; font-size: 0.9em; margin: 0;'>TITLE</p>"
                f"<p style='color: #ff006e; font-weight: 700; margin: 5px 0;'>{input_song_info['track_name']}</p>",
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown(
                f"<p style='color: #a0a0c0; font-size: 0.9em; margin: 0;'>ARTIST</p>"
                f"<p style='color: #ffffff; font-weight: 700; margin: 5px 0;'>{input_song_info['artists']}</p>",
                unsafe_allow_html=True
            )
        with col_c:
            st.markdown(
                f"<p style='color: #a0a0c0; font-size: 0.9em; margin: 0;'>GENRE</p>"
                f"<span class='metric-badge'>{input_song_info['track_genre'].upper()}</span>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Recommendation cards
        st.markdown("<h3>🎯 Recommended For You</h3>", unsafe_allow_html=True)

        for idx, (_, rec_row) in enumerate(recommendations.iterrows(), 1):
            similarity_score = scores[idx - 1] if idx - 1 < len(scores) else 0

            col_number, col_content = st.columns([0.06, 1], gap="small")
            with col_number:
                st.markdown(f"<div class='rec-number'>{idx}</div>", unsafe_allow_html=True)
            with col_content:
                st.markdown(f"""
                <div class='recommendation-card'>
                    <h3>{rec_row['track_name']}</h3>
                    <p>👤 <b>{rec_row['artists']}</b></p>
                    <p>🎸 {rec_row['track_genre'].upper()}</p>
                    <p style='color: #fb5607; font-weight: 600;'>⭐ Match Score: <b>{similarity_score:.1%}</b></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='text-align: center; padding: 40px; "
            "background: linear-gradient(135deg, rgba(255, 0, 110, 0.1) 0%, rgba(251, 86, 7, 0.1) 100%); "
            "border-radius: 15px; border: 1px solid rgba(255, 0, 110, 0.3);'>"
            "<h3 style='color: #ff006e;'>❌ Song Not Found</h3>"
            "<p style='color: #a0a0c0;'>Try searching for a different song</p></div>",
            unsafe_allow_html=True
        )

st.markdown("---")

# ── Visualizations ────────────────────────────────────────────────────────────
st.markdown("<h2>📊 Discover Your Music Space</h2>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🌌 PCA Galaxy View", "🎼 Audio DNA", "🎯 Genre Map"])

with tab1:
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["Genre"] = df_filtered_reset['track_genre'].values

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Genre",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'PC1': "Axis 1", 'PC2': "Axis 2"},
        title="Musical Universe - 2D PCA Projection",
        opacity=0.8,
        template="plotly_dark",
        size_max=12
    )
    fig.update_layout(
        plot_bgcolor="rgba(15, 15, 30, 0.5)",
        paper_bgcolor="rgba(15, 15, 30, 0)",
        hovermode='closest',
        font=dict(family="Poppins", size=12, color="#ffffff"),
        title_font_size=20,
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(255,255,255,0.3)")))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_feat1, col_feat2 = st.columns([2, 1])
    with col_feat1:
        selected_feature = st.selectbox("🎶 Select a feature to explore:", audio_features, key="feature_select")
    with col_feat2:
        st.metric("Feature", selected_feature.upper())

    fig = go.Figure()
    colors_gradient = ['#FF006E', '#FB5607', '#FF8C00', '#FFB703', '#FB5607']

    for i, genre in enumerate(sorted(df_filtered_reset['track_genre'].unique())):
        genre_data = df_filtered_reset[df_filtered_reset['track_genre'] == genre][selected_feature]
        fig.add_trace(go.Box(
            y=genre_data,
            name=genre.upper(),
            marker_color=colors_gradient[i % len(colors_gradient)],
            boxmean='sd'
        ))

    fig.update_layout(
        title=f"<b>{selected_feature.capitalize()} Distribution by Genre</b>",
        yaxis_title=selected_feature.capitalize(),
        xaxis_title="Genre",
        template="plotly_dark",
        showlegend=True,
        plot_bgcolor="rgba(15, 15, 30, 0.5)",
        paper_bgcolor="rgba(15, 15, 30, 0)",
        font=dict(family="Poppins", size=12, color="#ffffff"),
        title_font_size=18,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    genre_counts = df_filtered_reset['track_genre'].value_counts().sort_values(ascending=False)
    fig = px.pie(
        values=genre_counts.values,
        names=genre_counts.index,
        title="<b>Genre Distribution in Your Collection</b>",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_dark",
        hole=0.3
    )
    fig.update_layout(
        plot_bgcolor="rgba(15, 15, 30, 0)",
        paper_bgcolor="rgba(15, 15, 30, 0)",
        font=dict(family="Poppins", size=12, color="#ffffff"),
        title_font_size=18,
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='rgba(15, 15, 30, 0.8)', width=2))
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3>🚀 About This System</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='premium-card'>
        <p style='margin: 0; color: #e0e0e0; line-height: 1.6;'>
        This cutting-edge recommendation engine uses:
        </p>
        <ul style='color: #a0a0c0; margin-top: 10px;'>
            <li><b style='color: #ff006e;'>PCA</b> - Dimensionality reduction</li>
            <li><b style='color: #ff006e;'>Cosine Similarity</b> - Smart matching</li>
            <li><b style='color: #ff006e;'>Genre Filtering</b> - Cultural awareness</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>📈 Dataset Intelligence</h3>", unsafe_allow_html=True)

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(f"""
        <div class='stat-box'>
            <h4>Total Songs</h4>
            <p>{len(df_filtered_reset)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class='stat-box'>
            <h4>Genres</h4>
            <p>{df_filtered_reset['track_genre'].nunique()}</p>
        </div>
        """, unsafe_allow_html=True)

    col_stat3, col_stat4 = st.columns(2)
    with col_stat3:
        st.markdown(f"""
        <div class='stat-box'>
            <h4>Features</h4>
            <p>{len(audio_features)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col_stat4:
        st.markdown(f"""
        <div class='stat-box'>
            <h4>Dimensions</h4>
            <p>2D</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3>🎵 Genre Library</h3>", unsafe_allow_html=True)
    st.markdown("<div style='background: rgba(255,0,110,0.05); border-radius: 10px; padding: 15px;'>",
                unsafe_allow_html=True)
    for genre in sorted(df_filtered_reset['track_genre'].unique()):
        count = len(df_filtered_reset[df_filtered_reset['track_genre'] == genre])
        st.markdown(
            f"<p style='margin: 8px 0; color: #e0e0e0;'>"
            f"<span style='color: #ff006e;'>■</span> <b>{genre.capitalize()}</b> "
            f"<span style='color: #a0a0c0;'>({count})</span></p>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #a0a0c0; font-size: 0.9em;'>Built with ❤️ using Streamlit</p>",
        unsafe_allow_html=True
    )