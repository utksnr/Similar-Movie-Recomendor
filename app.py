import streamlit as st
import requests
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Preprocess Engine | Recommender", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced CSS to hide Streamlit elements and create a custom UI feel
st.markdown("""
    <style>
    /* Hide standard Streamlit header and main menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean metric styling */
    div[data-testid="stMetricValue"] { 
        font-size: 1.6rem; 
        font-weight: 600;
        color: #2c3e50; 
    }
    
    /* Custom divider */
    hr {
        margin-top: 1rem;
        margin-bottom: 2rem;
        border: 0;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Strict Developer Signature (Ledger format) */
    .footer {
        position: fixed;
        bottom: 20px;
        right: 20px;
        text-align: right;
        font-size: 12px;
        color: #666666;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        line-height: 1.4;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar (Clean & Technical)
with st.sidebar:
    st.markdown("### Preprocess Engine")
    st.caption("v1.0.0 | Semantic Search Module")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("#### Parameters")
    top_n = st.slider("Result Limit", min_value=2, max_value=20, value=5)
    min_votes = st.slider("Minimum Votes", min_value=0, max_value=500, value=30, step=10)
    min_rating = st.slider("Minimum Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("Backend: FastAPI + SentenceTransformers")

# 4. Main Title
st.markdown("## Content-Based Semantic Recommender")
st.caption("Utilizing NLP feature fusion to compute multi-dimensional similarities across movie metadata.")
st.markdown("<hr>", unsafe_allow_html=True)

# 5. Clean Tab Structure
tab_rec, tab_explore, tab_compare = st.tabs(["Recommendation", "Tag Analytics", "Pairwise Comparison"])

# ==========================================
# TAB 1: RECOMMENDATION
# ==========================================
with tab_rec:
    col1, col2 = st.columns([4, 1])
    with col1: 
        movie_name = st.text_input("Target Movie", "Matrix", key="rec_input", label_visibility="collapsed")
    with col2: 
        search_button = st.button("Run Analysis", use_container_width=True)

    if search_button:
        try:
            with st.spinner('Computing cosine similarities...'):
                res = requests.get("http://127.0.0.1:8000/recommend", params={"movie": movie_name, "top_n": top_n, "min_votes": min_votes, "min_rating": min_rating})
            
            if res.status_code == 200:
                df = pd.DataFrame(res.json())
                df.rename(columns={'title': 'Title', 'match_score_pct': 'Match (%)', 'avg_rating': 'Rating', 'vote_count': 'Votes'}, inplace=True)
                
                # Raw table display
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Statistical section (Minimalist)
                st.markdown("<br><b>Cluster Statistics</b>", unsafe_allow_html=True)
                stat1, stat2, stat3 = st.columns(3)
                stat1.metric("Mean Match (μ)", f"{df['Match (%)'].mean():.2f}%")
                stat2.metric("Std. Deviation (σ)", f"{df['Match (%)'].std():.2f}")
                stat3.metric("Mean Rating", f"{df['Rating'].mean():.2f}")
            else:
                st.error("Target entity not found in the dataset.")
        except: 
            st.error("Connection refused. Verify backend status.")

# ==========================================
# TAB 2: TAG ANALYTICS
# ==========================================
with tab_explore:
    t_col1, t_col2, t_col3 = st.columns([3, 1, 1])
    with t_col1: 
        tag_input = st.text_input("Metadata Tag", "cyberpunk", label_visibility="collapsed")
    with t_col2: 
        tag_min_votes = st.number_input("Min Votes", 0, 500, 50, key="tag_votes")
    with t_col3:
        tag_button = st.button("Query Database", use_container_width=True)
    
    if tag_button:
        try:
            with st.spinner('Querying...'):
                res = requests.get("http://127.0.0.1:8000/explore", params={"tag": tag_input, "top_n": 15, "min_votes": tag_min_votes})
            
            if res.status_code == 200:
                df_tag = pd.DataFrame(res.json())
                df_tag.rename(columns={'title': 'Title', 'avg_rating': 'Rating', 'vote_count': 'Votes', 'metadata': 'Features'}, inplace=True)
                
                view_col1, view_col2 = st.columns([1, 1])
                with view_col1:
                    st.dataframe(df_tag[['Title', 'Rating', 'Votes']], use_container_width=True, hide_index=True)
                with view_col2:
                    st.scatter_chart(df_tag, x="Votes", y="Rating", size="Rating", color="#2c3e50")
            else:
                st.info("No sufficient data points found for this criteria.")
        except: 
            st.error("Connection refused.")

# ==========================================
# TAB 3: PAIRWISE COMPARISON
# ==========================================
with tab_compare:
    c_col1, c_col2 = st.columns(2)
    with c_col1: 
        movie1 = st.text_input("Movie A", "Star Wars")
    with c_col2: 
        movie2 = st.text_input("Movie B", "Star Trek")
    
    if st.button("Calculate Similarity", use_container_width=False):
        try:
            res = requests.get("http://127.0.0.1:8000/compare", params={"movie1": movie1, "movie2": movie2})
            if res.status_code == 200:
                data = res.json()
                score = data['similarity_score']
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"#### Similarity Score: {score}%")
                st.caption(f"Calculated distance between '{data['movie1']}' and '{data['movie2']}' in the vector space.")
            else:
                st.error("One or both inputs are invalid.")
        except: 
            st.error("Connection refused.")

# 6. Strict Developer Footer
st.markdown("""
    <div class="footer">
        Developer<br>
        Utku Şener<br>
        Statistics Student at METU
    </div>
""", unsafe_allow_html=True)