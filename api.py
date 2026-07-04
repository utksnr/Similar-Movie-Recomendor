import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from fastapi import FastAPI, HTTPException
import uvicorn

warnings.filterwarnings('ignore')

# Engine
def load_and_preprocess_data(data_path: str = 'data/') -> pd.DataFrame:
    print("[INFO] Loading datasets...")
    movies = pd.read_csv(f'{data_path}movies.csv')
    tags = pd.read_csv(f'{data_path}tags.csv')
    ratings = pd.read_csv(f'{data_path}ratings.csv')

    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        vote_count=('rating', 'count')
    ).reset_index()

    movies = movies.merge(movie_stats, on='movieId', how='left')
    movies['avg_rating'] = movies['avg_rating'].fillna(0)
    movies['vote_count'] = movies['vote_count'].fillna(0)

    movies['genres'] = movies['genres'].str.replace('|', ' ').str.lower()
    movies['genres'] = movies['genres'].str.replace('(no genres listed)', '')

    tags.dropna(subset=['tag'], inplace=True)
    tags['tag'] = tags['tag'].astype(str).str.lower()
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

    movies = movies.merge(tags_grouped, on='movieId', how='left')
    movies['tag'] = movies['tag'].fillna('')

    movies['metadata'] = movies['genres'] + " " + movies['tag']
    movies.drop(['tag', 'genres'], axis=1, inplace=True)
    return movies

class SemanticRecommender:
    def __init__(self, df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2'):
        self.df = df.reset_index(drop=True)
        print(f"[INFO] Loading NLP Model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.similarity_matrix = None
        
    def fit(self) -> None:
        print("[INFO] Generating semantic embeddings...")
        self.embeddings = self.model.encode(self.df['metadata'].tolist(), show_progress_bar=True)
        print("[INFO] Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        print("[INFO] Engine is ready!")

    def recommend(self, movie_title: str, top_n: int = 5, min_votes: int = 20, min_rating: float = 3.0):
        try:
            movie_idx = self.df[self.df['title'].str.contains(movie_title, case=False, regex=False)].index[0]
        except IndexError:
            return "Movie not found"

        raw_scores = self.similarity_matrix[movie_idx]
        recommendations = self.df.copy()
        recommendations['match_score_pct'] = np.round(raw_scores * 100, 2)
        recommendations = recommendations.drop(movie_idx)
        recommendations = recommendations.sort_values(by='match_score_pct', ascending=False)

        filtered_recs = recommendations[
            (recommendations['vote_count'] >= min_votes) & 
            (recommendations['avg_rating'] >= min_rating)
        ]
        return filtered_recs[['title', 'match_score_pct', 'avg_rating', 'vote_count']].head(top_n)

    def explore_by_tag(self, tag: str, top_n: int = 15, min_votes: int = 20):
        mask = self.df['metadata'].str.contains(tag, case=False, regex=False)
        filtered = self.df[mask & (self.df['vote_count'] >= min_votes)].copy()
        
        if filtered.empty:
            return "No movies found with this tag"
            
        sorted_df = filtered.sort_values(by='avg_rating', ascending=False).head(top_n)
        return sorted_df[['title', 'avg_rating', 'vote_count', 'metadata']]

    def compare_movies(self, movie1: str, movie2: str):
        try:
            idx1 = self.df[self.df['title'].str.contains(movie1, case=False, regex=False)].index[0]
            title1 = self.df.iloc[idx1]['title']
        except IndexError:
            return f"'{movie1}' not found"
            
        try:
            idx2 = self.df[self.df['title'].str.contains(movie2, case=False, regex=False)].index[0]
            title2 = self.df.iloc[idx2]['title']
        except IndexError:
            return f"'{movie2}' not found"

        score = self.similarity_matrix[idx1][idx2]
        return {
            "movie1": title1,
            "movie2": title2,
            "similarity_score": float(np.round(score * 100, 2))
        }

#FastAPI backend
app = FastAPI(title="Movie Recommender API")
recommender = None

@app.on_event("startup")
def startup_event():
    global recommender
    movies_df = load_and_preprocess_data()
    recommender = SemanticRecommender(movies_df)
    recommender.fit()

@app.get("/recommend")
def get_recommendations(movie: str, top_n: int = 5, min_votes: int = 20, min_rating: float = 3.0):
    result = recommender.recommend(movie, top_n=top_n, min_votes=min_votes, min_rating=min_rating)
    if isinstance(result, str): raise HTTPException(status_code=404, detail=result)
    return result.to_dict(orient="records")

@app.get("/explore")
def explore_tag(tag: str, top_n: int = 15, min_votes: int = 20):
    result = recommender.explore_by_tag(tag, top_n=top_n, min_votes=min_votes)
    if isinstance(result, str): raise HTTPException(status_code=404, detail=result)
    return result.to_dict(orient="records")

@app.get("/compare")
def compare_two_movies(movie1: str, movie2: str):
    result = recommender.compare_movies(movie1, movie2)
    if isinstance(result, str): raise HTTPException(status_code=404, detail=result)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)