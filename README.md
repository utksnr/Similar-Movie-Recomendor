# Similar-Movie-Recomendor
Similar movie recomendation engine.

Hi, this is a content-based recommendation engine and data exploration dashboard I built using Python, FastAPI, and Streamlit. 

Instead of relying on basic collaborative filtering or just matching genres, this project uses Natural Language Processing (NLP) to understand the underlying themes of movies based on user-generated tags.

I intentionally developed this on the **MovieLens Small Dataset** (100k ratings, ~10k movies) as an MVP. The main goal here was to design a mathematically sound, modular architecture and get the API logic right. Once the algorithm's stability is proven on this scale, the next step is migrating the similarity matrix to a vector database (like FAISS) to handle the 33M+ dataset without memory bottlenecks.

## How the Algorithm Works

Here is a breakdown of the statistical and machine learning approach used in the engine:

* **Feature Fusion:** I combined explicit categorical variables (official genres) with implicit, crowdsourced text features (user tags). This creates a single, comprehensive text profile for each movie.
* **Vector Embeddings:** Used Hugging Face's `sentence-transformers` (`all-MiniLM-L6-v2`) to encode these fused text profiles into dense, high-dimensional vectors.
* **Similarity Search:** The engine calculates the Cosine Similarity between these vectors to find movies that are actually thematically close, not just superficially similar.
* **Statistical Filtering:** To prevent outliers from ruining the recommendations (e.g., a terrible movie that happens to have a single 5-star rating), I added dynamic thresholds for minimum vote counts and average ratings.
* **Cluster Analytics:** The Streamlit frontend doesn't just show a list of movies; it provides the mean match score ($\mu$) and standard deviation ($\sigma$) of the recommended cluster. A low standard deviation indicates that the algorithm's choices are stable and closely packed in the vector space.

## Tech Stack

* **ML / NLP:** `sentence-transformers`, `scikit-learn`
* **Data Handling:** `pandas`, `numpy`
* **Backend:** `FastAPI`, `uvicorn`
* **Frontend:** `Streamlit`

## Running the Project Locally

1. Clone the repo to your machine.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
