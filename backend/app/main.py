from fastapi import FastAPI
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
from app.functions import return_search_result_index

def load_model():
    model_name = "all-MiniLM-L6-v2"
    model_path = f'app/data/{model_name}'

    try:
        print(f"Loading model in '{model_path}'...")
        model = SentenceTransformer(model_path)
        print(f"Model was found.")
    except:
        print("Model could not be found!")
        print(f"Creating a new model and storing it in '{model_path}'...")
        model = SentenceTransformer(model_name)
        model.save(model_path)
        print("Successful.")
    
    return model

def run(app: FastAPI, model: SentenceTransformer):

    # API operations
    @app.get("/")
    def health_check():
        return {'health_check': 'OK'}
    
    @app.get("/info")
    def info():
        return {'name': 'yt-search', 'description': 'Search API for Shaw Talebi\'s YouTube videos.'}
    
    @app.get("/search")
    def search(query: str):

        # load video index
        df = pl.scan_parquet('app/data/video-index.parquet')

        # create distance metric object
        dist_name = "manhattan"
        dist = DistanceMetric.get_metric(dist_name)

        idx_result = return_search_result_index(query, df, model, dist)
        return df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False)

model = load_model()

# create FastAPI object
app = FastAPI()

run(app, model)
