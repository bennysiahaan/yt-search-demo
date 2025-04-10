import json
import os
import requests

import polars as pl

from config import *
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer


def parse_video_records(response: requests.models.Response):
    """
        Function to extract YouTube video data from GET response
    """
    video_record_list = []

    for raw_item in json.loads(response.text)['items']:
        # only execute for YouTube videos
        if raw_item['id']['kind'] != "youtube#video":
            continue

        video_record = {}
        video_record['video_id'] = raw_item['id']['videoId']
        video_record['datetime'] = raw_item['snippet']['publishedAt']
        video_record['title'] = raw_item['snippet']['title']

        video_record_list.append(video_record)
    
    return video_record_list

def get_video_records(config: Config) -> pl.DataFrame:
    """
        Function to fetch video metadata via YouTube data API
    """
    channel_id = config.apicalls[config.APICALLS_YT_CONFIG]["ChannelID"]
    url = config.apicalls[config.APICALLS_YT_CONFIG]["DataV3URL"]
    api_key = os.getenv(YT_API_KEY_ENV_NAME)

    page_token = None
    video_records = []

    while page_token != 0:
        params = {
            'key': api_key,
            'channel_id': channel_id,
            'part': ["snippet", "id"],
            'order': "date",
            'maxResults': 50,
            'pageToken': page_token,
        }

        response = requests.get(url, params=params)

        video_records += parse_video_records(response)

        try:
            page_token = json.loads(response.text)['nextPageToken']
        except:
            page_token = 0
    
    return pl.DataFrame(video_records)

def extract_text(transcript: list) -> str:
    """
        Function to extract text from transcript dictionary
    """

    text_list = [transcript[i]['text'] for i in range(len(transcript))]
    return ' '.join(text_list)

def get_video_transcripts(df: pl.DataFrame) -> pl.DataFrame:
    """
        Function to fetch video transcripts via YouTube API
    """
    ytt_api = YouTubeTranscriptApi()

    transcript_text_list = []

    for i, video_id in enumerate(df['video_id']):
        try:
            transcript = ytt_api.fetch(video_id=video_id, languages=["en"]).to_raw_data()
            transcript_text = extract_text(transcript)
        except:
            transcript_text = "n/a"
        
        transcript_text_list.append(transcript_text)
    
    return df.with_columns(pl.Series(name="transcript", values=transcript_text_list))

def set_data_types(df: pl.DataFrame) -> pl.DataFrame:
    """
        Function to change data types of columns in the dataframe
    """
    # change datetime to pl.Datetime dtype
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime))

    return df

def handle_special_characters(config: Config, df: pl.DataFrame) -> pl.DataFrame:
    """
        Function to handle special character strings in video
        transcripts and titles
    """
    data_transform_config = config.models[config.MODELS_DATA_TRANSFORM_CONFIG]
    special_strings = json.loads(data_transform_config["SpecialStrings"])
    special_strings_replacement = json.loads(data_transform_config["SpecialStringsReplacement"])

    for i in range(len(special_strings)):
        df = df.with_columns(df['title'].str.replace(
            special_strings[i],
            special_strings_replacement[i],
        ).alias('title'))
        df = df.with_columns(df['transcript'].str.replace(
            special_strings[i],
            special_strings_replacement[i],
        ).alias('transcript'))
    
    return df

def transform_data(config: Config, df: pl.DataFrame) -> pl.DataFrame:
    """
        Function to transform data
    """
    df = set_data_types(df)
    df = handle_special_characters(config, df)
    return df

def generate_text_embeddings(config: Config, df: pl.DataFrame) -> pl.DataFrame:
    """
        Function to generate text embeddings from title and
        transcripts columns
    """
    model_name = config.models[config.MODELS_SENTENCE_TRANSFORMERS_CONFIG]["ModelName"]
    column_name_list = ['title', 'transcript']

    model = SentenceTransformer(model_name)

    for column_name in column_name_list:
        embedding_arr = model.encode(df[column_name].to_list())

        schema_dict = {f'{column_name}_embedding-{i}': float for i in range(embedding_arr.shape[1])}
        df_embedding = pl.DataFrame(embedding_arr, schema=schema_dict)

        df = pl.concat([df, df_embedding], how='horizontal')
    
    return df
