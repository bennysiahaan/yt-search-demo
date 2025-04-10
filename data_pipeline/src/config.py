import configparser
import os

YT_API_KEY_ENV_NAME = "YT_DATA_V3_API_KEY"

class Config():

    APICALLS_CONFIG_FILE = "apicalls.ini"
    MODELS_CONFIG_FILE = "models.ini"

    APICALLS_YT_CONFIG = "YouTubeAPI"
    MODELS_DATA_TRANSFORM_CONFIG = "DataTransform"
    MODELS_SENTENCE_TRANSFORMERS_CONFIG = "SentenceTransformers"

    def __init__(self):
        self.apicalls = configparser.ConfigParser()
        self.apicalls.read(os.path.join("data_pipeline", self.APICALLS_CONFIG_FILE))

        self.models = configparser.ConfigParser()
        self.models.read(os.path.join("data_pipeline", self.MODELS_CONFIG_FILE))
