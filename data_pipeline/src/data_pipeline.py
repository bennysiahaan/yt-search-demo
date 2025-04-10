from functions import *
import time
import datetime

DATA_STORAGE_DIR = os.path.join("backend", "app", "data")

if __name__ == "__main__":
    print(f"Starting data pipeline at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("---------------------------------------------")

    # Step 0: Parse config files
    t0 = time.time()
    config = Config()
    t1 = time.time()
    print("Parsing configuration files: Done")
    print(f"---> Completed in {t1-t0} seconds.\n")

    # Step 1: Extract video IDs
    t0 = time.time()
    video_records = get_video_records(config)
    t1 = time.time()
    print("Step 1 - Extracting video IDs: Done")
    print(f"---> Completed in {t1-t0} seconds.\n")

    # Step 2: Extract transcripts for videos
    t0 = time.time()
    video_records = get_video_transcripts(video_records)
    t1 = time.time()
    print("Step 2 - Extracting transcripts for videos: Done")
    print(f"---> Completed in {t1-t0} seconds.\n")

    # Step 3: Transform data
    t0 = time.time()
    video_records = transform_data(config, video_records)
    t1 = time.time()
    print("Step 3 - Transforming data: Done")
    print(f"---> Completed in {t1-t0} seconds.\n")

    # Step 4: Generate text embeddings
    t0 = time.time()
    video_records = generate_text_embeddings(config, video_records)
    t1 = time.time()
    print("Step 4 - Generating text embeddings: Done")
    print(f"---> Completed in {t1-t0} seconds.\n")

    # Step 5: Store data
    t0 = time.time()
    video_records.write_parquet(os.path.join(DATA_STORAGE_DIR, "video-index.parquet"))
    t1 = time.time()
    print("Step 5 - Storing data: Done")
    print(f"---> Completed in {t1-t0} seconds.\n")
