import numpy as np
import polars
import sklearn

# helper function
def return_search_result_index(
    query: str,
    df: polars.LazyFrame,
    model,
    dist: sklearn.metrics._dist_metrics,
) -> np.ndarray:
    """
        Function to return the indices of top search results
    """

    # embed query
    query_embedding = model.encode(query).reshape(1, -1)

    # get column names without triggering schema resolution warning
    column_names = df.collect_schema().names()

    # compute distances between query and titles/transcripts
    dist_arr = dist.pairwise(df.select(column_names[4:388]).collect(), query_embedding)
    dist_arr += dist.pairwise(df.select(column_names[388:]).collect(), query_embedding)

    # search parameters
    threshold = 40 # eye balled threshold for manhattan distance
    top_k = 5

    # evaluate videos close to query based on threshold
    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()

    # keep top k closest videos
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    return idx_below_threshold[idx_sorted][:top_k]
