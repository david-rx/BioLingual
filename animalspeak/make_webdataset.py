"""
Make a webdataset for training or eval with CLAP
"""
import multiprocessing
import csv
import json
import os
import shutil
import tarfile
from typing import List
from pydub import AudioSegment
import numpy as np
import pandas as pd
import os

AUDIO_PATH_COLUMN = "path"
CAPTION_COLUMN = "caption"
CAPTION_COLUMN_2 = "caption2"

def is_json_valid(file_path):
    try:
        with open(file_path, 'r') as json_file:
            json.load(json_file)
        return True
    except json.JSONDecodeError:
        return False


def make_sizes_json(sizes: dict, sizes_output_path: str):
    """Given a list of sizes, makes a sizes json files like
    {
    "path1": size1,
    "path2": size2,
    ...
    }
    """
    # Write the dictionary as JSON to the output file
    with open(sizes_output_path, 'w') as json_file:
        json.dump(sizes, json_file)

# function to create a single shard
def create_shard(df, output_tar_path, shard_id):
    size = 0
    with tarfile.open(output_tar_path, "w") as tar:
        for index, row in df.iterrows():
            audio_path = row[AUDIO_PATH_COLUMN]
            audio_path = "/Users/davidrobinson/Code/animals/beans/" + audio_path
            caption = row[CAPTION_COLUMN]
            caption2 = row[CAPTION_COLUMN_2]
            captions = [caption, caption2]
            if pd.isna(caption):
                continue
            if pd.isna(caption2):
                captions = [caption]

            audio_basename = os.path.basename(audio_path)
            audio_id = os.path.splitext(audio_basename)[0]
            try:

                if not os.path.isfile(audio_path):
                    
                    continue
                
                with open(f"{audio_id}.json", "w") as jsonfile:
                    jsonfile.write(json.dumps({"text": captions}))

                if not is_json_valid(f"{audio_id}.json"):
                    print("invalid json", audio_id)
                    continue
                
                tar.add(f"{audio_id}.json", arcname=f"{audio_id}.json")
                tar.add(audio_path, arcname=f"{audio_id}.flac")
                
                size += 1
                os.remove(f"{audio_id}.json")
            except Exception as e:
                print("exception", e)
                continue
    return size

# function to convert CSV data to webdataset format
def csv_to_webdataset(csv_path, output_path, max_files_per_shard=50000):
    df = pd.read_csv(csv_path)
    n_shards = (len(df) // max_files_per_shard) + (len(df) % max_files_per_shard != 0)

    sizes = {}

    # Create a process pool
    with multiprocessing.Pool() as pool:
        results = []
        for shard_id in range(n_shards):
            start = shard_id * max_files_per_shard
            end = min((shard_id + 1) * max_files_per_shard, len(df))
            df_shard = df.iloc[start:end]
            output_tar_path = os.path.join(output_path, f"{shard_id}.tar")
            results.append(pool.apply_async(create_shard, args=(df_shard, output_tar_path, shard_id)))

        pool.close()
        pool.join()

        # collect sizes
        for shard_id, result in enumerate(results):
            sizes[f"{shard_id}.tar"] = result.get()

    make_sizes_json(sizes, os.path.join(output_path, "sizes.json"))

# usage
if __name__ == "__main__":
    csv_to_webdataset("../animals/beans/test_set.csv", "animals_webdataset/test/") # make webdataset for retrieval eval
