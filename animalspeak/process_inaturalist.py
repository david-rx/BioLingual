import os
import pandas as pd
import requests
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import concurrent

from animalspeak.chatgpt.client import ChatGPTClient
from animalspeak.captioning.captioner import Captioner

DESCRIPTION_COLUMN = "description"
LABEL_COLUMN = "common_name"
NEW_LABEL_COLUMN = "species"
URL_COLUMN = "sound_url"
NEW_URL_COLUMN = "url"
CAPTION_COLUMN = "caption"
BANNED_WORDS = ["siento", "description", "information", "caption"]
SPECIES_COUNT_THRESHOLD = 30

def label_to_caption(label):
    return f"The sound of a {label}"

def get_species_count(df, species):
    counts = df[df["species"] == species]["count"].values
    if counts:
        return counts[0]
    else:
        return 0

def split_audio(audio, duration_ms, max_chunks = 4):
    duration = len(audio)
    chunks = []
    start = 0
    while start < duration and len(chunks) < max_chunks:
        chunk = audio[start:start+duration_ms]
        if len(chunk) < 2000 and len(chunks) > 0:  # Exclude chunk if less than 2 seconds unless it's the only chunk
            break
        if len(chunk) < duration_ms:  # Pad last chunk if it's less than the desired duration
            padding = AudioSegment.silent(duration=duration_ms-len(chunk))
            chunk += padding
        chunks.append(chunk)
        start += duration_ms
    return chunks

def crop_audio_files(directory, duration_ms=10000):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio_path = os.path.join(directory, filename)
            audio = AudioSegment.from_wav(audio_path)

            # Crop if audio duration is more than duration_ms
            if len(audio) > duration_ms:
                audio = audio[:duration_ms]  # Crop audio to duration_ms

            # Overwrite the original file
            audio.export(audio_path, format="wav")

def download_and_convert_audio(url, output_dir, species, species_df, new_rate=48000, crop=True, duration_ms = 10000):
    try:
        # Fetch the content from the url
        
        original_file = os.path.join(output_dir, url.split('/')[-1])
        new_file = original_file.rsplit('.', 1)[0] + ".wav"
        species_count = get_species_count(species_df, species)
        new_files = []

        if species_count > SPECIES_COUNT_THRESHOLD and os.path.isfile(new_file):
            new_files.append(new_file)
            return new_files
        if species_count < SPECIES_COUNT_THRESHOLD:
            for i in range(5):
                new_file_i = f"{new_file.rsplit('.', 1)[0]}_{i}.wav"
                if os.path.isfile(new_file_i):
                    new_files.append(new_file_i)
        if new_files:
            print("already chunked!")
            return new_files
        
        if os.path.isfile(new_file):
            os.remove(new_file)
        response = requests.get(url)

        
        # Save the audio file
        with open(original_file, 'wb') as fp:
            fp.write(response.content)
        
        # Load audio file with pydub
        audio = AudioSegment.from_file(original_file)

        if len(audio) > duration_ms:
            if species_count > SPECIES_COUNT_THRESHOLD:
                audio = audio[:duration_ms]
                audio = audio.set_frame_rate(new_rate)
                audio.export(new_file, format="wav")
                new_files.append(new_file)
            else:
                chunks = split_audio(audio, duration_ms)
                for i, chunk in enumerate(chunks):
                    chunk_file = f"{new_file.rsplit('.', 1)[0]}_{i}.wav"
                    chunk = chunk.set_frame_rate(new_rate)
                    chunk.export(chunk_file, format="wav")
                    new_files.append(chunk_file)
        
        else: # short file, just save
            
            # Convert to .wav and resample
            new_files.append(new_file)
            audio = audio.set_frame_rate(new_rate)
            
            audio.export(new_file, format="wav")
            
            # Remove the original file
            os.remove(original_file)
        
        return new_files
    
    except Exception as e:
        print(f"Failed to process {url} due to {str(e)}")
        print("response was ", response.content)
        return []
    
def filter_applicable(df):
    print("starting len", len(df))
    allowed_licenses = ["CC0", "CC-BY", "CC-BY-NC"]
    filtered_df = df[df["license"].isin(allowed_licenses)]
    filtered_df = filtered_df[~filtered_df[URL_COLUMN].isna()]
    print("filtered len", len(filtered_df))
    return filtered_df
    
def append(file_paths, captions, urls, labels, descriptions, file_path, caption, url, label, description):
    file_paths.append(file_path)
    captions.append(caption)
    descriptions.append(description)
    urls.append(url)
    labels.append(label)

def load_ner():
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp

def has_location(text, pipeline):
    ner_results = pipeline(text)
    for result in ner_results:
        if "LOC" in result["entity"] and result["score"] > 0.92:
            print(result)
            return True
    return False

def needs_recaptioning(label, description, caption, filter_location=True, nlp=None):
    # pattern = r"^The sound of ((?!a\s|an\s).+)$"
    if not description or pd.isna(description):
        return False
    if not caption or pd.isna(caption):
        print(f"needs recaptioning cap {caption} label {label} == no caption")
        return True
    if filter_location:
        if has_location(caption, nlp) and not has_location(label, nlp):
            print(f"needs recaptioning {caption} -- has location")
            return True
        
    elif label.lower() not in caption.lower():
        print(f"needs recaptioning {caption} -- bad label {label}")
        return True
    return False

def process_row(row, output_dir, captioner, species_df, nlp=None):
    results = []
    try:
        url = row[NEW_URL_COLUMN]
        row_needs_recaptioning = False
        new_files = download_and_convert_audio(url, output_dir, row[NEW_LABEL_COLUMN], species_df)
        if new_files:
            label = row[NEW_LABEL_COLUMN]
            description = row[DESCRIPTION_COLUMN]
            current_caption = row[CAPTION_COLUMN]
            if needs_recaptioning(label, description, current_caption, nlp=nlp):
                row_needs_recaptioning = True
                caption = captioner.get_caption(species=label, description=description, current_caption=current_caption)[0]
                print(f"description {description} and {label} getting caption {caption}")
            else:
                caption = current_caption
            caption = postprocess_caption(caption=caption, description=description, species=label)

            for new_file in new_files:
                
                results.append({
                    "dataset": "inaturalist",
                    "location": new_file,
                    "caption": caption,
                    "species": label,
                    "description": description,
                    "url": url,
                    "paths": new_file,
                    "needs_recaptioning": row_needs_recaptioning
                })
        else:
            return None
        return results
    except Exception as e:
        print("exception in process row", e)
        return None
    
def postprocess_caption(caption, description, species):
    lower_caption = caption.lower()
    for word in BANNED_WORDS:
        if word in lower_caption:
            return species
    return caption

def get_caption_2(row):
    caption = row["caption"]
    scientific_name = row["scientific_name"]
    if pd.isna(scientific_name) or type(scientific_name) == float:
        return caption
    caption2 = caption.replace(row["species"], scientific_name)
    return caption2


def process_inaturalist(captioner: Captioner):
    """
    Download audio files in Inaturalist, caption them, and save the new dataset
    and csv.
    """
    species_counts = pd.read_csv("combined_species_counts.csv")
    df = pd.read_csv("inaturalist_dataset_final.csv")

    output_dir = "audios/inaturalist"
    
    os.makedirs(output_dir, exist_ok=True)

    jsonl_data = []
    file_paths = []
    captions = []
    urls = []
    labels = []
    descriptions = []

    ner = load_ner()

    total_rows = len(df)
    processed_rows = 0
    needs_recaptioning_count = 0

    # Use ThreadPoolExecutor to download and process files in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_row, row, output_dir, captioner, species_counts, ner): i for i, row in df.iterrows()}
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            if results is not None:
                for result in results:
                    jsonl_data.append(result)
                    file_paths.append(result["paths"])
                    captions.append(result["caption"])
                    urls.append(result["url"])
                    labels.append(result["species"])
                    descriptions.append(result["description"])
                    if result["needs_recaptioning"]:
                        needs_recaptioning_count += 1
                    processed_rows += 1
                    if processed_rows % 100 == 0:
                        print(f"{processed_rows} completed out of {total_rows}")

                    if processed_rows % 100 == 0:
                        output_dict = pd.DataFrame.from_dict({"species": labels, "caption": captions, "url": urls, "description": descriptions, "path": file_paths})
                        output_dict.to_csv("chunked_inaturalist_dataset_downloaded_v2.csv")

    print("needs recaptioning count", needs_recaptioning_count)
    
    output_dict = pd.DataFrame.from_dict({"species": labels, "caption": captions, "url": urls, "description": descriptions, "path": file_paths})
    output_dict.to_csv("chunked_inaturalist_dataset_downloaded_v2.csv")


if __name__ == "__main__":
    client = ChatGPTClient()
    captioner = Captioner(client)
