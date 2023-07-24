import pandas as pd
from animalspeak.chatgpt.client import ChatGPTClient
from animalspeak.captioning.captioner import Captioner

import re

def needs_recaption(text):
    # Search for coordinates
    coordinates_patterns = [r'[NS]\d{1,3}°\s?\d{1,3}\'\s?[EW]', r'\d{1,3}°\d{1,3}\'[NS],\s?\d{1,3}°\d{1,3}\'[EW]', r'coordinates']
    for pattern in coordinates_patterns:
        match = re.search(pattern, text)
        if match:
            print('Coordinates found.')
            return True

    # Search for CD, tape, session, hydrophone #, or cut
    keywords = ['CD', 'tape', 'session', 'hydrophone #', 'cut', "West", 'North', 'East', 'at location', 'recorded at']
    for keyword in keywords:
        if keyword in text:
            print(f'Keyword "{keyword}" found.')
            return True

    return False


def recaption_watkins(captioner: Captioner):
    df = pd.read_csv("watkins_captions.csv")
    recaption_count = 0
    for index, row in df.iterrows():
        caption = row["caption"]
        if needs_recaption(caption):
            print(f"needs new caption", caption)
            recaption_count += 1
            new_caption = captioner.recaption_watkins(caption)
            print("new caption is", new_caption)
            row["caption"] = new_caption
    print(f"{recaption_count} out of {len(df)} recaptioned")
    df.to_csv("watkins_captions_pass_2.csv")

if __name__ == "__main__":
    client = ChatGPTClient()
    captioner = Captioner(client)
    recaption_watkins(captioner)
