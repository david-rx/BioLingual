import pandas as pd
from animalspeak.chatgpt.client import ChatGPTClient, ChatHistory

CAPTION_PROMPT_PATH = "animalspeak/chatgpt/prompts/new_inaturalist_captions.jsonl"
WATKINS_CAPTION_PROMPT_PATH = "animalspeak/chatgpt/prompts/watkins_captions.jsonl"
WATKINS_RECAPTION_PROMPT_PATH = "animalspeak/chatgpt/prompts/watkins_pass_2.jsonl"

class Captioner:
    def __init__(self, client: ChatGPTClient) -> None:
        self.chatgpt_client = client
    
    def get_caption(self, species: str, description: str, current_caption = ""):
        if not description or pd.isna(description):
            return species, True

        message_history = ChatHistory.from_jsonl(CAPTION_PROMPT_PATH)
        message_history.messages[-1]["content"] = message_history.messages[-1]["content"].format(
            species=species, description=description)
        try:
            response = self.chatgpt_client.chat(message_history.messages)
        except Exception as e:
            print("ChatGPT exception", e)
            if current_caption:
                return current_caption, False
            return f"The sound of {species}", False
        caption = self.parse_response(response)
        print(f"old caption {current_caption} getting new caption {caption}")
        if "sorry" in caption or "apologize" in caption or "Sorry" in caption:
            return species, True
        return caption, True
    
    def get_caption_watkins(self, species, genus_species, behavior, num_animals, notes, signal_type):
        message_history = ChatHistory.from_jsonl(WATKINS_CAPTION_PROMPT_PATH)
        message_history.messages[-1]["content"] = message_history.messages[-1]["content"].format(
            species=species, genus_species=genus_species, signal_type=signal_type, behavior=behavior, num_animals=num_animals, notes=notes)
        try:
            response = self.chatgpt_client.chat(message_history.messages)
        except Exception as e:
            print("exception from chatgpt", e)
            return None, False
        caption = self.parse_response(response)
        return caption, True

    def recaption_watkins(self, old_caption):
        message_history = ChatHistory.from_jsonl(WATKINS_RECAPTION_PROMPT_PATH)
        message_history.messages[-1]["content"] = message_history.messages[-1]["content"].format(
            caption=old_caption)
        try:
            response = self.chatgpt_client.chat(message_history.messages)
        except Exception as e:
            print("exception from chatgpt", e)
            return old_caption
        caption = self.parse_response(response)
        return caption


    def parse_response(self, response: str):
        """
        Response should come
        """
        return response.strip()
    

def caption_inaturalist():
    df = pd.read_csv("/Users/davidrobinson/Code/datasets/inaturalist.csv")
    client = ChatGPTClient()
    captioner = Captioner(client)
    captions = []
    descriptions = []
    species_list = []
    audio_paths = []
    count = 0
    for index, row in df.iterrows():
        if count > 25:
            break
        count += 1
        description = row["description"]
        species = row["common_name"]
        caption, success = captioner.get_caption(species=species, description=description)
        captions.append(caption)
        species_list.append(species)
        descriptions.append(description)
        audio_paths.append(row["sound_url"])
    
    captioned_df = pd.DataFrame.from_dict({"species": species_list, "description": descriptions, "caption": captions, "sound_url": audio_paths})
    captioned_df.to_csv("captioned_inaturalist.csv")
    

if __name__ == "__main__":
    caption_inaturalist()