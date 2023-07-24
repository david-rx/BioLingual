from typing import Dict, List
import os
import json
import pandas as pd
import re
from tqdm import tqdm

from animalspeak.chatgpt.client import ChatGPTClient
from animalspeak.captioning.captioner import Captioner

CAPTION_PROMPT = """Summarize the metadata about some animal noises into a short, single-sentence caption for an audio clip. Don't include background information about the tape or location, only about the sounds in the recording and the animals heard. Do not mention animals seen if not heard.
For example, if you are given 'species: Common Dolphin, noise type: clicks, behavior: feeding, num_animals: 2+',
Notes: recorded on hydrophone 2452, ship noise in background, from Ray's 1970 tape. Killer whale seen in area but not heard.' a good caption would be "At least two common dolphins clicking while feeding, with ship noise in the background."
Please respond with just the caption for the following metadata: 'species list': {genus_species} 'species': {species}, 'behavior': {behavior}, 'number of animals': {num_animals}, 'notes': {notes}'
"""

CODE_TO_COMMON = {"CC3A": "Gray Seal", "BD3A": "Long Beaked (Pacific) Common Dolphin", "BD15A": "Pantropical Spotted Dolphin", "CC14A": "Ross Seal", "BD4A": "Grampus- Risso's Dolphin", "BD6B": "White-beaked Dolphin", "BF6A": "Finless Porpoise", "CC5A": "Weddell Seal", "BE3D": "Short- Finned( Pacific) Pilot Whale", "CA1F": "New Zealand Fur Seal", "CC12L": "Spotted Seal", "AA3B": "Southern Right Whale", "CC4A": "Leopard Seal", "BD10A": "Melon Headed Whale", "BE9A": "False Killer Whale", "AC1A": "Minke Whale", "BD1A": "Commerson's Dolphin", "CA3B": "Steller Sea Lion", "BD15B": "Clymene Dolphin", "BD15L": "Spinner Dolphin", "AB1A": "Gray Whale", "CC12H": "Ringed Seal", "AA1A": "Bowhead Whale", "BB1A": "Beluga- White Whale", "BD17A": "Rough- Toothed Dolphin", "BD12B": "Tucuxi Dolphin", "BE3C": "Long- Finned Pilot Whale", "BD6H": "Dusky Dolphin", "BD3B": "Common Dolphin", "BB2A": "Narwhal", "AC2A": "Humpback Whale", "BE7A": "Killer Whale", "X": "Killer Whale", "CD1A": "Sea Otter", "BD15C": "Striped Dolphin", "BD6A": "White-sided Dolphin", "CC12F": "Ribbon Seal", "AA3A": "Northern Right Whale", "BA2A": "Sperm Whale", "BG2A": "Boutu- Amazon River Dolphin", "CC1A": "Hooded Seal", "BF4A": "Dall's Porpoise", "CB1A": "Walrus", "DB1B": "West Indian Manatee", "BD1C": "Heaviside's Dolphin", "BD15F": "Atlantic Spotted Dolphin", "BF2A": "Harbor Porpoise", "BD19D": "Bottlenose Dolphin", "CC12G": "Harp Seal", "BD5A": "Fraser's Dolphin", "CC2A": "Bearded Seal"}
SCIENTIFIC_TO_COMMON = {"Halichoerus grypus": "Gray Seal", "Delphinus bairdii": "Long Beaked (Pacific) Common Dolphin", "Stenella attenuata": "Pantropical Spotted Dolphin", "Stenella attenuata  BD15A": "Sperm Whale", "Physeter catodon  BA2A": "Bottlenose Dolphin", "Ommatophoca rossi": "Ross Seal", "Hydrurga leptonyx  CC4A": "Leopard Seal", "Ommatophoca rossi  CC14A": "Leopard Seal", "Ice  X": "Bearded Seal", "Grampus griseus": "Grampus- Risso's Dolphin", "Grampus griseus  BD4A": "Grampus- Risso's Dolphin", "Lagenorhynchus albirostris": "White-beaked Dolphin", "Lagenorhynchus albirostris  BD6B": "White-beaked Dolphin", "Neophocaena phocaenoides": "Finless Porpoise", "Leptonychotes weddelli": "Weddell Seal", "Leptonychotes weddelli  CC5A": "Weddell Seal", "Leptonychotes weddellii": "Weddell Seal", "Globicephala macrorhynchus": "Short- Finned( Pacific) Pilot Whale", "Globicephala scammoni": "Short- Finned( Pacific) Pilot Whale", "Globicephala macrorhynchus  BE3D": "Sperm Whale", "Arctocephalus forsteri": "New Zealand Fur Seal", "Phoca largha": "Spotted Seal", "Eubalaena australis": "Southern Right Whale", "Eubalaena australis  AA3B": "Southern Right Whale", "Hydrurga leptonyx": "Leopard Seal", "Crustacea  O": "Irawaddy Dolphin", "Sotalia borneensis  BD12A": "Irawaddy Dolphin", "Peponocephala electra": "Melon Headed Whale", "Peponocephala electra  BD10A": "Sperm Whale", "Pseudorca crassidens": "False Killer Whale", "Pseudorca crassidens  BE9A": "False Killer Whale", "Balaenoptera acutorostrata": "Minke Whale", "Balaenoptera acutorostrata  AC1A": "Minke Whale", "Cephalorhynchus commersonii": "Commerson's Dolphin", "Eumetopias jubatus": "Steller Sea Lion", "Stenella clymene": "Clymene Dolphin", "Stenella clymene  BD15B": "Clymene Dolphin", "Stenella longirostris": "Spinner Dolphin", "Stenella longirostris  BD15L": "Sperm Whale", "Eschrichtius robustus": "Gray Whale", "Phoca hispida  CC12H": "Narwhal", "Phoca hispida": "Ringed Seal", "Balaena mysticetus": "Bowhead Whale", "Balaena mysticetus  AA1A": "Bearded Seal", "Balaena mysticetus AA1A": "Bearded Seal", "Erignathus barbatus  CC2A": "Bearded Seal", "Delphinapterus leucas": "Beluga- White Whale", "Delphinapterus leucas  BB1A": "Beluga- White Whale", "Steno bredanensis": "Rough- Toothed Dolphin", "Steno bredanensis  BD17A": "Rough- Toothed Dolphin", "Sotalia fluviatilis": "Tucuxi Dolphin", "Globicephala melaena": "Long- Finned Pilot Whale", "Globicephala melaena  BE3C": "Sperm Whale", "Globicephala melaena   BE3C": "Sperm Whale", "Lagenorhynchus  obscurus": "Dusky Dolphin", "Delphinus delphis  BD3B": "Common Dolphin", "Delphinus delphis": "Common Dolphin", "Monodon monoceros": "Narwhal", "Monodon monoceros  BB2A": "Narwhal", "Megaptera novaeangliae": "Humpback Whale", "Megaptera novaeangliae  AC2A": "Humpback Whale", "Orcinus orca": "Killer Whale", "Orcinus orca  BE7A": "Killer Whale", "Orcinus orca  BE7A; Ambient": "Killer Whale", "Enhydra lutris": "Sea Otter", "Stenella coeruleoalba": "Striped Dolphin", "Stenella coeruleoalba  BD15C": "Sperm Whale", "Lagenorhynchus acutus": "White-sided Dolphin", "Lagenorhynchus acutus  BD6A": "White-sided Dolphin", "Phoca fasciata": "Ribbon Seal", "Phoca fasciata  CC12F": "Bearded Seal", "Arctocephalus philippii  CA1P": "Juan Fernandez Fur Seal", "Tursiops truncatus  BD19D": "Bottlenose Dolphin", "Eubalaena glacialis  AA3A": "Northern Right Whale", "Eubalaena glacialis  AA3A  Blow": "Northern Right Whale", "Eubalaena glacialis": "Northern Right Whale", "Eubalaena glacialis  AA3A   Slap": "Northern Right Whale", "Physeter catodon": "Sperm Whale", "Delphinidae  BD": "Fin- Finback Whale", "Homo sapiens  E": "Sperm Whale", "Inia geoffrensis": "Boutu- Amazon River Dolphin", "Cystophora cristata": "Hooded Seal", "Phocoenoides dalli": "Dall's Porpoise", "Odobenus rosmarus": "Walrus", "Odobenus rosmarus  CB1A": "Bearded Seal", "Balaenoptera physalus  AC1F": "Fin- Finback Whale", "Balaenoptera musculus  AC1E": "Blue Whale", "Balaenoptera physalus": "Fin- Finback Whale", "Ambient  X": "Fin- Finback Whale", "Trichechus manatus": "West Indian Manatee", "Cephalorhynchus heavisidii": "Heaviside's Dolphin", "Stenella frontalis": "Atlantic Spotted Dolphin", "Phocoena phocoena": "Harbor Porpoise", "Phocoena phocoena  BF2A": "Harbor Porpoise", "Tursiops truncatus": "Bottlenose Dolphin", "Phoca groenlandica": "Harp Seal", "Lagenodelphis hosei  BD5A": "Fraser's Dolphin", "Lagenodelphis hosei": "Fraser's Dolphin", "Erignathus barbatus": "Bearded Seal"}
NT = "nt"
MAPPIING = {"NT": "noise type",
            "BH": "behavior",
            "ST": "signal type",
            "GS": "genus species",
            "SC": "Signal class",
            "NT": "Note",
            "NA": "Number of Animals"
            }

SIGNAL_TYPE_MAPPING = {
        'BL': 'broadband long signals',
        'BS': 'broadband short signals',
        'NL': 'narrowband long signals',
        'NS': 'narrowband short signals',
        'FM': 'frequency modulated signals',
        'CH': 'chirp signals',
        'PU': 'pulsed signals',
        'SE': 'series of sounds',
        'SO': 'song signals',
    }


WATKINS_BASE_DIR = "watkins_data"

FILE_PATH_COLUMN = "file_path"
SCIENTIFIC_NAME_COLUMN = "label"
SPECIES_COLUMN = "species"
CAPTION_COLUMN = "caption"

def extract_species_code(genus_species):
    first_section = genus_species.split(" | ")[0]
    species_code = first_section.split(" ")[-1]
    scientific_name = " ".join(first_section.split(" ")[:-1]).strip()
    return scientific_name, species_code

def remove_code(text):
    for code in CODE_TO_COMMON.keys():
        text = text.replace(code, "")
    return text.strip()

def replace_code(text):
    for code, common in CODE_TO_COMMON.items():
        text = text.replace(code, common)
    return text.strip()

def get_common_name(species):
    if species in SCIENTIFIC_TO_COMMON:
        return SCIENTIFIC_TO_COMMON[species]
    return species

def replace_scientific(text):
    for scientific, common in SCIENTIFIC_TO_COMMON.items():
        text = text.replace(scientific, common)
    return text

def extract_background_species(genus_species, use_common_name: bool):
    all_species = genus_species.split(" | ")
    if len(all_species) == 1:
        return []
    cleaned_background_species = [remove_code(species) for species in all_species[1:]]
    if use_common_name:
        return [get_common_name(species) for species in cleaned_background_species]
    
def caption_background_species(caption, background_species: List[str]):
    if not background_species:
        return caption
    caption += f" with background sounds from {', '.join(background_species)}"
    return caption

def map_to_common_name(species_name):
    # Special case for names containing parenthesis
    if "LongBeaked" in species_name:
        return "Long Beaked (Pacific) Common Dolphin"
    
    # Split at capital letters, except at the start of a string or preceded by an underscore
    parts = re.findall('[a-zA-Z][^A-Z]*', species_name)
    
    # Join the parts with a space and replace underscore with hyphen
    common_name = " ".join(parts).replace("_", "-")
    
    return common_name

def caption_num_animals(caption, num_animals):
    try:
        num = int(num_animals)
        if num == 1:
            caption += "a "
        else:
            caption += f"{num} "
    except ValueError:
        # num_animals could not be converted to an integer, so just return the original caption
        return caption + f"{num_animals} "
    return caption

def get_call_caption(caption, signal_type):
    if signal_type and signal_type in SIGNAL_TYPE_MAPPING:
            caption += f"{SIGNAL_TYPE_MAPPING[signal_type]} of "
    else:
        caption += "Sound of "
    return caption

def caption_behavior(caption, behavior):
    print("behavior start", behavior)
    behavior = remove_code(behavior).lower()
    print("cleaned behavior", behavior)
    if behavior:
        caption += f"{behavior} "
    return caption

def caption_species(caption, species):
    return caption + species

def create_caption2(metadata, species, use_scientific_name):
    genus_species = metadata.get('GS:', species)
    scientific_name, species_code = extract_species_code(genus_species)
    species_name = scientific_name if use_scientific_name else species

    signal_type = remove_code(metadata.get("ST:", ""))
    num_animals = metadata.get("NA:", "").split("  ")[0]
    behavior = metadata.get("BH:", "")
    print("species is ", species)
    caption = get_call_caption("", signal_type) #{Call} of
    caption = caption_num_animals(caption, num_animals)
    caption = caption_behavior(caption, behavior)
    caption = caption_species(caption, species_name) #{Call} of {number} {species}

    background_species = extract_background_species(genus_species, not use_scientific_name)
    caption = caption_background_species(caption, background_species)
    

    if metadata.get("NT:", ""):
        pass
        # caption += f" Note: {metadata['NT:']}."
    return caption


def create_caption_lm(metadata, species, pipeline, tokenizer, use_scientific_name=False):
    scientific_name, species_code = extract_species_code(metadata.get('GS:', species))
    species_name = scientific_name if use_scientific_name else species

    signal_type = replace_code(metadata.get("ST:", ""))
    num_animals = replace_code(metadata.get("NA:", "").split("  ")[0])
    behavior = replace_code(metadata.get("BH:", ""))
    notes = replace_code(metadata.get("NT:", ""))
    prompt = CAPTION_PROMPT.format(species=species_name, behavior=behavior, notes=notes, num_animals=num_animals, signal_type=signal_type)
    print(prompt)
    exit()

def create_caption_chatgpt(metadata, species, captioner: Captioner, use_scientific_name=False):
    genus_species = metadata.get('GS:', species)
    cleaned_genus_species = remove_code(genus_species)
    
    scientific_name, species_code = extract_species_code(genus_species)
    species_name = scientific_name if use_scientific_name else species
    

    signal_type = replace_code(metadata.get("ST:", ""))
    num_animals = replace_code(metadata.get("NA:", "").split("  ")[0])
    behavior = replace_code(metadata.get("BH:", ""))
    notes = replace_code(metadata.get("NT:", ""))
    
    if not use_scientific_name:
        cleaned_genus_species = replace_scientific(cleaned_genus_species)
        behavior = replace_scientific(behavior)
        num_animals = replace_scientific(num_animals)
        notes = replace_scientific(notes)

    caption = captioner.get_caption_watkins(species = species, genus_species = cleaned_genus_species, behavior=behavior, notes=notes, num_animals=num_animals, signal_type=signal_type)
    return caption

def metadata_to_raw_captions(metadata: Dict):
    caption = f"A "
    raw_captions = []
    for key, value in metadata.items():
        key = key.replace(":", "")
        if key in MAPPIING:
            raw_captions.append(f"{MAPPIING[key]}: {value}")
    return ", ".join(raw_captions)

def metadata_to_captions(metadata: Dict, species: str):
    return metadata_to_raw_captions(metadata, species)

def main(captioner: Captioner):
    files = []
    species_labels = []
    scientific_labels = []
    captions = []
    species_code_dict = {}
    code_to_common = {}
    scientific_to_common = {}
    count = 0
    for species_folder in tqdm(os.listdir(WATKINS_BASE_DIR)):
        for year_folder in os.listdir(os.path.join(WATKINS_BASE_DIR, species_folder)):
            for file in os.listdir(os.path.join(WATKINS_BASE_DIR, species_folder, year_folder)):
                if file.endswith(".wav"):
                    count += 1
                    with open(os.path.join(WATKINS_BASE_DIR, species_folder, year_folder, file + ".json")) as fp:
                        metadata = json.load(fp)
                        print(metadata)

                    common_name = map_to_common_name(species_folder)
                    scientific_name, code = extract_species_code(metadata["GS:"])
                    species_code_dict[code] = scientific_name
                    code_to_common[code] = common_name

                    caption = create_caption_chatgpt(metadata=metadata, species = common_name, captioner=captioner)[0]
                    print("caption is", caption)
                    files.append(os.path.join(WATKINS_BASE_DIR, species_folder, year_folder, file))
                    species_labels.append(common_name)
                    scientific_labels.append(scientific_name)
                    captions.append(caption)
                    scientific_to_common[scientific_name] = common_name
                    if count % 10 == 0:
                        df = pd.DataFrame.from_dict({FILE_PATH_COLUMN: files, SPECIES_COLUMN: species_labels, SCIENTIFIC_NAME_COLUMN: scientific_labels, CAPTION_COLUMN: captions})
                        df.to_csv("watkins_captions.csv", index=False)

    df = pd.DataFrame.from_dict({FILE_PATH_COLUMN: files, SPECIES_COLUMN: species_labels, SCIENTIFIC_NAME_COLUMN: scientific_labels, CAPTION_COLUMN: captions})
    df.to_csv("watkins_captions.csv", index=False)


    with open("code_to_common.json", "w") as codefile:
        json.dump(code_to_common, codefile)
    with open("scientific_to_common.json", "w") as commonfile:
        json.dump(scientific_to_common, commonfile)


if __name__ == "__main__":
    client = ChatGPTClient()
    captioner = Captioner(client)
    main(captioner)