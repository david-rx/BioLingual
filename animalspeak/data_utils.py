import pandas as pd
import requests

def get_common_name(scientific_name, mapping=None):
    if mapping and scientific_name in mapping:
        return mapping[scientific_name]
    
    species_id = get_gbif_species_id(scientific_name)
    gbif_common_name = get_gbif_common_name(species_id)
    if gbif_common_name:
        return gbif_common_name
    
    return scientific_name


def get_gbif_species_id(scientific_name):
    url = f"https://api.gbif.org/v1/species?name={scientific_name}"
    try:
        response = requests.get(url)
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            return data['results'][0]['key']
        else:
            return ""
    except Exception as e:
        print(f"failed to get species id for {scientific_name} due to {str(e)}")
        return ""

def get_gbif_common_name(species_id):
    #check if species_id is pandas nan
    if not species_id or pd.isna(species_id):
        return None
    url = f"https://api.gbif.org/v1/species/{species_id}/vernacularNames"
    try:
        response = requests.get(url)
    
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            for result in data['results']:
                if 'language' in result and result['language'] == 'eng':
                    return result['vernacularName']
        return None
    except Exception as e:
        print(f"failed to get common name for {species_id} due to {str(e)}")
        return None