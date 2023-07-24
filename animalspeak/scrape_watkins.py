"""
---- Pulled from -----
https://github.com/paloukari/OrcaDetector/blob/master/README.md

Script to pull all data from the WHOI website (by permission of site owners)
https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm

W251 (Summer 2019) - Spyros Garyfallos, Ram Iyer, Mike Winton
"""

import multiprocessing
import os
import sys
import urllib
from urllib.request import urlopen

import pandas as pd
import requests
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
import json
import re

def downloadSample(samplename, filename):
    sys.stdout.write('-')
    sys.stdout.flush()

    if not os.path.exists(filename):
        urllib.request.urlretrieve(
            'http://cis.whoi.edu/' + samplename, filename)
        
def downloadMetadata(metadata_url, filename):
    # Fetch the metadata page
    metadata_r = requests.get(metadata_url)
    print(metadata_r.text)
    metadata_soup = BeautifulSoup(metadata_r.text, 'lxml')

    # Find the first table (the header table)
    header_table = metadata_soup.find('table')

    # Find the next table after the header table (the metadata table)
    metadata_table = header_table.find_next_sibling('table')

    # Initialize an empty dictionary to store the metadata
    metadata = {}

    # Loop through each row in the metadata table
    for i, row in enumerate(metadata_table.find_all('tr')):
        # Skip the first row (header row)
        if i == 0:
            continue
        # Find the 'td' elements in the row (the key and value)
        key_td, value_td = row.find_all('td')

        # Get the text of the key and value
        key = key_td.get_text(strip=True)
        value = value_td.get_text(strip=True)

        # Store the key-value pair in the metadata dictionary
        metadata[key] = value

    # Write the metadata to a json file
    with open(filename + '.json', 'w') as f:
        json.dump(metadata, f)



def downloadTable(url, name, year):
    # Scrape the HTML at the url
    r = requests.get(url)

    # Turn the HTML into a Beautiful Soup object
    soup = BeautifulSoup(r.text, 'lxml')

    # Create four variables to score the scraped data in
    location = []
    date = []

    # Create an object of the first object that is class=database
    table = soup.find(class_='database')

    downloadData = []
    for row in table.find_all('tr')[1:]:
        col = row.find_all('a', href=True)

        samplename = col[0]['href']
        samplename_parts = col[0]['href'].split('/')

        dir = 'watkins_data/' + name + '/' + year
        filename = dir + '/' + samplename_parts[-1:][0]

        if not os.path.exists(dir):
            os.makedirs(dir)
        downloadData.append([samplename, filename])

        #get metadata
        metadata_link = row.find('a', text='Metadata')
        if metadata_link:
            metadata_js = metadata_link['href']
            metadata_rel_url = metadata_js.split("'")[1]
            metadata_url = 'https://cis.whoi.edu/science/B/whalesounds/' + metadata_rel_url
            downloadMetadata(metadata_url, filename)
        else:
            print("metadata not found")

    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores*2)(
        delayed(downloadSample)(data[0], data[1]) for data in downloadData)

    print('->')


def downloadAllAnimals(url):
    r = requests.get(url)

    soup = BeautifulSoup(r.text, 'lxml')

    # Loop over species
    list = soup.find(class_='large-4 medium-4 columns left')

    for species in list.find_all('option')[1:]:
        url_end = species['value']
        name = species.string.strip()

        print("Downloading " + name)

        name = name.replace(' ', '')
        name = name.replace('-', '_')
        name = name.replace(',', '_')

        # Loop over years
        ryears = requests.get(
            "http://cis.whoi.edu/science/B/whalesounds/" + url_end)

        soupYears = BeautifulSoup(ryears.text, 'lxml')

        listYears = soupYears.find(class_='large-4 medium-4 columns')

        for years in listYears.find_all('option')[1:]:
            urlFin = years['value']
            year = years.string.strip()

            print("         " + "\t" + year)

            downloadTable(
                "http://cis.whoi.edu/science/B/whalesounds/" + urlFin, name, year)

url = 'http://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm'
downloadAllAnimals(url)