import requests
import re
import os
import concurrent.futures
from bs4 import BeautifulSoup

url_list = [
    "https://www.debates.org/voter-education/debate-transcripts/september-29-2020-debate-transcript/",
    "http://debates.org/voter-education/debate-transcripts/vice-presidential-debate-at-the-university-of-utah-in-salt-lake-city-utah/",
    "https://www.debates.org/voter-education/debate-transcripts/october-22-2020-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/september-26-2016-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-4-2016-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-9-2016-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-19-2016-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-3-2012-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-11-2012-the-biden-romney-vice-presidential-debate/",
    "https://www.debates.org/voter-education/debate-transcripts/october-16-2012-the-second-obama-romney-presidential-debate/",
    "https://www.debates.org/voter-education/debate-transcripts/october-22-2012-the-third-obama-romney-presidential-debate/",
    "https://www.debates.org/voter-education/debate-transcripts/2008-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/2008-debate-transcript-2/",
    "https://www.debates.org/voter-education/debate-transcripts/october-7-2008-debate-transcrip/",
    "https://www.debates.org/voter-education/debate-transcripts/october-15-2008-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-13-2004-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-8-2004-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-5-2004-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/september-30-2004-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-3-2000-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-5-2000-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-11-2000-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-17-2000-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-17-2000-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-6-1996-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-9-1996-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-16-1996-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-11-1992-first-half-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-11-1992-second-half-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-13-1992-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-15-1992-first-half-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-15-1992-second-half-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-19-1992-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/september-25-1988-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-5-1988-debate-transcripts/",
    "https://www.debates.org/voter-education/debate-transcripts/october-13-1988-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-7-1984-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-11-1984-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-21-1984-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/september-21-1980-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-28-1980-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/september-23-1976-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-6-1976-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-22-1976-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/september-26-1960-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-7-1960-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-13-1960-debate-transcript/",
    "https://www.debates.org/voter-education/debate-transcripts/october-13-1960-debate-transcript/"]


#    "https://edition.cnn.com/2024/06/27/politics/read-biden-trump-debate-rush-transcript/index.html"


def fetch_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return None
    
def fetch_all_urls(urls, max_workers = 10):
        # This allows for parallel fetches, much faster than standard for loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map the fetch_url function to all URLs
            responses = list(executor.map(fetch_url, urls))
        # Only returning succesful fetches
        return [response for response in responses if response is not None]

def extract_names(text):
    pattern = r'(?:PARTICIPANTS:|SPEAKERS:)\s*\n(.*?)(?=[A-Z][A-Z\s]+:)'
    #pattern = r'(\s*[A-Z]+:)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        participants = match.group(1).strip()
        # Replace spaces and newlines with underscores
        participants = participants.replace(' ', '_').replace('\n', '_').replace('*','')
        return participants
    else:
        return None
    
def extract_date(soup):
    h1_tag = soup.find("h1")
    if h1_tag is None:
        date_text = "October_22,_2020"
        print("No h1 tag")
        return date_text
    date_text = h1_tag.get_text(strip=True)
    date_text = date_text.replace(' ', '_').replace(',','').replace('.','')
    return date_text

def save_scraped_text(raw_text, identifier, date, backup_identifier, base_directory='scraped_debates'):
    """
    Save scraped text to a file with a timestamped filename.
    
    Args:
        raw_text (str): The text content to be saved
        identifier (str): Unique identifier
        date (str): Extracted date from debate
        backup_identifier (str): A fallback identifier to ensure file gets saved even without extracted name
        base_directory (str, optional): Directory to save text files. Defaults to 'scraped_debates'.

    
    Returns:
        str: Full path to the saved file
    """
    # Create the base directory if it doesn't exist
    os.makedirs(base_directory, exist_ok=True)

    if identifier is not None:
        filename = f"{identifier}_{date}.txt"
    else:
        filename = f"{backup_identifier}_{date}.txt"
    full_path = os.path.join(base_directory, filename)
    
    # Save the text with UTF-8 encoding to support various characters
    try:
        with open(full_path, 'a', encoding='utf-8') as file:
            file.write(raw_text)
        print(f"Text successfully saved to {full_path}")
        # Returning file path is useful for loading in data later
        return full_path
    except IOError as e:
        print(f"Error saving file: {e}")
        return None

    

responses = fetch_all_urls(url_list)

names_in_document = []

for index, document in enumerate(responses):
    soup = BeautifulSoup(document.content, "html.parser")

    # Extract date
    date = extract_date(soup)
    if date is None:
        date = f"no_date_{index}"  # fallback for missing <h1> tag

    # Extract clean speaker text
    clean_text = soup.get_text()

    # Extract name (may be None)
    name = extract_names(clean_text)

    # Save text regardless of whether name was found
    save_scraped_text(
        raw_text=clean_text,
        identifier=name,
        date=date,
        backup_identifier=index
    )
