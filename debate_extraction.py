import re
import os
import glob

def save_scraped_text(raw_text, identifier, date, base_directory='debates'):
    """
    Save scraped text to a file with a timestamped filename.
    
    Args:
        raw_text (str): The text content to be saved
        identifier (str): Unique identifier
        base_directory (str, optional): Directory to save text files. Defaults to 'scraped_debates'.
        prefix (str, optional): Prefix for the filename. Defaults to 'debate'.
    
    Returns:
        str: Full path to the saved file
    """
    # Create the base directory if it doesn't exist
    os.makedirs(base_directory, exist_ok=True)
    
    # Generate a unique filename
    filename = f"{identifier}_{date}.txt"
    full_path = os.path.join(base_directory, filename)
    
    # Save the text with UTF-8 encoding to support various characters
    try:
        with open(full_path, 'w', encoding='utf-8') as file:
            if isinstance(raw_text, list):
                file.write('\n'.join(raw_text))
            else:
                file.write(raw_text)
        print(f"Text successfully saved to {full_path}")
        # Returning file path is useful for loading in data later
        return full_path
    except IOError as e:
        print(f"Error saving file: {e}")
        return None

folder = "C:/Dev/Python/scraping/scraped_debates"

trump_debates = glob.glob(f"{folder}/*.txt")

def load_and_extract(path):
    # Read the file
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Locate the starting line for the data (after 'Good evening')
    start_line = 0
    for i, line in enumerate(lines):
        if "Good evening" in line:
            start_line = i
            break

    # Extract data lines and join into a string
    data_lines = lines[start_line:]
    raw_text = '\n'.join(data_lines)
    return raw_text

def extract_date(filename):
    date_pattern = r'([A-Z][a-z]+_\d{1,2}_\d{4})'
    match = re.search(date_pattern, filename, re.DOTALL)
    if match:
        date = match.group(1).strip()
    else:
        print(f"No date extracted from {filename}")
        date = None
    return date

# Regex pattern to capture everything after TRUMP: up until the next speaker label
# I need a list of regexes to extract all relevant speakers

speaker_patterns = [
    r'NIXON:\s*(.*?)(?=\n[A-Z]+:)',
    r'FORD:\s*(.*?)(?=\n[A-Z]+:)',
    r'CARTER:\s*(.*?)(?=\n[A-Z]+:)',
    r'REAGAN:\s*(.*?)(?=\n[A-Z]+:)',
    r'BUSH:\s*(.*?)(?=\n[A-Z]+:)',
    r'CLINTON:\s*(.*?)(?=\n[A-Z]+:)',
    r'OBAMA:\s*(.*?)(?=\n[A-Z]+:)',
    r'TRUMP:\s*(.*?)(?=\n[A-Z]+:)',
    r'BIDEN:\s*(.*?)(?=\n[A-Z]+:)'
]

raw_texts = []
for debate in trump_debates:
    raw_text = load_and_extract(debate)
    raw_texts.append(raw_text)

# This loops through entire folder and saves all dates
dates = []
for debate in trump_debates:
    date = extract_date(debate)
    if date is None:
        print(f"Skipping file {debate}, no date extracted.")
        continue
    dates.append(date)

# Iterate over each speaker pattern
for speaker_pattern in speaker_patterns:
    # Extract the speaker's name from the pattern (e.g., 'TRUMP' from 'TRUMP:\s*(.*?)(?=\n[A-Z]+:)')
    speaker_name = re.match(r'([A-Z]+):', speaker_pattern).group(1)

    # Dictionary to store utterances by debate index for the current speaker
    speaker_text_by_debate = {}

    for i, text in enumerate(raw_texts):  # Enumerate to get the index of the debate
        matches = re.findall(speaker_pattern, text, re.DOTALL)
        if matches:
            speaker_text_by_debate[i] = matches  # Store matches for this debate index

    # Save each group's text into a separate file
    for debate_index, speaker_texts in speaker_text_by_debate.items():
        # Get the corresponding date for the debate
        if debate_index < len(dates):
            date = dates[debate_index]
        else:
            date = "unknown_date"  # Fallback if no date is available

        # Save the text for this debate
        save_scraped_text(
            speaker_texts,
            identifier=f"{speaker_name.lower()}_debate_{debate_index}",
            date=date
        )
