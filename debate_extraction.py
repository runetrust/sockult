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

speaker_patterns = [
    r'(?:MR\. |MS\. )?NIXON:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?FORD:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?CARTER:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?REAGAN:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?BUSH:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?CLINTON:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?OBAMA:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?TRUMP:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?BIDEN:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. )?THE PRESIDENT:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)',
    r'(?:MR\. |MS\. |SENATOR )?KENNEDY:\s*(.*?)(?=\n(?:MR\. |MS\. )?[A-Z]+:)'
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

for speaker_pattern in speaker_patterns:
    # Extract name from regex with optional prefixes
    match = re.search(r'(?:MR\. |MS\. )?([A-Z]+):', speaker_pattern)
    if match:
        speaker_name = match.group(1)  # Extract the speaker's name
        speaker_name = speaker_name.replace(".", "_")
    else:
        print(f"Invalid speaker pattern: {speaker_pattern}")
        continue 

    speaker_text_by_debate = {}
    for i, text in enumerate(raw_texts):
        matches = re.findall(speaker_pattern, text, re.DOTALL)
        if matches:
            speaker_text_by_debate[i] = matches

    # Save each into a separate file
    for debate_index, speaker_texts in speaker_text_by_debate.items():
        # Getting date
        if debate_index < len(dates):
            date = dates[debate_index]
        else:
            date = "unknown_date"

        # Saveing to unique folder and name structure 
        save_scraped_text(
            speaker_texts,
            identifier=f"{speaker_name.lower()}_debate_{debate_index}",
            base_directory=f"{speaker_name.lower()}",
            date=date
        )