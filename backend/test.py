import korean_news_scraper as kns
import csv
import os

kns.extract_article_content("data")
directory = f"{os.getcwd()}/data/article_contents/"


# Get the list of files in the directory.
file_list = os.listdir(directory)

# Iterate through each file and delete it.
for file_name in file_list:
    # Get the full path.
    file_path = os.path.join(directory, file_name)
    
    data: list = []

    # open file
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        
        for row in csvreader:
            data.append(row)

    # return the data.
    for row in data:
        print(row)

    try:
        # Remove files.
        os.remove(file_path)
        print(f"Deleted {file_name}.")
    except Exception as e:
        print(f"An error occurred while deleting {file_name}: {e}")


[os.remove(f"{os.getcwd()}/data/article_contents/{file}") for file in os.listdir(directory)]
[os.remove(f"{os.getcwd()}/data/article_links/{file}") for file in os.listdir(f"{os.getcwd()}/data/article_links/")]

