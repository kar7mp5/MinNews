# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# /get_news/
import korean_news_scraper as kns
from read_csv import read_csv
import os

# /metadata/
import requests
from bs4 import BeautifulSoup

# /minGPT/
# LLM library
from minGPT import*



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Keyword(BaseModel):
    keyword: str
    number: int

# /service
@app.post("/get_news")
async def get_news(data: Keyword) -> dict:
    """get news; run on /service
    
    Args:
        keyword (str): keyword; search for news
        number (int): article number

    Returns:
        dict: The return article links.
    
    TODO:
        1. get keyword
        2. get article links
        3. get text
    """

    # check `(param)keyword` is empty.
    # True : raise HTTPException
    # False: pass
    if (data.keyword == ""):
        raise HTTPException(status_code=400, detail="Empty Keyword")

    # file path
    csv_file_path = f"data/article_links/{data.keyword}.csv"

    # Check if the file already exists
    if os.path.exists(csv_file_path):
        # File already exists, no need to save article links again
        print("CSV file already exists. Skipping saving article links.")
    else:
        # File does not exist, save article links to CSV file
        kns.save_article_links([data.keyword], "data", lang="ko-KR")
    
    # get articles links.
    result: list = read_csv(f"/data/article_links/{data.keyword}.csv", data.number)

    return {
            "links": result
            }


class Metadata(BaseModel):
    title: str
    description: str
    image: str
    url: str


@app.get("/metadata")
async def get_metadata(url: str) -> Metadata:
    """Fetch metadata for a given URL.
    
    Args:
        url (str): The URL for which metadata is to be fetched.
    
    Returns:
        Metadata: The metadata extracted from the URL.
    """

    try:
        # Fetch HTML content of the URL
        response = requests.get(url)
        html_content = response.content

        # Parse HTML content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract metadata
        title = soup.find('meta', property='og:title')
        description = soup.find('meta', property='og:description')
        image = soup.find('meta', property='og:image')

        # Initialize metadata fields with empty strings
        title_content = description_content = image_content = ""

        # Assign content to metadata fields if found in the HTML
        if title:
            title_content = title.get('content')
        if description:
            description_content = description.get('content')
        if image:
            image_content = image.get('content')

        # Return the extracted metadata
        return Metadata(title=title_content, description=description_content, image=image_content, url=url)
    
    except Exception as e:
        # If any error occurs during metadata extraction, return default values
        print(f"Error fetching metadata for URL: {url}. Error: {str(e)}")

        return Metadata(title="", description="", image="", url=url)


