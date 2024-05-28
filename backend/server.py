# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import korean_news_scraper as kns

from read_csv import read_csv
import os


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
    """/
    get news; run on /service
    
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
