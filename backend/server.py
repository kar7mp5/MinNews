# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


import korean_news_scraper as kns
from read_csv import read_csv


app = FastAPI()


class Keyword(BaseModel):
    keyword: str

# /service
@app.post("/get_news")
async def get_news(data: Keyword):
    """/
    get news; run on /service
    
    Args:
        keyword (str): keyword; search for news

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

    keywords = ["Iran"]
    kns.save_article_links(keywords, "data", lang="en-EN")
    kns.extract_article_content("data")
    read_csv()

    return {
            "links": "Hello World"
            }
