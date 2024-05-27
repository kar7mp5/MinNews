# server.py
from fastapi import FastAPI

app = FastAPI()

# /service
@app.post("/get_news")
async def get_news():
    """get news; run on /service

    """
    
    return {"message": "Hello World"}