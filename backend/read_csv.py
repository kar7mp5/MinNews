# read_csv.py
import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor

def resolve_url(url: str, timeout: float = 30) -> str:
    """Resolve URL asynchronously.

    Args:
        url (str): URL to resolve.
        timeout (float, optional): Timeout duration in seconds. Defaults to 10.

    Returns:
        str: Resolved URL or original URL if timeout occurs.
    """
    try:
        response = requests.get(url, allow_redirects=True, timeout=timeout)
        return response.url
    except requests.Timeout:
        print(f"Timeout error: Request took longer than {timeout} seconds.")
        return url  # Return original URL if timeout occurs
    except requests.RequestException as e:
        print(f"Error resolving URL: {e}")
        return url  # Return original URL in case of other request exceptions



def read_csv(path: str, article_num: int) -> list:
    """Read CSV file and resolve URLs asynchronously.

    Args:
        path (str): Relative path to the CSV file.
        article_num (int): Number of articles to read.

    Returns:
        list: List of resolved article URLs.
    """
    _path = f"{os.getcwd()}/{path}"
    _lines = pd.read_csv(_path, usecols=["News Links"], encoding="utf8")

    # Limit article_num to the length of _lines
    article_num = min(article_num, len(_lines))

    # Function to resolve URL asynchronously
    def resolve_url_async(url):
        return resolve_url(url)

    # Use ThreadPoolExecutor to perform URL resolution in parallel
    with ThreadPoolExecutor() as executor:
        resolved_urls = list(executor.map(resolve_url_async, _lines["News Links"][:article_num]))

    return resolved_urls

if __name__ == "__main__":
    # Example usage
    resolved_links = read_csv("/data/article_links/Iran.csv", 2)
    for link in resolved_links:
        print(link)
