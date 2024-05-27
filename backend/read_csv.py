# read_csv.py
import csv
import os

def read_csv(path: str, article_num: int) -> list:
    """\
    read_csv

    Args:
        path (str): absolute csv path
        article_num (int): return article number 
        
    Returns:
        list: The return article links. 
        
        The list length is affected by the number of articles. 
    """
    path = f"{os.getcwd()}/data/article_links/Iran.csv"
    with open(path, newline='') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            print(row[1])


if __name__=="__main__":
    read_csv()