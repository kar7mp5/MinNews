# read_csv.py
import pandas as pd
import os

def read_csv(path: str, article_num: int) -> list:
    """\
    read_csv

    Args:
        path (str): relative csv path
        article_num (int): return article number 
        
    Returns:
        list: The return article links. 
        
        The list length is affected by the number of articles. 
    """
    _path = f"{os.getcwd()}{path}" # convert relative path to absolute path.

    _lines = pd.read_csv(
                        filepath_or_buffer=_path, 
                        usecols=["News Links"], 
                        encoding="utf8"
                        )

    _links: list = []
    # check `(param)article_num` over `(param)lines` length.
    # True for return `(param)lines`
    # False for return `(param)article_num`
    if (article_num > _lines.__len__()):
        return [_lines.values[i][0] for i in range(_lines.__len__())]
    else:
        return [_lines.values[i][0] for i in range(article_num)]


if __name__=="__main__":
    read_csv("/data/article_links/Iran.csv", 2)