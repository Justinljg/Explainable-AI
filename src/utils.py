import pandas as pd
import PIL.Image as Image
import os

def prepare_df(path:str) -> pd.DataFrame:

    onlyfiles = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(path) for f in filenames]
    df = pd.DataFrame({'file_path': onlyfiles})
    df['file_path'] = df['file_path'].astype(str)
    df["label"] = df["file_path"].apply(get_label)
    df["usage"] = df["file_path"].apply(get_data_use)
    df["size"] = df["file_path"].apply(get_image_size)
    df["aspect_ratio"] = df["size"].apply(lambda x: x[0] / x[1])
    return df

def get_label(file_path:str) -> str:
    if "bacteria" in file_path:
        return "bacteria"
    elif "virus" in file_path:
        return "virus"
    elif "normal" in file_path:
        return "normal"
    else:
        return None
    
def get_data_use(file_path:str) -> str:
    if "train" in file_path:
        return "train"
    elif "test" in file_path:
        return "test"
    elif "val" in file_path:
        return "val"
    else:
        return None

def get_image_size(file_path:str) -> tuple:
    with Image.open(file_path) as img:
        return img.size
