import json
from helpers import author_dict
import pandas as pd

AUTHOR_COUNTER = 0


def get_media_mapping(media_type):
    if media_type == "sticker" :return 1
    elif media_type == "annimation" : return 2
    elif media_type == "video_message": return 3
    if media_type == "voice_message" : return 4
    elif media_type == "video_file" : return 4
    else: return 1000

def get_message_class(media_type, photo_attached):
    """returns class identifier of the message
    0 - plain message, only text
    1 - sticker
    2 - annimation
    3 - video_message
    4 - voice_message
    5 - video_file
    6 - photo
    1000 - other
    """
    if bool(photo_attached): return 6
    if not bool(media_type): return 0
    return get_media_mapping(media_type)

def owns_by(user):
    
    """returns author class identifier of the message
    """
    global AUTHOR_COUNTER
    if user not in author_dict: 
        author_dict[user] = AUTHOR_COUNTER
        AUTHOR_COUNTER = AUTHOR_COUNTER + 1
    
    return author_dict[user] 

def preprocess_data(file_path):

    with open(file_path, encoding='UTF-8') as data_file:
        data = json.load(data_file)
        comp_df = pd.DataFrame(data['data'])

    comp_df = comp_df[["id", "type", "from", "text", "media_type", "photo", "date"]].fillna("")

    # Apply the transformations and fill NaN values with 0
    comp_df["owns_by"] = comp_df.apply(lambda x: owns_by(x["from"]), axis=1)
    comp_df["message_class"] = comp_df.apply(lambda x: get_message_class(x["media_type"], x["photo"]), axis=1)

    # Fill NaN values with 0 in specific columns
    columns_to_fill = ["id", "type", "owns_by", "text", "message_class"]
    comp_df[columns_to_fill] = comp_df[columns_to_fill].fillna(0)
    comp_df['text'] = comp_df['text'].astype(str)
    return comp_df