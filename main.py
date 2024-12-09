import cv2
import json
import os
from trainer import train_model
from inference import infer_model
from utils import get_config_dict
from infer_config import config as infer_config_dict




CONFIG_FILE_PATH = "config.json"
TRAIN_MODEL = False
INFER_MODEL = True



if __name__ == "__main__":

    # LOAD CONFIG DICT 
    config_dict = get_config_dict(CONFIG_FILE_PATH)

    # TRAIN MODEL
    if TRAIN_MODEL:
        train_model(config_dict)

    # TILE LEVEL INFERENCE
    if INFER_MODEL:
        data_path = infer_config_dict["training_data"]
        output_path = infer_config_dict["output_dir"]
        os.makedirs(output_path, exist_ok = True)
        infer_model(infer_config_dict, data_path, gt_path=infer_config_dict["gt_path"],
                    output_path=output_path, fog_score=infer_config_dict.get("fog_score"))
        
    


        


    
