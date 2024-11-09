import json
import numpy as np




def get_config_dict(CONFIG_FILE_PATH):

    '''Read Config File'''

    with open(CONFIG_FILE_PATH, 'r') as file:
        json_string = file.read()
        config_dict = json.loads(json_string)

    return config_dict


def print_global_scores(dice_metric, val_loss_list, classes, split):

    mean_val_loss = np.mean(val_loss_list)
    classwsie_global_val_dice = dice_metric.calculate_global_dice()
    dice_metric.reset()
    print(f"{split } Loss: {mean_val_loss:.3f}")
    for class_index in range(classes):
        print(f"Dice Score {class_index}: {classwsie_global_val_dice[class_index]:.3f}")


