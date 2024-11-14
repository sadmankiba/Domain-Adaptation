config = {

    # Paths
    # "output_dir": "/mnt/prj002/Dev Kumar/Liver/Ano_Data/BD_MG_EMH_OP/",
    "output_dir": "/users/Sadman/Test_Time_Domain_Adaptation/Experiments/Foggy_test_image_outputs/", 
    "training_data":"/users/Sadman/Test_Time_Domain_Adaptation/data/leftImg8bit_foggyDBF_short/train/hamburg/",
    "gt_path": "/users/Sadman/Test_Time_Domain_Adaptation/data/gtFine/train/hamburg/",#"/mnt/prj002/Arshad/Model_Generalization_Datasets/Liver/Microgranuloma/Dataset_v5/Training_Dataset/train/train_data_all_512/Label/",

    # Model Parameters
    #"model_path":"/home/kuldeepg/Projects/Frameworks/DeepLearning/Experiments/Training/Bileduct/Training_Data_V6/config_4/config_4.wts",
    # With aug
    # "model_path": "/users/Sadman/Test_Time_Domain_Adaptation/Experiments/Baseline for Cityscape with 20 classes with Basic Augmentation/best_model_epoch_17_loss_0.388.model",
    # Without aug
    "model_path": "/users/Sadman/Test_Time_Domain_Adaptation/Experiments/Baseline for Cityscape with 20 classes/epoch_17_loss_0.412.model",
    
    "model_architecture": "Unet",
    "backbone":"resnet50",
    "preprocessing_config": {"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]},
    #"preprocessing_config": {"divideBy":255, "mean":[0.406, 0.456, 0.485], "std_dev":[0.225, 0.224, 0.229]},
    #"preprocessing_config": {"divideBy":255, "mean":[ 0.695, 0.470, 0.704], "std_dev":[ 0.119 , 0.190,0.138]},
    "preprocessing_type": "global",
    "model_weights":False,


    # Device
    "classes":20,

    "tile_size": 1024,
    "device_id":0,

}