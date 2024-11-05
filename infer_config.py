config = {

    # Paths
    "output_dir": "/mnt/prj002/Dev Kumar/Liver/Ano_Data/BD_MG_EMH_OP/", 
    "training_data":"/mnt/prj002/Dev Kumar/Liver/Ano_Data/Data/",
    "gt_path": None,#"/mnt/prj002/Arshad/Model_Generalization_Datasets/Liver/Microgranuloma/Dataset_v5/Training_Dataset/train/train_data_all_512/Label/",

    # Model Parameters
    #"model_path":"/home/kuldeepg/Projects/Frameworks/DeepLearning/Experiments/Training/Bileduct/Training_Data_V6/config_4/config_4.wts",
    "model_path":"/mnt/imgproc/PROJECTS/Tissue_Triage/Wistar_Rat/Liver (Ameya, Kuldeep, Arshad, Digant)/For_HPC/BD_EMH_MG/best_model_EMH_MG_BD.model",
    "model_architecture": "Unet",
    "backbone":"timm-efficientnet-b4",
    "preprocessing_config": {"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]},
    #"preprocessing_config": {"divideBy":255, "mean":[0.406, 0.456, 0.485], "std_dev":[0.225, 0.224, 0.229]},
    #"preprocessing_config": {"divideBy":255, "mean":[ 0.695, 0.470, 0.704], "std_dev":[ 0.119 , 0.190,0.138]},
    "preprocessing_type": "global",
    "model_weights":False,


    # Device
    "classes":4,

    "tile_size": 1024,
    "device_id":3,

}