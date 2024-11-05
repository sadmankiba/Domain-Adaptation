def get_model(model_architecture, backbone, classes, device_id):

    # UNet Models
    if model_architecture == "Unet":
        import torch
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name= backbone, in_channels=3, classes=classes, encoder_weights= 'imagenet', activation=None).cuda(device_id)
        return model

    # SEGFORMER MODELS
    if model_architecture  == "Segformer":
        import torch
        from transformer import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(backbone, num_labels=classes)
        return model


    # UNet Plus Plus Model
    elif model_architecture == "UnetPlusPlus":
        import torch
        import segmentation_models_pytorch as smp
        model = smp.UnetPlusPlus(encoder_name=backbone,in_channels=3,classes=classes)
        return model