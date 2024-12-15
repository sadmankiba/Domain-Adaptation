# Unsupervised Domain Adaptation for Semantic Segmentation

We implemented Adaptive Batch Normalization (AdaBN) to adapt an image segmentation model on previously unseen distribution of test data. 

To run training, set path and configuration in `config.json`. Then, set `TRAIN=True` in `main.py` and run `python3 main.py`. 

To run inference, set the configuration in `infer_config.py`. Set, `INFER=True` in `main.py` and run `python3 main.py`.

## Report 
See [Report.pdf](./Report.pdf)
