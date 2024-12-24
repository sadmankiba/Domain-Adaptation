# Unsupervised Domain Adaptation for Semantic Segmentation

We implemented Adaptive Batch Normalization (AdaBN) to adapt an image segmentation model on previously unseen distribution of test data. 

## Run 

To run training, set path and configuration in `config.json`. Then, set `TRAIN=True` in `main.py` and run `python3 main.py`. 

To run inference, set the configuration in `infer_config.py`. Set, `INFER=True` in `main.py` and run `python3 main.py`.

## Method 

* **Ada-BN:** Re-calculated Batch norm layer statistics using EMA with multiple pass over test dataset.
* **Test-Train Mix:** Mixed training dataset of different ratios during batchnorm statistics calculation for test dataset.
 
## Results 

See [presentation](./Presentation.pdf) and [report](./Report.pdf)


## Contributors

* [Arshad Kazi](https://github.com/Arshad-Kazi)
* Bhuyashi Deka
* [Sadman Sakib](https://github.com/sadmankiba)