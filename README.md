Language-led VG for HCI
========

## Preparation
Please refer to [get_started.md](docs/get_started.md) for the preparation of the datasets and pretrained checkpoints.




## Training

The following is an example of model training on the RefCOCOg dataset.
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --config configs/LVVVG_R50_gref.py
```
We train the model on 2 GPUs with a total batch size of 64 for 90 epochs. 
The model and training hyper-parameters are defined in the configuration file ``LVVVG_R50_gref.py``. 
We prepare the configuration files for different datasets in the ``configs/`` folder. 




## Evaluation
Run the following script to evaluate the trained model with a single GPU.
```
python test.py --config configs/LVVVG_R50_gref.py --checkpoint LVVVG_R50_gref.pth --batch_size_test 16 --test_split val
```
Or evaluate the trained model with 2 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env test.py --config configs/LVVVG_R50_gref.py --checkpoint LVVVG_R50_gref.pth --batch_size_test 16 --test_split val
```




## owledgement
Part of our code is based on the previous works [DETR](https://github.com/facebookresearch/detr)，[ReSC](https://github.com/zyang-ur/ReSC) and [VLTVG](https://github.com/yangli18/VLTVG).

