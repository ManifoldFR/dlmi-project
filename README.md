## _Deep Learning for Medical Imaging_


## Dependencies

We use PyTorch, TorchVision, OpenCV, PIL, and the [albumentations](https://github.com/albumentations-team/albumentations) library for extensive data augmentation.
```bash
conda install -c pytorch torchvision captum
conda install -c conda-forge imgaug
conda install -c albumentations albumentations
```

We use PyTorch's TensorBoard integration to see metrics and some segmentation results during training
```bash
tensorboard --logdir runs/
```

## Training

To launch the training script:
```bash
python train.py --model attunet --loss combined --lr 0.001 -E 80
```


## Data

DRIVE data is located in
```
data/drive
    - training
    - test
```
Its mean is `[0.5078, 0.2682, 0.1613]`, stdev is `[0.3378, 0.1753, 0.0978]`

We also use the STARE dataset with manual vessel annotations: http://cecas.clemson.edu/~ahoover/stare/probing/index.html.
```bash
wget http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar
wget http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar
```
The layout of the data is
```
data/stare/
    - images  # images
    - labels/ # labels
        - labels_ah
        - labels_vk
        - results_hoover
```

CHASE dataset:
```
data/chase
```
The 1st manual annotation set is `*_1stHO.png`, the second manual annotations are `*_2ndHO.png`.


Test that the dataset and data augmentation works:
```bash
python utils/loaders.py
```

The data augmentation can be visualized in a [notebook](augmentations-demo.ipynb).
