## _Deep Learning for Medical Imaging_


## Dependencies

We use PyTorch, TorchVision, OpenCV, PIL, and the [albumentations](https://github.com/albumentations-team/albumentations) library for extensive data augmentation.
```bash
conda install -c pytorch torchvision captum
conda install -c conda-forge imgaug
conda install -c albumentations albumentations
```

Test that the dataset and data augmentation works:
```bash
python utils/loaders.py
```

## Data

DRIVE data is located in
```
data/drive
    - training
    - test
```

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

### Data augmentation

The data augmentation can be visualized in a [notebook](augmentations-demo.ipynb).
