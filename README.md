# Slot-Attention
This is a minimalistic PyTorch implementation of [Object-Centric Learning with Slot Attention](https://arxiv.org/pdf/2006.15055.pdf) for Tetrominoes dataset

# Training Results
Below are the slot visualizations of sample test_images after training for 100k steps

Left to right: Input, reconstruction and 4 slots are visualized for each sample test_image.

![Slot visualization after training slot attention model for 100k steps](result_imgs/slots_at_100000.jpg)

# Data
This repo trained Slot Attention only on Tetrominoes dataset which is part of Google Multi-Object Datasets and available as TFRecords here: https://github.com/deepmind/multi_object_datasets.

This code uses only `h5py` versions of this data that @pemami4911 created by modifying the TFRecords, available at this [Dropbox link](https://www.dropbox.com/s/y95oyh3cxl8nk5h/clevr6_with_masks_train.h5?dl=0) for download. ( tetrominoes.h5 and tetrominoes_test.h5 )

# Usage 
Training with default hyperparameters (for hyperparams check argparser in main.py) for Tetrominoes Dataset

`python main.py --train`

For testing on validation data 

`python main.py --test`

For visualizing slots of sample images 

`python main.py --visualize`
