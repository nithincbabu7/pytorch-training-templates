# Pytorch Training Templates
Python code templates for deep learning tasks. Includes configurable modules that can deal with multiple loss functions and models. Supports both epoch based and iteration based training approaches.

## Usage
Epoch based training:
```
python train_epoch.py --exp_name test --num_epochs 10 --batch_size 16
```
Iteration based training:
```
python train_iter.py --exp_name test --num_iterations 1000 --batch_size 16
```
