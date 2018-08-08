## Success classification for deformable medical image registration outcomes

This codebase tries to learn what images are successfully registered to a target image following deformable image registration. Our implementation currently is specific to medical images, in particular to CT images. **This is a work in progress.**

ct_classifier.py: Defines the dataloader and the neural network
ct_classifier_train.py: Trains the network on a specified number of training data and saves the trained model
ct_classifier_test.py: Tests the network on a left out dataset

**Dependencies**:
- [Torch](https://pypi.org/project/torchvision/): pip install torchvision
- [NiBabel](http://nipy.org/nibabel/): pip install nibabel
- [SciPy](https://www.scipy.org/) pip install scipy
- [Matplotlib](https://matplotlib.org/) pip install matplotlib

**Run**:
- Clone this repository
- If no training data, run *ct_classifier_test.py*. This will use our pretrained network to test on your data
- If training data is available, first run *ct_classifier_train.py*. This will train on the given training data and save the trained model. Next run *ct_classifier_test.py* using the new trained model.
