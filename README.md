[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![MIT License][license-shield]][license-url]
[![Coverage][coverage-shield]][license-url]
# KERAS CLASSIFY

Image classification for projects and researches

<!-- ABOUT THE PROJECT -->
## About The Project

Image classification is a commonly used problem in the experimental part of scientific papers and also frequently appears as part of the projects. With the desire to reduce time and effort, Keras Classify was created.


<!-- GETTING STARTED -->
## Getting Started

### Installation

1. Clone the repo:  https://github.com/nguyentruonglau/keras-classify.git

2. Install packages
   ```
   > python -m venv <virtual environments name>
   > activate.bat (in scripts folder)
   > pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Todo List:

- [x] Cosine learning rate scheduler
- [x] Gradient-based Localization
- [x] Sota models
- [x] Synthetic data
- [x] Smart Resize
- [x] Support Python 3.X and Tf 2.X
- [x] Use imagaug for augmentation data
- [x] Use prefetching and multiprocessing to training.
- [x] Analysis Of Input Shape
- [x] Compiled using XLA, auto-clustering on GPU
- [x] Receiver operating characteristic


## Quick Start

### Analysis Of Input Shape

If your data has random `input_shape`, you don't know which `input_shape` to choose, the analysis program is the right choice for you. The algorithm is applied to analyze: [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation).

### Convert Data

From tensorflow 2.3.x already support auto fit_generator, however moving the data to npy file will make it easier to manage.
The algorithm is applied to shuffle data: [Random Permutation](https://en.wikipedia.org/wiki/Random_permutation). Read more [here](https://github.com/nguyentruonglau/keras-classify/blob/main/convert/readme.md).

Run: python convert/convert_npy.py


### Training Model.

Design your model at `model/models.py`, we have made EfficientNetB0 the default. Adjust the appropriate hyperparameters and run: python train.py

### Evaluate Model.


* Statistics number of images per class after suffle on test data.

* Provide model evalution indicators such as: Accuracy, Precesion, Recall, F1-Score and AUC (Area Under the Curve).

* Plot training history of Accuracy, Loss, Receiver Operating Characteristic curve and Confusion Matrix.


### Explainable AI.

[Grad-CAM](https://arxiv.org/abs/1610.02391): Visual Explanations from Deep Networks via Gradient-based Localization. "We propose a technique for producing 'visual explanations' for decisions from a large class of CNN-based models, making them more transparent" Ramprasaath R. Selvaraju ... Read more [here](https://github.com/nguyentruonglau/keras-classify/blob/main/explain/readme.md).


## Example Code


### Use for projects
  ```
  from keras.preprocessing.image import load_img, img_to_array
  from keras.preprocessing.image import smart_resize
  from tensorflow.keras.models import load_model
  import tensorflow as tf
  import numpy as np

  #load pretrained model
  model_path = 'data/output/model/val_accuracy_max.h5'
  model = load_model(model_path)

  #load data
  img_path = 'images/images.jpg'
  img = load_img(img_path)
  img = img_to_array(img)
  img = smart_resize(img, (72,72)) #resize to HxW
  img = np.expand_dims(img, axis=0)

  #prediction
  y_pred = model.predict(img)
  y_pred = np.argmax(y_pred, axis=1)

  #see convert/output/label_decode.json
  print(y_pred)
  ```


### Smart resize (tf < 2.4.1)

  ```
  from tensorflow.keras.preprocessing.image import img_to_array
  from tensorflow.keras.preprocessing.image load_img
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import image_ops
  import numpy as np
  
  def smart_resize(img, new_size, interpolation='bilinear'):
      """Resize images to a target size without aspect ratio distortion.

      Arguments:
        img (3D array): image data
        new_size (tuple): HxW

      Returns:
        [3D array]: image after resize
      """
      # Get infor of the image
      height, width, _ = img.shape
      target_height, target_width = new_size

      crop_height = (width * target_height) // target_width
      crop_width = (height * target_width) // target_height

      # Set back to input height / width if crop_height / crop_width is not smaller.
      crop_height = np.min([height, crop_height])
      crop_width = np.min([width, crop_width])

      crop_box_hstart = (height - crop_height) // 2
      crop_box_wstart = (width - crop_width) // 2

      # Infor to resize image
      crop_box_start = array_ops.stack([crop_box_hstart, crop_box_wstart, 0])
      crop_box_size = array_ops.stack([crop_height, crop_width, -1])

      img = array_ops.slice(img, crop_box_start, crop_box_size)
      img = image_ops.resize_images_v2(
          images=img,
          size=new_size,
          method=interpolation)
      return img.numpy()
  ```


<!-- CONTRIBUTING -->
## Contributor

1. BS Nguyen Truong Lau
2. PhD Thai Trung Hieu

<!-- LICENSE -->
## License

Distributed under the MIT License. See [LICENSE](https://github.com/nguyentruonglau/keras-classify/blob/main/LICENSE.txt) for more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://github.com/nguyentruonglau/keras-classify/blob/main/images/license-MIT-green.svg
[license-url]: https://github.com/nguyentruonglau/keras-classify/blob/main/LICENSE.txt
[coverage-shield]: https://github.com/nguyentruonglau/keras-classify/blob/main/images/coverage-93%25-green.svg
[coverage-url]: https://github.com/nguyentruonglau
