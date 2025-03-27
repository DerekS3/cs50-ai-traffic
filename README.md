# CS50 AI Traffic

Traffic project involves building a neural network using TensorFlow to classify road signs from images. The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different road signs. 

## Contributions

`traffic.py`:

`load_data`: Loads and resizes images from directories corresponding to different traffic sign categories. Returns a tuple containing image arrays (resized to a consistent size) and their corresponding labels.

`get_model`: Constructs and returns a compiled neural network model for traffic sign classification, with convolutional and pooling layers, and an output layer matching the number of categories in the dataset.

### Testing

A test script (`test_traffic.py`) has been developed to verify the correct operation of all listed functions.

### Technologies Used

- `Unittest`
- `TensorFlow`
- `Scikit-learn`

### Usage

- main: `python3 traffic.py data_dir`
- test: `python3 test_traffic.py`