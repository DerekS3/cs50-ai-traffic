import unittest
from traffic import *


class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.images, self.labels = load_data('gtsrb-small')
        
    def test_load_images(self):
        self.assertIsInstance(self.images[0], np.ndarray)

    def test_load_labels(self):
        self.assertEqual(self.labels[0], 0)


class TestGetModel(unittest.TestCase):
    def setUp(self):
        self.model = get_model()
        
    def test_return_model_instance(self):
        self.assertIsInstance(self.model, tf.keras.Model)

    def test_input_shape(self):
        expected_result = (None, IMG_WIDTH, IMG_HEIGHT, 3)
        self.assertEqual(self.model.input_shape, expected_result)

    def test_output_shape(self):
        expected_result = (None, NUM_CATEGORIES)
        self.assertEqual(self.model.output_shape, expected_result)


if __name__ == '__main__':
    unittest.main()