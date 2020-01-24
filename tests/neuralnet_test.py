import unittest
import numpy as np
from fashion import neuralnet

from scipy import special

class TestNeuralNet(unittest.TestCase):
    def test_one_hot_encoding(self):
        labels=[0,3,4,5,6,1,2,3,4]
        encoded = neuralnet.one_hot_encoding(labels, 10)
        self.assertEqual(encoded.shape, (9,10))
        self.assertTrue(np.array_equal(np.array(labels), np.argmax(encoded, axis=1)))

    def test_normalize_data(self):
        tm = np.array([[30,255,243,210],[10,36,78,24]])
        norm = neuralnet.normalize_data(tm)
        self.assertEqual(norm.shape, tm.shape)
        self.assertTrue(np.all(norm.max(axis=1)==np.ones(len(tm))))
        self.assertTrue(np.all(norm.min(axis=1)==np.zeros(len(tm))))

    def test_softmax(self):
        tm = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        out = neuralnet.softmax(tm)
        self.assertEqual(out.shape, tm.shape)
        target = special.softmax(tm, axis=0)
        self.assertTrue(np.allclose(target,out))

if __name__=='__main__':
    unittest.main()