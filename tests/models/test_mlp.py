import unittest
from os import remove

from numpy import array

from ecnet.models.mlp import MultilayerPerceptron


class TestMLP(unittest.TestCase):

    def test_mlp(self):

        print('\nUNIT TEST: MultilayerPerceptron')
        mlp = MultilayerPerceptron()
        mlp.add_layer(2, 'relu', 2)
        mlp.add_layer(5, 'relu')
        mlp.add_layer(5, 'relu')
        mlp.add_layer(1, 'linear')
        mlp.fit(array([[1, 0], [0, 1]]), array([[1], [0]]), epochs=15000)

        results = mlp.use(array([[1, 0], [0, 1]]))
        self.assertAlmostEqual(results[0][0], 1, 3)
        self.assertAlmostEqual(results[1][0], 0, 3)

        mlp.save()
        mlp_saved = MultilayerPerceptron()
        mlp_saved.load()
        results = mlp_saved.use(array([[1, 0], [0, 1]]))
        self.assertAlmostEqual(results[0][0], 1, 3)
        self.assertAlmostEqual(results[1][0], 0, 3)
        remove('model.h5')


if __name__ == '__main__':

    unittest.main()
