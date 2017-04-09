"""Unit test for DataLoader (public methods only)"""

import unittest
import numpy as np

from dataloader import DataLoader

class TestDataLoader(unittest.TestCase):

    def test_size(self):
        """Test DataLoader.size() """

        dl = DataLoader('./test_data', 2)
        self.assertTrue(dl.size() == 4)

    def test_next(self):
        """Test DataLoader.next() """

        dl = DataLoader('./test_data', 2)

        # both minibatches are properly sized
        i1, m1 = dl.next()
        self.assertTrue(i1.shape == (2,256,256) and m1.shape == (2,256,256))
        i2, m2 = dl.next()
        self.assertTrue(i2.shape == (2,256,256) and m2.shape == (2,256,256))
        self.assertTrue(dl.next() is None)

        # all 4 images were returned in the minibatches
        n_unique_imgs = np.unique([np.sum(i1[0]), np.sum(i1[1]), np.sum(i2[0]), np.sum(i2[1])])
        self.assertTrue(len(n_unique_imgs) == 4)

    def test_reset(self):
        """Test DataLoader.reset() """

        dl = DataLoader('./test_data', 4)
        i1, m1 = dl.next()
        dl.reset()
        i2, m2 = dl.next()

        # all 4 images were returned in both epochs
        n_unique_imgs1 = np.unique([np.sum(i1[0]), np.sum(i1[1]), np.sum(i1[2]), np.sum(i1[3])])
        n_unique_imgs2 = np.unique([np.sum(i2[0]), np.sum(i2[1]), np.sum(i2[2]), np.sum(i2[3])])
        self.assertTrue(len(n_unique_imgs1) == 4 and len(n_unique_imgs2) == 4)

        # and they were in different order
        self.assertTrue(np.linalg.norm(i1-i2) != 0 and np.linalg.norm(m1-m2) != 0)


if __name__ == '__main__':
    unittest.main()