"""Unit test for DataLoader (public methods only)"""

import unittest
import numpy as np

from dataloader import DataLoader

class TestDataLoader(unittest.TestCase):

    def test_size(self):
        """Test DataLoader.size() """

        dl = DataLoader('./test_data', 2)
        self.assertTrue(dl.size() == 4)
        
        dl = DataLoader('./test_data', 2, True)
        self.assertTrue(dl.size() == 1)

    def test_next(self):
        """Test DataLoader.next() """

        dl = DataLoader('./test_data', 2)

        # both minibatches are properly sized
        i1, mi1, mo1 = dl.next()
        self.assertTrue(i1.shape == (2,256,256) and mi1.shape == (2,256,256) and mo1 == None)
        i2, mi2, mo2 = dl.next()
        self.assertTrue(i2.shape == (2,256,256) and mi2.shape == (2,256,256) and mo2 == None)
        self.assertTrue(dl.next() is None)

        # all 4 images were returned in the minibatches
        n_unique_imgs = np.unique([np.sum(i1[0]), np.sum(i1[1]), np.sum(i2[0]), np.sum(i2[1])])
        self.assertTrue(len(n_unique_imgs) == 4)
        
    def test_next_inout_contour(self):
        """Test DataLoader.next(), inner and outer contours """

        dl = DataLoader('./test_data', 1, True)

        # both minibatches are properly sized and masks are different
        i1, mi1, mo1 = dl.next()
        self.assertTrue(i1.shape == (1,256,256) and mi1.shape == (1,256,256) and mo1.shape == (1,256,256))
        self.assertTrue(np.linalg.norm(mi1-mo1) != 0)     

    def test_reset(self):
        """Test DataLoader.reset() """

        dl = DataLoader('./test_data', 4)
        i1, m1, _ = dl.next()
        dl.reset()
        i2, m2, _ = dl.next()

        # all 4 images were returned in both epochs
        n_unique_imgs1 = np.unique([np.sum(i1[0]), np.sum(i1[1]), np.sum(i1[2]), np.sum(i1[3])])
        n_unique_imgs2 = np.unique([np.sum(i2[0]), np.sum(i2[1]), np.sum(i2[2]), np.sum(i2[3])])
        self.assertTrue(len(n_unique_imgs1) == 4 and len(n_unique_imgs2) == 4)

        # and they were in different order
        self.assertTrue(np.linalg.norm(i1-i2) != 0 and np.linalg.norm(m1-m2) != 0)


if __name__ == '__main__':
    unittest.main()