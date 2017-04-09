"""Unit test for parsing methods"""

import unittest
import numpy as np
from scipy import misc

import parsing

class TestParsingMethods(unittest.TestCase):

    def test_integration(self):
        """Test all 3 methods by creating blended image of mask and dicom image.
           Result needs to be verified manually in test_data/merged*.png. """

        for id in (68,108,148,179):
            dcm_dict = parsing.parse_dicom_file('test_data/dicoms/dicom1/{:d}.dcm'.format(id))
            self.assertTrue(dcm_dict is not None)
            img = dcm_dict['pixel_data']

            coords_lst = parsing.parse_contour_file('test_data/contourfiles/folder1/i-contours/IM-0001-{:04d}-icontour-manual.txt'.format(id))
            self.assertTrue(len(coords_lst) > 0)

            mask = parsing.poly_to_mask(coords_lst, img.shape[1], img.shape[0])
            self.assertTrue(np.sum(mask) > 0)
            self.assertTrue(mask.shape == img.shape)

            imgRGB = np.tile(img, (3,1,1))
            imgRGB[1][mask] = 0
            misc.imsave('test_data/merged{:d}.png'.format(id), imgRGB) #needs to be verified manually


if __name__ == '__main__':
    unittest.main()