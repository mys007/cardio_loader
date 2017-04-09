"""Data loader for DICOM images and associated (internal) masks"""

import logging
from os import path
import csv
import glob
import numpy as np
import random
import parsing


class DataLoader(object):
    """Data loader for DICOM images and associated (internal) masks.

    Data is read from `base_dir`. DICOM images without matching annotations are skipped.

    Samples are returned in minibatches of size `batch_size`. In each epoch, every sample is read once.
    If the dataset size is not divisible by batch size, samples which would form the last incomplete minibatch
    are skipped in the particular epoch. The dataset is randomly reshuffled before start of each epoch.

    Example use:
        loader = DataLoader('./final_data', 8) # data is in 'final_data'
        for inputs, targets in iter(loader.next, None): # sample minibatches of size 8
            print inputs.shape
        loader.reset() # start new epoch by reshuffling the data
    """

    def __init__(self, base_dir, batch_size):
        """Constructor

        :param base_dir: data directory containing 'link.csv' file
        :param batch_size: minibatch size
        :return: list of tuples (dicom dir, contour dir)
        """
        self._base_dir = base_dir
        self._bs = batch_size

        patients = self.__read_patients(path.join(self._base_dir, 'link.csv'))
        self._samples = self.__assemble_samples(patients)
        self.reset()


    def __read_patients(self, filename):
        """Reads CSV list of patients

        :param filename: CSV filename
        :return: list of tuples (dicom dir, contour dir)
        """

        patient_tuples = []

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                patient_tuples.append((row['patient_id'], row['original_id']))

        return patient_tuples


    def __assemble_samples(self, patient_tuples):
        """Iterates over patients and finds all inner contours with matching DICOM files.

        :param patient_tuples: list of tuples (dicom dir, contour dir)
        :return: list of tuples (contour filepath, DICOM filepath)
        """
        sample_tuples = []

        for did, cid in patient_tuples:
            icontours_paths = glob.glob(path.join(self._base_dir, 'contourfiles', cid, 'i-contours/*.txt'))
            for cpath in icontours_paths:
                parts = path.basename(cpath).split('-')
                if parts[0]!='IM' or parts[1]!='0001':
                    logging.warning('Unknown naming pattern: ' + cpath)

                snum = parts[2].lstrip('0')
                dpath = path.join(self._base_dir, 'dicoms', did, snum + '.dcm')
                if path.isfile(dpath):
                    sample_tuples.append((cpath, dpath))
                else:
                    logging.warning('Non-existing dicom for contour: ' + dpath)

        return sample_tuples

    def __load_sample(self, cpath, dpath):
        """Loads a sample from given files

        :param cpath: filepath to the contour file
        :param dpath: filepath to the DICOM file
        :return: tuple (image, mask), both numpy arrays of shape (height, width).
                 None in case of error.
        """

        dcm_dict = parsing.parse_dicom_file(dpath)
        if dcm_dict is None:
            logging.warning('Dicom file invalid: ' + dpath)
            return None

        coords_lst = parsing.parse_contour_file(cpath)
        if len(coords_lst) == 0:
            logging.warning('Contour file empty: ' + coords_lst)
            return None

        img = dcm_dict['pixel_data']
        mask = parsing.poly_to_mask(coords_lst, img.shape[1], img.shape[0])

        logging.debug('Loaded: ' + dpath)
        return img, mask

    def size(self):
        """Return the number of samples in dataset (though not all may be loaded successfully).

        :return: number of samples
        """

        return len(self._samples)

    def next(self):
        """Samples a new minibatch of fixed `batch_size` (given in constructor).

        :return: tuple (images, masks), both numpy arrays of shape (batch_size, height, width).
                 None if the current epoch is finished or in case of error.
        """

        images = []
        masks = []

        while len(images) != self._bs:
            if self._pos >= len(self._samples): # epoch done
                if len(images) > 0:
                    logging.info('Incomplete batch skipped')
                return None

            img, mask = self.__load_sample(*self._samples[self._pos])
            self._pos = self._pos + 1
            if img is None: # move over incorrectly read samples
                continue

            images.append(img)
            masks.append(mask)

        try:
            return np.stack(images), np.stack(masks)
        except ValueError:
            logging.error('Batch consisted of samples of different size')
            return None

    def reset(self):
        """Starts a new epoch by reshuffling the dataset.
        """
        random.shuffle(self._samples)
        self._pos = 0
