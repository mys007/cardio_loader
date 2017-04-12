"""Data loader for DICOM images and associated inner (and optionally outer) masks."""

import logging
from os import path
import csv
import glob
import numpy as np
import random
import parsing


class DataLoader(object):
    """Data loader for DICOM images and associated inner (and optionally outer) masks.

    Data is read from `base_dir`. DICOM images without matching annotations are skipped. If `both_contours`
    is given, only DICOM images with both contours will be read.

    Samples are returned in minibatches of size `batch_size`. In each epoch, every sample is read once.
    If the dataset size is not divisible by batch size, samples which would form the last incomplete minibatch
    are skipped in the particular epoch. The dataset is randomly reshuffled before start of each epoch.

    Example use:
        loader = DataLoader('./final_data', 8) # data is in 'final_data'
        for inputs, imasks, omasks in iter(loader.next, None): # sample minibatches of size 8
            print inputs.shape   # omasks is None
        loader.reset() # start new epoch by reshuffling the data
    """

    def __init__(self, base_dir, batch_size, both_contours=None, patient_subset=None):
        """Constructor

        :param base_dir: data directory containing 'link.csv' file
        :param batch_size: minibatch size
        :param both_contours: Boolean whether to load samples with both contours only
        :param patient_subset: Optional list of 0-based indexes in 'link.csv' to filtering loading to selected patients
        :return: list of tuples (dicom dir, contour dir)
        """
        self._base_dir = base_dir
        self._bs = batch_size
        self._both_contours = both_contours == True
        self._pos = 0

        patients = self.__read_patients(path.join(self._base_dir, 'link.csv'), patient_subset)
        self._samples = self.__assemble_samples(patients)


    def __read_patients(self, filename, patient_subset):
        """Reads CSV list of patients

        :param filename: CSV filename
        :param patient_subset: Optional list of 0-based indexes in 'link.csv' to filtering loading to selected patients
        :return: list of tuples (dicom dir, contour dir)
        """

        patient_tuples = []
        i = 0

        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if patient_subset is None or i in patient_subset:
                    patient_tuples.append((row['patient_id'], row['original_id']))
                i = i + 1

        return patient_tuples


    def __assemble_samples(self, patient_tuples):
        """Iterates over patients and finds all inner contours with matching DICOM files.
        If `both_contours` is given (in constructor), only DICOM files with both inner and outer contours will be found.

        :param patient_tuples: list of tuples (dicom dir, contour dir)
        :return: list of tuples (inner contour filepath, outer contour filepath, DICOM filepath),
                 outer contour filepath is None unless `both_contours` is True
        """
        sample_tuples = []

        for did, cid in patient_tuples:
            icontours_paths = glob.glob(path.join(self._base_dir, 'contourfiles', cid, 'i-contours/*.txt'))
            for cipath in sorted(icontours_paths):
                parts = path.basename(cipath).split('-')
                if parts[0]!='IM' or parts[1]!='0001':
                    logging.warning('Unknown naming pattern: ' + cipath)

                copath = cipath.replace('icontour', 'ocontour').replace('i-contours', 'o-contours') if self._both_contours else None
                if copath and not path.isfile(copath):
                    logging.debug('Missing outer contour: ' + copath)
                    continue

                snum = parts[2].lstrip('0')
                dpath = path.join(self._base_dir, 'dicoms', did, snum + '.dcm')
                if path.isfile(dpath):
                    sample_tuples.append((cipath, copath, dpath))
                else:
                    logging.warning('Non-existing dicom for contour: ' + dpath)

        return sample_tuples

    def __load_sample(self, cipath, copath, dpath):
        """Loads a sample from given files

        :param cipath: filepath to the inner contour file
        :param copath: filepath to the outer contour file (may be None)
        :param dpath: filepath to the DICOM file
        :return: tuple (image, inner mask, outer mask), all numpy arrays of shape (height, width), outer mask None if copath None.
                 tuple of Nones in case of error.
        """

        dcm_dict = parsing.parse_dicom_file(dpath)
        if dcm_dict is None:
            logging.warning('Dicom file invalid: ' + dpath)
            return None, None, None
        img = dcm_dict['pixel_data']

        coords_lst = parsing.parse_contour_file(cipath)
        if len(coords_lst) == 0:
            logging.warning('Inner contour file empty: ' + cipath)
            return None, None, None
        imask = parsing.poly_to_mask(coords_lst, img.shape[1], img.shape[0])

        if copath:
            coords_lst = parsing.parse_contour_file(copath)
            if len(coords_lst) == 0:
                logging.warning('Outer contour file empty: ' + copath)
                return None, None, None
            omask = parsing.poly_to_mask(coords_lst, img.shape[1], img.shape[0])
        else:
            omask = None

        logging.debug('Loaded: ' + dpath)
        return img, imask, omask

    def size(self):
        """Return the number of samples in dataset (though not all may be loaded successfully).

        :return: number of samples
        """

        return len(self._samples)

    def next(self):
        """Samples a new minibatch of fixed `batch_size` (given in constructor).

        :return: tuple (images, inner masks, outer masks), all numpy arrays of shape (batch_size, height, width),
                 outer masks is None unless `both_contours` is True.
                 None if the current epoch is finished or in case of error.
        """

        images = []
        imasks = []
        omasks = []

        while len(images) != self._bs:
            if self._pos >= len(self._samples): # epoch done
                if len(images) > 0:
                    logging.info('Incomplete batch skipped')
                return None

            img, imask, omask = self.__load_sample(*self._samples[self._pos])
            self._pos = self._pos + 1
            if img is None: # move over incorrectly read samples
                continue

            images.append(img)
            imasks.append(imask)
            omasks.append(omask)

        try:
            return np.stack(images), np.stack(imasks), np.stack(omasks) if self._both_contours else None
        except ValueError:
            logging.error('Batch consisted of samples of different size')
            return None

    def reset(self):
        """Starts a new epoch by reshuffling the dataset.
        """
        random.shuffle(self._samples)
        self._pos = 0
