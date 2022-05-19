import os.path
import numpy as np
import pandas as pd
from skimage.io import imread
from darwin.config import *
import darwin.cohort_selection as cohort_selection
from darwin.perform_split import perform_split
from torchxrayvision.datasets import Dataset, normalize
from sklearn.model_selection import train_test_split

def limit_to_selected_views(self, views):
    """This function is called by subclasses to filter the
    images by view based on the values in .csv['view']
    """
    if type(views) is not list:
        views = [views]
    if '*' in views:
        # if you have the wildcard, the rest are irrelevant
        views = ["*"]
    self.views = views

    # missing data is unknown
    self.csv.view.fillna("UNKNOWN", inplace=True)

    if "*" not in views:
        self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


class MIMIC_Pneumonia_Dataset(Dataset):
    """MIMIC-CXR Dataset
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S.
    MIMIC-CXR: A large publicly available database of labeled chest radiographs.
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """
    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 option='train',
                 pre_processed=False,
                 seed=0,
                 remove_duplicate_hadm=False,
    ):
        self.pathologies = ["Viral",
                            "Bacterial"]

        super(Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.raw_csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)
        
        if os.path.exists(pneumo_processed_full_path) and pre_processed:
            before_split = pd.read_csv(pneumo_processed_full_path)
        else:
            before_split = cohort_selection.get_pneumonia_cohort(self.raw_csv, self.metacsv)
        train, val, test = perform_split(before_split, export_to_csv=True, remove_duplicate_hadm=remove_duplicate_hadm)
               
        if option == 'train':
            self.csv = train
        elif option == 'test':
            self.csv = val
        elif option == 'valid':
            self.csv = test
       
        self.labels = []

        for pathology in self.pathologies:
            mask = self.csv.etiology - 1 == self.pathologies.index(pathology)
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # print('LABELS :', self.labels)

        self.b_labels = []
        for i, elem in enumerate(self.labels):
            if elem[0].astype(int) == 0:
                self.b_labels.append(1)
            else:
                self.b_labels.append(0)

        self.b_labels = np.asarray(self.b_labels).T
        self.b_labels = self.b_labels.astype(np.float32)
        # print('Binary Labels', self.b_labels)

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self), self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.b_labels[idx]
        sample["hadm_id"] = self.csv.iloc[idx]["hadm_id"]
        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return sample
