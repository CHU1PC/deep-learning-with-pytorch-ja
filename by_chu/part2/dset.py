import os
import sys
import csv
import glob

import SimpleITK as sitk

import torch
from torch.utils.data import Dataset

from collections import namedtuple


LUNA16 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LUNA16")


CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple", "target, diameter_mm, series_uid, center_xyz"
)


def _load_data():
    mhd_path = glob.glob(
        os.path.join(
            LUNA16, "subset*", "subset*", "*.mhd"))
    mhd_list = {os.path.split(p)[-1][:-4] for p in mhd_path}

    cands = []

    with open(os.path.join(LUNA16, "annotations.csv")) as f:
        reader = list(csv.reader(f))
        for row in reader[1:]:

            ano_series_uid = row[0]
            if ano_series_uid not in mhd_list:
                continue

            diameter = row[-1]
            ano_xyz = (map(int, row[1:4]))

            for i in range(3):

    with open(os.path.join(LUNA16, "candidates.csv")) as f:
        reader = list(csv.reader(f))
        for row in reader[1:]:

            can_series_uid = row[0]
            if can_series_uid not in mhd_list:
                continue

            target = bool(int(row[-1]))
            cand_xyz = (map(int, row[1:4]))


                


class LunaData(Dataset):
    def __init__(self, postive_ratio=3):
        mhd_list = glob.glob(
            os.path.join(
                LUNA16, "subset*", "subset*", "*.mhd"))

        ct_mhd = sitk.ReadImage(mhd_list[0])
        print(ct_mhd)


LunaData()
