# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for VOC dataset module."""

from tensorflow_datasets.image import voc
import tensorflow_datasets.testing as tfds_test


class Voc2007Test(tfds_test.DatasetBuilderTestCase):
  DATASET_CLASS = voc.VOC
  SPLITS = {  # Expected number of examples on each split from fake example.
      "train": 10,
      "val": 10,
      "test": 10,
  }
  # If dataset `download_and_extract` more than one resource:
  DL_EXTRACT_RESULT = {
      "trainval": "VOC2007trainval_fake.tar",
      "test": "VOC2007test_fake.tar"
  }
  BUILDER_CONFIG_NAMES_TO_TEST = ["voc2007"]

if __name__ == "__main__":
  tfds_test.test_main()
