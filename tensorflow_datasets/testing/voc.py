# coding=utf-8
# Copyright 2018 The TensorFlow Datasets Authors.
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

r"""Generate VOC like files.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tarfile
import tempfile

from absl import app
from absl import flags
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.utils import py_utils


MIN_HEIGHT_WIDTH = 10
MAX_HEIGHT_WIDTH = 15
CHANNELS_NB = 3
NUMBER_IMAGES_PER_SPLIT = 10

flags.DEFINE_string("tfds_dir", py_utils.tfds_dir(),
                    "Path to tensorflow_datasets directory")
FLAGS = flags.FLAGS

def voc_output_dir():
  return os.path.join("tensorflow_datasets", "testing", "test_data", "fake_examples", "voc")


def remake_dirs(d):
  if tf.gfile.Exists(d):
    tf.gfile.DeleteRecursively(d)
  tf.gfile.MakeDirs(d)


def _get_random_picture():
  height = random.randrange(MIN_HEIGHT_WIDTH, MAX_HEIGHT_WIDTH)
  width = random.randrange(MIN_HEIGHT_WIDTH, MAX_HEIGHT_WIDTH)
  return np.random.randint(
      256, size=(height, width, CHANNELS_NB), dtype=np.uint8)


def _get_random_jpeg():
  image = _get_random_picture()
  jpeg = tf.image.encode_jpeg(image)
  with utils.nogpu_session() as sess:
    res = sess.run(jpeg)
  fobj = tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.JPEG')
  fobj.write(res)
  fobj.close()
  return fobj.name, image.shape[0], image.shape[1]

TEMPLATE = """
<annotation>
	<folder>VOC2007</folder>
	<filename>{filename}</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>1234567</flickrid>
	</source>
	<owner>
		<flickrid>dummy</flickrid>
		<name>dummy</name>
	</owner>
	<size>
		<width>{width}</width>
		<height>{height}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>{_class}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{xmin}</xmin>
			<ymin>{ymin}</ymin>
			<xmax>{xmax}</xmax>
			<ymax>{ymax}</ymax>
		</bndbox>
	</object>
</annotation>
"""

def _get_random_annotation(filename, width, height):
  xmin = random.randrange(0, width // 2)
  ymin = random.randrange(0, height // 2)
  xmax = random.randrange(width // 2, width)
  ymax = random.randrange(height // 2, height)

  annotation = TEMPLATE.format(
    filename=filename,
    width=width,
    height=height,
    xmin=xmin,
    ymin=ymin,
    xmax=xmax,
    ymax=ymax,
    _class="dog"
  )

  fobj = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.xml')
  fobj.write(annotation)
  fobj.close()
  return fobj.name


def _get_set_file(set_items):
  fobj = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.xml')
  for item in set_items:
    fobj.write("%s\n" % item)
  fobj.close()
  return fobj.name

def generate_voc_data(name, indices, year):
  output_dir = voc_output_dir()
  tar_name = os.path.join(output_dir, name)

  with tarfile.open(tar_name, "w") as tar:
      for idx in indices:

        # Create images
        jpeg, width, height = _get_random_jpeg()
        tar.add(jpeg, arcname=os.path.join("VOCdevkit/VOC" + year, "JPEGImages", str(idx) + ".jpg"))

        # Create annotations
        annot = _get_random_annotation(str(idx) + ".jpg", width, height)
        tar.add(annot, arcname=os.path.join("VOCdevkit/VOC" + year, "Annotations", str(idx) + ".xml"))

      if "trainval" in name:
        set_file = _get_set_file(indices[:NUMBER_IMAGES_PER_SPLIT])
        tar.add(set_file, arcname=os.path.join("VOCdevkit/VOC" + year, "ImageSets", "Main", "train.txt"))

        set_file = _get_set_file(indices[NUMBER_IMAGES_PER_SPLIT:])
        tar.add(set_file, arcname=os.path.join("VOCdevkit/VOC" + year, "ImageSets", "Main", "val.txt"))

      elif "test" in name:
        set_file = _get_set_file(indices)
        tar.add(set_file, arcname=os.path.join("VOCdevkit/VOC" + year + "/ImageSets/Main", "test.txt"))




def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  output_dir = voc_output_dir()
  remake_dirs(output_dir)
  indices = list(range(3 * NUMBER_IMAGES_PER_SPLIT))
  random.shuffle(indices)
  generate_voc_data(name="VOC2007trainval_fake.tar", indices=indices[:2 * NUMBER_IMAGES_PER_SPLIT], year="2007")
  generate_voc_data(name="VOC2007test_fake.tar", indices=indices[2* NUMBER_IMAGES_PER_SPLIT:], year="2007")
  generate_voc_data(name="VOC2012trainval_fake.tar", indices=indices[:2 * NUMBER_IMAGES_PER_SPLIT], year="2012")


if __name__ == "__main__":
  app.run(main)