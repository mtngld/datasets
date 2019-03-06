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

"""VOC datasets."""

import os
import tarfile

from lxml import etree
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core import api_utils


# Source: https://github.com/tensorflow/models/blob/32e7d660a813c11da61a2ad35055d85df8f14b63/research/object_detection/utils/dataset_util.py#L63
def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.
  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.
  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree
  Returns:
    Python dictionary holding XML contents.
  """
  if not len(xml):
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

class VOCConfig(tfds.core.BuilderConfig):
  """BuilderConfig for VOC."""

  @api_utils.disallow_positional_args
  def __init__(self, voc_year, **kwargs):
    """BuilderConfig for IMDBReviews.
    Args:
      text_encoder_config: `tfds.features.text.TextEncoderConfig`, configuration
        for the `tfds.features.text.TextEncoder` used for the IMDB `"text"`
        feature.
      **kwargs: keyword arguments forwarded to super.
    """
    super(VOCConfig, self).__init__(**kwargs)
    if voc_year not in ["2007", "2012"]:
      raise ValueError("Unknown VOC year %s" % voc_year)
    self.voc_year = voc_year


class VOC(tfds.core.GeneratorBasedBuilder):
  """Short description of my dataset."""

  BUILDER_CONFIGS = [
      VOCConfig(
          name="voc2007",
          version="1.0.0",
          description="voc2007",
          voc_year="2007"
      ),
      VOCConfig(
          name="voc2012",
          version="1.0.0",
          description="voc2012",
          voc_year="2012"
      )]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=("This is the dataset for Pascal VOC. The "
                     "images are kept at their original dimensions."),
        features=tfds.features.FeaturesDict({
            'height': tf.int64,
            'width': tf.int64,
            'filename': tfds.features.Text(),
            'image': tfds.features.Image(),
            'objects': tfds.features.SequenceDict(
                {
                    'bbox': tfds.features.BBoxFeature(),
                    'class': tfds.features.ClassLabel(names=VOC_CLASSES),
                    'difficult': tf.int64,
                    'truncated': tf.int64,
                    'view': tfds.features.Text(),
                }
            )
        }),
        urls=["http://host.robots.ox.ac.uk/pascal/VOC/"],
        citation=r"""@Article{Everingham10,
                    author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
                    title = "The Pascal Visual Object Classes (VOC) Challenge",
                    journal = "International Journal of Computer Vision",
                    volume = "88",
                    year = "2010",
                    number = "2",
                    month = jun,
                    pages = "303--338",
                  }""",
        )

  def _split_generators(self, dl_manager):
    """Create split generators for VOC."""
    if self.builder_config.voc_year == "2007":
      trainval_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
      test_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar'
      extracted_path = dl_manager.download(
          {
              "trainval": trainval_url,
              "test": test_url
          }
      )

    elif self.builder_config.voc_year == "2012":
      trainval_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
      extracted_path = dl_manager.download(
          {
              "trainval": trainval_url,
          }
      )

    train_split_gen = tfds.core.SplitGenerator(
            name="train",
            num_shards=10,
            gen_kwargs={
                "split": "train",
                "tar_file": extracted_path["trainval"],
                "extract_path": dl_manager._extract_dir
            })

    val_split_gen = tfds.core.SplitGenerator(
            name="val",
            num_shards=10,
            gen_kwargs={
                "split": "val",
                "tar_file": extracted_path["trainval"],
                "extract_path": dl_manager._extract_dir
            })

    # VOC2012 test split is not publicly available
    if self.builder_config.voc_year == "2007":
      test_split_gen = tfds.core.SplitGenerator(
              name="test",
              num_shards=10,
              gen_kwargs={
                  "split": "test",
                  "tar_file": extracted_path["test"],
                  "extract_path": dl_manager._extract_dir
              })
      return [train_split_gen, val_split_gen, test_split_gen]
    else:
      return [train_split_gen, val_split_gen]


  def _parse_annotation(self, img_path, annotation_path):
    """Parse annotation function, based on https://git.io/fhpK4"""

    with tf.gfile.GFile(annotation_path, 'r') as fid:
      xml_str = fid.read()

    xml = etree.fromstring(xml_str)
    data = recursive_parse_xml_to_dict(xml)['annotation']

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    class_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' in data:
      for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        difficult_obj.append(int(difficult))
        xmin.append(int(obj['bndbox']['xmin']))
        ymin.append(int(obj['bndbox']['ymin']))
        xmax.append(int(obj['bndbox']['xmax']))
        ymax.append(int(obj['bndbox']['ymax']))
        class_text.append(obj['name'])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'])

    def build_bbox(obj, width, height):
      return tfds.features.BBox(
          ymin=float(obj['bndbox']['ymin']) / height,
          xmin=float(obj['bndbox']['xmin']) / width,
          ymax=float(obj['bndbox']['ymax']) / height,
          xmax=float(obj['bndbox']['xmax']) / width,
      )

    return {
        "image": img_path,
        "filename": data["filename"],
        "width": width,
        "height": height,
        "objects": [{
            "bbox": build_bbox(obj, width, height),
            "class": obj["name"],
            "truncated": int(obj['truncated']),
            "view": obj["pose"],
            "difficult": bool(int(obj['difficult']))
        } for obj in data["object"]]
    }

  def _generate_examples(self, split, tar_file, extract_path):
    with tarfile.open(tar_file) as f:
      f.extractall(path=extract_path)

    root_dir = os.path.join(
      extract_path, "VOCdevkit", "VOC" + self.builder_config.voc_year)

    if split == "train":
      set_file = os.path.join(root_dir, "ImageSets", "Main", "train.txt")
    elif split == "val":
      set_file = os.path.join(root_dir, "ImageSets", "Main", "val.txt")
    else:
      set_file = os.path.join(root_dir, "ImageSets", "Main", "test.txt")

    with tf.gfile.Open(set_file, "r") as f:
      for line in f:
        line = line.rstrip()
        img_path = os.path.join(root_dir, "JPEGImages", line + ".jpg")
        annotation_path = os.path.join(root_dir, "Annotations", line + ".xml")
        parsed = self._parse_annotation(
            img_path=img_path, annotation_path=annotation_path)
        yield parsed
