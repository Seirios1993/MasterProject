# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

# year:’2007’
# image _set:’trainval’
# devkit_path:’data/VOCdevkit2007’
# data _path:’data/VOCdevkit2007/VOC2007’
# classes:(…)_如果想要训练自己的数据，需要修改这里_
# class _to _ind:{…} _一个将类名转换成下标的字典 _
# image _ext:’.jpg’
# image _index:[‘000001’,’000003’,……]_根据trainval.txt获取到的image索引_
# roidb _handler:<Method gt_roidb >
# salt:  <Object uuid >
# comp _id:’comp4’
# config:{…}

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg


class pascal_voc(imdb):                                  #定义pascal_voc类
  def __init__(self, image_set, year, use_diff=False):
    name = 'voc_' + year + '_' + image_set
    if use_diff:
      name += '_diff'
    imdb.__init__(self, name)
    self._year = year
    self._image_set = image_set
    self._devkit_path = self._get_default_path()                            #数据集的路径
    self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)   #数据集子目录
    self._classes = ('__background__',  # always index 0                    #语义
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))# {类别名：类别索引}字典
    self._image_ext = '.jpg'
    self._image_index = self._load_image_set_index()                        #图片索引号
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb                                     #得到roi图片信息,重载imdb中
    self._salt = str(uuid.uuid4())                  #生成一个随机的uuid，即对于分布式数据，每个数据都有自己对应的唯一的标识符
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': use_diff,
                   'matlab_eval': False,
                   'rpn_file': None}

    assert os.path.exists(self._devkit_path), \
      'VOCdevkit path does not exist: {}'.format(self._devkit_path)         #检测目录存在性
    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])                #返回图片绝对路径

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, 'JPEGImages',
                              index + self._image_ext)                 #通过图片索引号+图片格式返回绝对路径
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                  self._image_set + '.txt')           #目录索引文件（txt）
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)  #x.strip()就是当括号内为空就删除x开头与结尾的（'/n'，'/t',' '）
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]                #通过不同的索引编号确定训练集测试集等分组
    return image_index                                  #返回的image_index为一个列表，包含该数据集图片名称信息

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """                        #由config.py可知_get_default_path返回的是 Fsater-RCNN_TF/data/VOCdevkit+self._year
    return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

  def gt_roidb(self):                                    #得到ROI组成的database
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """                             #例如Fsater-RCNN_TF/data/cache/voc_2007_train__gt_roidb.pkl
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):                      #如果缓存文件存在，就从缓存读取
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb                                    #如果缓存文件不存在，就新建缓存文件，方便下次使用
                                                     # self._load_pascal_annotation(index)返回的是该图片信息dict
    gt_roidb = [self._load_pascal_annotation(index)  #然后按顺序存进一个list，对应图片信息引索与self.image_index引索相对应
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:               #将gt_roidb存入临时文件cache_file
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):                              #使用rpn_roidb这种method从imdb中获取roidb数据
    if int(self._year) == 2007 or self._image_set != 'test':     #数据集名称包含2007年且不是test数据集可以使用这种method
      gt_roidb = self.gt_roidb()                    #获得ground_truth的roidb，其实是xml解析的
      rpn_roidb = self._load_rpn_roidb(gt_roidb)    #生成rpn_roidb
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)#融合
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']              #获取rpn_file
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)  #调用超类imdb产生roidb

  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """                #定位Fsater - RCNN_TF / data / VOCdevkit +'year'/'VOC' + self._year/Annotations/000001.xml
    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)                                   #使用ElementTree处理XML
    objs = tree.findall('object')
    if not self.config['use_diff']:
      # Exclude the samples labeled as difficult  xml文件中该object有一个属性difficult，1表示目标难以区分，0表示容易识别
      non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0]   #该操作就是要把有difficult的目标给剔除
      # if len(non_diff_objs) != len(objs):
      #     print 'Removed {} difficult objects'.format(
      #         len(objs) - len(non_diff_objs))
      objs = non_diff_objs
    num_objs = len(objs)
                                                            # 初始化boxes，先建立一个shape为(num_objs, 4)的全零矩阵
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)        #num_objs为该引索图片中物体的个数，如有两只猫，则num_objs=2
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32) #self.num_classes为之前定义的所有分类的个数
    # "Seg" area for pascal is just the box area            box的面积
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):                        #枚举  目录，值
      bbox = obj.find('bndbox')
      # Make pixel indexes 0-based
      x1 = float(bbox.find('xmin').text) - 1
      y1 = float(bbox.find('ymin').text) - 1
      x2 = float(bbox.find('xmax').text) - 1
      y2 = float(bbox.find('ymax').text) - 1
      cls = self._class_to_ind[obj.find('name').text.lower().strip()]   #取出当前obj的name，变小写，去除字符串头尾'/n','/t',' '
      boxes[ix, :] = [x1, y1, x2, y2]                      #bounding boxes坐标
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0                              #因为这里的box就是gt，所以重叠率设为1;这样子overlaps就成了一个单位矩阵
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)        #计算bbox面积

    overlaps = scipy.sparse.csr_matrix(overlaps)           #将overlaps稀疏矩阵压缩
                                                           #                   总结类型
    return {'boxes': boxes,                                #bounding boxes坐标  array
            'gt_classes': gt_classes,                      #类别及索引           array
            'gt_overlaps': overlaps,                       #重叠                scipy.sparse.csr.csr_matrix
            'flipped': False,                              #图片翻转             bool
            'seg_areas': seg_areas}                        #bounding boxes面积  array

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):            #结果存储路径设置
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'VOC' + self._year,
      'Main',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'Annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      self._devkit_path,
      'VOC' + self._year,
      'ImageSets',
      'Main',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.pascal_voc import pascal_voc

  d = pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
