# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

class args:
  """
  Parse input arguments
  """
  TRAIN_IMDB = "voc_2007_trainval"                 #训练集
  TEST_IMDB = "voc_2007_test"                      #测试集
  net = 'res101'                                   #网络
  ITERS = 1000                                     #迭代次数
  anchors = [8,16,32]
  ratios = [0.5,1,2]

  data_path = '/home/seirios/GitHub/tf-faster-rcnn'
  cfg_file = os.path.join(data_path,'experiments/cfgs',net+'.yml')
  imdb_name = TEST_IMDB
  max_per_image = 100
  model = os.path.join(data_path,'output',net,TRAIN_IMDB,'default',
                       net+'_faster_rcnn_iter_'+('{}').format(ITERS)+'.ckpt')      #加载训练好的Faster-Rcnn模型
  tag = ''
  comp_mode = False
  set_cfgs = ['ANCHOR_SCALES', anchors, 'ANCHOR_RATIOS', ratios]


if __name__ == '__main__':


  print('Called with args:')

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the initialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  imdb = get_imdb(args.imdb_name)
  imdb.competition_mode(args.comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if args.net == 'vgg16':
    net = vgg16()
  elif args.net == 'res50':
    net = resnetv1(num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(num_layers=152)
  elif args.net == 'mobile':
    net = mobilenetv1()
  else:
    raise NotImplementedError

  # load model
  net.create_architecture("TEST", imdb.num_classes, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)

  if args.model:
    print(('Loading model check point from {:s}').format(args.model))
    saver = tf.train.Saver()
    saver.restore(sess, args.model) #加载训练好的模型
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(args.weight))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')

  test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)

  sess.close()