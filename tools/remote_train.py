# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import os
import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

class args:
  """
  Parse input arguments
  """
  TRAIN_IMDB = "voc_2007_trainval"            #训练集
  TEST_IMDB = "voc_2007_test"                 #测试集
  net = 'res101'                              #网络框架
  ITERS = 1000                                #迭代次数
  anchors = [8,16,32]                         #
  ratios = [0.5,1,2]                          #
  stepsize = [50000]                          #

  data_path = '/home/seirios/GitHub/tf-faster-rcnn'
  cfg_file = os.path.join(data_path,'experiments/cfgs',net+'.yml')
  weight = os.path.join(data_path,'data/imagenet_weights',net+'.ckpt')   #使用ImageNet训练好的权值
  imdb_name = TRAIN_IMDB
  imdbval_name = TEST_IMDB
  max_iters = ITERS
  tag = None
  set_cfgs = ['ANCHOR_SCALES', anchors, 'ANCHOR_RATIOS', ratios, 'TRAIN.STEPSIZE', stepsize] #输入是list或者string都能识别

# 融合roidb，roidb来自于数据集（实验可能用到多个），所以需要combine多个数据集的roidb
def combined_roidb(imdb_names):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)  # 设置proposal方法，这里是selective search（config.py）
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)  # 得到用于训练的roidb,定义在train.py,进行了水平翻转，以及为原始roidb添加了一些说明性的属性
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]  # 这里进行combine roidb
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)  # get_imdb方法定义在dataset/factory.py,通过名字得到imdb
    return imdb, roidb


if __name__ == '__main__':


    print('Called with args:')

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # train set
    imdb, roidb = combined_roidb(args.imdb_name)
    print('{:d} roidb entries'.format(len(roidb)))

    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    _, valroidb = combined_roidb(args.imdbval_name)
    print('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip

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

    train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=args.weight,
              max_iters=args.max_iters)

# 代码的执行流程是先读取cfg_file所指定的yml文件来配置部分超参量。执行函数为cfg_from_file(args.cfg_file),
# 它把yml中的超参数合并到config.py中定义的__C对象中，它是类EasyDict的对象。
# 然后，通过cfg_from_list(args.set_cfgs)配置__C对象中的变量。
# 接下来，开始处理训练集，通过combined_roidb(args.imdb_name)收集训练集，
# 它通过调用lib/datasets/factory.py中的get_imdb（）获得数据集，获得类pascal_voc的对象imdb，
# 再设置区域推荐的方式，默认为gt，通过lib/model/train_val.py中的函数get_training_roidb（）获得roidb，
# 即每张图片中的区域推荐样本，其为实际为imdb中的一个变量。打印出区域推荐样本的数量
# 接下来设置训练好的模型和tensorboard文件的存储路径，再获取验证集的数据，前面的训练的数据是经过数据增强的，
# 每张图片都经过旋转，验证集不进行数据增强。
# 接下来，配置vgg16网络的batch数量，默认是设置为1。
# 最后调用train_val.py中的train_net（）函数开启训练。