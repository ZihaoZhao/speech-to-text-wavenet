'''
Author: your name
Date: 2020-09-22 15:44:28
LastEditTime: 2020-09-22 17:02:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /speech-to-text-wavenet/test.py
'''
import glob
import json
import os
import time

import glog
import tensorflow as tf

import dataset
import utils
import wavenet

flags = tf.app.flags
flags.DEFINE_string('config_path', 'config/english-28.json', 'Directory to config.')
flags.DEFINE_string('dataset_path', 'data/v28/test.record', 'Path to wave file.')
flags.DEFINE_integer('device', 1, 'The device used to test.')
flags.DEFINE_string('ckpt_dir', 'model/v28', 'Path to directory holding a checkpoint.')
FLAGS = flags.FLAGS


def main(_):
  utils.load(FLAGS.config_path)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
 # with tf.device(FLAGS.device):
  test_dataset = dataset.create(FLAGS.dataset_path, repeat=False, batch_size=1)
  waves = tf.reshape(tf.sparse.to_dense(test_dataset[0]), shape=[1, -1, utils.Data.num_channel])
  labels = tf.sparse.to_dense(test_dataset[1])
  sequence_length = tf.cast(test_dataset[2], tf.int32)
  vocabulary = tf.constant(utils.Data.vocabulary)
  labels = tf.gather(vocabulary, labels)
  logits = wavenet.bulid_wavenet(waves, len(utils.Data.vocabulary))
  decodes, _ = tf.nn.ctc_beam_search_decoder(
    tf.transpose(logits, perm=[1, 0, 2]), sequence_length, merge_repeated=False)
  outputs = tf.gather(vocabulary,  tf.sparse.to_dense(decodes[0]))
  save = tf.train.Saver()

  evalutes = {}
  if os.path.exists(FLAGS.ckpt_dir + '/evalute.json'):
    evalutes = json.load(open(FLAGS.ckpt_dir + '/evalute.json', encoding='utf-8'))

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    status = 0
    while True:
      filepaths = glob.glob(FLAGS.ckpt_dir + '/*.index')
      filepaths.sort()
      filepaths.reverse()
      filepath = filepaths[0]
      max_uid = 0
      for filepath in filepaths:
        model_path = os.path.splitext(filepath)[0]
        uid = os.path.split(model_path)[-1]
        if max_uid <= int(uid.split("-")[1]):
          max_uid = int(uid.split("-")[1])
          max_uid_full = uid
          max_model_path = model_path
          # print(max_uid)
      status = 2
      sess.run(tf.global_variables_initializer())
      sess.run(test_dataset[-1])
      save.restore(sess, max_model_path)
    #   sa print(tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    #  ve.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
      evalutes[max_uid_full] = {}
      tps, preds, poses, count = 0, 0, 0, 0
      while True:
        try:
          count += 1
          y, y_ = sess.run((labels, outputs))
          y = utils.cvt_np2string(y)
          y_ = utils.cvt_np2string(y_)
          tp, pred, pos = utils.evalutes(y_, y)
          tps += tp
          preds += pred
          poses += pos
        #  if count % 1000 == 0:
        #    glog.info('processed %d: tp=%d, pred=%d, pos=%d.' % (count, tps, preds, poses))
        except:
        #  if count % 1000 != 0:
        #    glog.info('processed %d: tp=%d, pred=%d, pos=%d.' % (count, tps, preds, poses))
          break

      evalutes[max_uid_full]['tp'] = tps
      evalutes[max_uid_full]['pred'] = preds
      evalutes[max_uid_full]['pos'] = poses
      evalutes[max_uid_full]['f1'] = 2 * tps / (preds + poses + 1e-20)
      json.dump(evalutes, open(FLAGS.ckpt_dir + '/evalute.json', mode='w', encoding='utf-8'))
      evalute = evalutes[max_uid_full]
      glog.info('Evalute %s: tp=%d, pred=%d, pos=%d, f1=%f.' %
                (max_uid_full, evalute['tp'], evalute['pred'], evalute['pos'], evalute['f1']))
      if status == 1:
        time.sleep(60)
      status = 1


if __name__ == '__main__':
  tf.app.run()
