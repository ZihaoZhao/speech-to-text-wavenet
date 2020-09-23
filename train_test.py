'''
Author: your name
Date: 2020-09-21 11:26:53
LastEditTime: 2020-09-22 15:21:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /speech-to-text-wavenet/train.py
'''
import glob
import os
import glog
import tensorflow as tf
import utils
import wavenet
import dataset
import json

flags = tf.flags
flags.DEFINE_string('config_path', 'config/english-28.json', 'Directory to config.')
flags.DEFINE_string('dataset_path', 'data/v28/train.record', 'Filepath to train dataset record.')
flags.DEFINE_integer('batch_size', 32, 'Batch size of train.')
flags.DEFINE_integer('display', 20, 'Step to display loss.')
flags.DEFINE_integer('snapshot', 20, 'Step to save model.')
flags.DEFINE_string('device', "0", 'The device used to train.')
flags.DEFINE_string('pretrain_dir', 'pretrain', 'Directory to pretrain.')
flags.DEFINE_string('ckpt_path', 'model/v28/ckpt', 'Path to directory holding a checkpoint.')
flags.DEFINE_string('ckpt_dir', 'model/v28', 'Path to directory holding a checkpoint.')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate of train.')
FLAGS = flags.FLAGS


def main(_):
  os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
  utils.load(FLAGS.config_path)
  global_step = tf.train.get_or_create_global_step()
  train_dataset = dataset.create(FLAGS.dataset_path, FLAGS.batch_size, repeat=True)

  # bug tensorflow!!!  the  train_dataset[0].shape[0] != FLAGS.batch_size once in a while
  # waves = tf.reshape(tf.sparse.to_dense(train_dataset[0]), shape=[FLAGS.batch_size, -1, utils.Data.num_channel])
  waves = tf.sparse.to_dense(train_dataset[0])
  waves = tf.reshape(waves, [tf.shape(waves)[0], -1, utils.Data.num_channel])
  # print(waves.shape)
  # for i, t in enumerate(train_dataset):
  #   print(i, t)
  # exit()
  labels = tf.cast(train_dataset[1], tf.int32)
  sequence_length = tf.cast(train_dataset[2], tf.int32)
  logits = wavenet.bulid_wavenet(waves, len(utils.Data.vocabulary), is_training=True)
  loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, sequence_length, time_major=False))

  vocabulary = tf.constant(utils.Data.vocabulary)
  decodes, _ = tf.nn.ctc_beam_search_decoder(
    tf.transpose(logits, [1, 0, 2]), sequence_length, merge_repeated=False)
  outputs = tf.gather(vocabulary, tf.sparse.to_dense(decodes[0]))
  labels = tf.gather(vocabulary, tf.sparse.to_dense(labels))

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  save = tf.train.Saver(max_to_keep=1000)
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True

  # # test
  test_dataset = dataset.create(FLAGS.dataset_path, repeat=False, batch_size=1)
  twaves = tf.reshape(tf.sparse.to_dense(test_dataset[0]), shape=[1, -1, utils.Data.num_channel])
  tlabels = tf.sparse.to_dense(test_dataset[1])
  tsequence_length = tf.cast(test_dataset[2], tf.int32)
  tvocabulary = tf.constant(utils.Data.vocabulary)
  tlabels = tf.gather(tvocabulary, tlabels)
  tlogits = wavenet.bulid_wavenet(waves, len(utils.Data.vocabulary))
  tdecodes, _ = tf.nn.ctc_beam_search_decoder(
    tf.transpose(logits, perm=[1, 0, 2]), tsequence_length, merge_repeated=False)
  toutputs = tf.gather(tvocabulary,  tf.sparse.to_dense(tdecodes[0]))
  # tsave = tf.train.Saver()

  
  with tf.control_dependencies(update_ops):
    optimize = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss=loss, global_step=global_step)

  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_dataset[-1])
    # if os.path.exists(FLAGS.pretrain_dir) and len(os.listdir(FLAGS.pretrain_dir)) > 0:
    #   save.restore(sess, tf.train.latest_checkpoint(FLAGS.pretrain_dir))
    ckpt_dir = os.path.split(FLAGS.ckpt_path)[0]
    if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)
    # if len(os.listdir(ckpt_dir)) > 0:
    #   save.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    losses, tps, preds, poses = 0, 0, 0, 0
    gp = 0
    while gp < 2002:
      if len(os.listdir(ckpt_dir)) > 0:
        save.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
      _, ll, uid, ot, ls, _ = sess.run((global_step,  labels, train_dataset[3], outputs, loss, optimize))
      tp, pred, pos = utils.evalutes(utils.cvt_np2string(ot), utils.cvt_np2string(ll))
      tps += tp
      losses += ls
      preds += pred
      poses += pos
      if gp % FLAGS.display == 0:
        glog.info("Step %d: loss=%f, tp=%d, pos=%d, pred=%d, f1=%f." %
                  (gp, losses if gp == 0 else (losses / FLAGS.display), tps, preds, poses,
                   2 * tps / (preds + poses + 1e-10)))
        losses, tps, preds, poses = 0, 0, 0, 0
      if (gp+1) % FLAGS.snapshot == 0 and gp != 0:
        save.save(sess, FLAGS.ckpt_path, global_step=global_step)

      if (gp+1) % FLAGS.snapshot == 0 and gp != 0:      
        filepaths = glob.glob(FLAGS.ckpt_dir + '/*.index')
        filepaths.sort()
        filepath = filepaths[0]
        print(filepaths)
        model_path = os.path.splitext(filepath)[0]
        sess.run(tf.global_variables_initializer())
        sess.run(test_dataset[-1])
        save.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        evalutes = {}
        ttps, tpreds, tposes, tcount = 0, 0, 0, 0
        while True:
          # try:
            tcount += 1
            y, y_ = sess.run((tlabels, toutputs))
            y = utils.cvt_np2string(y)
            y_ = utils.cvt_np2string(y_)
            tp, pred, pos = utils.evalutes(y_, y)
            ttps += tp
            tpreds += pred
            tposes += pos
            if tcount % 10 == 0:
              glog.info('test processed %d: tp=%d, pred=%d, pos=%d.' % (tcount, tps, preds, poses))
          # except:
          # #  if count % 1000 != 0:
          # #    glog.info('processed %d: tp=%d, pred=%d, pos=%d.' % (count, tps, preds, poses))
          #   break


        evalutes['tp'] = ttps
        evalutes['pred'] = tpreds
        evalutes['pos'] = tposes
        evalutes['f1'] = 2 * ttps / (tpreds + tposes + 1e-20)
        json.dump(evalutes, open(FLAGS.ckpt_dir + '/evalute.json', mode='w', encoding='utf-8'))
        evalute = evalutes
        glog.info('Evalute %s: tp=%d, pred=%d, pos=%d, f1=%f.' %
                  (uid, evalute['tp'], evalute['pred'], evalute['pos'], evalute['f1']))
      gp =gp + 1

if __name__ == '__main__':
  tf.app.run()
