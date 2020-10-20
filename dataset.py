#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-20 11:19:57
# FilePath     : /speech-to-text-wavenet/dataset.py
# Description  : 
#-------------------------------------------# 
import tensorflow as tf


def create(filepath, batch_size=1, repeat=False, buffsize=1000):
  def _parse(record):
    keys_to_features = {
      'uid': tf.FixedLenFeature([], tf.string),
      'audio/data': tf.VarLenFeature(tf.float32),
      'audio/shape': tf.VarLenFeature(tf.int64),
      'text': tf.VarLenFeature(tf.int64)
    }
    features = tf.parse_single_example(
      record,
      features=keys_to_features
    )
    audio = features['audio/data'].values
    shape = features['audio/shape'].values
    audio = tf.reshape(audio, shape)
    audio = tf.contrib.layers.dense_to_sparse(audio)
    text = features['text']
    return audio, text, shape[0], features['uid']

  dataset = tf.data.TFRecordDataset(filepath).map(_parse).batch(batch_size=batch_size)
#   print(len(dataset))
#   exit()
  if buffsize > 0:
    dataset = dataset.shuffle(buffer_size=buffsize)
  if repeat:
    dataset = dataset.repeat()
  iterator = dataset.make_initializable_iterator()
  return tuple(list(iterator.get_next()) + [iterator.initializer])
