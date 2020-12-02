import tensorflow as tf
import os
from solver import Solver

Flags = tf.flags.FLAGS

tf.flags.DEFINE_bool('is_train', True, 'define model mode dafault = True')
tf.flags.DEFINE_integer('batch_size', 8, 'define batch size default = 32')
tf.flags.DEFINE_string('dataset', 'brainM2C', 'define datsetName default = brainM2C')
tf.flags.DEFINE_string('mode', 'mri_to_ct', 'define train model mri_to_ct or ct_to_mri  default = mri_to_ct')
tf.flags.DEFINE_string('test_dataset_path', '../tc2mData/train/tfrecords/train.tfrecords',
                       'define train tensorflow records data path')
tf.flags.DEFINE_string('train_dataset_path', '../tc2mData/test/tfrecords/test.tfrecords',
                       'define test tensorflow records data path')
tf.flags.DEFINE_string('model_output_path', None, 'folder or save model to continue learning default = None')
tf.flags.DEFINE_integer('epoch', 1000, 'set epoch number default = 1000')


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # define solver object
    solver = Solver(Flags)

    if Flags.is_train:
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    main()
