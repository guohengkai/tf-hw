from __future__ import print_function
from __future__ import division
import os
import sys
import time
import tensorflow as tf

sys.path.append("..")
from common.common import DATA_DIR
from hw1.not_mnist_dataset import NotMnistDataset
from hw2.lr_model import LRModel

def train(model, datasets, save_dir, log_dir, num_steps, batch_size, learning_rate):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with tf.Graph().as_default():
        images_pl, labels_pl = model.get_data_input(batch_size)
        logits = model.get_model(images_pl)
        loss = model.get_loss(logits, labels_pl)
        optimizer = model.get_optimizer(loss, learning_rate)
        evaluation = model.get_evaluation(logits, labels_pl)
        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)
        sess.run(init)
        print("Initialized")
        for step in xrange(num_steps):
            start_time = time.time()
            feed_dict = model.fill_feed_dict(datasets.train, images_pl,
                    labels_pl, batch_size)
            _, loss_value, acc_value = sess.run([optimizer, loss, evaluation], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print("Step %d: loss = %0.4f acc = %0.4f (%0.3f sec)" % (step, loss_value, acc_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == num_steps:
                saver.save(sess, save_dir, global_step=step)
                print("Validation dataset:")
                precision = model.do_eval(sess, evaluation, images_pl, labels_pl, datasets.validation, batch_size)
                print("  Precision: %0.06f" % precision)
                print("Test dataset:")
                precision = model.do_eval(sess, evaluation, images_pl, labels_pl, datasets.test, batch_size)
                print("  Precision: %0.06f" % precision)

def main():
    save_dir = "snapshot/"
    log_dir = "log/"
    datasets = NotMnistDataset(DATA_DIR).datasets
    num_steps = 50000
    batch_size = 128
    learning_rate = 0.5
    lr_model = LRModel()
    train(lr_model, datasets, save_dir, log_dir, num_steps, batch_size, learning_rate);


if __name__ == '__main__':
    main()
