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
from hw2.mlp_model import MLPModel

def train(model, datasets, save_dir, log_dir, num_steps, batch_size, learning_rate):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if os.path.isdir(log_dir):
        import shutil
        shutil.rmtree(log_dir)
    with tf.Graph().as_default():
        images_pl, labels_pl = model.get_data_input(batch_size)
        logits = model.get_model(images_pl, False)
        tf.get_variable_scope().reuse_variables()
        loss = model.get_loss(logits, labels_pl)
        optimizer = model.get_optimizer(loss, learning_rate, batch_size, datasets.train.count)
        evaluation = model.get_evaluation(logits, labels_pl, False)
        test_logits = model.get_model(images_pl, True)
        test_evaluation = model.get_evaluation(test_logits, labels_pl, True)
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
                precision = model.do_eval(sess, test_evaluation, images_pl, labels_pl, datasets.validation, batch_size)
                print("  Precision: %0.06f" % precision)
                print("Test dataset:")
                precision = model.do_eval(sess, test_evaluation, images_pl, labels_pl, datasets.test, batch_size)
                print("  Precision: %0.06f" % precision)

def main(model_name, num_steps, batch_size, learning_rate):
    save_dir = "snapshot/"
    log_dir = "log/"
    datasets = NotMnistDataset(DATA_DIR).datasets
    if model_name == "lr":
        model = LRModel()
    elif model_name == "mlp":
        model = MLPModel(1024)
    else:
        raise ValueError("invalid model name: " + model_name)
    train(model, datasets, save_dir, log_dir, num_steps, batch_size, learning_rate)

import argparse
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model_name", help="model name", default="lr")
    parser.add_argument("--max_step", help="max steps for iteration",
            type=int, default=100000)
    parser.add_argument("--batch_size", help="batch size",
            type=int, default=128)
    parser.add_argument("--learning_rate", help="learning rate",
            type=float, default=0.5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args.model_name, args.max_step, args.batch_size, args.learning_rate)
