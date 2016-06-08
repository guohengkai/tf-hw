from __future__ import print_function
from __future__ import division
import sys
import time
import tensorflow as tf

sys.path.append("..")
from common.common import DATA_DIR, get_sample_idx
from hw1.not_mnist_dataset import NotMnistDataset
from hw2.lr_model import LRModel

def fill_feed_dict(dataset, images_pl, labels_pl, batch_size):
    images_feed, labels_feed = dataset.next_batch(batch_size)
    feed_dict = {
            images_pl: images_feed,
            labels_pl: labels_feed,
    }
    return feed_dict

def do_eval(sess, evaluation, images_pl, labels_pl, dataset, batch_size):
    true_count = 0
    steps_per_epoch = dataset.count // batch_size
    count = steps_per_epoch * batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(dataset, images_pl, labels_pl, batch_size)
        true_count += sess.run(evaluation, feed_dict=feed_dict)
    precision = true_count / count
    print("  Num examples: %d  Num correct: %d  Precision: %0.06f" %
            (count, true_count, precision))

def main():
    is_flatten = True
    save_dir = "snapshot/"
    datasets = NotMnistDataset(DATA_DIR).get_datasets(is_flatten)
    num_steps = 3001
    batch_size = 128
    learning_rate = 0.5
    lr_model = LRModel(is_flatten)
    with tf.Graph().as_default():
        images_pl, labels_pl = lr_model.get_data_input(batch_size)
        logits = lr_model.get_model(images_pl)
        loss = lr_model.get_loss(logits, labels_pl)
        optimizer = lr_model.get_optimizer(loss, learning_rate)
        evaluation = lr_model.get_evaluation(logits, labels_pl)
        summary_op = tf.merge_all_summaries()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.train.SummaryWriter(save_dir, sess.graph)
        sess.run(init)
        print("Initialized")
        for step in xrange(num_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(datasets.train, images_pl,
                    labels_pl, batch_size)
            _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print("Step %d: loss = %0.2f (%0.3f sec)" % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == num_steps:
                saver.save(sess, save_dir, global_step=step)
                print("Training dataset:")
                do_eval(sess, evaluation, images_pl, labels_pl, datasets.train, batch_size)
                print("Validation dataset:")
                do_eval(sess, evaluation, images_pl, labels_pl, datasets.validation, batch_size)
        print("Test dataset:")
        do_eval(sess, evaluation, images_pl, labels_pl, datasets.test, batch_size)

if __name__ == '__main__':
    main()
