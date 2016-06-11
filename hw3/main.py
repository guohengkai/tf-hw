from __future__ import print_function
from __future__ import division
import os
import sys
import tensorflow as tf

sys.path.append("..")
from common.common import DATA_DIR
from hw1.not_mnist_dataset import NotMnistDataset
from hw2.main import train
from hw3.deep_mlp_model import DeepMLPModel

def main(hidden_num, num_steps, batch_size, learning_rate):
    save_dir = "snapshot/"
    log_dir = "log/"
    datasets = NotMnistDataset(DATA_DIR).datasets
    model = DeepMLPModel(hidden_num)
    train(model, datasets, save_dir, log_dir, num_steps, batch_size, learning_rate)

import argparse
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--hidden_num", help="list of hidden number",
            default="1024,1024")
    parser.add_argument("--max_step", help="max steps for iteration",
            type=int, default=200000)
    parser.add_argument("--batch_size", help="batch size",
            type=int, default=128)
    parser.add_argument("--learning_rate", help="learning rate",
            type=float, default=0.01)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(map(int, args.hidden_num.split(",")),
            args.max_step, args.batch_size, args.learning_rate)
