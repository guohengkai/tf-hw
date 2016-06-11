from __future__ import print_function
from __future__ import division
import os
import sys
import tensorflow as tf

sys.path.append("..")
from common.common import DATA_DIR
from hw1.not_mnist_dataset import NotMnistDataset
from hw2.main import train
from hw4.deep_conv_model import DeepConvModel
from hw4.stack_lstm_model import StackLSTMModel

def main(model_name, conv_num, lstm_num, hidden_num, num_steps, batch_size, learning_rate):
    save_dir = "snapshot/"
    log_dir = "log/"
    datasets = NotMnistDataset(DATA_DIR).datasets
    if model_name == "conv":
        model = DeepConvModel(conv_num, hidden_num)
    elif model_name == "lstm":
        model = StackLSTMModel(lstm_num, hidden_num)
    else:
        raise ValueError("invalid model name: " + model_name)
    train(model, datasets, save_dir, log_dir, num_steps, batch_size, learning_rate)

import argparse
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--model_name", help="model name", default="conv")
    parser.add_argument("--conv_num", help="list of convolution stage number",
            default="3,4,16,16,32,32,3,3,64,96,128")
    parser.add_argument("--lstm_num", help="LSTM hidden number",
            default="128,256")
    parser.add_argument("--hidden_num", help="list of hidden number",
            default="1024,1024")
    parser.add_argument("--max_step", help="max steps for iteration",
            type=int, default=200000)
    parser.add_argument("--batch_size", help="batch size",
            type=int, default=128)
    parser.add_argument("--learning_rate", help="learning rate",
            type=float, default=0.01)
    return parser.parse_args()

def parse_conv_num(conv_num):
    conv_num = map(int, conv_num.split(","))
    res = []
    idx = 0
    while idx < len(conv_num):
        stage = []
        stage.append(conv_num[idx])
        idx += 1
        count = conv_num[idx]
        idx += 1
        stage.append(count)
        while count:
            stage.append(conv_num[idx])
            count -= 1
            idx += 1
        res.append(stage)
    print("Stage result:", res)
    return res

if __name__ == '__main__':
    args = parse_args()
    main(args.model_name, parse_conv_num(args.conv_num),
            map(int, args.lstm_num.split(",")),
            map(int, args.hidden_num.split(",")),
            args.max_step, args.batch_size, args.learning_rate)
