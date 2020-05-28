#!/bin/bash
test_img() {
  python Test.py --model_name "$1" "$2" "$3"
}
train_mlp_mnist() {
  python Train.py --epoch 20 --dataset mnist --network mlp --loss_func "$2" --model_name "$3" 784 "$1"
}
train_cnn_mnist() {
  python Train.py --epoch 20 --dataset mnist --network conv --loss_func "$2" --model_name "$3" 1 "$1"
}
train_mlp_cifar() {
  python Train.py --epoch 20 --dataset cifar10 --network mlp --loss_func "$2" --model_name "$3" 3072 "$1"
}
train_cnn_cifar() {
  python Train.py --epoch 20 --dataset cifar10 --network conv --loss_func "$2" --model_name "$3" 3 "$1"
}