# Amazon SageMaker - Bring Your Own Model ハンズオンワークショップ

## 概要

このハンズオンでは、深層学習フレームワークのサンプルコードを題材として、実際に Amazon SageMaker 上へ移行するための手順を説明します。

## 対象
- Amazon SageMaker を使うことを検討しているが、具体的なコードの移行方法が分からない方

### 前提知識
- Python のコードがある程度書け、簡単なデバッグができること
- 機械学習、深層学習のコードがある程度読めること
- Amazon SageMaker や関連する AWS のサービスについて、概要を理解していること

## コンテンツ
- MXNet + Gluon (MNIST MLP) で [やってみる](./mxnet-gluon/mxnet-gluon.ipynb) [[Original code](https://github.com/apache/incubator-mxnet/tree/master/example/gluon/mnist)]
- Keras (MNIST MLP) で[やってみる](./tensorflow-keras/tensorflow-keras.ipynb) [[Original code](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)]
- Chainer (MNIST MLP) [[Original code](https://github.com/chainer/chainer/tree/master/examples/mnist)]
- TensorFlow (MNIST CNN)で[やってみる](./tensorflow/tensorflow.ipynb) [[Original code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py)]
- PyTorch (MNIST CNN) [[Original code](https://github.com/pytorch/examples/blob/master/mnist/main.py)]


### 手順
1. トレーニングスクリプトの書き換え
1. Notebook 上でのデータ準備
1. Local Mode によるトレーニングとコードの検証
1. トレーニングジョブの発行
1. 推論エンドポイントのデプロイ
