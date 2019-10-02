# TensorFlow-Keras モデルを Amazon SageMaker で実行するハンズオンワークショプ
TensorFlow™ によって開発者は素早く簡単に深層学習をクラウド環境で開始することができます。TensorFlow は様々な業界で使われており、特にコンピュータービジョンや自然言語処理や、機械翻訳の領域で人気のある深層学習フレームワークの一つとなっています。Amazon SageMaker を用いることで、構築、学習、デプロイといった機械学習開発の一連の流れをスケール可能なフルマネージドな環境で、TensorFlow を使うことができます。


## Amazon SageMaker で機械学習フレームワークを使う
Amazon SageMaker Python SDK はいくつかの異なる機械学習や深層学習フレームワークを簡単に構築することができる、オープンソースのAPIとコンテナを提供しています。詳細は [Amazon SageMaker Python SDK API references](https://sagemaker.readthedocs.io/) を参照下さい。


## 実施内容
このワークショップでは TensorFlow のサンプルコードを Amazon SageMaker 上で実行する手順を紹介します。SageMaker Python SDK で TensorFlow を使うための説明は SDK のドキュメント にも多くの情報がありますので合わせてご活用下さい。

1. [TensorFlow 学習スクリプトを Amazon SageMaker 向けに書き換える](0_Running_TensorFlow_In_SageMaker.ipynb)
2. [学習ジョブをTensorBoardとAmazon CloudWatch メトリクスを使って監視する](1_Monitoring_your_TensorFlow_scripts.ipynb)
3. 【追加予定】SageMakerのパイプ入力モードを用いた学習ジョブの最適化
4. [Horovodを用いた分散学習の実行](3_Distributed_training_with_Horovod.ipynb)
5. [Amazon SageMakerを用いた学習済モデルのデプロイ](4_Deploying_your_TensorFlow_model.ipynb)


## サンプルコードについて
本ハンズオンは基本的には 「[TensorFlow 学習スクリプトを Amazon SageMaker 向けに書き換える](0_Running_TensorFlow_In_SageMaker.ipynb)」で SageMaker 向けにスクリプトを書き換えることを想定されて作られています。もし「[学習ジョブをTensorBoardとAmazon CloudWatch メトリクスを使って監視する](1_Monitoring_your_TensorFlow_scripts.ipynb)
」や「[Horovodを用いた分散学習の実行](3_Distributed_training_with_Horovod.ipynb)」 のみを実施したい場合には、下記の手順で「[TensorFlow 学習スクリプトを Amazon SageMaker 向けに書き換える](0_Running_TensorFlow_In_SageMaker.ipynb) をスキップすることが出来ます。

1. すでに書き換え済のスクリプトが `training_script/sample-codes` にあるので、TensorFlow Estimator の `entory_point` に指定
2. ノートブックインスタンス上で 
```!aws s3 cp --recursive s3://floor28/data/cifar10 ./data```
を実行しデータをダウンロード
3. 各ノートブックでコメントアウトされている下記をを実行してS3へデータをアップロード

```python
dataset_location = sagemaker_session.upload_data(path='data', key_prefix='data/DEMO-cifar10')
display(dataset_location)
```


## ライセンスについて
このサンプルコードはMIT-0ライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルをご確認下さい。


## 参考
[Running your TensorFlow Models in SageMaker Workshop](https://github.com/aws-samples/TensorFlow-in-SageMaker-workshop)
