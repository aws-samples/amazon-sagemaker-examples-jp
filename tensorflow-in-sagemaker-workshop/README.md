# TensorFlow-Keras モデルを Amazon SageMaker で実行するハンズオンワークショプ

TensorFlow™ によって開発者は素早く簡単に深層学習をクラウド環境で開始することができます。TensorFlow は様々な業界で使われており、特にコンピュータービジョンや自然言語処理や、機械翻訳の領域で人気のある深層学習フレームワークの一つとなっています。Amazon SageMaker を用いることで、構築、学習、デプロイといった機械学習開発の一連の流れをスケール可能なフルマネージドな環境で、TensorFlow を使うことができます。


## Amazon SageMaker で機械学習フレームワークを使う
Amazon SageMaker Python SDK はいくつかの異なる機械学習や深層学習フレームワークを簡単に構築することができる、オープンソースのAPIとコンテナを提供しています。詳細は[Amazon SageMaker Python SDK API references](https://sagemaker.readthedocs.io/)を参照下さい。


## 実施内容
このワークショップでは TensorFlow のサンプルコードを Amazon SageMaker 上で実行する手順を紹介します。SageMaker Python SDK で TensorFlow を使うための説明は SDK のドキュメント にも多くの情報がありますので合わせてご活用下さい。

1. [TensorFlow 学習スクリプトを Amazon SageMaker 向けに書き換える](0_Running_TensorFlow_In_SageMaker.ipynb)
2. 【追加予定】学習ジョブをTensorBoardとAmazon CloudWatch メトリクスを使って監視する
3. 【追加予定】SageMakerのパイプ入力モードを用いた学習ジョブの最適化
4. 【追加予定】Horovodを用いた分散学習の実行
5. 【追加予定】Amazon SageMakerを用いた学習済モデルのデプロイ

## ライセンスについて
このサンプルコードはMIT-0ライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルをご確認下さい。
