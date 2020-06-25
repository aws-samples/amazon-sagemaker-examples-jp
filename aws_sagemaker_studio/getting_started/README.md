# Getting Started with Amazon SageMaker Studio

このフォルダでは [Amazon SageMaker Studio のメインの機能を試す Jupyter notebook](xgboost_customer_churn_studio.ipynb) を提供しています。これは Studio で動かすことを想定して作られています。XGBoostを利用して Customer Churn (顧客離反分析) のモデルを作るサンプルです。

## 利用している SageMaker の機能

* [Amazon SageMaker Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html)
  * 複数の機械学習の試行を管理する
  * ハイパーパラメータに関する実験とグラフ化
* [Amazon SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html)
  * モデルのデバッグ
* [Model hosting](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html)
  * モデルから予測を得るための永続的なエンドポイントをセットアップ
* [SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
  * モデルの品質を監視する
  * モデルの品質に変化があった場合にアラートをあげる

## 準備

[Amazon SageMaker Studioの利用を開始する](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) をすでに完了して、ログインできるようにしておく必要があります。

## ノートブックの利用方法

1. [Amazon SageMaker Studio](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/studio/) へログイン

2. Studio のターミナルを開く

![ターミナルの開き方](./images/open_a_terminal.gif)

3. 以下のコマンドからこのレポジトリをクローンする

```bash
git clone https://github.com/awslabs/amazon-sagemaker-examples.git
```

![レポジトリのクローン](./images/clone_the_repo.gif)

4. Studioのファイルマネージャからノートブックを探して開く

![ノートブックを探す](./images/find_and_open_the_notebook.gif)
