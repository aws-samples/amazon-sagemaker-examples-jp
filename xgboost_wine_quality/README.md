# XGBoost によるワインの品質推定

## はじめに
本ハンズオンでは機械学習モデルの解釈性をテーマに、どの特徴量がどの程度の影響があるかについて解析する手法について Amazon SageMaker の組み込みアルゴリズム XGBoost を用いて実践します。

## 本ハンズオンで学べる内容
- SageMaker の組み込みアルゴリズムである XGBoost を用いた機械学習モデルの学習
- SageMaker を用いた際の XGBoost での特徴量重要度の可視化
- Partianl Dependency Plot(PDP) による特徴量の変化による目的変数への影響の可視化

## ノートブックについて
`xgboost_wine_quality_batch.ipynb`  ではPDPを描く際に、特定のカラムを任意の値に変更し、s3へアップロードした上でバッチ変換ジョブで推論を実施します。一方、`xgboost_wine_quality_endpoint.ipynb` では、推論エンドポイントへ変更後のデータを渡して推論を実施します。ハンズオンなどで使用できるインスタンスに制限がある場合には、適切なノートブックを選択下さい。

## 参考
- [XGBoost tutorial (var imp + partial dependence)](https://www.kaggle.com/chalkalan/xgboost-tutorial-var-imp-partial-dependence)
