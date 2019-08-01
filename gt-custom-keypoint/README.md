# SageMaker Ground Truth カスタムラベリング キーポイント
## 概要
SageMaker Ground Truthでは前・後処理のLambda関数とラベリングツールのHTMLテンプレートを変更するとで、様々なタスクに対応することができます。
今回は組み込みラベリングツールにはない姿勢推定タスクに対するラベリングジョブを、カスタムテンプレートを活用して構築します。

本ハンズオンは2019年8月20日に実施されました[Amazon SageMaker Ground Truth 体験ハンズオン](https://pages.awscloud.com/HandsOn-Ground-Truth-20190820_LandingPage.html)で公開されています。詳細な手順は[【WIP】スライド](xx)や[【WIP】動画](xxx)をご確認下さい。


## カスタムデータラベリングジョブの主なコンポーネント
### 1. HTMLテンプレート
- カスタムラベリングジョブでは定義済のテンプレートをカスタマイズして使う事が出来ます。
- 今回は"Keypoint"というHTMLテンプレートをカスタマイズして[template.html](https://github.com/tkazusa/gt-custom-pose/blob/master/web/template.html)を作成しました。

### 2. ラベリング対象のデータ
- ラベリングするデータです。S3に保存することでGround Truthでアノテーションすることができます。
- 今回は[こちら](https://20190820-handson-images.s3-ap-northeast-1.amazonaws.com/images.zip)を使用します。

### 3. 入力のマニフェスト
- Ground TruthでどのS3上にあるどのラベリング対象データを使うか、メタデータは何かなどのを記載したファイル。
- S3上に保存されたものを活用します。本ハンズオンではS3上に保存されたデータから作成します。
- プレラベリング Lambda関数が活用します。

### 4. プレラベリング Lambda関数
- 入力マニフェストエントリを処理して、Ground Truthのテンプレートエンジンに情報を渡すために呼び出す[プレラベリング Lambda関数](https://github.com/tkazusa/gt-custom-pose/blob/master/server/processing/sagemaker-gt-preprocess.py)を準備します。

### 5. ポストラベリング Lambda関数
- ワーカーがタスクを完了したら、Ground Truth は結果を [ポストラベリング Lambda関数](https://github.com/tkazusa/gt-custom-pose/blob/master/server/processing/sagemaker-gt-postprocess.py) に送信します。
- この Lambda は一般に、[注釈統合](https://docs.aws.amazon.com/ja_jp/sagemaker/latest/dg/sms-annotation-consolidation.html)に使用されます。

## 活用するデータ
データは[MPII Human Pose Dataset, Version 1.0]()から抜粋しています。
著作権は2015 Max Planck Institute for Informaticsに帰属し、Simplified BSD Licenseの条件のもとで活用されます。
詳細は[bsd.txt](https://github.com/tkazusa/amazon-sagemaker-examples-jp/blob/master/gt-custom-keypoint/server/data/bsd.txt)をご覧下さい。

## 参考資料
- [Amazon SageMaker Ground Truth を使ったカスタムデータラベリングワークフローの構築](https://aws.amazon.com/jp/blogs/news/build-a-custom-data-labeling-workflow-with-amazon-sagemaker-ground-truth/)
- [Build your own custom labeling workflow using SageMaker Ground Truth(Github repository)](https://github.com/nitinaws/gt-custom-workflow.git)
- [AWS Lambda を使用した処理](https://docs.aws.amazon.com/ja_jp/sagemaker/latest/dg/sms-custom-templates-step3.html)
