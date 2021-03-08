
---
title: "Amazon Rekognition"
weight: 1
date: 2021-01-05
description: >
  画像や動画の分析を行うための AI サービスです。
---

## 一般的なシーン認識・物体検知を行う
画像のどこに人、車、猫などの一般的な被写体が写っているかを分析したり、写真に何が写っているのかを分析したりできます。AWS コンソールから画像をアップロードして簡単に機能を試すことができます。

![Rekognition](rek-obj.png)

[AWS コンソールへのリンク（東京リージョン）](https://ap-northeast-1.console.aws.amazon.com/rekognition/home?region=ap-northeast-1#/label-detection)

### 参考情報

[Amazon Rekognition の開始方法](https://aws.amazon.com/jp/rekognition/getting-started/)<br>
Amazon Rekognition の利用を始めるためのドキュメントや、いくつかのユースケースを実現するための動画が公開されています。

## 独自のシーン認識・物体検知モデルを作る
Amazon Rekognition Custom Labels という機能を使って、お客様がお持ちの画像を使って、独自のシーン認識（画像分類）や物体検知モデルをノーコードで作成いただけます。

Amazon SageMaker Ground Truth で画像のラベリングを行い、その結果を使って Amazon Rekognition Custom Labels でモデルを作成することができます。

### 参考情報
[【Nyantech ハンズオンシリーズ】機械学習を使って写真に写っている猫を見分けてみよう！](https://aws.amazon.com/jp/builders-flash/202003/sagemaker-groundtruth-cat/)<br>
Amazon SageMaker Ground Truth で画像のラベリングをし、Amazon Rekognition Custom Labels を使って猫のタマとミケを見分ける機械学習モデルを作る方法がハンズオン形式で説明されています。

[Amazon Rekognition Custom Labelsを利用した動物の特徴的な行動検出](https://aws.amazon.com/jp/blogs/news/detecting-playful-animal-behavior-in-videos-using-amazon-rekognition-custom-labels/)
猫の動画を題材に、猫パンチをしているかしていないかを見分けるモデルを作っています。

[![](http://img.youtube.com/vi/h02X4ZH1wQI/0.jpg)](http://www.youtube.com/watch?v=h02X4ZH1wQI "")


## 