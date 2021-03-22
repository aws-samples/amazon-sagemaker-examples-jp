
---
title: "Amazon Rekognition"
weight: 1
date: 2021-01-05
description: >
  画像や動画の分析を行うための AI サービスです。
---

## 一般的なシーン認識・物体検知を行う
画像のどこに人、車、猫などの一般的な被写体が写っているかを分析したり、写真に何が写っているのかを分析したりできます。AWS コンソールからサンプル画像やアップロードした画像を使って簡単に機能を試すことができます。

![Rekognition](rek-obj.png)

[AWS コンソールへのリンク（東京リージョン）](https://ap-northeast-1.console.aws.amazon.com/rekognition/home?region=ap-northeast-1#/label-detection)

### 参考情報

[Amazon Rekognition の開始方法](https://aws.amazon.com/jp/rekognition/getting-started/)<br>
Amazon Rekognition の利用を始めるためのドキュメントや、いくつかのユースケースを実現するための動画が公開されています。

---
## 独自のシーン認識・物体検知モデルを作る
Amazon Rekognition Custom Labels（カスタムラベル）という機能を使って、お客様がお持ちの画像を使って、独自のシーン認識（画像分類）や物体検知モデルをノーコードで作成いただけます。

Amazon SageMaker Ground Truth で画像のラベリングを行い、その結果を使って Amazon Rekognition Custom Labels でモデルを作成することができます。

### 参考情報
[【Nyantech ハンズオンシリーズ】機械学習を使って写真に写っている猫を見分けてみよう！](https://aws.amazon.com/jp/builders-flash/202003/sagemaker-groundtruth-cat/)<br>
Amazon SageMaker Ground Truth で画像のラベリングをし、Amazon Rekognition Custom Labels を使って猫のタマとミケを見分ける機械学習モデルを作る方法がハンズオン形式で説明されています。

[Amazon Rekognition Custom Labelsを利用した動物の特徴的な行動検出](https://aws.amazon.com/jp/blogs/news/detecting-playful-animal-behavior-in-videos-using-amazon-rekognition-custom-labels/)<br>
猫の動画を題材に、猫パンチをしているかしていないかを見分けるモデルを作っています。

[![](http://img.youtube.com/vi/h02X4ZH1wQI/0.jpg)](http://www.youtube.com/watch?v=h02X4ZH1wQI "")

---
## 顔検出と分析
画像に写っている顔を検出し、その特徴を分析する機能です。年齢、目の大きさ、眼鏡濃霧、髭の有無などの属性を取得できます。動画では、これらの顔の属性が時間と共にどう変化するかも取得でき、例えば役者が示す感情のタイムラインを作成することができます。

![顔検出と分析](https://d1.awsstatic.com/product-marketing/Rekognition/Image%20for%20facial%20analysis.3fcc22e8451b4a238540128cb5510b8cbe22da51.jpg)

### 顔の比較
顔の比較の機能を使うことで、2枚の写真に写っている人物がどれくらい似ているかを知ることができます。AWS コンソールからサンプル画像を使ったり、画像をアップロードしたりしてこの機能を簡単に試すことができます。

![顔検の比較](compare-face.png)

[AWS コンソールへのリンク（東京リージョン）](https://ap-northeast-1.console.aws.amazon.com/rekognition/home?region=ap-northeast-1#/face-comparison)

### 参考情報
[顔検出機能で取得できる情報（開発者ガイド）](https://docs.aws.amazon.com/ja_jp/rekognition/latest/dg/faces-detect-images.html#detectfaces-response)<br>
顔検出機能の API (DetectFaces) を使って取得できる情報について記載されています。

[画像間の顔の比較（開発者ガイド）](https://docs.aws.amazon.com/ja_jp/rekognition/latest/dg/faces-comparefaces.html)<br>
顔の比較の機能の使い方が記載されています。

---
## 顔検索と検証
顔画像の特徴を記録するレポジトリを作成し、写真やビデオ中の人物をリポジトリに登録された人物と照合することができます。

![顔検索と検証](https://d1.awsstatic.com/product-marketing/Rekognition/Image%20for%20facial%20recognition.d14cf0759b26beed0e9731c93a4680954baf7310.jpg)

### 参考情報
[東海理研株式会社様 AWS/Rekognitionを活用した宿泊施設向け入退室管理システム構築事例](https://aws.amazon.com/jp/blogs/news/ml-images-usecase-seminar/)<br>
2020年9月2日に開催された「AWSの機械学習を使った画像データの業務活用セミナー」で登壇いただいたお客様事例の動画と資料を公開しています。

[Auto Check-In App（AWS ソリューションライブラリ）](https://aws.amazon.com/jp/solutions/implementations/auto-check-in-app/)<br>
顔の特徴を登録するリポジトリ（顔コレクション）を作成し、イベントのチェックインのための顔認証に必要な製品やサービスを自動的に構築するソリューションを公開しています。イベントチェックイン時に、イベント参加者の写真を撮影すると、このソリューションは Amazon Rekognition に顔画像を送信し、そこで事前登録された参加者の顔コレクションに照らし合わせてそれらの画像を検証します。

---
## テキスト検出
画像内のテキスト（英数字）を検出できる機能です。

![テキスト検出](https://d1.awsstatic.com/product-marketing/Rekognition/text-detection.e676acb975a2bb1bfb2fee0596aeee9fed23c6f3.png)

AWS コンソールからサンプル画像やアップロードした画像を使って簡単に機能を試すことができます。

![テキスト検出](text.png)

[AWS コンソールへのリンク（東京リージョン）](https://ap-northeast-1.console.aws.amazon.com/rekognition/home?region=ap-northeast-1#/text-detection)

### 参考情報
[スマートガレージ（AWS ブログ）](https://aws.amazon.com/jp/blogs/news/building-a-smart-garage-door-opener-with-aws-deeplens-and-amazon-rekognition/)<br>
Amazon Rekognition のテキスト検出機能を使って、特定のナンバープレートの車のみが使用できるガレージを実現する方法を紹介している記事です。

---
## コンテンツのモデレーション
写真に写っているものが暴力的、性的なものでないかなどを判断するための情報を取得できます。お客様は取得した情報をもとに、その写真が不適切なものかどうかを判断することができます。

自社サイトに意図しない画像がアップロードされることを防ぐために、この機能を使ってアップロード前に画像を検査するなどが考えられます。

![コンテンツのモデレーション](https://d1.awsstatic.com/product-marketing/Rekognition/Unsafe-Content.17ce677c09ac11e463053ad7ea3dcc80bc6372f1.jpg)

AWS コンソールからサンプル画像やアップロードした画像を使って簡単に機能を試すことができます。

![コンテンツのモデレーション](moderation.png)

[AWS コンソールへのリンク（東京リージョン）](https://ap-northeast-1.console.aws.amazon.com/rekognition/home?region=ap-northeast-1#/image-moderation)

### 参考情報
[この機能で取得できる情報（開発者ガイド）](https://docs.aws.amazon.com/ja_jp/rekognition/latest/dg/moderation.html)<br>
この機能を使うことによって、どのような情報が得られるかが記載されています。


---
## 有名人の認識

入力された動画や写真をライブラリに照らし合わせ、どの有名人が写っているかを認識します。

![有名人の認識](https://d1.awsstatic.com/product-marketing/Rekognition/Image%20for%20celebrity%20recognition_v3.2264009c637a0ee8cf02b75fd82bb30aa34073eb.jpg)

AWS コンソールからサンプル画像やアップロードした画像を使って簡単に機能を試すことができます。

![有名人の認識](celeb.png)

[AWS コンソールへのリンク（東京リージョン）](https://ap-northeast-1.console.aws.amazon.com/rekognition/home?region=ap-northeast-1#/celebrity-detection)

### 参考情報

[この機能で取得できる情報（開発者ガイド）](https://docs.aws.amazon.com/ja_jp/rekognition/latest/dg/celebrities-procedure-image.html)<br>
この機能を使うことによって、どのような情報が得られるかが記載されています。

---
## Personal Protective Equipment (PPE) の検出

写真に映っている人物がフェイスカバー (フェイスマスク)、ハンドカバー (手袋)、ヘッドカバー (ヘルメット) などの PPE を着用しているかどうか、またそれらの保護器具が、該当する身体の部分 (フェイスカバーでは鼻、ヘッドカバーでは頭、ハンドカバーでは手) を覆っているかどうかを自動的に検出します。

![PPE の検出](https://d1.awsstatic.com/logos/Screen%20Shot%202020-10-14%20at%206.17.20%20PM.b2bcb1899c20e0cc9ea3c48aee1ad37aca91c12c.png)

AWS コンソールからサンプル画像やアップロードした画像を使って簡単に機能を試すことができます。

![有名人の認識](ppe.png)

[AWS コンソールへのリンク（東京リージョン）](https://ap-northeast-1.console.aws.amazon.com/rekognition/home?region=ap-northeast-1#/ppe)

### 参考情報

[この機能で取得できる情報（開発者ガイド）](https://docs.aws.amazon.com/ja_jp/rekognition/latest/dg/ppe-request-response.html)<br>
この機能を使うことによって、どのような情報が得られるかが記載されています。
