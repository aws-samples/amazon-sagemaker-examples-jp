
---
title: "Amazon Textract"
weight: 1
date: 2021-01-05
description: >
  スキャンしたドキュメントなどの画像からテキストを抽出するための AI サービスです。
---

Amazon Textract は、スキャンしたドキュメントの画像に含まれているテキスト、手書きの文字、その他のデータを抽出するサービスです。単純な光学文字認識 (OCR) のではなく、テキストがフォームやテーブルの一部かどうかを認識、理解したうえで情報を関連づけて抽出することができます。現在、日本語テキストには対応していません。

[Amazon Rekognition のテキスト検出機能]({{< ref "/docs/ai-services/images/rekognition/_index.md#テキスト検出" >}}) と似ていますが、Amazon Textract はフォームやテーブルなど、書類のフォーマットに応じたテキストの抽出が可能なことが特徴です。

AWS コンソールからサンプル画像やアップロードした画像を使って簡単に機能を試すことができます。

![Textract](textract.png)

[AWS コンソールへのリンク（バージニア北部リージョン）](https://console.aws.amazon.com/textract/home?region=us-east-1#/demo)

### 参考情報

[開発者ガイド](https://docs.aws.amazon.com/textract/latest/dg/what-is.html)<br>
Amazon Textract の機能や使い方が説明されているドキュメントです。