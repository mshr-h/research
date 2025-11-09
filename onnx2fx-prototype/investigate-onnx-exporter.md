# dynamo ベース torch.onnx.export における control-flow / Sequence 変換調査

## 背景
- dynamo=True で使われる torch.export ルートはテンソル計算のみを含むグラフを生成し、Python の制御フローとデータ構造を原則排除した上で ONNX へ翻訳する設計になっている（`docs/source/onnx_export.md:15-20`）。  
- FAQ でも「ループを含むモデルは torch.cond を参照せよ」と案内されており、従来の Python ループ／if がそのまま ONNX Control-Flow になるわけではない（`docs/source/onnx.md:62-64`）。

## 制御フロー (Control-Flow) の変換状況
- `torch.ops.higher_order.cond` が ONNX `If` へ直接ローワーされる実装が `_torchlib/ops/hop.py` に入っている。true/false subgraph を ir.Graph として構築し、`call_op("If", …)` でノード生成している（`torch/onnx/_internal/exporter/_torchlib/ops/hop.py:63-95`）。  
  - 実際の e2e テストでも `torch.cond` を含むモジュールを export すると ONNX グラフ内に `If` が現れることを検証している（`test/onnx/exporter/test_small_models_e2e.py:83-160`）。  
  - モジュール内にネストした `torch.cond` や複数出力の `torch.cond` も同じテストでカバーされており、dynamo 経由で ONNX Control-Flow（If）が生成できることが確認済み。
- `torch.ops.higher_order.scan` は ONNX `Scan` にローワーされる。body subgraph の入出力を明示的に作り、Scan の属性 `num_scan_inputs` や `scan_input_directions` を設定している（`torch/onnx/_internal/exporter/_torchlib/ops/hop.py:97-157`）。  
  - `test_small_models_e2e.py` では距離計算などを `torch.ops.higher_order.scan` に書き換えたモデルが export できるか、動的 shape を含めて検証している（`test/onnx/exporter/test_small_models_e2e.py:623-712`）。  
  - 生の Python ループを scan に書き換えて export するテクニックもテスト内で紹介されており、`torch.compiler.is_exporting()` で分岐して while/for を置換している（`test/onnx/exporter/test_small_models_e2e.py:670-712`）。
- 逆に `torch.ops.higher_order.while_loop` やその他の Loop 系 HOP には onnx_impl が登録されておらず、`_torchlib/ops/hop.py` で扱っているのは cond と scan のみである。このため ONNX `Loop` ノードはまだ生成できず、ループを表現したい場合は `torch.cond` や `torch.ops.higher_order.scan` への書き換えが必要になる。

## Sequence Ops の扱い
- グラフビルダーは Python の `list/tuple` をそのまま保持せず、ONNX で Sequence 型が要求されている入力に対しては `SequenceConstruct` ノードを自動生成する。`_process_python_sequences` で schema を調べ、`allowed_types` が Sequence であれば `opset.SequenceConstruct(*arg)` へ差し替える（`torch/onnx/_internal/exporter/_building.py:367-405`）。  
- この振る舞いはユニットテストでも確認されており、`SequenceAt` へ Python list を渡すと `SequenceConstruct` が追加されること、variadic 引数や 0 次元テンソルとの混在ケースで適切に Constant / Concat を挿入することが検証されている（`test/onnx/exporter/test_building.py:82-133`）。
- ONNX にしか存在しない Sequence 演算を明示的に使いたい場合は、symbolic API を使って FX グラフへ直接挿入できる。`torch.onnx.ops.symbolic` / `symbolic_multi_out` は任意のドメイン＋op 名を受け取り ONNX ノードを生成するユーティリティであり、`torch.onnx.is_in_onnx_export()` ガード内で呼び出すことが推奨されている（`docs/source/onnx_ops.md:7-35`、実体は `torch/onnx/ops/__init__.py`）。これにより SequenceConstruct/Insert/Erase など ONNX 独自演算を PyTorch 側から要求できる。

## 実務メモ
1. Python の if/for ではなく `torch.cond` や `torch.ops.higher_order.scan` で制御フローを表現すれば、ONNX `If` / `Scan` ノードとして落ちる。`torch.ops.higher_order.while_loop` を含むモデルは現状 export できないので、scan か cond へのリライトが必要。  
2. 既存の Python list/tuple を ONNX Sequence にしたい場合は、ONNX オペレータのシグネチャが Sequence を要求していることを確認する。そうであれば exporter が自動で `SequenceConstruct` を差し込む。  
3. Sequence やその他 ONNX 固有演算を追加したいときは `torch.onnx.ops.symbolic` を使い、`torch.onnx.is_in_onnx_export()` で囲って推論時には実行されないようにする。  
4. ループを残したいケースでは、FX グラフ上で `torch.compiler.is_exporting()` を使って export 時だけ scan 版の実装に差し替えるやり方（`test_small_models_e2e.py:670-712`）が実例として機能している。

