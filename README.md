# Monster Arena HD-2D (`monster_arena_2d`)

ドット絵スプライトと WebGL1 による **HD-2D 3D ジオラマ／ポストエフェクト** を融合した、ブラウザ動作のリアルタイム／ターン制モンスター対戦アクションゲームおよびスプライト・技デバッグビューアです。

- **ゲーム本編**: `game/index.html`（ルート `index.html` から自動リダイレクト）
- **デバッグ画面**: `game/debug.html`（ルート `debug.html` から自動リダイレクト）
- **GitHub Pages 公開 URL（デプロイ設定後）**:
  - ゲーム: `https://hiro-s-2000.github.io/monster_arena_2d/`
  - デバッグ: `https://hiro-s-2000.github.io/monster_arena_2d/debug.html`

---

## 起動方法

### ローカルで起動する場合（macOS）

ドット絵・アニメーション GIF の画素読み取り（`fetch` / `CanvasRenderingContext2D.getImageData`）および WebGL 3D 描画のため、HTTP(S) 経由で開きます。

- **ゲーム起動**: リポジトリ直下の `ゲーム起動.command` をダブルクリック（Python3 のローカル HTTP サーバーを立てて `game/index.html` を開きます。同一 Wi-Fi 上のスマホ接続用 URL も表示されます）
- **デバッグ起動**: `デバッグ起動.command` をダブルクリック（`game/debug.html` を開きます）

> ※ `game/debug.html` のみ `file://` 直開き時の簡易フォールバック（2D 平面表示＋`data/sprite_bounds.js` 参照）にも対応していますが、3D ジオラマと GIF アニメ・ドット処理を使う場合はローカル HTTP サーバー経由で開いてください。

### 任意の OS / 手動起動

```bash
python3 -m http.server 8787
# ゲーム:   http://127.0.0.1:8787/game/index.html
# デバッグ: http://127.0.0.1:8787/game/debug.html
```

---

## ゲームモード（4 種類）

タイトル画面（夕焼けのドット絵背景上でモンスターが歩くエキシビション＋デモ対戦）の `PRESS START` 後、以下の 4 モードを選択できます（`game/js/menu.js`）。

| モード | 内容 |
| --- | --- |
| **バトル** | 最大 **6 対 6** のリアルタイム対戦。自分で操作する「プレイ」と CPU 同士の「かんせん」を切り替え可能。観戦中も画面上のトレーナーやコマンド枠をタップして赤／青コーナーに指示を出せます。 |
| **トーナメント** | **8 体** によるシングルエリミネーション勝ち抜き戦（`ROUND ONE` → `ROUND TWO` → `FINAL ROUND`）。うち 1 体だけを自分で操作することも可能です（`game/js/tournament.js`）。 |
| **ミニバトル** | **3 対 3** のコマンド選択式ターン制バトル（`たたかう` / `モンスター` / `にげる`）。技のエフェクトは実際のバトルフィールド上で再生されます（`game/js/turn.js`）。 |
| **トレーニング** | 最大 6 対 1 で、相手が攻撃してこない練習モード。 |

---

## 主な実装機能

### 1. HD-2D 3D ジオラマ＆ポストエフェクト（`game/js/scene3d.js`, `post.js`, `scenes/*.js`）
- **本物の 3D ステージ背景**: 全 11 ステージそれぞれに専用の 3D 定義ファイル（`game/js/scenes/<id>.js`）を用意。観客席・照明塔・建物・樹木・岩・遠景などを 3D メッシュと正対ビルボードで構築し、地面（`z = 0`）は遠近投影、モンスター・トレーナー・小物はカメラ正対ビルボードで描画します。
- **HD-2D 質感＆ライティング**: 地面ドット絵の 3×3 バイラテラル補間と疑似法線陰影、素材ごとのマイクロサーフェス、流れ動く木漏れ日（雲影）、動的ライト、ブルーム、チルトシフト被写界深度（DoF）、フォグ、ゴッドレイ、空気中のちり、ビネット、強打時の放射ブラー＆ネガ反転フラッシュを WebGL1 シェーダーで合成しています。
- **白飛び抑制**: 強い光源下でもモンスターの配色が潰れないよう、モンスター層のみ過剰光圧縮（`SPR_OVER`）とブルーム／光筋の減衰（`uSprMask`）を適用しています。
- **2D 平面フォールバック**: URL に `?flat=1` を付けると従来の 2D 平面描画に切り替わります。

### 2. 3D 空中エフェクト層（`game/js/fx.js` `FXV`, `game/js/render.js`）
- エフェクトを **地面層**（予告範囲・地面の衝撃輪・影）と **空中層**（ビーム・弾・爆発・柱・落雷・飛散粒子）に分離。
- ワールド座標のまま `FXV.at(ctx, x, y)`（正対ビルボード）および `FXV.seg(ctx, x0, y0, h0, x1, y1, h1)`（2 点を結ぶビーム・糸）で遠近座標へ投影し、Bayer 網点ディザ＆減色を通して描画します。
- 炎系技の色は使い手の正面スプライト色相（`data/flame_hues.js` 収録 1,185 件、または実行時ピクセル解析）に合わせて動的に変化します。

### 3. 収録データ・スプライトシステム
- **モンスター**: 第 1 〜第 9 世代 **1,217 フォーム**（通常 `1,026` / メガシンカ `99` / リージョン `58` / キョダイマックス `34`）、**19 タイプ**（`data/monsterdex.js`）。
- **技**: **453 技**（`data/monsterdex.js` / `game/js/moves.js`）。近接・突進・弾・追尾・拡散・連続噴出（ストーンエッジ等）・ビーム・サイコブレイク・サイコキネシス（つかみ）・跳躍・地面隆起（ワイドフォース等）・落雷／光柱・設置ゾーン・状態異常・能力変化・回復・まもる・天候変化などを表現。
- **トレーナー**: **247 人**（HGSS `181` 人 / FRLG `66` 人、`data/trainers.js`）。選択画面で名前検索や左右切替が可能で、登場時のボール投擲や戦況に応じたリアクションを行います。
- **8 方向＆アクションスプライト（`data/dir8.js`, `game/js/dir8.js`, `assets.js`）**:
  - 1,217 フォームすべてに対応（`8dir` シート `976` 体、`fb` 正面・背面フルフレームシート `133` 体、`actions_8dir` 8 方向×アクションシート `108` 体）。
  - GIF 本来のドット解像度を潰さずに保持し、図鑑の「高さ・重さ」と姿勢補正からトレーナー身長（1.7m）基準の実寸バランスでフィールド上に配置します。

### 4. バトルギミック＆戦術 AI
- **ビーム＆弾の衝突「⚡ BEAM CLASH!」**: ビーム同士が正面衝突すると拮抗コアが発生し、ボタン連打（`mashT`）で押し返せます。
- **地形破壊・永続デカール（`game/js/slabs.js`, `ground.js`）**: ワイドフォース／サイコキネシスではフィールド床が複数の岩盤スラブに割れて持ち上がり、着弾時のクレーター・焦げ跡・亀裂・氷の融解痕がフィールド上に残ります。
- **天候＆とくせい（`game/js/ability.js`）**: はれ・あめ・すなあらし・あられの天候と、いかく・ふゆう・がんじょう・ひでり/あめふらし/すなおこし/ゆきふらし・かそく・じしんかじょう・マルチスケイル・ふしぎなまもり・テクニシャン・てきおうりょく・ちくでん/ちょすい/もらいび・いたずらごころ 等を実装し、発動時に専用バーを表示します。
- **戦術 CPU（`game/js/ai.js`）**: 技構成と能力値から `zoner` / `rusher` / `allrounder` を自動判定し、危険範囲の予告退避・直角回避・敵の硬直を狙う差し返し（Whiff-punish）・カウンタービームによるビームクラッシュ誘発を行います。
- **HUD＆プロシージャルサウンド（`game/js/hud.js`, `audio.js`, `style.css`）**: デフォルトの **eSports 中継スキン** と **レトロ DQ スキン** を切り替え可能。BGM・観客歓声・タイプ別 SE はすべて Web Audio API でリアルタイム合成しています（外部音声ファイル不要）。

---

## アリーナ一覧（11 種類）

`game/js/arena.js` と `game/js/scenes/<id>.js` で定義されています。

| ID | 日本語名 | 英語名 | 主な特徴・ギミック |
| --- | --- | --- | --- |
| `stadium` | モンスターリーグ・スタジアム | League Stadium | 芝コート・照明塔・電光掲示板・夜間観客席 |
| `dirt` | つちのグラウンド | Clay Pitch | 土のフィールド・岩障害物 |
| `stone` | いしだたみコート | Stone Court | 石畳・木箱障害物 |
| `colosseum` | サンド・コロシアム | Sand Colosseum | 砂地・外周トラック |
| `pokeball` | ボールドーム | Ball Dome | 天窓のある屋内ドーム・ボール模様コート |
| `ice` | こおりのリンク | Frost Rink | 氷床（移動が滑る）・氷塊（炎で融解痕） |
| `volcano` | マグマのふんか口 | Magma Pit | 溶岩池 4 か所（飛行・ほのおタイプ以外はダメージ） |
| `water` | ウォーターフィールド | Water Field | 水上コート＋浮き台（みずタイプは加速、それ以外は水中で減速） |
| `cavern` | クリスタル・ケイブ | Crystal Cavern | 星空の泉と発光クリスタル群 |
| `cosmic` | うちゅうステージ | Cosmic Stage | 宇宙空間・低重力（ふっとびやすい） |
| `meadow` | よるのそうげん | Night Meadow | 夜の草原・ホタル・霧 |

---

## デバッグ画面（`game/debug.html`）

開発・検証用のオールインワンビューアです。画面下部バーのボタンで 3 つのパネルを個別に開閉できます。

- **モンスター選択（`#side`）**: 名前・図鑑 No.・フォルム（通常 / メガ進化 / キョダイマックス / リージョン）・世代・8 方向シート有無で絞り込み、`↑` `↓` / `PageUp` `PageDown` で切替。
- **技選択（#moves）**: 全 453 技を検索し、クリックひとつでターゲット相手に発射して軌道・ヒットストップ・3D エフェクトを確認。
- **スプライト確認（#dirs）**: 上段 `#strip`（8 方向またはアクション各セル）と下段 `#official`（`front.gif`, `front-reverse.gif`, `back.gif`, `back-reverse.gif`）の 2 段横並び・高さ `30vh` 固定表示。コマ送り・再生/停止・倍率（自動 / ×1 / ×2 / ×3）に対応。
- **各種トグル**: 色違い切替、HP むげん、相手移動/攻撃、ステージ・天候・相手・トレーナー変更、メガ/フォルムチェンジ、再生速度（×0.1 〜 ×2）、1/60 秒コマ送り、HUD 表示切替。

---

## 操作方法

| 操作 | キー / 入力 |
| --- | --- |
| **移動** | `W` `A` `S` `D` / 矢印キー / タッチ仮想スティック / ゲームパッド左スティック |
| **技 1 〜 4** | `Z` `X` `C` `V` / `1` 〜 `4` / マウス左・右クリック / スキルカード / ゲームパッド `X` `A` `Y` `B` |
| **回避（テレポート）** | `Space` / `Shift` |
| **交代** | `Q` / `Tab` / 控えボールアイコン |
| **メガシンカ / 姿切替** | `M` / `G` |
| **カメラズーム** | `−` / `＝` / `0`（100% リセット） / マウスホイール / 画面上部ボタン |
| **HUD スキン切替** | `H` / 画面上部 `HUD` ボタン |
| **ポーズ** | `Esc` / `P` |
| **ミニバトル（ターン制）** | 矢印キー / `WASD` で選択、`Z` / `Enter` / `Space` で決定、`X` / `Esc` で戻る |

---

## URL クエリパラメータ

テストや特定シチュエーションの直接起動に利用できます。

### `game/index.html`

```text
game/index.html?auto=charizard,blastoise;mewtwo,gengar&arena=stadium&mode=watch&hud=esports&ff=12&freeze=1&seed=42
```

- `auto=<team0>;<team1>`: カンマ区切りスラッグ（末尾 `*` で色違い）を指定して即バトル開始
- `mode=watch|play|turn`: 観戦・プレイ・ターン制を指定
- `arena=<id>`: ステージ ID（`stadium`, `water`, `ice` 等）
- `weather=sun|rain|sand|hail`: 初期天候
- `skill=<0..1>` / `time=<sec>` / `hud=esports|retro` / `zoom=<mul>`
- `ff=<sec>` / `freeze=1` / `seed=<int>` / `cam=<x,y>`: 指定秒数早送りして静止・乱数シード固定・カメラ座標指定
- `flat=1` / `pitch=<deg>` / `fov=<deg>` / `dbg=1`: 2D 平面モード・カメラ角度変更・FPS 表示

### `game/debug.html`

```text
game/debug.html?mon=charizard&foe=snorlax&arena=stadium&mode=walk&hud=1
```

- `mon=<slug>` / `foe=<slug>` / `arena=<id>` / `weather` / `mode=auto|face|spin|walk` / `hud=0|1` / `flat=1` / `pitch` / `fov`

---

## GitHub Pages 対応の仕組み

`monsterSprite/` には生成用中間ファイル（`_work/`）を含め約 4.5 GB（ゲーム参照分だけでも約 3.48 GB）のデータが存在するため、GitHub Pages の **1 GB 公開サイズ上限** に収まるよう専用のビルド構成を採用しています。

- **ビルドスクリプト（`tools/build_pages.py`）**:
  - ルートエントリー（`index.html`, `debug.html`, `.nojekyll`）、`game/`、`data/`、`trainers/`、アイコン・静止画 PNG・非 `fb` GIF・色違い GIF を `_site/` へ収集。
  - `*-reverse.gif` は同梱せず、`game/js/assets.js` および `debug.js` が実行時に元の GIF を左右反転して表示。
  - 8 方向・アクション・`fb` シートは `cwebp -lossless -z 4` で劣化なく WebP 圧縮（WebP の 16,383px 制限を超える 5 シートは PNG を維持）。
  - 総出力を `--max-mib 820`（約 860 MB）以下に抑え、入りきらない重量級 `fb` シート（メガ／キョダイの一部）と `debug.html` の `fb` 公式 GIF プレビューは `data/pages_assets.js`（`window.ARENA_PAGES`）経由で `raw.githubusercontent.com` から CORS 対応の `Blob` URL としてオンデマンド取得します。
- **GitHub Actions（`.github/workflows/pages.yml`）**:
  - `main` へのプッシュ時に `--filter=blob:none` ＋ sparse-checkout（`_work/` と `*-reverse.gif` を除外）でチェックアウトし、`actions/cache@v4` で変換済み WebP キャッシュを再利用して `actions/deploy-pages@v4` でデプロイします。

---

## ディレクトリ構成

```text
monster_arena_2d/
├── index.html                  # ルートエントリー（./game/index.html へリダイレクト）
├── debug.html                  # ルートデバッグ入口（./game/debug.html へリダイレクト）
├── .nojekyll                   # GitHub Pages 用 Jekyll 無効化マーカー
├── ゲーム起動.command          # macOS 用ローカルサーバー＆ゲーム起動スクリプト
├── デバッグ起動.command        # macOS 用ローカルサーバー＆デバッグ起動スクリプト
├── .github/workflows/pages.yml # GitHub Pages 自動ビルド・デプロイワークフロー
├── data/
│   ├── monsterdex.js           # 1,217 フォーム・453 技・19 タイプ相性データ
│   ├── trainers.js             # 247 トレーナー定義（HGSS / FRLG）
│   ├── dir8.js                 # 8 方向・アクション・fb シートのメタデータ
│   ├── sprite_bounds.js        # スプライト描画境界キャッシュ（file:// 互換等）
│   ├── flame_hues.js           # 炎色相テーブル（1,185 件）
│   └── pages_assets.js         # GitHub Pages 用アセット解決マップ（ローカル時は null）
├── game/
│   ├── index.html              # ゲーム本編 HTML
│   ├── debug.html              # デバッグビューア HTML
│   ├── style.css               # UI・eSports / レトロ DQ HUD スタイル
│   └── js/
│       ├── main.js             # 起動・メインループ・展示デモ・URL 自動起動
│       ├── battle.js           # バトル進行・交代・メガシンカ・天候・BEAM CLASH
│       ├── fighter.js          # ファイター状態・身長スケール・歩き方プロファイル
│       ├── ai.js               # 戦術 CPU（ロール分類・予告回避・差し返し）
│       ├── ability.js          # とくせい効果処理
│       ├── moves.js            # 技アーキタイプ・描画・予告範囲・ヒット判定
│       ├── data.js             # 図鑑データ検索・タイプ相性・ダメージ計算
│       ├── assets.js           # GIF/PNG/WebP ローダー・Pages 反転/Blob 解決
│       ├── gif.js              # インメモリ GIF デコーダ
│       ├── dir8.js             # 8 方向・アクション・fb シートのフレーム解決
│       ├── motion.js           # 技種別ごとの初動モーションキーフレーム
│       ├── pixel.js            # ドットプリミティブ・ビットマップ文字・輪郭抽出
│       ├── arena.js            # 11 アリーナの地形・天候・グレード設定
│       ├── scene3d.js          # WebGL1 3D ジオラマエンジン
│       ├── scenes/*.js         # 11 ステージそれぞれの 3D 背景定義
│       ├── post.js             # WebGL1 HD-2D ポストプロセスシェーダー
│       ├── render.js           # ワールド/スプライト/空中エフェクト描画統合
│       ├── fx.js / vfx.js      # パーティクル・FXV 3D 投影・固有技 VFX
│       ├── slabs.js / ground.js / water.js # 岩盤隆起・永続デカール・水面効果
│       ├── hud.js / turn.js / title.js / menu.js / tournament.js
│       ├── audio.js / input.js / util.js
│       └── action_fit.js / debug.js # デバッグビューア本体・アクション枠補正
├── monsterSprite/              # 世代別モンスター画像・GIF・8 方向/アクションシート
├── monsterData/                # 図鑑データ元ファイル（data/monsterdex.js の生成元）
├── trainers/                   # トレーナードット絵（HGSS / FRLG）
├── tools/
│   ├── build_pages.py          # GitHub Pages 向け WebP 圧縮・上限パックツール
│   └── ...                     # データ生成・8 方向/アクションシート生成スクリプト群
└── md/                         # 開発ドキュメント・仕様正本（AGENTS.md / scene3d.md 等）
```

---

## ライセンス・クレジット

- 本プロジェクトはファンメイドの非公式デモ実装です。
- Pokémon © Nintendo / Creatures / GAME FREAK.
