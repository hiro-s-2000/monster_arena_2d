# GEMINI.md — arena_revenge_claude

作業ルール・ノウハウの正本は `md/` フォルダ内の `AGENTS.md` / `CLAUDE.md` / `GEMINI.md` です。ノウハウを更新する時は、`md/` フォルダ内のこれらの中身を更新すること。

@AGENTS.md

メガラティアスのアクション（2026-11-03）は `AGENTS.md` の「メガラティアス アクション v2」に書いた。`tools/forge_latias_mega_actions.py` は再実行しない。

gen4 のアクションはメガ11体のあと通常107体を、1体ごとにコンテキストを切って作る。手順は `md/gen4-mega-forge.md`、進行は `monsterSprite/gen4/_forge_queue.json`。pokeAnim は使わない。攻撃は体の平面回転と、横幅を潰す振り向きをやめ、絵にある手・足・牙・羽で打つ（2026-11-04）。`data/dir8.js` に `actions_8dir` を追加・更新したときは、`data/sprite_bounds.js` の `s[<id>]` も `agent-sprite-forge/sheet.png` の縦8行（0コマ目）の実測値で更新すること（古い `pmd` の境界値が残ると `debug.html` で足元が大きく下にずれる）。

gen5 のアクションは 1体ごとに別プロセスで、同時に最大16プロセスで作る。2026-11-04 21:35 JST 以降は新しい体を始めない。手順は `md/gen5-forge.md`、進行は `monsterSprite/gen5/_forge_queue.json`、ランナーは `tools/gen5_forge_pool.py`。pokeAnim は使わない。攻撃は咆哮、等倍の背面への振り向き、絵にある部位の打撃。週次使用率が 95% 以上なら次を始めない。見た目は debug.html でユーザーが確認する。作業側は画像を開かず、数値（静止差分・パレット外・中間アルファ・枠接触が 0）で publish する。

キョダイマックスリザードンのgen1先行実装（2026-11-04）は `AGENTS.md` の同名項目。再生成は `tools/forge_gen1_0006_charizard_gmax_actions.py`。2dは `monsterSprite/forge-manifest.js` でForgeを接続し、原front/backは保持する。walkのgait自動評価警告を合格扱いにしない。

キョダイマックス フシギバナ・カメックスは原画像忠実版 `tools/forge_gmax_original_actions.py --publish` で再生成（2026-11-04）。旧AGY原画版を再実行しない。元画像の前後2ビュー＋反転の8行互換で、8方向新規原画ではない。カメックスwalkのgait自動評価警告は合格扱いしない。詳細はAGENTS.mdの同名項目。

キョダイマックス フシギバナ・カメックスのAntigravity CLI再作成依頼（2026-11-04、ユーザーがCLIを明示）は未完了。各フォームの `art/agy_v3/` に原front/backコピー・prompt.txt・scoped-image.log・status.jsonを保存。両体の生成が429 RESOURCE_EXHAUSTEDで停止、解除予定2026-11-05 00:17:14 JST。新規画像・ゲーム差し替えは無し。再開は同フォルダ内で、権限省略オプションなしの `agy -p <prompt.txtの内容> --add-dir <同フォルダ絶対パス> --sandbox --print-timeout 1400s`。原画を見てから専用リグを作る。以前のoriginal_rig_v2を今回の完成品扱いにしない。

2026-11-04の最新指示は「Antigravity CLIを使用せずに」。フシギバナ・カメックスは `forge_gmax_original_actions.py --publish` のoriginal_rig_v4。腹の節線・白い下地の問題を修正、16コマ1周期歩行と技エフェクトを改善。新規方向絵は未生成、前後2ビュー＋反転の互換配置。AGY再開待ちの記述は旧依頼の記録。

【最新・ユーザー指定 2026-11-04】キョダイマックス フシギバナ／カメックスだけを8方向アクション採用前へ戻した。revenge data/dir8.js の2項目は _backup_gengar_mega_taunt_20261104/dir8.js の旧idle登録と同一、境界値は既存旧シートから再計測。2dのforge-manifest.jsから2体の上書き登録を削除して従来アニメへ復帰。生成ファイルは保管のみで不使用。上記の作成・再開メモは履歴であり、この2体のForge/8方向アクションを明示の再依頼なしに再採用しない。リザードンその他は変更しない。今回の直前状態は ../_backup_gmax_pair_before_rollback_20261104/ に保管。

- **キョダイマックスの GIF を削除（2026-11-04、ユーザー指示「変なエネルギーや顔が歪んで動く front.gif / front-reverse.gif / back.gif / back-reverse.gif などは全て削除」）**: `monsterSprite/**/gigantamax/<form>/` 直下の GIF 268 個（34 フォーム × 通常・色違い・反転）を削除。退避先は `../_backup_gmax_gifs_deleted_20261104/`（同じ相対パス）。`actions/`・`agent-sprite-forge/`・`art/` 内の GIF と、PNG（front/back の静止画・8方向シート）は残した。キョダイマックスの GIF を作り直したり戻したりしないこと（ユーザーの直接指示がある場合を除く）。
  - debug.html：「front.gif / front-reverse.gif / back.gif / back-reverse.gif」欄と、8方向シートが無いときの8方向欄は、GIF が無ければ `No Image` の枠（`.noimg`）を出す（`game/js/debug.js` の `noImage()`）。また、`#strip` の**8方向スプライト欄・アクションスプライト欄ともに**、×1/×2/×3 指定時を除き `gifSizedBox()` で4種 GIF（無ければ同名 PNG、それも無ければ `96×96`）と同じ枠サイズに揃える。
  - ゲーム（`game/js/assets.js` の `pngFallback()`、debug の file:// 用 `fileGet()`）：GIF が無いときは同名の `front.png` / `back.png` の静止画を使い、`-reverse` はそれを左右反転する。PNG も無ければ従来どおり failed。

- **メガ進化の GIF も削除（2026-11-04、ユーザー指示「メガ進化でキョダイマックスと同じような動きをする front.gif / front-reverse.gif / back.gif / back-reverse.gif は全て削除」）**: 見分け方は「front.gif が 19〜20 コマ」。この型はキョダイマックスと同じで、口からエネルギーが出て顔がゆがむ。14〜16 コマや 40 コマ以上のメガ GIF は普通の動きなので残した。対象 58 フォームの GIF 388 個（通常・色違い・反転）を `../_backup_mega_gifs_deleted_20261104/` へ退避して削除。例外：ミュウツーYの back 系（22 コマ、しっぽが揺れるだけ）は残した。メガラティアス（v3、1コマ）・メガライチュウX/Y は対象外。
  - debug.html の No Image 枠と、ゲームの PNG 代替（上のキョダイマックスの項）がそのまま効く。対象フォームはすべて front.png / back.png がある。
  - 削除したメガ GIF を作り直したり戻したりしないこと（ユーザーの直接指示がある場合を除く）。

- **通常・リージョンの GIF も削除（2026-11-04、ユーザー指示「リージョンと通常もキョダイマックスと同じような動きをする front.gif / front-reverse.gif / back.gif / back-reverse.gif は全て削除」）**: 正確な見分け方はコマの表示時間。「19 コマで3コマ目だけ 130ms、他は 60ms」か「20 コマすべて 60ms」の GIF がこの型（口からエネルギーが出て顔がゆがむ）。キョダイマックス・メガで消したものも同じ。対象は 188 フォーム（通常 166・リージョン 22）、GIF 1,500 個。退避先は `../_backup_normal_regional_gifs_deleted_20261104/`。
  - 残したもの：プテラの back 系4つ（17〜24 コマ、別の型）。19〜20 コマでも表示時間が違うオニドリル・ゲッコウガ・アゴジムシは普通の動きなので残した。
  - debug.html の No Image 枠とゲームの PNG 代替がそのまま効く。対象フォームはすべて front.png / back.png がある。
  - 削除した GIF を作り直したり戻したりしないこと（ユーザーの直接指示がある場合を除く）。

- **debug.html：`#dirs` の 2 段固定・`30vh` 高さ固定（2026-11-15、ユーザー指示「3列（3段）になるのは絶対にやめましょう。2列（2段）が好ましいのと、枠の高さは画面の30%くらいで固定」）**: `#strip.acts` の4個折り返しを廃止して全アクションを横1行（`grid-auto-flow: column`）に統一し、`#official`（4種 GIF）も `flex-wrap: nowrap` で横1行に固定して常に 2 段表示とした。また `#app` / `#dirs` を `30vh` 固定とし、`#dirs .head` を1行省略表示、自動ズーム時は `autoRowScale()` で `#strip` と `#official` を同率縮小フィットさせ、`×1/×2/×3` 指定時は `#dirs-body` 内をスクロールするようにした。
- **サンダー（`0145_zapdos`）の GIF をユーザー提供のものに差し替え（2026-11-04、ユーザー指示）**: `~/Desktop/zapdos-black-white-animated.gif` → `front.gif`、`~/Desktop/zapdos-black-white-back-animated.gif` → `back.gif`（どちらも 9 コマ・100ms、そのままコピー）。`front-reverse.gif` / `back-reverse.gif` は各コマを左右反転して作った（既存の `-reverse` も正面の左右反転なので同じ規則）。`data/sprite_bounds.js` の `g` の4項目を1コマ目の実測値に更新。色違いの GIF は提供が無いので作っていない（ゲームは `front-shiny.png` / `back-shiny.png` の静止画を使う）。8方向シート（`0145_zapdos.png`）は変更していない。

---

## ゲーム体験・演出ブラッシュアップのノウハウ（2026-11-11：fable 5.2 ＋ BigKidNation ＋ Ivandanke_Rivals 反映）

詳細は `AGENTS.md` の同名セクションを参照。今後の改修でも以下の5原則を維持すること。

1. **黄金比スタジアム縮尺とカメラのメリハリ（`assets.js` / `render.js`）**:
   - トレーナー身長 `1.7m = 48px` を基準物差し（メニュー・顔・リーチ用）とし、**`FIELD_DRAW = 0.75`（`Assets.onFieldPx`）** でコートと観客席を広く見せる。`1.7m` 未満は線形縮小（`BODY_MIN = 12px`、フィールド最小 `10px`）、`1.7m` 以上は重量バルク補正（上限 `107px` / フィールド `80px`）、キョダイマックスは `195〜268px`（フィールド `146〜201px`）。トレーナーはフィールド上 `18px`、画面外ワイプ最大 `58*dpr`（モバイル `30*dpr`）に調整し、手元から回収ビーム＋ボール投擲トレイル＋展開バーストを描画する。 **（2026-11-15 夜の最新: `FIELD_DRAW = 0.6`・フィールド最小 `8px`・1.7m ＝ `29px`。詳細は md/AGENTS.md）**
   - 通常時は画面酔いを防ぐワイド俯瞰を維持し、**ドラマチックなズームインは「強打・急所（`camFocus`）」「ビーム競り合い（`activeClash`）」「登場（`focus`）」「K.O.決着（`koCam`）」の4局面のみ**に限定する。
2. **「棒立ち撃ち」をなくす3段階のゲームフィール（`fighter.js` / `moves.js` / `battle.js` / `render.js`）**:
   - **Stage 1（予備動作＆テレグラフ）**: 溜め前半65%まで追尾、残り35%で照準角度ロック＋地面に危険範囲（`Moves.drawWindups` / `Moves.windupDanger`）を表示し、見てから回避可能にする。
   - **Stage 2（色付きシルエット残像）**: 回避・突進・跳躍・近接踏み込み時に `f.ghosts` 残像を描画（`actions_8dir` シートには `squash` を二重掛けしない）。
   - **Stage 3（着弾の重み）**: 段階的ヒットストップ（`0.036s / 0.052s / 0.072s`）、純白シルエットフラッシュ、ノックバック土煙、`zoomPunch`、重量級（`mass >= 2.2`）のスーパーアーマー。
3. **ビーム＆弾の正面衝突「⚡ BEAM CLASH!」（`Ivandanke_Rivals` 反映：`moves.js` / `arena.js` / `render.js` / `hud.js`）**:
   - ビーム同士の衝突で拮抗点（`B.clashes` / `B.activeClash`）を生成し、脈動コア（`drawClashNexus`）・動的ライト・放射状衝撃痕（`clash` デカール、氷融解対応）・上部 `#clash-tag`（`⚡ BEAM CLASH!`）・連打押し返し（`clashPush`）を動かす。弾同士の空中相殺・対ビーム干渉も維持。
4. **競技仕様の戦術AI（`ai.js`）**:
   - `zoner`（240〜300引き撃ち）・`rusher`（65〜95踏み込み＆離脱）・`allrounder` のロール自動分類、`Moves.windupDanger` / `B.dangerAt` による予告退避＆直角無敵回避、敵の技後硬直（`recoverT`）を狙う**差し返し（Whiff-punish）**、敵ビームへの**カウンタービーム（Beam Clash 誘発）**を維持。
5. **eスポーツ中継HUD・大型K.O.カットイン・永続フィールド破壊・観戦介入（`BigKidNation` ＋ `Ivandanke_Rivals` 反映：`index.html` / `style.css` / `hud.js` / `battle.js` / `render.js` / `main.js`）**:
   - デフォルトは `#hud.esports`（斜めカット赤青パネル・OKLabグラデーション・ゴースト付きHP・ENERGYバー・6体ボール・扇形CD・英日コンボ表示）。`#opt-hud` / `#hud-skin-btn` / `H` キー / `?hud=retro|esports` でレトロDQスキンと切替可能。
   - 撃破時は `0.14s` ヒットストップ＋スロー＋`koCam` ズーム＋斜めスラッシュ帯 `#ko-cutin`（`KNOCKOUT!! / K.O.!`）と永続クレーター生成（足元沈み込み付き）。
   - 頭上2段ミニバー（HP＋ENERGY＋コーナー色＋操作マーカー）、画面外トレーナーの左右ワイプ枠（`trainerWipes`）、観戦モード（`watch`）でのトレーナータップによる指示コーナー切替（`commandSide`）＆ `M` キー姿サイクル（通常→メガ→キョダイ→通常）、`body.touch` モバイル最適化を維持。

- **バトルフィールドの 3D 化（2026-11-16、ユーザー指示「HD-2Dらしさが全然足りない。背景を 3D 化」）**: まわりの景色は 3D の立体（`game/js/scene3d.js` ＋ `game/js/scenes/<id>.js`）、カメラは遠近つき。仕様の正本は `md/scene3d.md`、要点は `md/AGENTS.md` の同名の項。画面上の位置・倍率は `Renderer.worldToScreen(x, y, z)` / `toWorld` / `sclAt` を使い、`S * cam.zoom` を直接使わない。`?flat=1` で以前の平面表示。

- **debug.html が 3D にならないとき（2026-11-16）**: `debug.html` をダブルクリック（`file://`）で開くと、ブラウザの制限でスプライトの絵を WebGL に渡せないため `game/js/debug.js` が `Post.init` を止め、平面表示になる（コードでは直せない制限）。3D で見るには `デバッグ起動.command`（`ゲーム起動.command` と同じローカル http サーバーで `game/debug.html` を開く）を使う。`file://` で開いたときは画面右下の注意書きにも同じ案内が出る。

- **モンスターが光で白く飛ばないようにした（2026-11-16、ユーザー指示「モンスターが光り過ぎて見えない時がある。光源が強いのはもう少しモンスターが見える様に」）**: 原因は 4 つの重なり（照明の絵がモンスターを最大 2 倍に明るくする／まひ等の状態異常の単色かぶせ／ブルーム／光の筋）。(1) `game/js/post.js` の `spr3`（3D）と `compose`（平面）：モンスターだけ、等倍を超えた光を 0.3 倍に圧縮（最大 2.0 → 1.3、`SPR_OVER`）。地面・立体はそのまま。(2) `post.js` の `final`：モンスターがいる画素（`uSprM` のアルファ、3D のときだけ `uSprMask = 1`）では、ブルームを 0.4 倍、光の筋・ちりを 0.25 倍、上下端のぼかしを 0.3 倍に弱める。(3) `game/js/render.js`：状態異常の色かぶせを 0.25±0.15 → 0.14±0.08 に薄くした。被弾の白フラッシュ・技のエフェクト自体は変えていない。ファイル直開き（WebGL なし）の 2D 表示に効くのは (3) だけ。もっと抑えるなら `SPR_OVER` を下げる。元の強さへ戻さないこと。
