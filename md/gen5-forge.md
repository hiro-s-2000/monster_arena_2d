# gen5 アクション（1体ずつ、プロセスを切る）

2026-11-04。ユーザー指示: `monster_arena_2d/monsterSprite/gen5` のモンスターを全て、pokeAnim を使わず、gen4 と同じ agent-sprite-forge の部位リグで作る。連続して描くと再現度が落ちるので、**1体が終わるたびにそのプロセスを終了し、次のまっさらなプロセスがキューの次の1体だけを再開する。** 攻撃は咆哮と、背面が見える振り向き。体を平面に倒す回転はしない。手・足・牙・羽・角・尾・ハサミなど、絵にある部位で打つ。

キュー: `monsterSprite/gen5/_forge_queue.json`  
ランナー: `tools/gen5_forge_pool.py`（同時に最大8プロセス。各プロセスは `--session-id` が新規で、1体だけ作って終了する）

## 範囲

- 図鑑 494 から 649。フォルダにある `front.png` のうち、色違い・`box-icons`・`legends-arceus` は作らない。
- 通常、メガ、リージョナル、キョダイマックスは入れる。順番は図鑑番号順で、同じ番号ではメガ、キョダイ、リージョナル、通常。
- 1プロセスにつき1体だけ。終わったらそのプロセスでは次を始めない。同時に動かすのは最大8プロセス（2026-11-04、ユーザー指示で8人に増やした）。進行中の体は別プロセスが奪わない。
- 公式の front/back の png/gif（色違い・reverse 含む）は上書きしない。

## 禁止

- pokeAnim を開かない、実行しない、置き換えない。`monster_arena_2d` の戦闘は pokeAnim のまま。同期するのは `agent-sprite-forge/` の成果物だけ（`_work` は同期しない）。
- `tools/forge_latias_mega_actions.py`、`tools/forge_latios_mega_actions_v2.py`、`tools/forge_anatomical_actions.py` を実行・import しない。
- `tools/gen4_attack_strike.py` を編集しない。第5世代のピークは `tools/gen5_attack_strike.py` と `tools/gen5_attack_peaks.json`。
- Grok Imagine（image_gen / image_edit / image_to_video）、route_media、agy を使わない。
- 他種の色マスクやリグをコピーしない。絵に無い足・ダチョウ脚・ロケット伸ばしを足さない。
- `data/dir8.js` と `data/sprite_bounds.js` は `tools/gen5_publish.py` だけで、合格したキー1個を足す。ファイル全体を整形しない。`tools/build_sprite_bounds.js` は走らせない。
- debug.html の gen2 横4列は変えない。git push しない。`0.0.0.0` で待受しない。サブエージェントを起動しない。別の grok を起動しない。

## 出力

作者側は revenge の `<form>/agent-sprite-forge/`。2d の同じ相対パスへファイルだけコピーする。

ゲームが読むのは `agent-sprite-forge/sheet.png`。

- 4ブロック縦積み: idle, walk, attack, hurt。各8行 S SE E NE N NW W SW。
- コマ数 16, 16, 16, 12。静止画は多くの種が 96×96。違う寸法の種はその寸法に合わせ、セルは回転と振り向きで切れない大きさ（周囲約16px、足元ラインの下に約20px）。
- 併記: `preview.png`, `preview_motion.png`, `meta.json`, `idle.png`, `walk.png`, `attack.png`, `hurt.png`, 各 `*_8dir.png`, 各 `*_S_strip.png`。`_work/state.json` と `_work/frames/<action>/<dir>/NN.png`。`actions/preview.png` は目視用に同じ絵でよい。`actions/sheet.png` は書かない。

## 方向

静止画の目と、鼻・口・顔の先端の x を測ってから反転を固定する。静止が右向きなら S SE E NE はそのまま、N NW W SW は左右反転。左向きなら逆。E は画面右、W は画面左。N は背面が見える行。

## 動き

- 待機: その種らしい姿勢。0コマとループ端は静止画と一致。足裏は地面に着ける。足が無い種は浮く。
- 歩き: 絵にある足で歩く。足が無い種は滑空やうねり。足を生やさない。
- 攻撃:
  - 0–4 咆哮。口がある種は口のくさびとリング。口が無い種は口を発明せず、既にある顎・牙・トサカ・翼を上げる。
  - 5–8 後ろを向く。横幅を `|cos|` で潰さない。体を横倒し・逆さにしない。N 行と同じ、等倍の背面（左右反転した静止の向き）へ、全身のシルエットを保ったまま振り向く。
  - 9–12 正面側の姿勢に戻り、絵にある手・足・牙・羽・角・尾・ハサミで打つ。予備は 8–22°。打撃は最大約34°で、そのスクリプトが既に描いている角度を超えない。`gen5_attack_strike.finish()` が攻撃の `spin` を 0 にする。11–12 は衝撃。
  - 13–15 復帰。0コマの待機に戻る。
- ダメージ: 12コマ。のけぞりと復帰。回転は最大約38°。腹を反転しない。
- 頭は整数ピクセル移動だけ。通常は head_dx も head_dy も 0。
- 波は 0 コマとループ端で必ず 0: `amp * (sin(2π * cycles * f / n + phase) - sin(phase))`。

## パレットと合成

- 不透明ピクセルは、静止画の alpha≥128 の色への最近傍。アルファは 0 か 255 だけ。距離は int32。
- 回転のピボットは胴体に一番近い接触ピクセル。4倍に拡大してから最近傍で戻す。
- 欠ける関節は、部位の下で胴が空だった場所だけ胴色で埋める。静止の合成は元絵と差分 0。
- ゴミ取りは回転のあとだけ、4近傍が全部空のピクセルだけ消す。
- Python は `/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge/.venv/bin/python -I`。

## 再開と検査

- 見本: `tools/forge_gen5_<slug>_actions.py --samples`。画像は開かない。ログの静止差分・パレット外・中間アルファ・枠接触がすべて 0 なら 480 コマに進む。
- `--force` はその体の `_work/frames` だけ消す。ハッシュが一致しサイズが 0 より大きいコマは飛ばす。`NN.tmp.png` に書いてから置き換える。
- 合格前に `preview.png` は開かない。パレット外 0、枠への接触 0、静止差分 0、中間アルファ 0 をログで確認する。見た目はユーザーが debug.html で確認する（2026-11-04）。
- 不合格なら `gen5_publish` を呼ばず、`tools/gen5_queue.py mark <id> needs_fix`。
- 合格したら `gen5_publish.publish_dir8` と `publish_bounds`（待機0コマを8方向ぶん実測）。`layout` は `actions_8dir`。`an` は `[16,16,16,12]`。`ad` は `[0.08,0.062,0.05,0.055]`。`sheetDir` と `src` は `agent-sprite-forge`。`raw` は 1。`v` は `<slug>_v1`。そのあと `gen5_queue.py mark <id> done`。

## 週次

`~/.grok/statusline-usage.json` の `creditUsagePercent` が 95 以上なら、次の体を claim しない。ランナーは `WEEKLY_HOLD` で止まる。2026-11-04 21:35 JST 以降も次の体を始めない（`CLOCK_HOLD`）。進行中の体は終了まで走らせる。この枠では同時 16 プロセス（`GEN5_SLOTS=16`）。週次 95% は従来どおり。リセットは 21:41 JST。
