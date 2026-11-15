# gen4 アクション（1体ずつ、コンテキストを切る）

2026-11-03。ユーザー指示: `monster_arena_2d/monsterSprite/gen4` のモンスターを全て、pokeAnim を使わず、gen3 などのメガ進化と同じ agent-sprite-forge の部位リグで作る。連続して描くと再現度が落ちるので、**1体が終わるたびにコンテキストを切り、次のまっさらなコンテキストがキューの次の1体だけを再開する。** 動きは方向転換と咆哮。攻撃は部位の打撃（2026-11-04。体を平面に回して横向き・逆さにしない。手・足・牙・羽など、絵にある部位で打つ）。

キュー: `monsterSprite/gen4/_forge_queue.json`

## 範囲

- 順序はメガ11体（図鑑順、先頭 `0398_staraptor-mega`）、そのあと通常フォルム107体（`0387_turtwig` から `0493_arceus`）。
- 1コンテキストにつき1体だけ。終わったらそのコンテキストでは次を始めない。再開は次のまっさらな実行がキューを読む。
- 色違い、`legends-arceus/`、メス差分、`box-icons/` は作らない。`0469_yanmega` は通常フォルムであり、メガではない。
- ガブリアスとルカリオは mega と mega-z で絵が違う。リグを流用しない。通常フォルムも、他種の色マスクやリグをコピーしない。

## 禁止

- `tools/forge_latias_mega_actions.py` を実行・import しない。ラティアス v2 のスクリプトも import しない。構造だけ読む。
- `tools/forge_latios_mega_actions_v2.py` の GIF 上書き（back.gif 付近）をコピーしない。
- `tools/forge_anatomical_actions.py` をこのメガに向けない（gen2 用で、`data/dir8.js` を全部書き換える）。
- front/back の png/gif（色違い・reverse 含む）と box-icons を上書きしない。色違いシートは作らない。
- `pokeAnim` を置き換えない。2d の戦闘は pokeAnim のまま。同期するのは `agent-sprite-forge/` の成果物だけ（`_work` は同期しない）。
- Grok Imagine（image_gen / image_edit / image_to_video）と route_media を使わない。ローカルのパレットロックと部位リグだけ。
- 絵に無い足・ダチョウ脚・ロケット伸ばしを足さない。ゴマゾウの鼻の接ぎ木を他種に使わない。
- `data/dir8.js` は、その体のシートを目視して合格したあと、そのキー1個だけを文字列で差し込む。ファイル全体を整形しない。他種の項目は触らない。
- debug.html の gen2 横4列は変えない。git push しない。`0.0.0.0` で待受しない。

## 出力

ゲームが読むのは `agent-sprite-forge/sheet.png`（`actions/sheet.png` は書かない）。

- 4ブロック縦積み: idle, walk, attack, hurt。各8行 S SE E NE N NW W SW。
- コマ数 16, 16, 16, 12。足りない行の右側は透明。
- 静止画は 96×96。セルは回転しても切れない大きさ（目安 176×168 前後）。静止で周囲約16px、足元ラインの下に約20px。南の待機0コマの不透明高さを dir8 の `h` にする。
- 併記: `preview.png`, `preview_motion.png`, `meta.json`, `_work/state.json`, `_work/frames/<action>/<dir>/NN.png`。`actions/preview.png` は目視用に同じ絵でよい。

## 方向

静止画が右向きなら、S SE E NE はそのまま、N NW W SW は左右反転。左向きなら逆。E は画面右、W は画面左を向く。ムクホークメガはトサカが右、目がその左で、嘴は左を向く。嘴先の x が目より左であることを測ってから反転を固定する。

## 動き

- 待機: その種らしい姿勢（鳥はホバー）。0コマとループ端は静止画と一致。
- 歩き: 足が無い種は滑空。鳥は機首を下げた滑空と羽ばたき。元の爪の形を残す。
- 攻撃: 0–4 咆哮（口のくさび＋リング）。5–12 は体を回さず、そのスクリプトが既に描いている手・足・牙・羽・角・尾・ハサミの角度で打つ。8–22°の予備動作だけ、読める打撃（最大約34°、既にそれより大きい角度は超えない）まで開いてよい。ピクセル移動（口・先端の 1–4px）は度数にしない。11–12 は衝撃、13–15 は復帰。体全体の平面回転は攻撃に使わない。
- ダメージ: 12コマ。のけぞりと復帰。回転は最大約38°。腹を反転しない。`|cos|` で 0.34 まで潰すと細い絵が砕ける。
- 頭は整数ピクセル移動だけ。通常は head_dx も head_dy も 0（−1 でも細い首が切れる）。嘴・顎は頭と同じ移動。口は頭に付いたくさび。攻撃で体全体は回さない。
- 波は 0 コマとループ端で必ず 0: `amp * (sin(2π * cycles * f / n + phase) - sin(phase))`。

## パレットと合成

- 不透明ピクセルは、静止画の alpha≥128 の色への最近傍。アルファは 0 か 255 だけ。距離は int32（int16 は 255² で溢れて色が壊れる）。
- 回転のピボットは胴体に一番近い接触ピクセル。4倍に拡大してから最近傍で戻す。
- 欠ける関節は、翼や腕の下で胴が空だった場所だけ胴色で埋める。部位は胴のあとから描き、静止の合成は元絵と差分 0。
- ゴミ取りは回転のあとだけ、4近傍が全部空のピクセル（n==0）だけ消す。目と輪郭を消さない。
- 回転後に連結成分を見る。体は主成分が1つ。リングとスピード線は別成分でよい。

## 再開と検査

Python は `/Users/honmahiroshi/Desktop/ai_workspace/agent-sprite-forge/.venv/bin/python -I`。

- `--force` はそのモンスターの `_work/frames` だけ消す。
- ハッシュが一致し png のサイズが 0 より大きいコマは飛ばす。`NN.tmp.png` に書いてから置き換える。
- `--samples` は見本だけ書いて終了し、done にしない。
- `--no-sync` は 2d へコピーしない。`--skip-qa` は gait を飛ばす。
- 見本で静止差分、嘴のつながり、きりもみで体が砕けていないことを見てから 480 コマ（8方向×(16+16+16+12)）に進む。
- gait は `agent-sprite-forge/skills/video2dsprite/scripts/gait_loop.py select --frames-dir ... --fps 12 --kind hover --no-aids`。出力先が既存なら使わない。
- 合格前に `preview.png` を目で見る。パレット外 0、枠への接触 0、静止差分 0、ダメージで体が粉々でないこと。不合格なら dir8 に入れず、キューを `needs_fix` にする。
- 合格したらそのキーだけ dir8 に入れ、キューを `done` にする。`v` は `<variant>_v1`。`layout` は `actions_8dir`。`raw` は 1。`an` は `[16,16,16,12]`。`sheetDir` は `agent-sprite-forge`。あわせて `data/sprite_bounds.js` の `s["<id>"]` も `agent-sprite-forge/sheet.png` の 0〜7 行（S…SW の待機0コマ）の不透明矩形 `[x, y, w, h, ×8]` で更新する（旧 `pmd` シートの値が残ると `debug.html` でモンスターがヒットポイントより下にずれる）。

dir8 の形（数値はその体の実測）:

```json
{"dex":398,"form":"mega","variant":"staraptor-mega","gen":"gen4","dir":"monsterSprite/gen4/0398_staraptor/mega/staraptor-mega","sheetDir":"agent-sprite-forge","layout":"actions_8dir","nf":16,"cw":176,"ch":168,"w":176,"h":0,"fy":148,"an":[16,16,16,12],"ad":[0.08,0.062,0.05,0.055],"raw":1,"src":"agent-sprite-forge","v":"staraptor_mega_v1"}
```
