# gen1 アクション（メガライチュウX と同じ流れ、1体ずつ）

2026-11-04。ユーザー指示：「`monsterSprite/gen1/0026_raichu/mega/raichu-mega-x` の実装を参考にして、`monster_arena_2d/monsterSprite/gen1` のアクションを、クオリティの高いアクションで agent-sprite-forge の機能を使って作成し直す。Antigravity CLI でできる内容で。Agent は 8 人くらい」。

- 1体＝1エージェント（まっさらなコンテキスト）。同時に動かすのは 8 体まで（8 レーン）。
- 手本は `tools/forge_raichu_mega_x_actions.py` と `monsterSprite/gen1/0026_raichu/mega/raichu-mega-x/art/agy_prompt.md`。流れを写し、**部位・色・攻撃はその種の絵から作る**。他の種のリグや色マスクをコピーしない。
- 状態：`monsterSprite/gen1/_forge_status/<key>.json`（`{"key","status":"done|needs_fix|failed","v","note"}`）。done の体は飛ばす。

## 範囲

- `monster_arena_2d/monsterSprite/gen1` の `front.png` があるフォーム（通常・メガ・キョダイマックス・リージョナル）。色違い・`box-icons`・`legends-arceus` は作らない。
- 作らない：メガライチュウX・Y（完成済み）、メガピクシー（帽子の修正をユーザーが指定済み）、メガゲンガー（挑発の待機をユーザーが指定済み）。
- 作業は revenge 側（このフォルダ）の `monsterSprite/gen1/<…>/` で行い、`agent-sprite-forge/` だけを `monster_arena_2d` の同じ相対パスへコピーする（`_work` は除く）。

## 手順（1体）

1. 参照：そのフォームの `front.png` と `back.png`（無ければ `front.png` だけ）を最近傍で 8 倍にして `art/ref_front.png`・`art/ref_back.png` に置く。絵を見て、体の形・色・部位（足の数、腕、尾、羽、角、ヒレ、ツル、殻…）を数える。
2. Antigravity CLI：`art/agy_prompt.md` を書き（手本と同じ形式。**固有名詞は書かない**。形と色と部位の数を具体的に書き、「数を増やさない」ルールを入れる）、
   `cd art && agy -p "$(cat agy_prompt.md)" --add-dir <art の絶対パス> --dangerously-skip-permissions --sandbox --print-timeout 1400s`
   で `art/out/keys_2x2.png`（左上 S、右上 E か W の真横、左下 NE、右下 N）と `art/out/sw_single.png`（SW）を作らせる。生成物を目で見て、部位の数・色・向きが違えば 1 回だけ作り直させる。2 回目も駄目なら `needs_fix`。
   - **並列時の取り違えに注意（2026-11-04 判明）**：agy は生成画像（`~/.gemini/antigravity-cli/brain/<会話ID>/<名前>_<数字>.jpg`）を、サンドボックスの都合で共有の `/tmp/keys_2x2.png`・`/tmp/sw_single.png` を経由して `art/out/` にコピーしていた。8 レーン同時だとこの `/tmp` のファイルが他の体の絵で上書きされ、「別の生き物を描いた」（リザードンにカメ、など）ように見えていた。絵そのものは正しく描けていることが多い。
     - プロンプトに「一時ファイルは `/tmp` に固定名で置かない。使うなら `/tmp/agy_<key>/` を使う」と書く。
     - 「違う生き物」が出たら作り直す前に、agy の出力や brain の transcript（`.system_generated/logs/transcript_full.jsonl` の `Final Image Path`）から元の jpg を探し、`sips -s format png` で `art/out/` に入れ直す（画像生成の回数を使わない）。元の jpg は `art/agy_raw/` に控える。
     - 画像モデルの上限（429 RESOURCE_EXHAUSTED）に当たったら、その体は `failed` にして次へ進まない（解除時刻を note に書く）。
3. スクリプト `tools/forge_gen1_<slug>_actions.py`（slug は key の `-` を `_` に）を、手本の構成で書く：白背景の抜き（外周につながる白だけ）→ 5 枚を同じ縮尺に縮小 → 残り 3 方向は反転 → `palette_tool.py` build/lock/apply → 部位リグ → 4 倍で姿勢を作って最近傍で戻す → QC → 書き出し → 2d へ同期。
   - 大きさ：S の静止の体の高さを、公式 `front.png` の不透明部分の高さに合わせる（ゲーム内の大きさを変えない）。
   - 回転のピボットは胴とのつなぎ目。隠れる継ぎ目にはクッション（近い自分の色）。アルファは 0 か 255。新しい色は作らない（技の光だけ予約色で可）。
4. アクション（各 16 コマ、`an` [16,16,16,16]、`ad` [0.08,0.062,0.05,0.055]）：
   - 待機：その種らしい呼吸・揺れ。0 コマとループ端は静止と一致。足は地面に着ける。足が無い種は浮く。
   - 歩き：絵にある足で歩く（左右交互）。足が無い種は滑空・うねり・跳ね。足を生やさない。
   - 攻撃：構え → 絵にある部位（手・足・牙・角・尾・羽・ツル・ハサミ・殻・口からの技）で打つ → 衝撃・エフェクト → 戻り。体を平面にくるっと回さない。横幅を潰した振り向きをしない。口を開けるときは `tools/mouth_open.py`（あごを下げない、四角い口にしない、最大 2px）。
   - ダメージ：白フラッシュ、のけぞって後退、戻り。回転は最大約 38°。体が砕けない。
   - 波は 0 コマとループ端で 0：`amp * (sin(2π·cycles·f/n + phase) − sin(phase))`。
5. QC：パレット外 0、枠に触れるコマ 0、静止差分 0、`gait_loop.py` を待機・歩きで実行。`preview.png` を必ず目で見る（足の消失・融合、尾の千切れ、顔の崩壊、部位の数の間違いが無いこと）。
6. 出力（フォーム直下）：`8dir_stills.png`、`art/`、`agent-sprite-forge/`（`sheet.png`＝4 動作×8 行、`preview.png`、`meta.json`、`idle.png`… 手本と同じ一式、`_work/`）。**公式の front/back の png/gif（色違い・reverse 含む）は上書きしない。**
7. 公開：`tools/gen5_publish.py` の `publish_dir8(key, entry)` と `publish_bounds(key, measure_idle_bounds(...))` だけで、そのキー 1 個を書き換える（ロック付き。ファイル全体を整形しない、`tools/build_sprite_bounds.js` は走らせない）。
   entry：`{"dex","form","variant","gen":"gen1","dir","sheetDir":"agent-sprite-forge","layout":"actions_8dir","nf":16,"cw","ch","w","h","fy","an":[16,16,16,16],"ad":[0.08,0.062,0.05,0.055],"raw":1,"src":"agent-sprite-forge","v":"<slug>_agy_v1"}`。
8. 状態ファイルを書く。不合格なら公開しない。

## 禁止

- pokeAnim を開かない・使わない。debug.html の 4 列表示（固定仕様 1）を変えない。メガラティアスの固定仕様 3 に触れない。
- `tools/gen4_attack_strike.py`、`tools/gen5_*`（publish の import 以外）、他の種の `forge_*.py` を編集・実行しない。
- サブエージェントを起動しない。git push しない。
