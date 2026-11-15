# AGENTS.md — arena_revenge_claude

このフォルダの作業ルール・ノウハウの正本は **`md/` フォルダ内の `AGENTS.md` / `CLAUDE.md` / `GEMINI.md`** です。作業の前に必ず読んでください（特に「ユーザーが固定した仕様（変更禁止）」）。

- ノウハウ・ドキュメントを更新する時は、`md/` フォルダ内の以下の3ファイルの中身を更新すること：
  - `/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude/md/GEMINI.md`
  - `/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude/md/AGENTS.md`
  - `/Users/honmahiroshi/Desktop/ai_workspace/arena_revenge_claude/md/CLAUDE.md`

## ユーザーが固定した仕様（変更禁止）

以下はユーザーが明示的に決めた仕様です。**ユーザーがその項目の変更を直接指示した場合を除き、リファクタ・機能追加・「整理」のついでであっても変更・巻き戻ししないこと。**
別の作業でこの部分に触れる必要が出たら、変更せずにユーザーへ確認すること。

1. **debug.html：アクションスプライト／8方向スプライト枠（`#dirs`）は常に 2 段（1段目 `#strip` 横1行・2段目 `#official` 横1行）とし、枠高さは画面の `30vh` 固定で表示する（2026-11-15 更新）。**
   - 縦並び（1行に1動作）は画面を圧迫するため却下済み（2026-11-02、2回指示あり）。
   - さらに 2026-11-15 のユーザー指示により、メガバシャーモのようにアクション数が5個以上（攻撃1・攻撃2 等）ある場合でも 4 個で折り返して 3 段構成にすることは禁止し、全アクションを1行横並びにして下の `#official`（`front.gif` / `front-reverse.gif` / `back.gif` / `back-reverse.gif`）行と合わせた 2 段表示を維持する。
   - `#dirs` 自体の高さは `30vh` 固定（`#app` の `grid-template-rows: minmax(0, 1fr) 30vh`）とし、`#dzoom` が「自動」のときは `#strip` と `#official` を等倍以下で同率縮小フィットさせて 2 段とも 30% 枠内に収める（`×1`/`×2`/`×3` 指定時は `#dirs-body` 内をスクロール）。
   - 実装箇所：
     - `game/debug.html` の CSS `#app` / `#dirs` / `#dirs-body` / `#strip.acts { grid-auto-flow: column; grid-auto-columns: max-content; }` / `#official { flex-wrap: nowrap; }`
     - `game/js/debug.js` の `Strip.show()`・`Strip.layout()`・`autoRowScale()`（1動作 = `.cell.play` 1つ、ラベルはセル下端の `.lb`）
   - `.actrow` / `.name` を使った縦並びや、4列折り返しによる3段表示に戻さないこと。
   - コマ数の読み方（`m.an`）やシートの中身を変えるのは構わない。並び方と固定高さだけは維持すること。
   - コード内の `[USER-LOCKED LAYOUT]` コメントが付いた箇所がこの仕様。

2. **モンスターのアクションスプライト（待機・歩き・攻撃・ダメージ）作成時の必須ルール：**
   - どのモンスターであっても、アクションを作成・修正する際は必ず `/Users/honmahiroshi/.gemini/config/skills/monster-action-forge-architect/SKILL.md`（`monster-action-forge-architect` スキル）に従うこと。
   - `agent-sprite-forge`（パレットロック・`gait_loop.py` ループ評価）と解剖学的部位リグ（`tools/forge_anatomical_actions.py`）を用い、モンスターの状態・形態（4足歩行・2足歩行・飛行猛禽・蛇/とぐろ・ハサミ/角/顎/爪などの攻撃部位）に応じて、足や尻尾・腕・頭が独立して自然に動くように作ること。
   - 生成後は必ず `actions/preview.png` を目視確認し、足の消失・融合・尻尾の千切れ・顔の崩壊がないことを確認すること。

3. **メガラティアス（`0380_latias-mega`）は、2026-11-04 にユーザーが渡した2枚の絵（公式アート `mega-latias.webp` と `ref_gemini.jpeg`）から Antigravity CLI で作った v3（`latias_mega_agy_v3`）を使う。**
   - ユーザーの直接指示（2026-11-04）で、それまでの v2（目を描き足した 120×120 の絵と、そのアクション）を置き換えた。front/back の png/gif（色違い含む）も v3 で上書き済み（ユーザー了承）。
   - 作り直しは `tools/forge_latias_mega_agy_actions.py` だけを使う。**`tools/forge_latias_mega_actions.py`（v1）・`tools/forge_latias_mega_actions_v2.py`・`tools/mega_latias_sprite.py` は実行しないこと**（v3 の front/back やシートが古い絵に戻る）。
   - v2 一式は `../_backup_latias_mega_20261104_before_agy/` にある。元に戻すのはユーザーの指示があったときだけ。

## スプライト差し替えのメモ

- **メガラティアス（2026-11-03、ユーザー指示）**: 公式アート（ユーザー提供の `mega-latias.webp`）を `tools/art_to_sprite.py` でドット絵化し、`monsterSprite/gen3/0380_latias/mega/latias-mega/` の front/back の .png/.gif（色違い含む）を置き換えた。
  - 背面の絵が無いため、背面は正面の絵（左右反転）を流用（ユーザー了承済み）。色違いは紫→緑の色相置換。
  - 古い GIF 由来の8方向シート（`0380_latias-mega(-shiny).png`）は外し、`data/dir8.js` の項目も削除。ゲームは新しい GIF から実行時に8方向を作る。
  - 元のファイルは `../_backup_latias_mega_20261103/` にある。
  - 実行例: `<agent-sprite-forge>/.venv/bin/python -I tools/art_to_sprite.py <画像> <出力フォルダ> 90`（Pillow が入っているのはこの venv）
- **メガラティアス v2（2026-11-03、ユーザー指示「顔や目の形がわかるように」）**: `tools/mega_latias_sprite.py` で作り直し（120×120 キャンバス、体 112px）。
  元絵の頭を首元基準で 1.3 倍に拡大してからドット化し、目（輪郭・琥珀色の虹彩・黒い瞳・白いハイライト）を手打ちの 7×5 ドットで描き、マスクの白い筋とあごの白を塗り直している。
  v1（art_to_sprite.py 版）は `../_backup_latias_mega_20261103/v1_from_art/` にある。
- **メガラティオス アクション v2（2026-11-03、ユーザー指示「咆哮・回転するかっこいいアクション」）**: `tools/forge_latios_mega_actions_v2.py`。
  Antigravity 版（`forge_latios_mega_actions.py`）の7部位リグをそのまま読み込んで使い、体全体の動きを足している：
  待機＝ホバー＋翼のゆれ＋ときどき唸り（口＋咆哮リング）、歩き＝機首下げの滑空＋羽ばたき＋スピード線、
  攻撃＝のけぞって咆哮（口を大きく開く・リング3重）→ 360° きりもみ回転で突進 → 急降下の一撃（衝撃エフェクト）→ 復帰、
  ダメージ＝白フラッシュ1コマ→ のけぞり＋半回転のよろめき＋火花。頭は常に整数移動だけなので顔・目は崩れない。
  - 出力の形式（120×96・足元 86・16コマ・S…SW）は v1 と同じ。`data/dir8.js` の `v` を `latios_mega_v2` に上げ、シート URL は `?v=<m.v>` でキャッシュを更新する。
  - **途中から再開できる**：コマごとに `agent-sprite-forge/_work/frames/` に保存し、`_work/state.json` で済んだ所を飛ばす（`--force` で全部作り直し）。QC（パレット外の色・枠へのはみ出し・agent-sprite-forge の `gait_loop.py` ループ評価）は `_work/qa/qa.json`。
  - 実行後に `monster_arena_2d` の同じフォルダへ自動で同期する（`--no-sync` で止める）。v1 の出力は `../_backup_latios_mega_actions_v1_20261103/` にある。
  - 実行: `<agent-sprite-forge>/.venv/bin/python tools/forge_latios_mega_actions_v2.py`
- **メガラティアス アクション v2（2026-11-03、ユーザー指示「方向・咆哮・回転」）**: `tools/forge_latias_mega_actions_v2.py`。v2 の 120×120（目が読める静止画）だけを使い、翼・頭・あご・前肢・横びれ・尾を分けて動かす。足は足さない。待機はホバー、歩きは機首を下げた滑空、攻撃は咆哮リングから 360° きりもみ、ダメージは白フラッシュとよろめき。頭はきりもみ以外では回さない。パレット外の色は 0。`front/back` の png/gif は触らない。
  - セル 200×188、足元 168、南の不透明高さ 108。方向は S SE E NE が右向き、N NW W SW が左右反転。`data/dir8.js` の `0380_latias-mega` は `layout: actions_8dir`、`v: latias_mega_v2`。
  - **途中から再開できる**。済んだコマは `agent-sprite-forge/_work/frames/`、`_work/state.json` のハッシュが変わったときだけ作り直す（`--force` で全部）。`--samples` は見本だけ。QC は `_work/qa/qa.json`（パレット、枠、`gait_loop.py`）。
  - 同じフォルダを `monster_arena_2d` の `agent-sprite-forge/` へ同期する（`--no-sync` で止める）。2d の `front.png` は 96×96 のまま。2d の戦闘は `pokeAnim` を再生するので、このシートは revenge の debug / 戦闘で見える。
  - 実行: `<agent-sprite-forge>/.venv/bin/python -I tools/forge_latias_mega_actions_v2.py`
- **gen4 のアクション（2026-11-03、ユーザー指示「gen4 のモンスターを全て。pokeAnim は使わない。1体ごとにコンテキストを切って再開。方向・咆哮・回転」）**: 手順は `md/gen4-mega-forge.md`。進行は `monsterSprite/gen4/_forge_queue.json`。先にメガ11体、そのあと通常フォルム107体。1コンテキストにつき1体だけ。色違い・legends-arceus・メス差分は作らない。front/back の png/gif は上書きしない。`forge_latias_mega_actions.py` は実行しない。
  - **2026-11-04 完了**：残りの通常フォルム 26 体（0467 ブーバーン〜0493 アルセウス、前回 needs_fix のルカリオ・マネネを含む）を、1体＝1エージェント（毎回まっさらなコンテキスト）の Workflow で順に作成。キューは 118 件すべて done、`data/dir8.js` にも 26 件とも `actions_8dir` で入り、debug.html で表示確認済み（エラー無し）。
  - 途中で別のセッションが「攻撃で体全体を平面回転しない」規則（`tools/gen4_attack_strike.py`、gen4-mega-forge.md）を追加したため、0477 ヨノワール以降の多くは 360° 回転ではなく、振り向き（正面→背面の絵の切り替え）や部位の打撃で攻撃している。旧 pmd 形式の項目（shiny の副項目を含む）は新しい項目で置き換えられた。
  - **攻撃（2026-11-04、ユーザー指示「回転して平面になるのはやめ、手・足・牙・羽で打つ。今よりカッコ悪くしない」）**: 攻撃で体全体を平面回転しない。横幅を約半分に潰して背面へ回す振り向き（narrow）も平面化なので使わない。`tools/gen4_attack_strike.py` が攻撃の `spin` を 0 にし、各スクリプトが既に描いている部位で 5–12 コマを打つ。待機・歩き・ダメージは変えない。ダメージののけぞり（最大約38°）は残す。描いていない角度まで上げない。1–4px の移動を度数として増幅しない。
  - **debug.html（file://）での足元位置（2026-11-04 修正）**: `data/sprite_bounds.js` の `s[<id>]` に旧 `pmd` シート（32×28 等、`ay≈25`）の値が残っていると、`actions_8dir` シート（176×168 等、本来の足元 `fy=148` 付近）のモンスターがヒットポイント・影・HPバーより大幅に下にずれて描画される。`data/sprite_bounds.js` の `s` を全 `actions_8dir` / `actions` シートの実測値（縦8行の0コマ目）で更新し、`tools/build_sprite_bounds.js` も `actions_8dir`（`agent-sprite-forge/sheet.png` の縦8行）と `actions` に対応させた。さらに `game/js/debug.js` の `fileSheet()` でも、万一 `BOUNDS.s` に古い `pmd` の値が残っていた場合は `m.fy` と `m.cw / 2` を優先するようフォールバックを入れた。新しい `actions_8dir` シートを `data/dir8.js` に追加するときは `data/sprite_bounds.js` の `s[<id>]` も実測値で更新すること。
- **gen5 のアクション（2026-11-04、ユーザー指示「gen5 を全て。pokeAnim は使わない。1体ごとにコンテキストを切って再開。咆哮と、後ろを向く回転」）**: 手順は `md/gen5-forge.md`。進行は `monsterSprite/gen5/_forge_queue.json`。ランナーは `tools/gen5_forge_pool.py`。1プロセスにつき1体で、同時に最大16プロセス（2026-11-04 21:35 JST まで。それ以降は新しい体を始めない）。色違い・`legends-arceus`・`box-icons` は作らない。通常・メガ・リージョナル・キョダイは作る。front/back の png/gif は上書きしない。攻撃の 0–4 は咆哮、5–8 は等倍の背面への振り向き（横幅を潰さない、体を平面に倒さない）、9–12 は絵にある部位の打撃。`tools/gen4_attack_strike.py` は編集しない。週次の `creditUsagePercent` が 95 以上なら次の体を始めない（2026-11-04、21時までの作業は 95% まで許容）。見た目はユーザーが debug.html で確認する。作業プロセスは preview を開かず、静止差分・パレット外・中間アルファ・枠接触が 0 なら publish する。
- **メガラティアス v3（2026-11-04、ユーザー指示「2枚の絵から agent-sprite-forge で8方向に作り直し、既存実装を消して作り直す。Antigravity CLI でできる内容で。front/back も置き換え」）**: `tools/forge_latias_mega_agy_actions.py`。元絵と生成物は `monsterSprite/gen3/0380_latias/mega/latias-mega/art/`（`agy_prompt.md`、`out/keys_2x2.png`＝南・西・北東・北、`out/sw_single.png`＝南西。南東・東・北西は反転）。
  - 白背景は「外枠につながる白」だけ抜く（胸の白灰色は 230 以下なので残る）→ 0.26 倍に面積縮小（南西は高さを合わせる）→ `palette_tool.py` build / lock / apply（30色、うち2色は念力エフェクト用に予約）。
  - 部位：頭、胴＋尾、左右の翼（腕の付け根と3本爪の手を含む）。胴の下に頭と胴の輪郭内を埋める下地があり、翼が動いても穴が空かない。正面系は翼が胴の後ろ、背面系は翼が手前。
  - 待機＝ホバーの上下とゆっくりした羽ばたき。歩き＝前傾の滑空と速い羽ばたき、スピード線。攻撃＝翼を振り上げて口から念力のリング（0–3）→ 8方向の本物の絵を順に見せる一回転（4–9、体を平面回転させない）→ 前へ突進して衝撃（10–11）→ 戻り。ダメージ＝白フラッシュ、のけぞって後退、翼をたたむ、光の粒。全アクション16コマ。
  - セル 232×172、足元（尾の先）150、`an` [16,16,16,16]、`v: latias_mega_agy_v3`。`front.png`＝南西、`back.png`＝北、色違いは紫→緑の色相置換。`8dir_stills.png` あり。`data/sprite_bounds.js` の `s` と `g` も更新済み。
  - QC：パレット外の色 0、枠へのはみ出し 0。`gait_loop.py`（hover）は待機 pingpong、歩き valid-cycle。debug.html で表示確認（エラー無し）。`monster_arena_2d` へ同期（シートと front/back）。
- **メガ・ゲンシの口の開き方（2026-11-04、ユーザー指示「口の開き方が長方形でしゃくれて見える。顔の形に沿って小さく口が開くように」）**: 共通の部品 `tools/mouth_open.py`（`open_mouth(img, line, amount, deep, inner, peak)` と `transform_points`）。下あごの層を下げて奥に長方形の口の中を塗るやり方は**やめる**。あごは頭と一緒に固定し、静止画で測った口の線（`MOUTH_LINE[view]`、口角→口先）の下に、両端が閉じたレンズ形のすき間を描く。体の輪郭を1px削った内側だけを塗り、色は元のパレット（輪郭色と既存の暗い赤など）だけを使う。開き幅は待機・歩き・ダメージが 1px、咆哮のピークでも最大 2px。背面系（口が見えない向き）は開けない。1体ずつ別のエージェントで作り直し、`data/dir8.js` の `v` を `_v1m` / `_v2m` に上げた。
  - 対象（fixed）：バンギラス・ボーマンダ・ジュカイン・ラグラージ・バシャーモ（`forge_blaziken_mega_kicks.py` で作り直し）・ヤミラミ・ボスゴドラ・チャーレム・ライボルト・サメハダー・バクーダ・ジュペッタ・アブソル・アブソルZ・オニゴーリ・メタグロス・ラティオス（`_v2.py` で作り直し）・ゲンシカイオーガ・ゲンシグラードン・レックウザ・オーダイル・ハガネール・エアームド・ヘルガー・ユキノオー・エルレイド・ガブリアスZ・ムクホーク・ルカリオ・ルカリオZ・エンブオーのメガ。ユキメノコは口が無いので変更なし（`no_mouth_motion`）。
  - 多くのスクリプトは毎回待機から front/back の gif を書き出す。今回は gif をバックアップから戻した（一部のスクリプトには `--gifs` などの書き出しスイッチを追加）。そのため front/back の gif には古いあごの動きが残っているものがある。
  - バックアップ：`../_backup_mouth_fix_20261104/<id>/`。ジュペッタとメタグロスの攻撃で枠に触れるコマ（3・5コマ）は以前からあるもの。
- **メガピクシー（0036_clefable-mega）の額のクルン隠し（2026-11-04、ユーザー指示「額がピクシーの様にクルンとしている。front.png・back.png の様に白い帽子で隠す」）**: `tools/fix_clefable_mega_cap.py`（入力＝バックアップ、出力＝シート）。8方向シート `0036_clefable-mega.png`（列＝方向、行＝8コマ）の目が見える40コマで、白い帽子を目の上 4px まで弧を描いて下ろし、額の渦巻きを覆う。縁にはシートの帽子と同じ薄いピンク白の影、その下の肌に薄い影。塗るのは帽子とつながった肌・渦巻きの色だけで、帽子の幅より外（耳・羽）と背面3方向は触らない。`data/dir8.js` に `v: clefable_mega_cap1`。元のシートは `../_backup_clefable_mega_cap_20261104/`。
- **メガゲンガー（0094_gengar-mega）の待機に挑発（2026-11-04、ユーザー指示「歯が見える時は、時折、口を開いて長いベロを出して相手を挑発する」）**: `tools/forge_gengar_mega_taunt.py`（入力＝`../_backup_gengar_mega_taunt_20261104/` の元シート、出力＝シートと `data/dir8.js` の `n`・`d`・`v`）。手作りの待機シート（列＝8方向、行＝コマ、229×196）のうち、歯が見える南・南東・東・西・南西を 32 コマにした：元の 8 コマの呼吸を 2 周（2.4 秒）→ 挑発 16 コマ（約 1.3 秒）。挑発は、歯の三日月に沿って上の歯は動かさず下の歯の列だけを弧形に下げて口を開け（両端は閉じたまま。四角い口・しゃくれにしない）、中を暗い紫で塗り、ベロを出して振ってから引っ込めて閉じる。背面3方向は 8 コマのまま。
  - v2（同日、ユーザー指摘「ベロが細くて可愛くない。顔の向きでベロの向きも変える」）：ベロは口元で細く先が丸く広いしゃもじ形（幅は歯の幅の約 38–50%）、濃い紫ピンクの輪郭・上側のハイライト・中央の溝・先のツヤ。向き別（`VIEWS`）：南＝まっすぐ下に垂らして先を左右に振る、南東/南西＝顔の向く側へ斜め下、東/西＝鼻先から真横に出して垂れ、先を上下に振る。口の開きの最大位置も顔の向く側へ寄せる（`peak`）。ベロはセルからはみ出さないよう長さで調整。
  - v3（同日、ユーザー指摘「ベロがリアルすぎる。絵柄の質に合わせて二次元に」）：ハイライト・中央の溝・先のツヤ・柔らかい輪郭をやめ、スプライトと同じ黒に近い 1px 輪郭＋ピンク1色＋下側の影1色だけのベタ塗り（アニメ塗り）にした。形と向き別の動きは v2 のまま。`v: gengar_mega_taunt3`。元の `dir8.js` も同じバックアップフォルダにある。
- **メガヒードラン（0485_heatran-mega）の攻撃を踏みつけに作り直し（2026-11-04、ユーザー指示「黄色い口が不自然に動く。足を勢いよく踏みつけるかっこいい動きに」）**: `tools/forge_heatran_mega_actions.py` v3。黄色い口の炎は全コマで回さない（ダメージも）。この種には共通フック `gen4_attack_strike.finish()` を適用しない（口を 54° 回していたため）。攻撃＝後ろ足を軸に体をのけぞらせて前足を上げる（0–4）→ 振り下ろして踏みつけ（5）→ 地面の衝撃の輪・溶岩のしぶき・岩の破片・地割れ・小さな揺れ（6–11）→ 戻り。`v: heatran_mega_v3`。前の版は `../_backup_heatran_mega_actions_20261104/`。
- **ベロベルト（0463）のベロ修正（2026-11-04、ユーザー指示「ベロが体から離れている」）**: `tools/forge_lickilicky_actions.py` v3。ベロ全体を端の一点で板のように回していたのをやめ、口側の付け根を固定して先ほど大きく曲げる（`bend_tongue`）。動かす部品はオレンジ・赤のベロと輪郭だけにし、白い胸毛・唇・両手の爪は胴に戻した。ベロの下の胸色の下塗りは体の輪郭の内側だけ。攻撃は口を開けてベロを振り上げ → 前へ伸ばして振り下ろす舌打ち。`v: lickilicky_v3`。前の版は `../_backup_lickilicky_actions_20261104/`。
- **メガバシャーモ 攻撃1・攻撃2（2026-11-03、ユーザー指示「回し蹴り・跳び蹴りを攻撃1・攻撃2として、細かいコマ割りで」）**: `tools/forge_blaziken_mega_kicks.py`。
  Antigravity 版（`forge_blaziken_mega_actions.py`）の10部位リグを読み込んで使う（元のスクリプトは編集しない）。待機・歩き・攻撃・ダメージは元のまま作り直し、2つ追加する：
  攻撃1＝回し蹴り（構え→8方向のリグを1コマずつ回して 360° ターン＋腰の炎の輪→上段回し蹴り＋炎の弧＋衝撃→戻り）、
  攻撃2＝跳び蹴り（しゃがみ→砂ぼこりを上げて跳ぶ→体を前に傾けて急降下キック＋炎の筋→着地の衝撃＋砂の輪→後ろへ跳んで構え）。どちらも 24 コマ（0.042 秒/コマ）。
  - 蹴り足は蹴りの間だけ腕より手前に描く。頭と手は整数移動だけなので顔が崩れない。パレット外の色 0、枠へのはみ出し 0。
  - ジャンプと伸ばした足が入るようにセルを 176×136（足元 128）に広げた。シートは 6 アクション × 8 方向 = 48 行、幅は最大 24 コマ。
  - `data/dir8.js` の `0257_blaziken-mega` に **`acts: ["idle","walk","attack","hurt","attack1","attack2"]`** を追加（`an`/`ad` も 6 個）。`acts` が無いシートは従来どおり 4 アクション。
  - ゲーム（`game/js/dir8.js`）：`acts` を名前で引く。攻撃が複数あるときは、技を出すたびに 攻撃1 → 攻撃2 → 攻撃 の順に使い、その攻撃を 1 コマ目から再生する（`render.js` の `frameOf` が `view().t` を使う）。
  - debug.html：`acts` の順にセルを並べる。2026-11-15 のユーザー指示（3段禁止・2段維持・枠高さ `30vh` 固定）により、攻撃1・攻撃2 も折り返さず 1 行目へ横並びにし、下の `#official`（4種 GIF）行と合わせて 2 段で表示する。`data/sprite_bounds.js` の `s["0257_blaziken-mega"]` も新しいセルに合わせて書き換えた。
  - 途中から再開できる（`_work/frames/`、`_work/state.json`、`--force`）。QC は `_work/qa/qa.json`。歩きは拳法のすり足なので `gait_loop.py --kind gait` は周期なし（no gait period）と出る。全コマの確認用画像は `_work/review_kicks_all_frames.png`。
  - `monster_arena_2d` の同じフォルダへ同期する（`--no-sync` で止める）。前の版は `../_backup_blaziken_mega_actions_v1_20261103/`。
  - 実行: `<agent-sprite-forge>/.venv/bin/python -I tools/forge_blaziken_mega_kicks.py`。**`forge_blaziken_mega_actions.py` を単独で再実行すると 4 アクション・旧セルのシートに戻り、`dir8.js` の `acts` とずれる**ので、メガバシャーモはこちらを使うこと。
- **バトルで agent-sprite-forge の動きを使う（2026-11-03、ユーザー指示「agent-sprite-forge があるモンスターはバトルでその動きで動くように」）**:
  - `data/dir8.js` に `layout: actions_8dir` の項目があるモンスター（メガシンカ後に切り替わる場合も含む）は、バトルで `agent-sprite-forge/sheet.png` の 待機・歩き・攻撃・ダメージ を使う。
  - `game/js/dir8.js` の `actionOf()`：攻撃は技を出した瞬間に1コマ目から始め、技が先に終わっても最後のコマまで再生する（ビームなど技が続いている間はくり返す）。ダメージも被弾の瞬間に1コマ目から最後まで再生し、攻撃中に被弾したらダメージが優先。攻撃が複数あるシート（`acts`）は 攻撃1 → 攻撃2 → 攻撃 の順。
  - `game/js/render.js`：アクションシートのモンスターには技の構え・被弾の伸び縮み（squash）をかけない（シートのコマに構えやのけぞりが描いてあるため）。前へ出る・浮く等の位置の動きはそのまま。
  - アクションシートが無いモンスターは従来どおり（GIF＋伸び縮み）。
- **メガライチュウX（2026-11-03、ユーザー指示「Antigravity CLI でできる内容で作り直し」）**: `tools/forge_raichu_mega_x_actions.py`。元絵と生成物は `monsterSprite/gen1/0026_raichu/mega/raichu-mega-x/art/`。
  - ユーザー提供の正面・背面の絵（`raimX_front.webp`・`raimX_back.webp`）を参照画像にして、**Antigravity CLI（`agy -p`、image-generator）** で8方向のキー画像を描かせた。手順は Antigravity のスキル `~/.gemini/config/skills/monster-8dir-sheet-architect`（固有名詞を使わず形・色・部位で説明する）。プロンプトは `art/agy_prompt.md`。
    - `art/agy_keys_2x2.png`：左上＝南（正面）、右上＝西（横向き・左向き）、左下＝北東（斜め後ろ）、右下＝北（背面）。`art/agy_sw_single.png`：南西（斜め前）。南東・東・北西はそれぞれ南西・西・北東の左右反転。
    - Codex CLI（agent-sprite-forge の画像ルート）は公式アートからの生成を安全フィルターで拒否したので使っていない。
  - 白背景を抜く（背景につながった白だけ抜く。お腹の白は残し、耳の渦巻きの穴は抜く。背面2方向の渦巻きの白は暗い灰色に）→ 面積平均でゲームの大きさに縮小（2×2 は 0.258 倍、南西は体の高さを合わせる）→ agent-sprite-forge `palette_tool.py` で build（30色＋火花用の2色）→ lock → apply。
  - 部位リグ（向きごとの多角形）：尻尾2本（根元で回転）、頭＋耳、こぶし（茶色のグローブ部分だけ）、胴、足。4倍に拡大して最近傍で回転し、1倍に戻す（色が増えない）。胴の下には「頭と胴の輪郭の内側」を埋める下地があるので、こぶしが動いても穴が空かない。
  - アクションは各16コマ。待機＝呼吸、尻尾・耳のゆれ、ほおの火花。歩き＝足を交互に上げ、こぶしと尻尾を振る。攻撃＝かみなりパンチ（ためて火花→踏み込んでパンチ→電気の爆発→戻り）。ダメージ＝白フラッシュ→のけぞり→体に電気。
  - セル 232×132、足元 122。`data/dir8.js` は `layout: actions_8dir`、`v: raichu_mega_x_agy_v1`。`front.png`＝南西、`back.png`＝北（＋反転・GIF）、`8dir_stills.png` に8方向の静止画を並べてある。`data/sprite_bounds.js` も更新済み。
  - QC（`agent-sprite-forge/_work/qa/qa.json`）：パレット外の色 0、枠へのはみ出し 0。`gait_loop.py` は待機が pingpong、歩きは「no loop window」（その場の足踏みで一定周期が取れない）。
  - 前の版（手作りシート、尻尾を足すスクリプト、最初の作り直し）はユーザーが却下。`../_backup_raichu_mega_x_20261103/`（`before_asf_rebuild/`、`before_agy_rebuild/`）に残してある。戻さないこと。
  - 途中から再開できる（`_work/state.json`、`--force` で全部作り直し）。`monster_arena_2d` の `agent-sprite-forge/` へ同期する（`--no-sync` で止める）。
  - 実行: `<agent-sprite-forge>/.venv/bin/python -I tools/forge_raichu_mega_x_actions.py`
- **メガライチュウY（2026-11-03、ユーザー指示「Antigravity CLI でできる内容で作り直し」）**: `tools/forge_raichu_mega_y_actions.py`（メガライチュウX と同じ流れ。X のスクリプトを元に、部位と攻撃を Y 用にしたもの）。元絵と生成物は `monsterSprite/gen1/0026_raichu/mega/raichu-mega-y/art/`。
  - ユーザー提供の `raimY_front.webp`・`raimY_back.webp` を参照画像にして、Antigravity CLI（`agy -p`）で `art/agy_keys_2x2.png`（南・西・北東・北）と `art/agy_sw_single.png`（南西）を生成（プロンプト `art/agy_prompt.md`）。南東・東・北西は反転。
  - 白背景を抜く（南西は元画像が大きいので、穴とみなす大きさの上限を 2.5 倍にしている。お腹の白は残す）→ 縮小（2×2 は 0.258 倍、南西は体の高さを合わせる）→ `palette_tool.py` build / lock / apply（30色）。
  - 部位：頭の左右の稲妻形の飾り（crest_l / crest_r）、頭（耳・顔）、胴（腕と細い尻尾のひもを含む）、足、尻尾の大きな稲妻の板（ひもの先を軸に回す）。
  - 攻撃＝尻尾で電気をためる雷撃（飾りと尻尾の板を振り上げてため→踏み込んで尻尾を前へ振り下ろす→前方に雷と爆発→戻り）。待機・歩き・ダメージは X と同じ考え方。
  - セル 304×140、足元 128。`data/dir8.js` は `layout: actions_8dir`、`v: raichu_mega_y_agy_v1`。前の手作りシート `0026_raichu-mega-y.png` は削除（前のファイルは `../_backup_raichu_mega_y_20261103/`）。
  - QC：パレット外の色 0、枠へのはみ出し 0。`gait_loop.py` は待機 pingpong、歩きは no loop window。`monster_arena_2d` へ同期済み。
  - 実行: `<agent-sprite-forge>/.venv/bin/python -I tools/forge_raichu_mega_y_actions.py`

- **メガリザードンY（2026-11-15、ユーザー指示「アクションがショボい。メガリザードンXのレベルに。今のSpriteの特徴や見た目は崩さない」）**: `tools/forge_gen1_0006_charizard_mega_y_actions.py`。**新しい原画は作らない**。素材は今までの8方向シート `0006_charizard-mega-y.png` の 0 コマ目（S / SW / W / NE / N、0.5 倍に面積平均、S の体高 96px。SE / E / NW は反転）。palette build/lock/apply（30色＋炎用3色）→ 部位リグ（翼2・頭・腕・胴・足2・尾＋尾の炎）→ 4動作×16コマ×8方向（セル 256×148、足元 132）。待機＝呼吸・羽ばたき・尾の揺れ・火の粉、歩き＝左右交互の足踏み、攻撃＝火炎放射（のけぞって口元にため → 踏み込んで口から炎）、ダメージ＝白フラッシュとのけぞり。公開は `gen5_publish.publish_dir8 / publish_bounds` でこのキーだけ（`v` = `0006_charizard_mega_y_rig_v1`）。古い登録内容は `../_backup_charizard_mega_y_20261115/old_entries.json`。古いシート png は残してある（素材なので消さない）。`monster_arena_2d` へは同期していない（`--no-sync` で実行）。
- **メガリザードンXの back.gif / back-reverse.gif（2026-11-15、ユーザー指示「back の動きを front.gif / front-reverse.gif にかなり近づける」）**: `tools/build_charizard_mega_x_back_gif.py --src <旧 back.gif> --apply`。旧 back.gif（20コマ・60ms、口から橙の火と火花が出る型）の 1 コマ目の絵だけを素材に、front.gif と同じ時間割（96コマ・100ms：18コマの羽ばたき×4 → 24コマの咆哮＝のけぞる → 踏み込む → 戻る）で動かす。**新しい絵は描かない・色は元の11色のまま・1 コマ目は元の絵と同じ**。体は部位（翼2・頭・尾・腕）の動きを重みをぼかしてつなぐ（部位を切り貼りすると関節でちぎれる）。炎2つは色で切り出した別レイヤー（体の重みに混ぜると隣の翼に引っぱられて割れる）。back-reverse は全コマの左右反転。キャンバスは 108→113×93（`data/sprite_bounds.js` の `g` 2項目を更新）。色違いの back GIF は触っていない。旧ファイルは `../_backup_charizard_mega_x_back_gif_20261115/`。
- **メガリザードンYの front / back GIF を新規作成（2026-11-15、ユーザーの直接指示「メガリザードンXのレベルの front.gif / front-reverse.gif / back.gif / back-reverse.gif を作成。Yの特徴は全て反映」）**: 2026-11-04 に「変な動き」で削除した GIF の作り直しではなく、`front.png` / `back.png`（96×96）の絵だけを素材にした新規作成。エンジンは `tools/gif_bone_warp.py`（やわらかい骨で曲げる汎用版。設定ファイルを渡す。使い方は先頭のコメント）、設定は `tools/charizard_mega_y_gif_front.py` / `tools/charizard_mega_y_gif_back.py`。実行は `<venv python> -I tools/gif_bone_warp.py <設定> --apply`（`--out DIR --sheet` で試作と確認用シート、`--grid` で方眼付きの原画）。96コマ・100ms、18コマの待機×4 → 24コマの咆哮（メガリザードンXと同じ時間割）。1 コマ目は原画と同一、色は原画の色だけ、`-reverse` は左右反転。3本の角・腕のひれ・大きな翼と青緑の膜・尾の炎は全コマで保つこと（頭は剛体で動かす）。色違いの GIF は作っていない。`data/sprite_bounds.js` の `g` 4項目を更新（更新前の控えは `../_backup_charizard_mega_y_20261115/sprite_bounds_before_gifs.js.bak`）。ゲーム内の表示は8方向アクションシートのままで、GIF は debug.html の下段などに出る。
- **キョダイマックスリザードン（2026-11-04、gen1再開の先行1体）**: `tools/forge_gen1_0006_charizard_gmax_actions.py`。メガライチュウXの生成構成を参考に、回収済みAntigravity CLI原画 `art/out/keys_2x2.png`（S/W/NE/N）と `sw_single.png` を使う。炎翼・尾・頭・腕・左右脚の専用リグ、agent-sprite-forge palette build/lock/apply、`forge_anatomical_actions.build_part_layer`、`mouth_open` を使用。待機・重量歩行・溜めから火炎攻撃・被弾の各16コマ、8方向。セル224×144、足元126、`v: charizard_gmax_agy_v1`。公式front/backは変更しない。
  - `agent-sprite-forge/` に成果物・`qa.json`、`_work/` に再開用フレームと詳細評価。`--force` で再生成、`--no-sync` / `--no-publish` で同期・revenge登録を個別に止められる。
  - `monster_arena_2d/monsterSprite/forge-manifest.js` で対象1体を登録。2dの`game/js/sprites.js`は4シート読込成功時だけForgeを使い、失敗時は従来アニメ一式へ戻す。`fighter.js`は再生中のシート自身の体高を使う。色違いは既存経路。manifest寸法は生成スクリプトで更新する。
  - QCはパレット外色・中間アルファ・セル端接触・静止差分を検査。`gait_loop.py` はidleでpingpong、walkで `no gait period`（8コマ反復の改善12%、判定基準15%）を報告するため自動周期判定合格とは扱わず、ループ境界と接地を別途確認する。2d成果物の`qa/browser-report.json`と`qa/game.png`に実ゲーム検証結果。
  - 実行: `<workspace>/agent-sprite-forge/.venv/bin/python -I tools/forge_gen1_0006_charizard_gmax_actions.py`。

- **キョダイマックス フシギバナ・カメックス 原画像版（2026-11-04、ユーザー指示「front/backからかけ離れているので作り直す」）**: 再生成は `tools/forge_gmax_original_actions.py --publish`（`--only venusaur` / `--only blastoise`）。元front/backの96px画像を直接部位リグ化し、Raichu Xと同じパレットロック・4倍最近傍変換・部位クッション・gait_loop評価を使う。旧生成原画の姿へ戻さない。
  - フシギバナ：花・冠・垂れ葉・頭・見えている脚を独立制御。歩きは対角位相の接地＋遊脚、攻撃は沈み込みから踏み込みと花びらの放出。カメックス：腹・頭・腕・脚・尾・甲羅を分離、砲口からの水流と甲羅の反動。全体平面回転なし。
  - **カメックスは現行front.pngが甲羅側、back.pngが腹側**。素材の内容に従い腹側をF、甲羅側をBに割当てる。原ファイルは変更しない。
  - 画像生成ツールはoutput moderationで拒否。生成原画は採用していない。**方向は前後2枚と左右反転による8行の互換配置であり、独立した8方向原画ではない**。正面・真横の新規絵を作ったと称さない。
  - セル224×160、足元128、4動作各16コマ、`v: original_rig_v2`。静止0コマは元画像と差分0、元パレット以外0、セル端接触0。フシギバナwalkはgait_loop cycle。カメックスwalkの自動判定はno loop windowのため合格と称さず、`qa.json`のexported_loopsと接地・目視で補足する。
  - `agent-sprite-forge/preview.gif` は4動作×前後2ビューの確認用。`actions/preview.png`にも確認画像を出す。`_original_v2/`に部位マスク、フレーム、パレット、gait評価を保存。`--publish`はrevengeのdir8/boundsと2dのforge-manifest対象項目・画像を更新。旧版バックアップは `../_backup_gmax_pair_actions_20261104_v1/`。

キョダイマックス フシギバナ・カメックスのAntigravity CLI再作成依頼（2026-11-04、ユーザーがCLIを明示）は未完了。各フォームの `art/agy_v3/` に原front/backコピー・prompt.txt・scoped-image.log・status.jsonを保存。両体の生成が429 RESOURCE_EXHAUSTEDで停止、解除予定2026-11-05 00:17:14 JST。新規画像・ゲーム差し替えは無し。再開は同フォルダ内で、権限省略オプションなしの `agy -p <prompt.txtの内容> --add-dir <同フォルダ絶対パス> --sandbox --print-timeout 1400s`。原画を見てから専用リグを作る。以前のoriginal_rig_v2を今回の完成品扱いにしない。

- **キョダイマックス フシギバナ・カメックス 原画像版v4（2026-11-04、ユーザー指示「Antigravity CLIを使用せずに」）**: `tools/forge_gmax_original_actions.py --publish`、`v: original_rig_v4`。今回はAntigravityは起動していない。組み込みimagegenで両体の新規原画がoutput moderationにより拒否されたため、元front/backの直接リグを改善。新規8方向原画は作っていない（前後2ビュー＋反転の8行互換）。AGY再開待ちはこの依頼に適用しない。
  - 16コマで1歩行周期（両脚2歩）、対角/左右逆位相の接地と遊脚。花びらは前方に螺旋で飛ぶ8枚、砲撃は溜めから発射、太さのある水流・泡・発射反動。腹の暗い節線を甲羅として切り出さず腹と一緒に保持し、白い甲羅縁を腹の下地へ伸ばさない。
  - 静止差分・パレット外・中間アルファ・セル端接触は0。`_original_v4/loop_input/`に実出力を3周繰り返してgait_loopを実行（16コマ周期を検出範囲に入れるため、ゲーム出力は16コマ）。最終qa.jsonを参照。プレビューはagent-sprite-forge/preview.gifとactions/preview.png。

【最新・ユーザー指定 2026-11-04】キョダイマックス フシギバナ／カメックスだけを8方向アクション採用前へ戻した。revenge data/dir8.js の2項目は _backup_gengar_mega_taunt_20261104/dir8.js の旧idle登録と同一、境界値は既存旧シートから再計測。2dのforge-manifest.jsから2体の上書き登録を削除して従来アニメへ復帰。生成ファイルは保管のみで不使用。上記の作成・再開メモは履歴であり、この2体のForge/8方向アクションを明示の再依頼なしに再採用しない。リザードンその他は変更しない。今回の直前状態は ../_backup_gmax_pair_before_rollback_20261104/ に保管。

- **キョダイマックスの GIF を削除（2026-11-04、ユーザー指示「変なエネルギーや顔が歪んで動く front.gif / front-reverse.gif / back.gif / back-reverse.gif などは全て削除」）**: `monsterSprite/**/gigantamax/<form>/` 直下の GIF 268 個（34 フォーム × 通常・色違い・反転）を削除。退避先は `../_backup_gmax_gifs_deleted_20261104/`（同じ相対パス）。`actions/`・`agent-sprite-forge/`・`art/` 内の GIF と、PNG（front/back の静止画・8方向シート）は残した。キョダイマックスの GIF を作り直したり戻したりしないこと（ユーザーの直接指示がある場合を除く）。
  - debug.html：「front.gif / front-reverse.gif / back.gif / back-reverse.gif」欄と、8方向シートが無いときの8方向欄は、GIF が無ければ `No Image` の枠（`.noimg`）を出す（`game/js/debug.js` の `noImage()`）。また、`#strip` の**8方向スプライト欄・アクションスプライト欄ともに**、×1/×2/×3 指定時を除き `gifSizedBox()` で4種 GIF（無ければ同名 PNG、それも無ければ `96×96`）と同じ枠サイズに揃える（ステージ高さ基準で巨大化させない）。
  - ゲーム（`game/js/assets.js` の `pngFallback()`、debug の file:// 用 `fileGet()`）：GIF が無いときは同名の `front.png` / `back.png` の静止画を使い、`-reverse` はそれを左右反転する。PNG も無ければ従来どおり failed。

- **メガ進化の GIF も削除（2026-11-04、ユーザー指示「メガ進化でキョダイマックスと同じような動きをする front.gif / front-reverse.gif / back.gif / back-reverse.gif は全て削除」）**: 見分け方は「front.gif が 19〜20 コマ」。この型はキョダイマックスと同じで、口からエネルギーが出て顔がゆがむ。14〜16 コマや 40 コマ以上のメガ GIF は普通の動きなので残した。対象 58 フォームの GIF 388 個（通常・色違い・反転）を `../_backup_mega_gifs_deleted_20261104/` へ退避して削除。例外：ミュウツーYの back 系（22 コマ、しっぽが揺れるだけ）は残した。メガラティアス（v3、1コマ）・メガライチュウX/Y は対象外。
  - debug.html の No Image 枠と、ゲームの PNG 代替（上のキョダイマックスの項）がそのまま効く。対象フォームはすべて front.png / back.png がある。
  - 削除したメガ GIF を作り直したり戻したりしないこと（ユーザーの直接指示がある場合を除く）。

- **通常・リージョンの GIF も削除（2026-11-04、ユーザー指示「リージョンと通常もキョダイマックスと同じような動きをする front.gif / front-reverse.gif / back.gif / back-reverse.gif は全て削除」）**: 正確な見分け方はコマの表示時間。「19 コマで3コマ目だけ 130ms、他は 60ms」か「20 コマすべて 60ms」の GIF がこの型（口からエネルギーが出て顔がゆがむ）。キョダイマックス・メガで消したものも同じ。対象は 188 フォーム（通常 166・リージョン 22）、GIF 1,500 個。退避先は `../_backup_normal_regional_gifs_deleted_20261104/`。
  - 残したもの：プテラの back 系4つ（17〜24 コマ、別の型）。19〜20 コマでも表示時間が違うオニドリル・ゲッコウガ・アゴジムシは普通の動きなので残した。
  - debug.html の No Image 枠とゲームの PNG 代替がそのまま効く。対象フォームはすべて front.png / back.png がある。
  - 削除した GIF を作り直したり戻したりしないこと（ユーザーの直接指示がある場合を除く）。

- **サンダー（`0145_zapdos`）の GIF をユーザー提供のものに差し替え（2026-11-04、ユーザー指示）**: `~/Desktop/zapdos-black-white-animated.gif` → `front.gif`、`~/Desktop/zapdos-black-white-back-animated.gif` → `back.gif`（どちらも 9 コマ・100ms、そのままコピー）。`front-reverse.gif` / `back-reverse.gif` は各コマを左右反転して作った（既存の `-reverse` も正面の左右反転なので同じ規則）。`data/sprite_bounds.js` の `g` の4項目を1コマ目の実測値に更新。色違いの GIF は提供が無いので作っていない（ゲームは `front-shiny.png` / `back-shiny.png` の静止画を使う）。8方向シート（`0145_zapdos.png`）は変更していない。

---

## ゲーム体験・演出ブラッシュアップのノウハウ（2026-11-11：fable 5.2 ＋ BigKidNation ＋ Ivandanke_Rivals 反映）

`fable 5.2` のゲーム制作・アクション手触りの設計ノウハウをベースに、`参考用データ/BigKidNation_game_videos`（全12動画）および `参考用データ/Ivandanke_Rivals`（2動画）の優れた体験要素を統合した設計・実装仕様。今後の改修時も以下の原則・構造を崩さないこと。

### 1. スケール感とカメラワーク（黄金比ワイドコート × ここぞのドラマチックズーム）
- **黄金比スタジアム縮尺（`game/js/assets.js`）**:
  - 基準物差しは人間のトレーナー身長 `TRAINER_M = 1.7m` ＝ `TRAINER_PX = 48px`（メニュー・ポートレート・攻撃リーチ計算用）。通常モンスターは体高＋体重バルク補正（100kg超で太さ・存在感増）により `20〜112px`、キョダイマックスは原種の `2.6倍（120〜200px）`。
  - バトルフィールド上の描画には **`FIELD_DRAW = 0.45`（`Assets.onFieldPx`：1.7m ≈ 22px、最小13px）** を適用する。これにより BigKidNation 動画のように「広いコート・センターサークル・観客席・永続クレーター群」が一画面に気持ちよく収まる俯瞰スケールを維持する。
  - **【最新・ユーザー指定 2026-11-15】大きさは data の height を大きく反映、ドットは GIF の細かさ**（上の数値より優先）:
    - 大きさ（`Assets.bodyArtPx`、2026-11-15 ユーザー指定の最新調整値）：`1.7m` 以上は `48px × (身長/1.7m) × 体重バルク`（上限 `107px`）、`1.7m` 未満は線形縮小 `48px × (身長/1.7m)`（下限 `BODY_MIN = 12px`）、キョダイマックスは大きく **`195〜268px`**（フィールド上 `146〜201px`）。フィールド上は **`FIELD_DRAW = 0.75` 倍（最小 `10px`）**。例：キャタピー `0.3m`・ピカチュウ `0.4m` ＝ `10px`、フシギダネ `0.7m` ＝ `15px`、ルカリオ `1.2m` ＝ `26px`、リザードン `1.7m` ＝ `36px`、カビゴン ＝ `68px`、イワーク ＝ `80px`（上限）、キョダイマックスリザードン `28m` ＝ `179px`（`snorlax-gmax` `201px`）。トレーナーはモンスター基準に引きずられて巨大化しないようフィールド上 **`18px`（`Renderer.trainerFieldPx`: `onFieldPx(trainerArtPx()) * 0.50`）**、画面外ワイプ枠 **`trainerWipes` `faceH` 最大 `58*dpr`（モバイル `30*dpr`）** とし、手元位置（`Battle.trainerHandPos`）から回収レーザー・回転トレイル付きボール投げ・着地展開バーストを行う（`audio.js` / `fx.js` / `fighter.js` / `battle.js`）。`FIELD_DRAW = 0.45` や最小 `9px` へ戻さない。
    - **【最新・ユーザー指定 2026-11-15 夜】トレーナーもモンスターも同じ比率で 2 割小さく（上の行のフィールド上の数値より優先）**: `FIELD_DRAW = 0.6`（最小 `8px`）。フィールド上はトレーナー・リザードン `1.7m` ＝ `29px`、ピカチュウ `8px`、カビゴン `46px`、イワーク `64px`。トレーナーだけ別の倍率にしない（1.7m の物差しのまま）。ドットを欠けさせないため、(1) カメラ倍率は縮小前の体の大きさ（`rawOld`）で決めてフィールドの映り方を変えず、GIF の 1 ドットが画面の 1 画素を下回らない下限を持つ（`render.js` `updateCamera`）、(2) スプライト用キャンバスの密度 `SPR` を体の縮小率に合わせて 4〜6 まで上げる（`sprNeed`）。このため小さいモンスター（ピカチュウ等）を追っているときは、これ以上画面上で小さくならない。`FIELD_DRAW = 0.75` へ戻さない。
    - **【さらに最新・ユーザー指定 2026-11-15 夜】トレーナーは少し小さく、リザードン以上は少し大きく（上の行の数値より優先）**: トレーナーは 1.7m の物差し（29px）の `TRAINER_DRAW = 0.72` 倍 ＝ フィールド上 `21px`（0.85 → 0.72 に再調整）（`render.js` `trainerBodyPx`）。`1.7m` 以上のモンスターは `BIG_BOOST = 1.25` 倍（1.12 → 1.25 に再調整、`BODY_MAX = 134`、リザードン `36px`・カビゴン `58px`・イワーク `80px`。以下の括弧内の旧数値より優先）（`assets.js` `bigBoost`。`1.4m`→`1.7m` でなだらかに上げる。上限 `BODY_MAX = 120`。キョダイマックスには掛けない）。フィールド上はリザードン `32px`、カビゴン `52px`、イワーク `72px`。`1.4m` 以下は変えない。カメラは上乗せぶんを打ち消さない（`updateCamera` の `nat`）。コートの広さの基準（1倍）は 1.7m のモンスター（`Battle.fieldScaleFor` の `ref`）。`Assets.trainerArtPx()` は 48 の定数。
    - **トレーナーの頭だけ小さく（2026-11-15、ユーザー指示「顔だけがデカすぎる」）**: フィールド上のトレーナーは、首から上だけ `TRAINER_HEAD = 0.72` 倍に縮めて描く（`render.js` `drawTrainers`、首の行は `trainerNeck` が立ち絵の幅から自動で決める。体はそのまま、頭は首の位置に下端をそろえる）。画面端のワイプ（顔のアップ）は元の絵のまま。
    - 移動速度：`Fighter.moveRate(p)` ＝ `MOVE_RATE 0.78` ＋ 体重に応じた上乗せ（10kg 以下 +0.10、300kg 以上 +0.03、間は対数補間。ピカチュウ 0.88、リザードン 0.83、カビゴン 0.81）。ユーザー指示「もう少し遅く」→「weight に応じた按配でもう少しだけ速く」による。
    - ドット：フィールド上のモンスターは GIF の元のドットをそのまま描く（`Assets.setFrames` は pixelize しない。22色への減色・2:1 の粗いドット化は「荒すぎる」と却下）。GIF 由来・raw の8方向シートもそのまま、手作りシートだけ体高 56〜96px 相当へ pixelize（`Assets.sheetScale`）。
    - 描画（`render.js` / `post.js`）：スプライト用キャンバスは「いま画面に映る範囲」だけを `SPR` 倍（1 art px あたり 4、引くと 3 → 2）で持つ（`sprWindow()`、原点 `sox/soy`、`sprT()` が基準変換）。シェーダは `uSprOff` / `uSprK` で読み、乗算済みアルファ、縮小時は線形・拡大時はシャープバイリニア。スプライト用キャンバスへ描くコードは `setTransform` を直接呼ばず `sprT()` を使うこと。
  - **【最新・ユーザー指定 2026-11-15】基準ズームは「GIF 等倍」、画面外のモンスターはワイプ**（下の `zoomMul ≈ 0.99〜1.06` のワイド固定・既定 161% / 180% より優先）:
    - 基準ズーム（`updateCamera` の `natZ`）＝追っているモンスターの元のドット（front.gif の解像度）が画面の 1 ドット（CSS px）になる倍率 `dpr / (f.fit · S)`。小さいモンスターには寄り、大きいモンスターからは引く。観戦で主役がいないときは両者の相乗平均。演出ズーム（強打・競り合い・登場・K.O.）とプレイヤー倍率 `cam.user` はその上に掛かる。フィールドが荒く見えないよう、等倍のための寄りは「1 art px = 3 CSS px」まで、引きはアリーナが画面を覆う `baseZ` まで（大型は GIF より大きく映る）。
    - `cam.user` の既定はバトルも debug.html も `100%`（＝GIF 等倍）、範囲 `0.4〜3`、保存先は `arena.camZoom.v3` / `arena.camZoom.debug.v2`。
    - **操作モンスターは常に画面の中心（2026-11-15、ユーザー指示）**: `updateCamera` の `lock`。自分が操作（指示）しているモンスターの体の中心をカメラの中心に置き、相手側への先回り・強打／競り合い／登場／K.O. での寄り道はしない（演出はズームだけ）。立ち回り中の倍率も動かさず GIF 等倍を保つ。フィールドの端でも中心を保つため、`cam.lock` の間は `computeView` のアリーナ内クランプを外す（外側は暗色）。操作モンスターがいない観戦は従来どおり中間追従＋クランプ。
    - 画面外のモンスターは `monsterWipes()`（`render.js`）が、いる方向の画面端に「いまの姿・HP・名前・方向の矢印」のワイプを出す（トレーナーのワイプと重なれば上下によける）。以前の四角い画面外マーカーは廃止。
  - **フィールドの広さはモンスターの大きさで決まる（2026-11-15、ユーザー指示）**: `Battle.fieldScaleFor(teams)`（`battle.js`）→ `Arenas.create(id, seed, scale)`（`arena.js`）。出場する全モンスターのうち、いちばん大きい体（フィールド上の高さ）をトレーナー（1.7m、常に同じ大きさ）と比べ、`0.78〜1.75` 倍にする（大きい側は 0.8 乗、小さい側は 0.5 乗）。**キョダイマックスは数えない（そのためには広げない）。ただしキョダイマックスがいるときは標準（1倍）より狭くしない。** 試合中は変えない（開始時の顔ぶれで決める）。メニューのサムネイルとタイトルは 1 倍。トレーナーの身長は常に一定（`onFieldPx(trainerArtPx())`）に保つこと。
  - **だいもんじ（2026-11-15、ユーザー指示「参考動画 sample6_サイトのミニバトル.mp4 のだいもんじのように、最高品質の VFX で」）**: `fire-blast` は strike のまま（r 130・delay 0.5、当たり判定は変えていない）、`sky: "daimonji"` で専用演出。`game/js/vfx.js` の `VFX.fireBlast`（空中レイヤーの描画）と `VFX.fireBlastTick`（光・火の粉・煙・画面フラッシュ）。流れ：口から太い炎の帯が火球を先頭に着弾点へ走る（delay の間）→ 白いギザギザの閃光 → 転がる火の爆発＋「大」の字の炎（上が短く、左右のはらいが長い 5 本）→ 黒い煙と黄色い十字の火花。帯は着弾後、口側から燃え尽きる。汎用の `fx.blast` / 輪は daimonji では出さない。ほかの炎 strike（既定 `sky: "blast"`）は従来のまま。 **同日の追加指示「大の字の炎をもう少し大きい炎に、大の字は少し小さく」**: 字の広がりを `a.r * 0.8` → `a.r * 0.66` に縮め、腕の炎の玉を太く（半径 13.5→19、1 本 9→12 個）、腕から立つ炎の舌を多く・高くした。上の画は埋もれないよう 0.5→0.72 に伸ばしてある。
  - **デバッグ画面の技（2026-11-15、ユーザー指示）**: `game/js/debug.js` `renderMoves` は、そうび・おぼえる技・ほじょ技に加えて、そのモンスターのタイプごとに「◯◯タイプの技 ぜんぶ」（全技データ＋ `dig` / `dive` / `whirlpool` から同タイプを威力順）を出す。おぼえない技でも同タイプなら全部ためせること。
  - **デバッグ画面のトレーナー（2026-11-15、ユーザー指示）**: `game/debug.html` の `#tr0`（自分側）/ `#tr1`（相手側）で `data/trainers.js` の全トレーナーをえらべる（先頭は「ランダム」）。既定は自分＝レッド（`trainers/hgss_0211.png`）、相手＝ヒビキ（`trainers/hgss_0000.png`）。選択は `localStorage` の `dbg.tr0` / `dbg.tr1`（画像パス）に保存。変更時はバトルを作り直さず `battle.trainers[side]` / `trainerRecs[side]` だけ差し替える（`game/js/debug.js` の `pickTrainers()`）。
- **カメラ制御のメリハリ（`game/js/render.js` の `updateCamera`）**:
  - 通常の立ち回り中は頻繁なズーム加減で画面酔いさせないよう、ワイドな俯瞰ショット（`zoomMul ≈ 0.99〜1.06`）を維持する。
  - **ダイナミックなズームインは以下の4局面に限定**して発動し、緩急をつける：
    1. **強打・急所ヒット時**（`B.camFocus`：約 `1.22〜1.25x` ＋ `fx.zoomPunch`）
    2. **ビーム競り合い中**（`B.activeClash`：`1.28x`、拮抗座標へフォーカス）
    3. **登場・注目演出時**（`B.focus`：`1.36x`）
    4. **K.O. 決着の瞬間**（`B.koCam`：`1.38x`）
  - **モンスター追従（2026-11-14、ユーザー指示）**: カメラは操作中／指示中のモンスター（`playerSide`、無ければ `commandSide`）を追い、相手方向へ最大110ワールド単位（距離の3割まで）先回りする。操作モンスターがいない観戦は両者の中間を追う。コート中央寄せ（旧 `lerp(中央, 中間, 0.22)`）には戻さない。`debug.html` も同じ `Renderer.updateCamera` を使い、表示中のモンスター（side 0）に追従・同じズーム操作。既定は debug だけ 180%（`Renderer.zoomProfile("arena.camZoom.debug", 1.8)`、保存先もバトルと別）（`#camzoom`、ホイール、`−` `＝` `0`）を持つ（2026-11-14、ユーザー指示）。
  - **プレイヤーのズーム倍率 `cam.user`（2026-11-14、ユーザー指示「もっとカメラを引きたい／寄せたい」「161% を default に」）**: `0.55〜2.2`、既定 `1.61`（`USER_DEFAULT`、`0` キーもここへ戻す）。上の演出ズームすべてに掛け算で乗る。操作はバトル中のマウスホイール、`−` / `＝` / `0`（標準に戻す）キー、HUD 上部の `− 100% ＋` ボタン（`#cam-zoom`、`main.js`）。`localStorage["arena.camZoom.v2"]` に保存、URL `?zoom=0.7` でも指定できる。
  - 画面ズームが 1 未満になるときは、低解像度バッファを `1/BZ` 倍に広げて描く（`render.js` の `BZ` / `alloc()`、シェーダの `uPad`）。バッファ基準の座標変換（`computeView` / `artToScreen` / `toWorld` / `fallback`）は画面中心を `W/(2·BZ)` として扱うこと。アリーナより広く引いた分は外側が暗色になる。

### 2. 「棒立ち撃ち」をなくす3段階のゲームフィール（Stage 1〜3）
1. **Stage 1：予備動作と「見てから回避できる」テレグラフ（`fighter.js` / `moves.js` / `battle.js`）**:
   - 技の溜め（wind-up）前半 **65%** までは相手の位置を自動追尾し、発射直前の残り **35%** で照準角度をロックする（`fighter.js`）。
   - 同時に地面レイヤーへ危険範囲（`Moves.drawWindups` / `Moves.windupDanger`：照射ライン・扇形・着弾円・魔法陣）を描画し、プレイヤーもCPUも「予告を見てから横移動や無敵回避（Space）でかわす」駆け引きを成立させる。
2. **Stage 2：高速移動・踏み込み時の色付きシルエット残像（`fighter.js` / `render.js`）**:
   - 回避ダッシュ・突進・跳躍・近接踏み込み中、過去フレームの座標・ポーズを `f.ghosts` にサンプリングして色付き半透明シルエット残像を描画する。
   - `actions_8dir` シートのモンスターはシート自体に構え・伸縮が描かれているため、残像および本体に余計な `squash` 変形を二重掛けしない（`render.js`）。
3. **Stage 3：着弾の重み・Juiciness（`battle.js` の `hit`）**:
   - 威力・ダメージ割合に応じた段階的ヒットストップ（基準値：弱 `0.036s` / 中 `0.052s` / 強・急所 `0.072s`。2026-11-14 のユーザー指示「もっと重く」により、`fx.js` の `FX.HITSTOP_MUL = 1.9` が `fx.stop()` 全体に掛かり、実際は約 `0.068s / 0.099s / 0.137s`、K.O. は `0.27s`、上限 `0.42s`。重さの調整はこの倍率で行う。URL `?hitstop=倍率` で試せる）、純白シルエットフラッシュ（`f.flash`）、ノックバック土煙、放射ブラー・単発ストロボを入れる。
   - **へこみ・水面の歩行（2026-11-14、ユーザー指示「埋もれた地面／水面を歩くのが不自然」）**: へこみ（`f.sinkG`、`fighter.js`）は足元5点の平均高さをなめらかに追い、体の大きさで上限（体高の14%・最大5px）を決め、体と影を床と一緒に下げるだけにする。体を地面の線で水平に切らない（切るのは `digSink` のもぐる演出だけ）。水（`f.wade` 0〜1）は入水・上陸で水位が徐々に変わり、体は水面に対して上下に揺れ（歩行中はストローク、静止中はゆっくり）、切り口は水面リングの手前側カーブに沿わせる。
   - **接地と浮遊（2026-11-15、ユーザー指示「モンスターが不自然に浮いて見える。全モンスターが地上を歩く様に。鳥型や蛇の様なドラゴンは浮いたままで、もう少し浮遊感を」）**: 浮くのは `Fighter.canFly`（`fighter.js`）= 空中にいられる種（ひこうタイプ・ふゆう・`FLOATERS`、`Fighter.airborne`）のうち体型が `HOVER_SHAPES`（`wings` 鳥型／`squiggle` 蛇型／`bug-wings` 羽根のある虫型＝同日ユーザー追加）のものだけ。それ以外（リザードン・ミュウツー・ゲンガー等）は地上の歩き方（体型で決まる gait）になり、水では泳ぎ、溶岩・氷の影響も受ける。浮く高さは `Fighter.hoverPx`（体高の11%、4〜8px。以前は一律1.5〜2px）で、ゆっくり上下し、倒れると地面に落ちる。影（`render.js` の `shadow()`）は、地上＝濃い接地影＋足の真下の芯、浮遊＝地面に残り体が高いほど小さく薄い影。浮かせる種を増減するときは `HOVER_SHAPES` を変える。羽根のある虫型（`bug-wings`）は、ひこうタイプでなくても浮く（スピアー・モルフォン・ウルガモス等）。歩く虫 `BUG_WALKERS`（ハッサム・コロトック・バサギリ）だけ地上。さらに同日ユーザー指定で、鳥でも地上を走る・歩く種は `GROUND_BIRDS`（ガラルサンダー、ドードー系、カモネギ系、デリバード、アーケン、オドリドリ、カラミンゴ）で地上に、脚のない浮遊体は `HOVER_SLUGS`（ゴース・ゴースト、コイル系、ドガース系）で体型に関係なく浮かせる。ゲンガーは脚があるので地上。
   - 重量級（`f.mass >= 2.2`）は軽い攻撃ではひるまないスーパーアーマーを持ち、歩行時や着地時に地響きを起こす。

### 2.5 技の VFX（`game/js/vfx.js`、2026-11-15、ユーザー指示「かえんほうしゃ・なみのり・れいとうビームなどがダサい。参考動画を参考に最高品質の VFX をレトロな HD-2D で」）
- 参考：`参考用データ/BigKidNation_game_videos`（sample10 の火炎放射＝色の帯がはっきりした炎の塊＋地面の照り返し、sample5 の波乗り＝白い波頭の大波、氷＝氷塊）と `参考用データ/Ivandanke_Rivals`。
- **水の技（なみのり・うずしお）は「1ドットずつ計算する水面」で描く（2026-11-15、ユーザー指示「うずしおがとてつもなくダサい。レトロだけど水の質感を綺麗に表現できている最高品質の HD-2D で。なみのりも同じレベルに」）**: `vfx.js` の `field()`。水面の高さ場（うねり・波の壁・渦のすり鉢）を奥から手前へ、手前向きの面ごと描いて立体に見せ、深さの濃淡・光の網目（`caustic`）・流れの筋（`vnoise`）・泡・きらめきを 8 色の水色パレット（`PAL`）へベイヤーディザで落とす。world キャンバスに置く（モンスターの下。アリーナのライトとブルームが乗るので、パレットは見せたい色より一段深く作ってある。水の技から `fx.light` を足すと白飛びするので足さない）。線・円・破線を重ねた旧表現（平たい、数珠に見える）へ戻さない。
  - なみのり `surfWorld`：水が範囲に広がり、中心から波が4本、外へ行くほど高くなる。波は外側が切り立ち内側がなだらかな断面で、頂に泡の帯、後ろに泡の尾。うずしお `whirlWorld`：対数らせんの畝が中心ほど速く回り、外周が盛り上がって中心は暗い目へ落ち込む。畝の頂に泡。
  - 計算量は 1 フレーム約 6ms（半径 250〜290）。これ以上重い式を足すときは `field()` の極座標の事前計算などで軽くしてから。
- 作り方の原則（炎・氷・水流のビーム）：fx キャンバス（アート解像度）に**暗い輪郭 → 濃色 → 中間色 → 明色 → 白**の順で「全部の塊を同じ色ごとに重ねる」。こうすると塊がつながって1つのドット絵に見える。グラデーションや加算の発光で形を作らない（発光は `fx.light` とブルームに任せる）。
- `VFX.flame`（`cone` のビーム：かえんほうしゃ・りゅうのいぶき）＝転がる炎の塊、立ち上る炎の舌、火の粉、先端の煙。`VFX.iceBeam`（こおりタイプのビーム）＝ぎざぎざの縁の氷の槍、まわりを回る氷片、きらめき、先端の結晶バースト。`VFX.waterBeam`（みずタイプのビーム）＝泡の縁と白い筋の水流、先端の水しぶきの冠。なみのり（`wave:true` の範囲技、同日ユーザーの再指示「範囲内に水が広がって、キャラクターの中心から波が徐々に大きくなっていく」）＝①足元から水柱が上がり、水が範囲いっぱいに広がる（`VFX.drawWorld`：world キャンバスに描くのでモンスターの下に敷かれる。水面は濃淡の短い筋が外へ流れるドット絵）②中心から波が4本、順に生まれ、外へ進むほど高くなり、後の波ほど大きい（`VFX.wave` / `crest`：4段の色の水の壁＋途切れない泡の帯＋手前へ倒れるカール）③縁で砕けて水しぶき・霧・画面の揺れ、水が引く。技の長さは `dur: 1.9`、当たり判定は1本目の波の位置（`VFX.waveHitR`）。輪が1つ広がるだけの旧表現や、泡の玉を等間隔に並べた波頭（数珠に見える）へ戻さない。水の色は薄い水色（`#4aa0ea`〜`#b4e4ff`。濃い青はユーザーが却下）。水面は明るい色なので、ライトは技全体で弱い1灯だけにする（毎フレーム `fx.light` を重ねると白飛びする）。
- うずしお（`whirlpool`、同日ユーザー指示「水がキャラクターの中心にらせん状に渦を巻くように集まり、敵にダメージを与えて自分に引き寄せる」）＝`kind:"spiral"` ＋ `pull`（引き寄せの速さ）。使用者を中心に半径 `r1: 250`、`dur: 2.6`。範囲内の相手を中心へ引き寄せ（流れに沿って少し回り込む、使用者に触れる距離で止まる）、0.42 秒ごとにダメージ。見た目は `VFX.drawWorld` → `whirlWorld`（地面の水面にらせんの濃淡の流れ・内へ流れる泡の筋・中心の目）と `VFX.whirlAir`（水面上を回る水しぶきの弧、使用者を取り巻いて上る水の輪。輪は手前半分だけ描いて使用者を隠さない）。技データに無かったので `moves.js` の `EXTRA` に登録し、`data.js` の `WHIRLERS` でギャラドス・タッツー系・メノクラゲ系・ニョロゾ系に配った。以前の「相手の足元に置く罠（zone）」には戻さない。
- つなぎ込み：`moves.js` の `drawBeam` は `VFX.styleOf(s)` があればそれで描く。更新側は `VFX.beamTick` / `VFX.waveTick` が粒子・ライトを出し、汎用の `fx.trail` / `fx.burst`（黒っぽい煙が出て汚く見えた）は出さない。それ以外のビーム（でんき・エスパー・ノーマルなど）は従来の `drawBeam` のまま。新しい技の見た目を足すときは `vfx.js` に関数を足して `styleOf` で振り分ける。

### 3. ビーム＆飛び道具の正面衝突「⚡ BEAM CLASH!」（`Ivandanke_Rivals` 反映）
- **ビーム同士の競り合い（`game/js/moves.js` / `arena.js` / `render.js` / `hud.js`）**:
  - 両者のビーム系技が交差・正面衝突すると `B.clashes`（`B.activeClash`）が生成され、互いの照射終端が衝突点（`clashEnd`）でロックされる。
  - 拮抗点では激しい脈動コア（`drawClashNexus`）・動的ポイントライト・放射状の地面衝撃痕（`Arenas.decal(..., "clash")`、氷上では氷を融解して蒸気化）が発生し、HUD中央上部に **`⚡ BEAM CLASH!`（`#clash-tag`）** がポップする。入力連打（`mashT`）や威力差で拮抗点を押し返せる（`clashPush`）。
- **飛び道具の空中相殺・対ビーム干渉**:
  - 弾同士の正面衝突による空中爆発相殺、および弾が敵ビームにぶつかった際の押し返し（`clashPush`）も `Moves.update` で処理する。

### 4. 競技仕様の戦術AI（`game/js/ai.js`）
- **ロール自動分類と間合い管理**:
  - 所持技と種族値から `zoner`（距離 `240〜300` で引き撃ち、不用意に突進しない）・`rusher`（距離 `65〜95` へ踏み込み、クールダウン中は一度離脱）・`allrounder` に分類して立ち回る。
- **予告回避・差し返し（Whiff-punish）・カウンタービーム**:
  - `B.dangerAt` と `Moves.windupDanger` をサンプリングして安全な退避ベクトル（`pickSafeVector`）へ移動し、直撃寸前の弾・突進・予告範囲には直角方向へ無敵回避（i-frame dodge）を切る。
  - 相手の大技後の硬直（`recoverT`）を検知して突進・近接・狙撃の優先度を跳ね上げる**差し返し（`punish`）**と、相手がビームを構えた際にこちらもビームで撃ち返して **Beam Clash** を起こすカウンター思考を持つ。

### 5. eスポーツ中継風HUD・K.O.カットイン・永続フィールド破壊・観戦介入（`BigKidNation` ＋ `Ivandanke_Rivals` 反映）
- **2種類のHUDスキン切替（`index.html` / `style.css` / `hud.js` / `menu.js`）**:
  - デフォルトは **eスポーツ中継スキン（`#hud.esports`）**。選択画面の `HUD` 設定（`#opt-hud`）、試合中の上部ボタン（`#hud-skin-btn`）、キーボードの `H` キー、URLクエリ `?hud=esports|retro` で従来の **レトロDQスキン** と即時切替できる。
  - `esports` スキンは斜めカット（`clip-path` / `skewX(±16deg)`）の赤・青コーナーパネル、OKLab色空間グラデーションフォールバック（`--in-oklab`）、遅延減少ゴーストバー付きHPゲージ、`ENERGY` バー、6体の手持ちボールアイコン、`conic-gradient` 扇形クールダウン、英日併記のコンボ表示（`Move ×N`）を備える。
- **大型 K.O. カットイン（`#ko-cutin` / `battle.js` の `knockOut`）**:
  - 撃破時は `0.14s` ヒットストップ ＋ スローモーション（`0.9s`）＋ `koCam` ズーム ＋ クレーター生成に加え、画面を斜めに横断する大型スラッシュ帯 `KNOCKOUT!! / K.O.!` カットイン（決着時は `.finish` 強調）を再生する。
- **頭上2段ミニバー・トレーナーワイプ・観戦コマンド介入・タッチ最適化（`render.js` / `battle.js` / `main.js` / `style.css`）**:
  - フィールド上の各モンスター頭上に「HP＋ENERGY＋赤青コーナー色＋操作中マーカー」の2段ミニバーを描画し、視線移動なしで戦況とエナジー残量を把握できるようにする。
  - サイドラインのトレーナーがカメラ枠外に出た際は、画面左右端に顔・名前入りのワイプ枠（`trainerWipes`）を自動表示する。
  - **観戦モード（`mode === "watch"`）での介入**: 観戦中でも片方のコーナー（初期は左＝赤コーナー `commandSide = 0`）へ技・回避・進化の手動指示を乗せられる。フィールド上のトレーナーまたは左右ワイプ枠をタップすると指示対象コーナーを赤⇄青で切り替えられ（`Renderer.trainerAt` → `battle.setCommandSide`）、`M` キーで「通常 → メガ → キョダイ → 通常」の姿・技サイクル（`battle.cycleForm`）も試せる。
  - `body.touch` および小画面メディアクエリにより、スマホ・タブレットでもHUD・コンボ・相手技カードが重ならず収まるレスポンシブ構造を維持する。
- **GIF があるモンスターは全て GIF で動かす（2026-11-15、ユーザー指示「バトルや debug.html で静止画のモンスターが多い。基本的に front.gif や back.gif を採用して、バトルで動く様に」）**: 原因は、gen2・gen5（一部）・gen6〜9 の 397 フォームに 8 方向シートが無く、debug.html をファイル直開き（file://）したとき GIF を解読できず 1 コマ目で止まっていたこと。`tools/build_dir8_from_gifs.py --gen <gen2|gen5|gen6|gen7|gen8|gen9> --apply` で、gen1・gen3 と同じ「4 つの GIF から作る 8 方向シート」（`src:"gif"`、コマ数・速さは GIF と同じ）を 397 フォームぶん作り、`data/dir8.js` / `data/sprite_bounds.js` に登録した。作業前のデータは `../_backup_dir8_before_gif_rebuild_20261115/before_gen2_5to9/`。`actions_8dir`（アクションシート）のモンスターは動いているので触っていない。**まだ静止画のもの**: 4 種の GIF が揃っていない 215 フォーム（GIF が 1 つも無い 208、front.gif だけの 7）。この大半は 2026-11-04 のユーザー指示で GIF を削除したもの・元から GIF が無い gen9 などで、作り直し・復元はユーザーの直接指示が要る。
- **gen9 の 8 方向を全て定義（2026-11-15、ユーザー指示「gen9 の 8 方向の定義ができてないものを、定義されているものを参考に gen9 だけ定義」）**: GIF が無く `front.png` / `back.png` だけの gen9 の 101 フォーム（通常 95・メガ 6）に、`tools/build_dir8_from_gifs.py --gen gen9 --png-fallback --apply` で 8 方向シート（`<id>.png`、1 コマ）を作り `data/dir8.js`（`src:"png"`, `raw:1`, `v:"png8d1"`）と `data/sprite_bounds.js` に登録。向きの作り方は GIF 由来シートと同じ（南西＝front、南東＝front の反転、北東＝back、北西＝back の反転、南・北＝横幅 80%、東・西＝横幅 90%＋前傾）。**静止画なので動かない**（待機アニメは無い）。`--png-fallback` は「4 種の GIF が 1 つも無く、dir8 の項目も無い」フォームだけが対象。gen9 以外には実行していない。作業前のデータは `../_backup_dir8_before_gif_rebuild_20261115/before_gen9_png/`。
- **gen8 の 8 方向を全て定義＋gen9 を GIF 版へ更新（2026-11-15、ユーザー指示「gen8 の 8 方向の定義ができてないものを、定義されているものを参考に gen8 だけ定義」）**: 作業時点で、別の作業により `monsterSprite/` 全世代へ GIF が追加されていた（2026-11-15 19:35〜19:40 の更新。退避は `../_backup_monsterSprite_gifs_before_showdown_20261115/`）。gen8 の未定義 49 フォームを `tools/build_dir8_from_gifs.py --gen gen8 --png-fallback --apply` で定義：4 種の GIF が揃う 44 は GIF から（動く）、`back.gif` が無い 5（キョダイゴリランダー・インテレオン・ウーラオス 2 種、メガタイレーツ）は、GIF がある向きは GIF・無い向きは `back.png` の静止画で埋めた（`v:"gifpng1"`、`still` に静止画の向き）。`--png-fallback` は「GIF が欠けた向きだけ静止画で埋める」に一般化（対象は dir8 の項目が無いか `src:"png"` のフォーム）。**gen9 も同時に更新**：直前に静止画で定義した 101 のうち 87 は GIF が揃ったので GIF から作り直した（静止画シートのままだと新しい GIF がバトルで使われないため）。静止画のまま残るのは gen9 の 14（GIF 無し 12、front だけ 2）。作業前のデータは `../_backup_dir8_before_gif_rebuild_20261115/before_gen8_define/`。
- **フーパ（ときはなたれしすがた）を追加（2026-11-15、ユーザー指示「pokestats.gg/sprites/hoopa-unbound が取り入れられてないので、front.gif / front-reverse.gif / back.gif / back-reverse.gif を Pixel, Black / White style で取り入れて 8 方向を用意」）**: `tools/add_hoopa_unbound.py --src <素材フォルダ> --apply` → `tools/build_dir8_from_gifs.py --gen gen6 --only 0720_hoopa-unbound --apply`。置き場所は `monsterSprite/gen6/0720_hoopa/form/hoopa-unbound/`（通常・メガ・キョダイ・リージョン以外の「別のすがた」は初めてなので `form/` を新設）。`front.gif` は pokestats の Black / White スタイル（`black-white/animated/10086.gif`、105×103・120 コマ・10 秒）をそのまま、`front-reverse.gif` はその左右反転、`front.png` / `back.png` は同スタイルの静止画（96×96）。**Black / White スタイルの背面には動く GIF が配布元に無い**ので、`back.gif` は `back.png` だけから作った：体から離れて浮いている 5 本の腕を整数ドット（上下 2・左右 1）で漂わせ、体を 1 ドット上下。回転・拡大・描き足しなし、0 コマ目＝静止画、長さは front と同じ 10 秒。`data/monsterdex.js` に `0720_hoopa-unbound` を 1 項目だけ手で追加（`cat:"gen6"`、エスパー／あく、80/160/60/170/130/80、6.5m・490kg、`shiny:false`、技は フーパの技から選んだもの。`hyperspace-fury` は技カタログに無い）。**`tools/build_monsterdex.js` で作り直すとこの項目は消える**ので、その後に `add_hoopa_unbound.py` を再実行する。`monsterSprite/manifest.json`・`monsterData/` は触っていない。色違い・box-icons は無い（アイコンは `front.png`）。作業前のデータは `../_backup_hoopa_unbound_20261115/`。
- **gen7・gen6 の 8 方向を全て定義（2026-11-15、ユーザー指示「gen7 だけ定義」「gen6 だけ定義」を続けて受けた）**: `tools/build_dir8_from_gifs.py --gen gen7 --png-fallback --apply`（未定義 22 フォーム）、同 `--gen gen6`（未定義 33 フォーム）。どちらも作業時点で 4 種の GIF が全フォームに揃っていた（別の作業で追加されたもの）ので、全て GIF から作成。静止画で埋めたものは無い。**既知の粗**: メガ 13 体（gen7: メガケケンカニ・メガジジーロン、gen6: メガブリガロン・メガマフォクシー・メガゲッコウガ・メガカエンジシ・メガフラエッテ ほか）は `front.gif` が 1 コマの小さい絵、`back.gif` が大きい多コマの絵で、前向きと後ろ向きの大きさ・動きが揃っていない（元の GIF の問題。シートは GIF どおり）。作業前のデータは `../_backup_dir8_before_gif_rebuild_20261115/before_gen7_define/` と `before_gen6_define/`。
- **【最新・ユーザー指定 2026-11-15】gen4 はアクションスプライトをやめ、全て GIF 由来の 8 方向スプライトにした（ユーザー指示「gen4 のアクションスプライトを全て、8 方向のスプライトに変更」。上の「gen4 のアクション」「agent-sprite-forge があるモンスターはバトルでその動きで動く」より、gen4 についてはこちらが優先）**: `tools/build_dir8_from_gifs.py --gen gen4 --include-actions --apply`。118 フォーム（通常 106・メガ 12）すべて、`data/dir8.js` の項目を `layout: actions_8dir` から `src:"gif"` の 8 方向シート（`<form>/<id>.png`）へ置き換え、`data/sprite_bounds.js` も更新。バトルでは待機の GIF の動き＋技の構え・被弾の伸び縮み（アクションシート無しのモンスターと同じ扱い）になり、専用の歩き・攻撃・ダメージのコマは使われない。`agent-sprite-forge/sheet.png` などアクションの生成物は消していない（使っていないだけ）。メガムクホークは `front.gif` が 1 コマなので前向きは動かない。戻すには `../_backup_dir8_before_gif_rebuild_20261115/before_gen4_actions_to_gif/` の `dir8.js` / `sprite_bounds.js` を戻す。gen4 の forge スクリプトを再実行するとアクションシートの登録に戻るので実行しないこと。
- **羽をたたんでいる鳥・虫は地上（2026-11-15、ユーザー指示「鳥や虫ポケモンの中で、羽を閉じているモンスターは地上に足をつける」。上の「接地と浮遊」への追加）**: `game/js/fighter.js` の `WINGS_FOLDED`（slug の完全一致）に入っている種は `Fighter.canFly` が false になり、地上の歩き方になる（水では泳ぎ、溶岩・氷の影響も受ける）。選び方は、浮いていた鳥型・羽根のある虫型 104 フォームの `front.gif`（0 コマ目と中間のコマ）を一覧にして、羽をたたんで足で立っている絵のものを目で選んだ 38：ポッポ、ピジョット、オニスズメ、ストライク、ホーホー、ヨルノズク、レディバ、レディアン、ネイティ、ネイティオ、ヤミカラス、エアームド、スバメ、オオスバメ、チルタリス、ムックル、ムクバード、ムクホーク、メガムクホーク、ドンカラス、ペラップ、マメパト、ハトーボー、ケンホロウ、コアルヒー、スワンナ、ワシボン、バルチャイ、バルジーナ、ヤヤコマ、モクロー、フクスロー、ツツケラ、ケララッパ、ドデカバシ、ココガラ、アーマーガア、ウッウ。同じ種でも絵が羽を広げているフォーム（ピジョン、メガピジョット、メガエアームド、キョダイアーマーガア等）は浮いたまま。絵（GIF）を差し替えたときはこの一覧も見直すこと。
- **ステージを実写寄りの HD-2D に加工（2026-11-15、ユーザー指示「最高の画像加工技術で、今のバトルステージを、特徴を維持したまま、かなりリアルに近い HD-2D のフィールドに加工」）**: ステージの絵（`arena.js` が作る粗いドットの床・観客席）は変えず、合成シェーダ（`game/js/post.js` の `compose`）で画面の解像度へ加工する。world レイヤーだけが対象で、モンスター（spr）・技（fx）・HUD はドットのまま。処理は 4 つ：(1) まわり 3×3 のドットを色が近いものだけ混ぜるなめらかな拡大（網点の濃淡が階調になる。白線・観客・電光掲示などの強い輪郭は残る）、(2) 明るさを高さとみなして左上から光を当てる起伏の陰影（石の継ぎ目・ひび・溶岩の割れ目が凹凸になる）、(3) ステージに固定した細かい肌（カメラが動いても流れない。`uOrg` = バッファ左上のステージ座標）、(4) 光の向きに傾いた面だけの照り返し。強さは床の素材ごとに `render.js` の `HD_FLOOR`（`[混ぜる色の近さ, 起伏, 照り返し, 肌, 肌の細かさ x, y, 照り返しの鋭さ]`）。先頭を 0 にすると元のドット表示に戻る。world テクスチャは LINEAR にした（ドットの中心を読む従来の経路は同じ色を返す）。WebGL が使えないとき（debug.html のファイル直開きなど、2D の fallback）は加工されない。
  - **ちらつきの修正（同日、ユーザー報告「debug.html が点滅する」）**: 素材の肌が画面の 1 画素より細かく（1 周期 1 画素未満まで）入っていたため、カメラが少し動くたびに全面の明暗が入れ替わってちらついていた（debug.html は既定でモンスターが歩き続けるので常に出る）。`compose` の `micro()` を粗くし（周期 0.5 / 1.1 / 2.3 回／ドット）、1 周期が画面の 2.5 画素を切る成分は消すようにした（`fine`）。肌を細かくし直すときはこの下限を守ること。切り分け用に、コンソールで `window.ARENA_HD_OFF = true` とすると加工だけ止まる。
- **8 方向シートの大きさに上限（2026-11-15、ユーザー報告「debug.html が点滅する」＋画面収録）**: 収録ではステージが真っ黒・8 方向欄が空・一瞬ページ全体が白くなる、というブラウザ側の描画崩れだった。原因と見たのは、同日に作った GIF 由来の 8 方向シートの大きさ：GIF の余白ごと全コマを並べていたため、解いた状態で最大 800MB（100MB 超が 56 フォーム、縦 97,524px）あり、しかも一度開いたシートを捨てていなかった。対策は 3 つ。(1) `tools/build_dir8_from_gifs.py`：各向きの全コマを「不透明部分を合わせた範囲」へ切りつめ（`crop_union`）、解いたときに `BUDGET_MB = 48` を超えるシートはコマを均等に間引く（`thin`、表示時間は合算）。全世代の GIF 由来シートを作り直した。(2) `Dir8.trim(keep)`（`game/js/dir8.js`）：`Battle` を作るたびに、その試合に出ないモンスターのシートを捨てる（debug.html の file:// 用キャッシュも同じ）。(3) 手作りシート（メガゲンガーの挑発、メガピクシーの帽子など）は GIF があっても置き換えない（`--include-hand` を付けたときだけ）。**作り直しの途中で gen1 の手作りシート 26 フォームを一度 GIF 版に置き換えてしまい、バックアップから元に戻した**（シート・`dir8.js`・`sprite_bounds.js`）。このとき未定義だった 10 フォーム（キョダイラプラス、メガプテラ、ヒスイバクフーン、ヒスイハリーセン、ヒスイニューラ、メガシビルドン、メガシャンデラ、ガラルマッギョ、メガゴルーグ、ヒスイウォーグル）にも GIF 由来の 8 方向が入った。作業前のデータは `../_backup_dir8_before_gif_rebuild_20261115/before_size_budget/`。アクションシート（gen5 などで最大 189MB）は作り直していない（(2) で溜まらなくはなる）。
- [2026-11-15 点滅バグの追加修正] `game/js/render.js` の `trainerStandBox` は、file:// ではドットを読めず失敗する。失敗を覚えずに毎回 canvas を作り直していたため、ファイル直開きの debug.html で毎秒およそ 360 枚の canvas が作られていた（動画の点滅の原因と判断）。失敗は `tr._standFail` に 1 回で覚える。描画のたびに呼ばれる関数で、失敗時に canvas を作り直す書き方をしない。
- [2026-11-15 ユーザー指示] キョダイマックスピカチュウ（`0025_pikachu-gmax`）の 8 方向シートは、手作りの 8 コマ待機シート（`v:"tailkeep1"`）をやめ、今の `front.gif` / `front-reverse.gif` / `back.gif` / `back-reverse.gif` から作り直した（`tools/build_dir8_from_gifs.py --gen gen1 --only 0025_pikachu-gmax --include-hand --apply`。108×164px/コマ、77 コマ、`src:"gif"`）。作り直し前のシートと `data/dir8.js`・`data/sprite_bounds.js` は `../_backup_dir8_before_gif_rebuild_20261115/before_pikachu_gmax_rebuild/`。
- [2026-11-15 ユーザー指示] キョダイマックスピカチュウの GIF 4 枚は、同じフォルダの `front.png` / `back.png` の色合いと光沢に塗り替えた（`tools/recolor_pikachu_gmax.py --view front|back --apply`。向きごとに 1 回ずつ実行）。形・コマ・表示時間は元のまま、使う色は 2 枚の PNG にある色だけ。元の GIF（暗い茶色の体・ピンクの縁取り）は `../_backup_pikachu_gmax_recolor_20261115/` にあり、スクリプトは毎回そこから読む。**このバックアップを消すと塗り替えをやり直せない。** 塗り替えたあとは `tools/build_dir8_from_gifs.py --gen gen1 --only 0025_pikachu-gmax --apply` で 8 方向シートを作り直す。配布元の GIF で上書きすると暗い色に戻るので、このフォームは GIF の一括更新の対象から外す。
- [2026-11-16 ユーザー指示・キョダイマックスピカチュウの最新] 上の「front.png / back.png の色に塗り替え」は取りやめ、GIF 4 枚を pokestats.gg の Generation VIII（Sword / Shield）のアニメで上書きした（`tools/import_pikachu_gmax_gen8.py --src <素材> --view front|back --apply`。素材の webp は `../_backup_pikachu_gmax_before_gen8_20261116/src_gen8/`）。正面 92×155・背面 92×159、50 コマ、30/30/40 ms のくり返し。8 方向シートは 106×159px/コマ・50 コマ。`tools/recolor_pikachu_gmax.py` は再実行しない（黄色の塗り替え版に戻ってしまう）。上書き前（塗り替え版）の GIF・シート・データは `../_backup_pikachu_gmax_before_gen8_20261116/`。色違いの GIF は作っていない。`front.png` / `back.png` は変えていない。
- [2026-11-16 ユーザー指示] キョダイマックス全 34 体の 8 方向シートを、今の `front.gif` / `front-reverse.gif` / `back.gif` / `back-reverse.gif` から作り直した（1 体ずつ `tools/build_dir8_from_gifs.py --gen <gen> --only <id> --include-actions --include-hand --apply`）。手作りの待機シート 7 体（フシギバナ・カメックス・ニャース・カイリキー・ゲンガー・キングラー・イーブイ）と、8 方向×4 アクションだった 3 体（リザードン・バタフリー・ダストダス）も GIF 由来の待機アニメに置き換えた（アクションシートのファイルは残っているが使わない）。GIF は 256×256 でコマ数が多いので、48MB の上限に合わせてコマを均等に間引いている（1 周の時間は GIF と同じ）。作り直し前のシートとデータ、対象の一覧、実行結果は `../_backup_dir8_before_gif_rebuild_20261115/before_gmax_rebuild_20261116/`。
- [2026-11-16 ユーザー指示] メガ進化（`dir` に `/mega/` をふくむ 99 体。ゲンシカイキをふくむ）の `front.gif` / `front-reverse.gif` / `back.gif` / `back-reverse.gif` を、pokestats.gg の Generation IX（Legends: Z-A、256×256 のアニメ）で上書きした（`tools/update_gifs_gen9_mega.py --only <id> --apply` を 1 体ずつ）。手作りだった GIF（メガリザードン X の背面、メガリザードン Y、メガラティアスなど）も上書きした。上書き前の GIF・対象一覧・実行結果は `../_backup_monsterSprite_gifs_before_gen9_mega_20261116/`。色違いの GIF と PNG は変えていない。
  - 8 方向シートは、GIF から作られていた 43 体だけ新しい GIF で作り直した（`../_backup_dir8_before_gif_rebuild_20261115/before_mega_gen9_rebuild_20261116/` に作り直し前・対象 `list.txt`・触っていない体 `kept.txt`）。アクションシート 45 体と手作りシート 11 体は触っていないので、バトル上の絵は以前のまま（GIF 欄だけ新しい絵）。
  - 新しい GIF は 256×256 で 70〜645 コマと長い。48MB の上限のため、作り直した 43 体のシートは 1 秒あたり 1.1〜9.4 コマまで間引かれていて動きが粗い（未解決。上限を上げるか、絵を縮めてコマを残すかはユーザーの判断待ち）。
- [2026-11-16] GIF／静止画から作った 8 方向シートの `v` は `bwgif2-<絵の中身の印 8 桁>` の形にした（`tools/build_dir8_from_gifs.py` が作り直すたびに付け直す）。ランチャー経由ではシートの URL に `?v=` が付くので、作り直しても同じ `v` のままだと古いシートがブラウザに残り、コマの大きさが合わず崩れるため。`v` の値をコードで比べている所は無い。
- [2026-11-16 ユーザー指示・メガ進化の 8 方向の最新] メガ進化 99 体すべての 8 方向シートを、今の GIF（Generation IX）から作り直した。上の「アクションシート 45 体と手作りシート 11 体は触っていない」は取り消し：この 56 体も GIF 由来の待機アニメに置き換えた（`--include-actions --include-hand`。歩き・攻撃・ダメージの専用アニメと手作りの待機アニメは使わなくなった。ファイルは残っている）。作り直し前のシートとデータ、対象 `list.txt`、実行結果は `../_backup_dir8_before_gif_rebuild_20261115/before_mega_all_rebuild_20261116/`。コマの間引きによる粗さ（1 秒あたり 1.1〜30 コマ、中央 5.1 コマ）は未解決のまま。
- [2026-11-16 ユーザー指示・メガ進化とキョダイマックスの 8 方向の最新（上の「GIF から作り直し」「コマの間引きは未解決」を置き換える）] 「8 方向の解像度・動き・大きさの質が悪過ぎる。4 つの GIF とまったく同じに」との指示で、メガ 99 体とキョダイマックス 34 体の 8 方向を作り直した（`tools/build_dir8_fb.py --only <id> --apply` を 1 体ずつ）。
  - **間引かない・変形しない**: GIF の全コマ・同じ表示時間。8 方向は GIF そのまま（南・西・南西 = front、南東・東 = front-reverse、北東・北 = back、北西 = back-reverse）。南北の横幅 80%、東西の前傾はしない。この 133 体を `tools/build_dir8_from_gifs.py` で作り直さない（間引きと変形が戻る）。
  - **シートの形（`layout:"fb"`）**: 正面の全コマ `<id>.png` と背面の全コマ `<id>-b.png` の 2 枚。反転は描くときに左右反転する。コマは `cols` 個ごとに折り返す。項目は `d1`（正面の秒）・`d2`（背面。同じなら省略）・`g`（GIF の枠の大きさと、その中の箱の位置）を持つ。読み込みは `game/js/dir8.js` の `sheetFb`（ファイル直開きは `debug.js` の `fileSheet` から同じ関数を呼ぶ）。
  - **メモリ**: コマの canvas は使うときに切り出し、新しい順に 200 枚だけ持つ（`FB_KEEP`）。色つきシルエットの控えも大きいコマは 24 枚まで（`assets.js` `tinted`）。
  - **GIF は 1 コマ目だけ読む**: ランチャー経由では、この 133 体の GIF（256px × 数百コマ）を全コマ解くと 1 体で数百 MB〜1GB 使って固まるので、`Assets.sprites` / `front` / `hd` は `still` で読む（`heavyGif`）。このためタイトル・選択画面・HUD のこの 133 体は動かない（色違いは別の小さい GIF なので動く）。
  - **debug.html の 8 方向欄**: `layout:"fb"` は GIF の枠そのものの大きさで、GIF の中と同じ位置にコマを描く（GIF の欄と同じ大きさ・同じ位置）。
  - シートの合計は約 1.1GB（133 体 × 2 枚）。作り直し前のシートとデータ、対象 `list.txt`、実行結果は `../_backup_dir8_before_fb_20261116/`。
- [2026-11-16 ユーザー指示] 「バトルフィールドを、HD-2D リメイク（参考画像: 木漏れ日・光の筋・長い影・手前と奥のぼかしのある村の絵）のような画質に作り直す」。ステージの形・床・観客・小物・昼夜は変えず、光と影と後処理を作り直した。変更前の `post.js` / `render.js` / `arena.js` は `../_backup_hd2d_field_20261116/`。
  - **ステージごとの光の設定**: `game/js/arena.js` の `HD2D`（`grade` に足す）。`sun` 日なた・日かげの強さ、`sunCol` / `shCol` その色、`rays` 光の筋、`rayCol` 光の色、`cast` 落ち影の濃さ、`mote` 光の中のちり、`split` 暗部を青く明部を黄色く、`tilt` ぼかし（1 より大きいと強い）。スタジアムだけ `ambient` も明るくした（夜のままだが照明で明るい）。
  - **日なた・日かげ**（`post.js` compose、`uSun`）: 地面だけに、左上→右下へ伸びた明暗の帯をステージに固定して落とす（ゆっくり流れる）。モンスターには掛けない。
  - **落ち影**（`render.js` `castShadow`）: モンスターの今のコマの形を、足もとから右下へ倒して地面に描く（高さ h の点は右へ 0.9h・下へ 0.32h）。足もとの丸い影は残している。倒れたモンスターには出さない。
  - **光の筋・ちり・色分け・ぼかし**（`post.js` final）: 左上からの幅の広いやわらかい光の束（`uRays`）、ただようちり（`uMote`）、暗部を青く明部を黄色く（`uSplit`）、手前と奥のぼかしを強く（`tilt > 1` でぼかしを 1 段足す）。
  - **ファイル直開き（WebGL なし）**: `render.js` `fallbackHd` が 2D で近似する（日なた・日かげの重ね、光の筋、上下のぼかし、四すみの暗がり）。あわせて、光の絵を掛けたあと 2 倍に戻すようにした（これまでファイル直開きは WebGL の半分の明るさだった）。
  - 全部を切るには `window.ARENA_HD_OFF = true`（地面の質感の加工と同じスイッチ）。個別に切るには `HD2D` の値を 0 にする。
  - [同日の調整・上の HD-2D の記述より優先] 別エージェントの検査（「地面の明暗が迷彩の縞に見える」「光の筋が見えない」「ぼかしが強すぎて観客がつぶれる」など）を受けて直した: 日なた・日かげは大きくやわらかい面にしてフィールドの中だけ（`uFld`）、強さを下げた。光の筋は見えるように式を直し、照明が消えている間は出さない。ぼかしは前と同じ強さ（`tilt` 1）に戻した。落ち影は体の動き（ため・突進・ゆれ）に付いていき、小物（木箱・岩など）にも付けた（`castProp`）。ファイル直開きは、日なた・日かげをモンスターより先に描き、ぼかしを継ぎ目のない方式に変え、光の帯の端が見えないようにした。`window.ARENA_HD_OFF = true` で、ランチャー経由でも作り直す前の見た目（光の筋・ブルーム・スタジアムの明るさも前の値）に戻る。
  - [同日] debug.html の 8 方向欄（`layout:"fb"`）は、絵を GIF の元の画素数（256×256）のまま持ち、表示の大きさだけ CSS で合わせる（高精細な画面で GIF の欄より粗くならないように）。

- [2026-11-16 ユーザー指示・バトルフィールドの 3D 化] 「HD-2Dらしさが全然足りない。各バトルフィールドの背景を、最高品質の画風で、HD-2D 化するために 3D 化する」。まわりの景色（観客席・壁・塔・木・岩・遠景）を本物の 3D の立体にし、遠近のあるカメラ（手前の上空から約 46° で見下ろす）で映すようにした。**仕様・道具・決まりは `md/scene3d.md`（正本）**。
  - エンジンは `game/js/scene3d.js`。ステージごとの立体は `game/js/scenes/<id>.js`（11 ステージ。1 ステージずつ別のエージェントが作った）。地面（床・傷・影）は今までの 2D の canvas を z = 0 の面として映し、モンスター・トレーナー・小物は画面に正対する板として、足もとの奥行きに合わせた倍率で描く（`render.js` の `anchor` / `sprT`）。
  - 画面と地面の対応は `Renderer.worldToScreen(x, y, z)` / `Renderer.toWorld`（3D では投影・逆投影）。モンスターの頭上のバーなど「その場所での倍率」が要る所は `sclAt`。`S * cam.zoom` を直接使う処理を新しく書かないこと（奥と手前で倍率が違う）。
  - 操作モンスターが画面の中心・100% = GIF 等倍、という決まりは、画面中央の点の倍率として保っている（奥のものは小さく、手前のものは大きく映る）。
  - タイトル・デモ・登場・K.O.・競り合いのときはカメラが低くなり（`pitch3`）、奥の景色が広く映る。
  - 天気（雨・砂・あられ）は 3D では画面にそのまま降らせる（`weather3`）。技のエフェクト（fx）は地面の面に乗る。
  - `?flat=1` で以前の平面表示。定義ファイルが無いステージ、WebGL が使えないとき（`file://` 直開き時）は平面表示になる。`game/debug.html` も `scene3d.js` + `scenes/*.js` + `hud.js` を読み込み本編と同じ 3D ジオラマ＋eSports HUD で描画する（モンスター選択 `#side`・技選択 `#moves`・スプライト確認 `#dirs` は初期非表示でバーのボタンから個別開閉、開いても `[USER-LOCKED LAYOUT]` の `#dirs` 高さ `30vh`・2段横並びは維持）。
  - 新しいステージを足す・作り直すときは、`md/scene3d.md` を読ませた新しいエージェントに 1 ステージだけ渡す（他のステージのファイルを見せない）。

- **debug.html が 3D にならないとき（2026-11-16）**: `debug.html` をダブルクリック（`file://`）で開くと、ブラウザの制限でスプライトの絵を WebGL に渡せないため `game/js/debug.js` が `Post.init` を止め、平面表示になる（コードでは直せない制限）。3D で見るには `デバッグ起動.command`（`ゲーム起動.command` と同じローカル http サーバーで `game/debug.html` を開く）を使う。`file://` で開いたときは画面右下の注意書きにも同じ案内が出る。

- **モンスターが光で白く飛ばないようにした（2026-11-16、ユーザー指示「モンスターが光り過ぎて見えない時がある。光源が強いのはもう少しモンスターが見える様に」）**: 原因は 4 つの重なり（照明の絵がモンスターを最大 2 倍に明るくする／まひ等の状態異常の単色かぶせ／ブルーム／光の筋）。(1) `game/js/post.js` の `spr3`（3D）と `compose`（平面）：モンスターだけ、等倍を超えた光を 0.3 倍に圧縮（最大 2.0 → 1.3、`SPR_OVER`）。地面・立体はそのまま。(2) `post.js` の `final`：モンスターがいる画素（`uSprM` のアルファ、3D のときだけ `uSprMask = 1`）では、ブルームを 0.4 倍、光の筋・ちりを 0.25 倍、上下端のぼかしを 0.3 倍に弱める。(3) `game/js/render.js`：状態異常の色かぶせを 0.25±0.15 → 0.14±0.08 に薄くした。被弾の白フラッシュ・技のエフェクト自体は変えていない。ファイル直開き（WebGL なし）の 2D 表示に効くのは (3) だけ。もっと抑えるなら `SPR_OVER` を下げる。元の強さへ戻さないこと。

- **技のエフェクトの 3D 化（2026-11-16、ユーザー指示「HD-2Dになったので、技のエフェクトも今のエフェクトが3Dになる様に」）**: 3D 表示では、空中のエフェクト（ビーム・弾・爆発・柱・落雷・岩・粒子・空中の輪）を地面に貼らず、画面に正対して立つ層（`render.js` の `airc` → `post.js` の `air3`）に描く。地面のエフェクト（予告範囲・地面の輪・影）は地面の面のまま、モンスターより先に描く。描く側は world 座標のままで、`fx.js` の `FXV.at(ctx, x, y)`（足もとに立つ板）/ `FXV.seg(ctx, x0, y0, h0, x1, y1, h1)`（2 点を結ぶビーム）で場所を決める。技ごとの割り当ては `moves.js` の `airView`。地面の円のたてつぶしは `FXV.gsq()`（3D では 1）。平面表示（`?flat=1`・file://）は変えていない。仕様の正本は `md/scene3d.md` の同名の項。空中エフェクトを足すときは `ctx.setTransform` を直接呼ばないこと。
