# CLAUDE.md — arena_revenge_claude

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

- **サンダー（`0145_zapdos`）の GIF をユーザー提供のものに差し替え（2026-11-04、ユーザー指示）**: `~/Desktop/zapdos-black-white-animated.gif` → `front.gif`、`~/Desktop/zapdos-black-white-back-animated.gif` → `back.gif`（どちらも 9 コマ・100ms、そのままコピー）。`front-reverse.gif` / `back-reverse.gif` は各コマを左右反転して作った（既存の `-reverse` も正面の左右反転なので同じ規則）。`data/sprite_bounds.js` の `g` の4項目を1コマ目の実測値に更新。色違いの GIF は提供が無いので作っていない（ゲームは `front-shiny.png` / `back-shiny.png` の静止画を使う）。8方向シート（`0145_zapdos.png`）は変更していない。

- **front / back GIF を pokestats.gg の Black / White スタイルへ更新（2026-11-15、ユーザー指示「全ての世代で更新」）**: 配布元は `https://assets.pokestats.gg/pokeapi/v1/pokemon/versions/generation-v/black-white/animated/<PokeAPI番号>.gif`（後ろは `/back/`）。1216 フォーム中、正面があるのは 943、後ろもあるのは 858。`front.gif` / `back.gif` は配布元とバイト一致、`-reverse.gif` は全コマの左右反転（コマ数・表示時間は同じ、パレットも保持）。配布元に無い側、色違い GIF、PNG、8方向シートは触っていない。メガラティアスは固定仕様3のため対象外。以前「変な動き」で削除したフォームのうち配布元にあるもの（正面 23・後ろ 11）は、配布元の GIF で入れ直した（配布元には 19〜20 コマ・60ms の型は無い）。`data/sprite_bounds.js` の `g` も更新済み。元ファイルは `../_backup_monsterSprite_gifs_before_pokestats_20261115/`。処理の記録用スクリプトは `tools/update_gifs_pokestats_bw.py`（作業用フォルダの `forms.json` と `dl/` を前提にしたもの。再実行するときは配布元から取り直す）。

- **pokestats に Pixel / Black・White のアニメ GIF が無い面だけ Pokémon Showdown で補完（2026-11-15、ユーザー指示）**: 上の Black / White アニメが 200 の面は残した。無かった 631 面のうち、作成分（メガラティアスの back、メガリザードンYの front/back、メガリザードンXの back）は上書きしていない。それ以外は `https://play.pokemonshowdown.com/sprites/ani/<spriteid>.gif` と `ani-back` のバイトを `front.gif` / `back.gif` にした。公式 URL が 404 で pokestats の `pokemon/other/showdown/<PokeAPI番号>.gif` にある 7 面（キョダイマックスの正面が中心。ストリンダー低音キョダイは後ろもある）だけそちらを使った。書いたのは 545 面（新規 449、既存の非BWを置換 96）。`-reverse.gif` は全コマの左右反転。`data/sprite_bounds.js` の `g` を 1090 件更新。色違い GIF、PNG、8方向シートは未変更。Showdown 側にも無いのは 82 面（新しいメガ、一部の後ろ、オーガポン、ペパー、テツノツツミ、テツノカイナの後ろなど）。退避は `../_backup_monsterSprite_gifs_before_showdown_20261115/`。スクリプトは `tools/update_gifs_showdown_fallback.py`。2026-11-04 の「キョダイマックス GIF を作り直さない」は、この指示の範囲（Showdown にある面）だけ例外にした。

- **pokestats に Pixel / Black・White も Pokémon Showdown も無い面だけ Generation IX の ANIMATED で補完（2026-11-15、ユーザー指示）**: pokestats の Generation IX ANIMATED は GIF ではなくアニメ WebP。表示用は `https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-ix/scarlet-violet/<PokeAPI番号>.webp`（後ろは `/back/`、256×256）。新しいメガは scarlet-violet に無く、`.../generation-ix/legends-za/<PokeAPI番号>.webp` にある。静止画の `.../v1/pokemon/versions/...webp` と、`.../animated/` の GIF は使わない（後者は 404）。ゲームは `.gif` しか読まないので、WebP の各コマを GIF に変換して `front.gif` / `back.gif` にした（表示時間は `webpmux -info` のミリ秒。GIF は 10ms 単位なので 33ms は 30ms）。`-reverse.gif` は全コマの左右反転。BW も Showdown も無い 82 面のうち、書いたのは 76 面（新規 65、既存の非BW・非Showdownを置換 11）。出典は legends-za 54、scarlet-violet 22。置換したのはメガライチュウX/Yの back、メガチリーンの back、メガアブソルZの front/back、テツノカイナ・テツノドクガ・テツノイバラ・テツノブジン・ミライドン・テラパゴスの back。`data/sprite_bounds.js` の `g` を 152 件更新。作成分（メガラティアス、メガリザードンY、メガリザードンXの back）は未変更。色違い GIF、PNG、8方向シートは未変更。画面上の大きさはキャンバスの 256px ではなく、`assets.js` が体の高さに合わせて縮める。まだ無いのはキョダイマックスの後ろ 6 面（フシギバナ 10195、カメックス 10197、ゴリランダー 10209、インテレオン 10211、ウーラオス連撃 10227、ウーラオス一撃 10226。3系統とも 404）。退避は `../_backup_monsterSprite_gifs_before_gen9_20261115/`。スクリプトは `tools/update_gifs_gen9_animated.py`。2026-11-04 の「キョダイマックス GIF を作り直さない」は、BW も Showdown も無く Generation IX ANIMATED がある面だけ例外にする。今回その後ろ 6 面は配布元が無いので作っていない。

- **移動中は対戦相手の方を向く（2026-11-15、ユーザー指示「移動する時、対戦相手の方向を必ず向く。相手も同じ」）**: リアルタイムは `game/js/fighter.js` の自由移動。入力が動いているあいだ `facing` は `lockTarget`（相手のアクティブ）への角度。プレイヤーと CPU は同じ `Fighter.update` なので両側に効く。止まっているときは今までどおり `aim`（マウス・右スティック、無ければ相手）。発射方向の `aim` は変えない。攻撃・ダッシュ・被弾・登場の向きは変えていない。表示の8方向は `dir8.js` が 1 段ずつ回す。ターン制の歩み寄り（`game/js/turn.js` の tween）も移動中は相手を向く。相手が倒れているときと、重なって距離が 1 未満のときだけ進行方向に戻る。

- **キョダイマックスカメックスの4枚の GIF を Generation VIII の Animated で上書き（2026-11-15、ユーザー指示）**: ページは `https://pokestats.gg/sprites/blastoise-gmax` の Generation VIII「Sword / Shield, Animated」。表示用は `https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-viii/sword-shield/10197.webp` と `.../back/10197.webp`（256×256、70コマのアニメ WebP）。ゲームは `.gif` なので変換して `monsterSprite/gen1/0009_blastoise/gigantamax/blastoise-gmax/` の `front.gif` / `back.gif` にした。`-reverse.gif` は全コマの左右反転。背面2枚は以前ファイルが無かったので新規。正面2枚の更新前は `../_backup_blastoise_gmax_gifs_before_gen8_20261115/`。`data/sprite_bounds.js` の `g` を4件更新。色違い、PNG、8方向シート `0009_blastoise-gmax.png` は、この時点では未変更。2026-11-04 の「キョダイマックス GIF を作り直さない」は、次の項で全キョダイマックスの4枚に広げた。

- **全キョダイマックスの4枚の GIF を Generation VIII の Animated で上書き（2026-11-16、ユーザー指示「全てのギガントマックス」）**: index の `-gmax` 34 フォーム。表示用は pokestats の Generation VIII「Sword / Shield, Animated」。URL は `https://assets.pokestats.gg/pokeapi/v1/anim/pokemon/versions/generation-viii/sword-shield/<pokemonId>.webp` と `.../back/<pokemonId>.webp`。静止画の `.../v1/pokemon/versions/...webp` と、クリック先の `.../pokeapi/orig/...webp` は使わない。ゲームは `.gif` しか読まないので WebP を GIF に変換した（表示時間は `webpmux -info`）。`-reverse.gif` は全コマの左右反転。68 面すべて書けた（置換 63、新規 5）。新規の背面はフシギバナ、ゴリランダー、インテレオン、ウーラオス連撃、ウーラオス一撃。カメックスの背面は直前の項で作ってあったので置換。`data/sprite_bounds.js` の `g` を更新。色違い GIF と `front.png` / `back.png` は未変更。退避は `../_backup_monsterSprite_gifs_before_gen8_gmax_20261115/`。スクリプトは `tools/update_gifs_gen8_gmax.py`（再実行は `--apply` で全34体を取り直す）。ストリンダー低音（10228）の配布ファイルはハイ（10219）とバイト一致なので、ローカルの front/back も同じになる。画面上の大きさはキャンバス寸法ではなく体の高さ。ピカチュウだけ別スクリプト `tools/import_pikachu_gmax_gen8.py` が orig の 512px を体の高さ 155 に縮小して上書きしたので、同じ表示用アニメ（256×256、50コマ）に戻した。この縮小スクリプトは再実行しない。

- **技の炎を使い手の色に合わせる（2026-11-16、ユーザー指示「メガリザードンXは青い炎」）**: 炎タイプの技だけ、使い手の正面 GIF の色に合わせる。色相は `data/flame_hues.js`（`tools/build_flame_hues.py` が `front.gif` / `front-shiny.gif` から作る。再生成して上書き）。debug.html は file:// で開くと絵の色を読めないので、戦闘中にピクセルを取らずこの表を使う。表に無い橙・赤（色相 62 以下、または 348 以上。リザードン、ヒトカゲ、ブーバー、メガリザードンY）は今までの炎のまま。青（150〜270。メガリザードンXは 205）は熱い芯を黄へ寄せない。りゅうのいぶきは紫のまま。やけど、溶岩、登場の炎は使い手を渡さないので橙のまま。体そのものの色（突進の残像、被弾フラッシュ）は変えない。読み込み直しは `flame_hues.js` と `fx.js` の `?v=20261116flame2`。
- **キョダイマックスピカチュウは体を残し、体の外側の白だけ透過（2026-11-15）**: 体・黄色い輪郭・稲妻の内側の白は残す。体の外側に接している近白（輪郭の外にはみ出た白）だけアルファ 0。シートは `0025_pikachu-gmax.png`。読み込み直しは `data/dir8.js` の `v: "tailkeep1"` と `debug.html` / `index.html` の `dir8.js?v=20261115-gifall9`。白だけ残して体を消した版は誤り。全身の退避は `../_backup_pikachu_gmax_sheet_20261115/`。

- **debug.html：`#dirs` の 2 段固定・`30vh` 高さ固定（2026-11-15、ユーザー指示「3列（3段）になるのは絶対にやめましょう。2列（2段）が好ましいのと、枠の高さは画面の30%くらいで固定」）**: `#strip.acts` の4個折り返しを廃止して全アクションを横1行（`grid-auto-flow: column`）に統一し、`#official`（4種 GIF）も `flex-wrap: nowrap` で横1行に固定して常に 2 段表示とした。また `#app` / `#dirs` を `30vh` 固定とし、`#dirs .head` を1行省略表示、自動ズーム時は `autoRowScale()` で `#strip` と `#official` を同率縮小フィットさせ、`×1/×2/×3` 指定時は `#dirs-body` 内をスクロールするようにした。
- **gen1 の8方向スプライトを 4 つの GIF から作り直し（2026-11-15、ユーザー指示「8方向のSpriteのアニメーションを front / front-reverse / back / back-reverse の GIF を基盤に作り直す。まずは gen1 だけ」）**: `tools/build_dir8_from_gifs.py --gen gen1 --apply`（Pillow のある `agent-sprite-forge/.venv/bin/python` で実行、1フォームずつ独立に処理、公開は `gen5_publish.py` の `publish_dir8` / `publish_bounds`）。gen1 の 220 フォーム中 164 を作り直した（`data/dir8.js` は `src:"gif"`、`v:"bwgif2"`、色違いシート無し）。向きはゲームの実行時ルールと同じ：SW=front、SE=front-reverse、NE=back、NW=back-reverse、S/N は横幅 80%、E/W は横幅 90%＋前傾。コマは GIF の全コマ（連続する同じ絵だけまとめる。間引かない）。**32 コマへの間引きは「GIF より鈍い」とユーザーが却下（2026-11-15）。8方向は GIF と同じコマ数・同じ速さで動かすこと（`MAX_FRAMES = 0`）。** シートは縦に長くなる（最大 152 行・約 15,200px）。
  - 触っていないもの：`layout:"actions_8dir"` のフォーム（24。待機・歩き・攻撃・ダメージのアクションシート。置き換えるなら `--include-actions`）、4 つの GIF が揃っていないフォーム（32。メガ・キョダイマックス・一部リージョン）。
  - 以前の手作りシート（AI 原画＋待機アニメ）は `../_backup_dir8_before_gif_rebuild_20261115/` にある（`dir8.js`・`sprite_bounds.js` も同じ場所）。メガピクシーの帽子・メガゲンガーの挑発は GIF が無いので対象外のまま。
  - debug.html：GIF から作ったシートは下の GIF と同じ倍率で描く（`game/js/debug.js` の `gifZ`）。他の世代は未実施。

---

## ゲーム体験・演出ブラッシュアップのノウハウ（2026-11-11：fable 5.2 ＋ BigKidNation ＋ Ivandanke_Rivals 反映）

詳細は `AGENTS.md` の同名セクションを参照。今後の改修でも以下の5原則を維持すること。

1. **黄金比スタジアム縮尺とカメラのメリハリ（`assets.js` / `render.js`）**:
   - トレーナー身長 `1.7m = 48px` を基準物差し（メニュー・顔・リーチ用）としつつ、**2026-11-15 最新調整で大きさバランスを改定：`1.7m` 未満は線形縮小（`BODY_MIN = 12px`、フィールド最小 `10px`、ピカチュウ `10px` / フシギダネ `15px`）、リザードン `1.7m = 36px` 基準・重量級上限 `107px`（フィールド `80px`）、キョダイマックスは `195〜268px`（フィールド `146〜201px`）**、`FIELD_DRAW = 0.75`。**トレーナーはフィールド上 `18px`（`trainerFieldPx`）、画面外ワイプ `58*dpr`（モバイル `30*dpr`）の控えめサイズとし、交換時の回収レーザー・トレーナー手元からのボール投擲トレイル・展開バーストを実装**（移動速度 `Fighter.moveRate`、`SPR` 最大 4 倍専用ウィンドウ維持）。詳細と数値は `AGENTS.md` の同セクション。 **（2026-11-15 夜の最新: `FIELD_DRAW = 0.6`・フィールド最小 `8px`・1.7m ＝ `29px`。詳細は md/AGENTS.md）**
   - 通常時は画面酔いを防ぐワイド俯瞰を維持し、**ドラマチックなズームインは「強打・急所（`camFocus`）」「ビーム競り合い（`activeClash`）」「登場（`focus`）」「K.O.決着（`koCam`）」の4局面のみ**に限定する。プレイヤーはホイール／`−` `＝` `0` キー／HUD の `#cam-zoom` で全体倍率 `cam.user` を変えられ、カメラは操作中のモンスターに追従する。**2026-11-15 のユーザー指定で、基準ズームは「追っているモンスターが front.gif と同じ大きさで映る倍率」（`cam.user` 100% = GIF 等倍、既定 100%）になり、画面外のモンスターは方向つきのワイプ（`monsterWipes`）で映す。詳細は `AGENTS.md`**（2026-11-14、詳細は `AGENTS.md`）。
2. **「棒立ち撃ち」をなくす3段階のゲームフィール（`fighter.js` / `moves.js` / `battle.js` / `render.js`）**:
   - **Stage 1（予備動作＆テレグラフ）**: 溜め前半65%まで追尾、残り35%で照準角度ロック＋地面に危険範囲（`Moves.drawWindups` / `Moves.windupDanger`）を表示し、見てから回避可能にする。
   - **Stage 2（色付きシルエット残像）**: 回避・突進・跳躍・近接踏み込み時に `f.ghosts` 残像を描画（`actions_8dir` シートには `squash` を二重掛けしない）。
   - **Stage 3（着弾の重み）**: 段階的ヒットストップ（基準 `0.036s / 0.052s / 0.072s` × `FX.HITSTOP_MUL = 1.9`、2026-11-14 ユーザー指示で重くした。調整は `fx.js` のこの倍率）、純白シルエットフラッシュ、ノックバック土煙、`zoomPunch`、重量級（`mass >= 2.2`）のスーパーアーマー。
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
