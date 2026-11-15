#!/bin/bash
# デバッグ画面（game/debug.html）用。3D のバトルフィールドは WebGL が要り、file:// 直開きでは使えないので http で開く。
# Serves this folder over a local http server (needed so the animated GIF
# sprites can be decoded) and opens the game. Close this window or press Ctrl+C to stop.
cd "$(dirname "$0")" || exit 1
PORT=8787
while lsof -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do PORT=$((PORT + 1)); done
# no-cache validates modified files immediately via 304 while keeping unchanged assets instant
python3 -c '
import http.server, sys
class S(http.server.ThreadingHTTPServer):
    request_queue_size = 256
    allow_reuse_address = True
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()
    def log_message(self, *a): pass
S(("0.0.0.0", int(sys.argv[1])), H).serve_forever()
' "$PORT" >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM
sleep 1
open "http://127.0.0.1:$PORT/game/debug.html"
LAN=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
echo "このパソコン: http://127.0.0.1:$PORT/game/debug.html"
[ -n "$LAN" ] && echo "同じWi-Fiのスマホ: http://$LAN:$PORT/game/debug.html"
echo "このウィンドウを閉じるとサーバーが止まります。"
wait $SERVER
