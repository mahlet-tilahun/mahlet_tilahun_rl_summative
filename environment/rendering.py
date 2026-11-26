"""
Rendering = 3D visualisation in three.js (assets/viewer.html).

Two ways to feed the 3D scene (both send plain JSON = the serialisation path / API):

1. LIVE (default):  `LiveSession` + `serve_viewer(..., session=...)` start a small web server.
   The three.js page calls   /api/start  and then  /api/step  once per animation step.
   Each /api/step makes the trained agent choose an action, steps the environment and returns the new state
   as JSON, so the 3D scene is driven by the agent while it plays.
2. REPLAY (fallback):  `record_episode` + `save_replay` write assets/replay.js, which the page can load
   without any server (used when the page is opened without the Python server).
"""
import functools
import http.server
import json
import os
import threading
import webbrowser
from urllib.parse import parse_qs, urlparse

from environment.custom_env import ACTION_NAMES, NEED_NAMES


# ----------------------------------------------------------------------------- replay (fallback)
def record_episode(env, policy_fn, seed=0, label=""):
    obs, _ = env.reset(seed=seed)
    frames = [dict(env.state_dict(), reward=0.0)]
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(policy_fn(obs))
        total += r
        frames.append(dict(env.state_dict(), reward=round(r, 3)))
        done = term or trunc
    return {"label": label, "seed": seed, "layout": env.layout(), "total_reward": round(total, 2),
            "actions": ACTION_NAMES, "needs": NEED_NAMES, "frames": frames}


def save_replay(episodes, path="assets/replay.js"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("window.REPLAYS = " + json.dumps(episodes) + ";")
    return path


# ----------------------------------------------------------------------------- live session
class LiveSession:
    """Holds one environment + the trained policies; the web page drives it step by step."""

    def __init__(self, env, policies, best, seed, on_start=None, on_step=None):
        self.env, self.policies, self.best, self.seed = env, policies, best, seed
        self.on_start, self.on_step = on_start, on_step
        self.lock = threading.Lock()
        self.sid, self.algo, self.obs, self.total, self.done = 0, None, None, 0.0, True

    def info(self):
        return {"live": True, "best": self.best, "seed": self.seed,
                "algos": [{"id": a, "label": a.upper() + (" ★" if a == self.best else "")} for a in self.policies]}

    def start(self, algo, seed):
        with self.lock:
            self.sid += 1
            self.algo, self.seed = algo, seed
            self.obs, _ = self.env.reset(seed=seed)
            self.total, self.done = 0.0, False
            if self.on_start:
                self.on_start(algo, seed, self.env)
            return {"sid": self.sid, "algo": algo, "seed": seed, "label": algo.upper(), "layout": self.env.layout(),
                    "actions": ACTION_NAMES, "needs": NEED_NAMES, "frame": dict(self.env.state_dict(), reward=0.0)}

    def step(self, sid):
        with self.lock:
            if sid != self.sid:
                return {"stale": True}
            if self.done:
                return {"stale": True}
            a = self.policies[self.algo](self.obs)
            self.obs, r, te, tr, _ = self.env.step(a)
            self.total += r
            self.done = bool(te or tr)
            if self.on_step:
                self.on_step(self.env, a, r, self.done, self.total)
            return {"stale": False, "frame": dict(self.env.state_dict(), reward=round(r, 3)),
                    "done": self.done, "total": round(self.total, 2)}


# ----------------------------------------------------------------------------- web server
def _handler(directory, session):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=directory, **k)

        def log_message(self, *a):          # keep the terminal clean for the agent's step log
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            u = urlparse(self.path)
            if not u.path.startswith("/api/"):
                return super().do_GET()
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if session is None:
                    raise KeyError("no live session")
                if u.path == "/api/ping":
                    out = session.info()
                elif u.path == "/api/start":
                    out = session.start(q.get("algo", session.best), int(q.get("seed", session.seed)))
                elif u.path == "/api/step":
                    out = session.step(int(q.get("sid", -1)))
                else:
                    raise KeyError(u.path)
                data, code = json.dumps(out).encode(), 200
            except Exception as e:           # noqa: BLE001
                data, code = json.dumps({"error": str(e)}).encode(), 404
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def serve_viewer(directory="assets", port=8000, open_browser=True, session=None):
    server = None
    for p in range(port, port + 20):         # if the port is busy (another program), try the next one
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", p), _handler(directory, session))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("No free port found between %d and %d" % (port, port + 19))
    url = f"http://127.0.0.1:{server.server_address[1]}/viewer.html"
    print(f"[three.js viewer] open {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
