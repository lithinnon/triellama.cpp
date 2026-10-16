#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

SERVER_BIN = os.path.abspath("build/bin/llama-server")
MODEL_PATH = os.path.abspath("models/test-stories260K-f32.gguf")
TEST_PORT = 28991
CACHE_DIR = f"/tmp/triellama_live_e2e_{int(time.time())}/radix"

def wait_for_server(url, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False

def send_completion(url, prompt, n_predict=5):
    payload = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/completion",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print(f"[test-live-server-e2e] Starting end-to-end integration test...")
    if not os.path.exists(SERVER_BIN):
        print(f"Error: Server binary not found at {SERVER_BIN}")
        return 1
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return 1

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    server_url = f"http://127.0.0.1:{TEST_PORT}"

    # Phase 1: Launch server with 0-RAM mode and disk tier enabled
    cmd1 = [
        SERVER_BIN,
        "-m", MODEL_PATH,
        "--port", str(TEST_PORT),
        "--cache-disk-dir", CACHE_DIR,
        "--cache-disk", "50",
        "-cram", "0",
        "--cache-idle-slots",
        "--no-warmup",
    ]
    print(f"[test-live-server-e2e] Launching server instance 1...")
    p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        if not wait_for_server(server_url):
            raise RuntimeError("Server 1 failed to reach healthy state within timeout")
        print(f" - Server instance 1 is healthy on {server_url}")

        prompt1 = "Once upon a time in a faraway forest, a little blue bird decided to build a nest."
        resp1 = send_completion(server_url, prompt1, n_predict=5)
        print(f" - Turn 1 full response: {json.dumps(resp1, indent=2)}")

        # Give background saver time to write idle slot to disk
        time.sleep(1.0)

        # Check that disk checkpoints were created
        ckpt_files = []
        for root, _, files in os.walk(CACHE_DIR):
            for f in files:
                if f.endswith(".ckpt"):
                    ckpt_files.append(os.path.join(root, f))
        print(f" - On-disk .ckpt files created: {len(ckpt_files)}")
        assert len(ckpt_files) > 0, "No checkpoint files written to disk tier in 0-RAM mode"

        # Turn 2: Query with extended prompt sharing the exact prefix
        prompt2 = "Once upon a time in a faraway forest, a little blue bird decided to build a nest. When the morning sun arose, it began to sing."
        resp2 = send_completion(server_url, prompt2, n_predict=5)
        tokens_cached2 = resp2.get("tokens_cached", 0)
        print(f" - Turn 2 response: tokens_cached={tokens_cached2}, text={resp2.get('content', '')[:30]!r}")
        assert tokens_cached2 > 0, f"Turn 2 failed to achieve prefix cache hit (got {tokens_cached2})"

    finally:
        p1.terminate()
        try:
            p1.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p1.kill()
        print(" - Server instance 1 stopped")

    # Phase 2: Restart server pointing to the same disk cache directory
    print(f"[test-live-server-e2e] Launching server instance 2 (Cold restart test)...")
    p2 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        if not wait_for_server(server_url):
            raise RuntimeError("Server 2 failed to reach healthy state within timeout")
        print(f" - Server instance 2 is healthy on {server_url}")

        # Turn 3: Query the prompt after cold restart -> must hit hydrated disk checkpoint
        prompt3 = "Once upon a time in a faraway forest, a little blue bird decided to build a nest. When the cold winter arrived, it flew south."
        resp3 = send_completion(server_url, prompt3, n_predict=5)
        tokens_cached3 = resp3.get("tokens_cached", 0)
        print(f" - Turn 3 (Cold restart hit): tokens_cached={tokens_cached3}, text={resp3.get('content', '')[:30]!r}")
        assert tokens_cached3 > 0, f"Turn 3 failed to achieve prefix cache hit after restart (got {tokens_cached3})"

    finally:
        p2.terminate()
        try:
            p2.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p2.kill()
        print(" - Server instance 2 stopped")

    # Clean up test cache
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print(f"[test-live-server-e2e] All end-to-end integration tests PASSED successfully!\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
