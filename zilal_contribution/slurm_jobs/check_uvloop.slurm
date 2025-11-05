pip install "uvloop>=0.17,<1.0"
# quick check
python - <<'PY'
import uvloop, asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
print("uvloop OK")
PY