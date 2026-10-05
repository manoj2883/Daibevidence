"""
Pre-builds the OFFLINE_MODE search index (data/offline_index_v3.npz): embeds every chunk in
corpus/v3_5k_recursive.json.gz with the local all-MiniLM-L6-v2 model. No API calls. Optional: the
app builds it on first offline start if it's missing.

    python -m scripts.build_offline_index
"""
import time

from src.rag.offline import OFFLINE_INDEX_PATH, build_offline_index

if __name__ == "__main__":
    t0 = time.monotonic()
    index = build_offline_index()
    print(f"Saved {len(index['ids'])} chunk vectors to {OFFLINE_INDEX_PATH} in {time.monotonic() - t0:.0f}s")
