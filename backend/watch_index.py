# watch_index.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import subprocess, time

DATA = Path("data")

class Handler(FileSystemEventHandler):
    def on_any_event(self, event):
        if not event.is_directory:
            print("Mudança detectada, reconstruindo índice…")
            subprocess.run(["python", "build_index_incremental.py"])

if __name__ == "__main__":
    obs = Observer()
    obs.schedule(Handler(), str(DATA), recursive=True)
    obs.start()
    print("Observando /data (Ctrl+C para sair)")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()
