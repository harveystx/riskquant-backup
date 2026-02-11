# from __future__ import annotations
#
# import os
# import tracemalloc
#
# import psutil
#
#
# def trace_memory_top_lines(top_n: int = 25) -> None:
#     tracemalloc.start()
#     snap = tracemalloc.take_snapshot()
#     stats = snap.statistics("lineno")
#
#     print(f"Top {top_n} allocation sites:")
#     for stat in stats[:top_n]:
#         size_kb = stat.size / 1024
#         print(f"{size_kb:10.1f} KiB  {stat.traceback.format()[-1]}")
#
#
# def print_process_memory(stage: str = "") -> None:
#     """
#     Print current process memory usage in MB.
#     Designed for diagnostics, not for hot paths.
#     """
#     p = psutil.Process(os.getpid())
#     mem_mb = p.memory_info().rss >> 20  # shift bits instead of float division
#     label = f"[{stage}] " if stage else ""
#     print(f"{label}Process memory (RSS): {mem_mb} MB")
