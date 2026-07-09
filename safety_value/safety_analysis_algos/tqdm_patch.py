from contextlib import contextmanager
import sys

@contextmanager
def tqdm_over(iterable, total=None, desc=None):
    try:
        from tqdm import tqdm
        yield tqdm(iterable, total=total, desc=desc)
    except ImportError:
        # fallback: no progress bar
        yield iterable
