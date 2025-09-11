"""Minimal tqdm stub for testing."""

class tqdm:
    def __init__(self, iterable=None, desc=None, total=None, **kwargs):
        self.iterable = iterable or []
        self.desc = desc
        self.total = total
    
    def __iter__(self):
        for item in self.iterable:
            yield item
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update(self, n=1):
        pass
    
    def set_description(self, desc):
        pass
    
    def close(self):
        pass