import os

def create_directories(type="landing"):
    """
    Create necessary directories if they do not exist.
    """
    os.makedirs(f"data/{type}", exist_ok=True)

def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time for {func.__name__} is : {end - start:.2f} seconds")
        return result
    return wrapper