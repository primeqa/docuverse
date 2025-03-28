import threading
import queue
import time

from sympy.codegen.fnodes import elemental
from tqdm import tqdm


def worker(thread_number, q, msg, size, results):
    """Worker function to process items in the queue."""
    with tqdm(desc=f"{msg}/thread {thread_number}", leave=False,
                  position=thread_number+1, total=2*size) as tk:
        while not q.empty():
            try:
                # Get an item from the queue
                item = q.get_nowait()
                # Simulate processing time
                # time.sleep(0.5)
                tk.update()
                # Process the item (e.g., square the number in this example)
                result = f"Processed {item}: {item ** 2}"
                # Save the result in the shared list
                results.append(result)
            except queue.Empty:
                break
            finally:
                q.task_done()


def main():
    # The input list to process
    # elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    elements = range(1, 100)

    # Create a thread-safe queue and populate it with elements
    q = queue.Queue()
    for element in elements:
        q.put(element)

    # Create a list to store results
    results = []

    # Create worker threads
    num_threads = 4  # Number of threads
    threads = []
    for pid in range(num_threads):
        t = threading.Thread(target=worker,
                             args=(pid, q, f"Processing {pid}", len(elements)/num_threads*2, results))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Print results
    print("All items processed:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
