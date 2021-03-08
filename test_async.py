from queue import Empty, Full
import time
import multiprocessing as mp
from multiprocessing import Queue, Process
import cv2

# async def count():
#     print("One")
#     await asyncio.sleep(1)
#     print("Two")

# async def main():
#     await asyncio.gather(count(), count(), count())

hehe = 2


def read_data(queue):
    global hehe
    cam = cv2.VideoCapture(0)
    while (True):
        ret, frame = cam.read()
        hehe = 3
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        if (ret == True):
            try:
                queue.put_nowait(frame)
            except Full:
                pass


if __name__ == "__main__":
    mp.set_start_method('spawn')
    q = Queue(maxsize=1)
    read_thread = mp.Process(target=read_data, args=(q, ))
    read_thread.start()
    while True:
        cv2.imshow("ASd", q.get())
        key = cv2.waitKey(1) & 0xFF
        print("hehe", hehe)
        time.sleep(0.5)
        if key == ord('q'):
            # read_thread.run()
            read_thread.terminate()
            read_thread.join()
            exit()
