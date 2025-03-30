import queue
import pickle

class BatchQueue:
    def __init__(self, save_file="batch_queue_file.pkl"):
        self.save_file = save_file
        self.batch_queue = queue.Queue(maxsize=4)
        self.loadBatches()

    def addToQueue(self, batch_data):
        if(self.batch_queue.full()):
            self.batch_queue.get()
        self.batch_queue.put(batch_data)
        with open(self.save_file, "wb") as f:
            pickle.dump(list(self.batch_queue.queue), f)

    def loadBatches(self):
        with open(self.save_file, "rb") as f:
            savedBatches = pickle.load(f)
            for batch in savedBatches:
                self.batch_queue.put(batch)