import os
import numpy as np
import pandas as pd


class TsWindowPublisher:
    def __init__(self, dataset: np.ndarray, window_length: int, subscribers: list, base_anomaly_save_path=None, debug=False) -> None:
        self.dataset = dataset
        self.update_index = window_length - 1
        self.window = dataset[:window_length]
        self.window_length = window_length
        self.subscribers = subscribers
        self.last_update_length = window_length
        self.last_added = dataset[:window_length]
        self.last_removed = np.zeros(())
        self.base_anomaly_save_path = base_anomaly_save_path
        self.debug = debug

    def update_window(self, step_size=1):
        if self.debug:
            from time import time
            t1 = time()
        self.update_index += step_size
        self.last_removed = self.window[:step_size]
        self.window = self.dataset[self.update_index - self.window_length + 1 : self.update_index + 1]
        self.last_added = self.window[-step_size:]
        self.last_update_length = step_size
        if self.debug:
            print(f'tsWindowPublisher at {hex(id(self))} update after {time()-t1:.6f}s')
        for subscriber in self.subscribers:
            subscriber.notify(step_size=step_size)
        return 0

    def insert_artificial_anomaly(self, offset=100):
        size, up, plateau, goals = 20, 8, 0, [10.0]
        samples, channels = self.dataset.shape
        down = size - up - plateau
        pos = min(len(self.dataset)-(down+100), self.update_index + offset)
        channel_means_stds = np.stack([np.mean(self.dataset[pos - (size + plateau) // 2:pos + (size + plateau) // 2], axis=0),
                                      np.std(self.dataset[pos - (size + plateau) // 2:pos + (size + plateau) // 2], axis=0)])
        for ch in range(channels):
            error = np.zeros((size + plateau, channels))
            goal = goals[np.random.randint(0, max(1, len(goals) - 1))]
            # print(f'Goal: {goal}')
            for i in range(up):
                error[i][ch] = error[max(0, i - 1)][ch] + np.random.normal(loc=goal / up + channel_means_stds[0, ch], scale=goal / (up * 2) + channel_means_stds[1, ch])
                # error[i][ch] = error[max(0, i - 1)][ch] + np.random.normal(loc=goal / up, scale=goal / (up * 2))
            for i in range(down + plateau):
                if plateau != 0 and down // 2 < i < down // 2 + plateau:
                    # Just slight variation (as loc=0.0) in plateau
                    error[up + i][ch] = error[up - 1 + i][ch] + np.random.normal(loc=0.0, scale=0.05 * goal)
                else:
                    error[up + i][ch] = error[up - 1 + i][ch] - \
                                        np.random.normal(loc=goal / down + channel_means_stds[0, ch], scale=goal / (down * 2) + channel_means_stds[1, ch])
            self.dataset[pos - (size + plateau) // 2:pos + (size + plateau) // 2, ch] += error[:, ch]

        if self.base_anomaly_save_path is not None:
            if not os.path.exists(self.base_anomaly_save_path):
                os.makedirs(self.base_anomaly_save_path)
            entry = pd.DataFrame(self.dataset[pos - (size + plateau) // 2:pos + (size + plateau) // 2], columns=[
                f'ch{ch}' for ch in range(channels)], index=[i for i in range(pos - (size + plateau) // 2, pos + (size + plateau) // 2)])
            save_path = f'{self.base_anomaly_save_path}/anomaly_{pos - (size + plateau) // 2}-{pos + (size + plateau) // 2}.csv'
            if not os.path.exists(save_path):
                entry.index.name = 'update_index'
                entry.to_csv(save_path, mode='w')
            else:
                entry.to_csv(save_path, mode='a', header=False)

    def add_subscriber(self, subscriber):
        self.subscribers.append(subscriber)
        return 0

    def remove_subscriber(self, subscriber):
        self.subscribers.remove(subscriber)
        return 0

    def get_update_index(self):
        return self.update_index

    def get_window(self):
        return self.window

    def get_window_length(self):
        return self.window_length