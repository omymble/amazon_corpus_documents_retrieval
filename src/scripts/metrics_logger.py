# scripts/metrics_logger.py
import matplotlib.pyplot as plt
import json
import os
import time
import torch
import csv

class MetricsLogger:
    def __init__(self, log_dir):
        self.metrics = {
            'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [],
            'gpu_usage': [], 'learning_rate': [], 'batch_time': []
        }
        self.start_time = time.time()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_metrics(self, train_loss=None, train_accuracy=None, val_loss=None, val_accuracy=None, learning_rate=None, batch_time=None):
        current_time = time.time() - self.start_time
        gpu_usage = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0

        if train_loss is not None:
            self.metrics['train_loss'].append((current_time, train_loss))
        if train_accuracy is not None:
            self.metrics['train_accuracy'].append((current_time, train_accuracy))
        if val_loss is not None:
            self.metrics['val_loss'].append((current_time, val_loss))
        if val_accuracy is not None:
            self.metrics['val_accuracy'].append((current_time, val_accuracy))
        if learning_rate is not None:
            self.metrics['learning_rate'].append((current_time, learning_rate))
        if batch_time is not None:
            self.metrics['batch_time'].append((current_time, batch_time))

        self.metrics['gpu_usage'].append((current_time, gpu_usage))

        self._save_metrics()

    def _save_metrics(self):
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)

        for metric_name, values in self.metrics.items():
            with open(os.path.join(self.log_dir, f'{metric_name}.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['time', metric_name])
                writer.writerows(values)

    def plot_metrics(self):
        for metric_name in self.metrics.keys():
            self._plot_metric(metric_name)

    def _plot_metric(self, metric_name):
        if not self.metrics[metric_name]:
            return

        times, values = zip(*self.metrics[metric_name])
        plt.figure()
        plt.plot(times, values, label=metric_name)
        plt.xlabel('Time (s)')
        plt.ylabel(metric_name.replace('_', ' ').capitalize())
        plt.title(f'{metric_name.replace("_", " ").capitalize()} over Time')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, f'{metric_name}.png'))
        plt.close()
