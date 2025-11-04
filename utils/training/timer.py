
from pathlib import Path
import time

from LumenGSLAM.utils.general import my_logger



class Timer:
    def __init__(self):

        self.start_time = 0.

        self.single_iter_time = 0.

        self.running_time = 0.
        self.num_iters = 0

        self.num_phases = 0

    def start(self):
        self.start_time = time.time()
        self.single_iter_time = 0.

    def iter_end(self):
        self.single_iter_time = time.time() - self.start_time

        self.num_iters += 1

    def phase_end(self):
        self.running_time += self.single_iter_time
        self.num_phases += 1

    def get_average_iter_time(self):
        return self.running_time / self.num_iters if self.num_iters > 0 else 0.

    def get_average_phase_time(self):
        return self.running_time / self.num_phases if self.num_phases > 0 else 0.

    def get_single_iter_time(self, n):
        return self.single_iter_time / n if n > 0 else 0.


class TrainingTimer:
    def __init__(self):

        self.tracking_timer = Timer()
        self.mapping_timer = Timer()

        self.total_time = 0.
        self.num_frames = 0

    def start(self):
        self.num_frames += 1

    def frame_end(self):
        self.total_time += self.tracking_timer.get_average_phase_time() + self.mapping_timer.get_average_phase_time()

    def tracking_start(self):
        self.tracking_timer.start()

    def tracking_iter_end(self):
        self.tracking_timer.iter_end()

    def tracking_end(self):
        self.tracking_timer.phase_end()

    def mapping_start(self):
        self.mapping_timer.start()

    def mapping_iter_end(self):
        self.mapping_timer.iter_end()

    def mapping_end(self):
        self.mapping_timer.phase_end()

    def get_tracking_iter_time(self):
        return self.tracking_timer.get_average_iter_time()

    def get_tracking_time(self):
        return self.tracking_timer.get_average_phase_time()

    def get_mapping_iter_time(self):
        return self.mapping_timer.get_average_iter_time()

    def get_mapping_time(self):
        return self.mapping_timer.get_average_phase_time()

    def get_tracking_single_iter_time(self, n):
        return self.tracking_timer.get_single_iter_time(n)

    def get_mapping_single_iter_time(self, n):
        return self.mapping_timer.get_single_iter_time(n)

    def get_average_frame_time(self):
        return self.total_time / self.num_frames

    def get_times(self):
        return ((self.get_tracking_iter_time(), self.get_tracking_time()) ,
                (self.get_mapping_iter_time(), self.get_mapping_time()),
                (self.get_average_frame_time(), self.total_time))

    def save_info(self, dst, name='time_report.txt'):
        dst = Path(dst)
        my_logger.info(f'saving times to {dst / name}')
        with open(dst / name, "w") as f:
            (track_iter, track_tot), (map_iter, map_tot), (frame_avg, tot) = self.get_times()
            if track_iter > 0.:
                f.write("TRACKING:\n")
                f.write(f"     - average iter time: {track_iter:.4f}\n")
                f.write(f"     - average tracking time: {track_tot:.4f}\n\n")

            f.write("MAPPING:\n")
            f.write(f"     - average iter time: {map_iter:.4f}\n")
            f.write(f"     - average mapping time: {map_tot:.4f}\n\n")

            f.write("TOTAL:\n")
            f.write(f"     - average frame time: {frame_avg:.4f}\n")
            f.write(f"     - total time: {tot:.4f}\n\n")


