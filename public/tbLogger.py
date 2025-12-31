# tensorboard_logger.py
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from torch.utils.tensorboard import SummaryWriter
from hurst import compute_Hc
from scipy.signal import resample
from typing import Optional
class TensorboardLogger:
    def __init__(self, model_name="run", base_dir="runs"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
        self.writer = SummaryWriter(log_dir=log_dir)

    # ---------- basic logging ----------
    def log_scalar(self, tag, value, epoch):
        self.writer.add_scalar(tag, value, epoch)

    def log_scalars(self, tag, values_dict, epoch):
        self.writer.add_scalars(tag, values_dict, epoch)

    def log_histogram(self, tag, values, epoch):
        self.writer.add_histogram(tag, values, epoch)

    def log_text(self, tag, text, epoch):
        self.writer.add_text(tag, text, epoch)

    def computefractalmetrics(ts, min_len: int = 128):
        """
        Returns (hurst, divider_dim) with guard rails.
        - Hurst via hurst.compute_Hc (fallback not needed here; you can add if you like)
        - Divider dimension via path-length vs scale slope
        """
        ts = np.asarray(ts, dtype=np.float64).ravel()
        if ts.size < min_len or not np.isfinite(ts).all() or np.all(ts == ts[0]):
            return np.nan, np.nan

        # normalize to stabilize estimators
        ts = (ts - ts.mean()) / (ts.std() + 1e-8)

        # Hurst: 'random_walk' tends to be stabler for discrete-ish signals
        try:
            H, _, _ = compute_Hc(ts, kind='random_walk', simplified=True)
            H = float(H)
        except Exception:
            H = np.nan

        # Divider dimension: total variation across scales
        eps_min = max(8 / ts.size, 1e-3)
        eps = np.logspace(np.log10(eps_min), -0.05, num=12)
        Ls, eps_used = [], []
        for e in eps:
            n = max(8, int(ts.size * e))
            tr = resample(ts, n)
            L = np.sum(np.abs(np.diff(tr)))
            if np.isfinite(L) and L > 0:
                Ls.append(L)
                eps_used.append(e)

        if len(Ls) < 2:
            divider = np.nan
        else:
            x = np.log(1.0 / np.asarray(eps_used))
            y = np.log(np.asarray(Ls))
            slope, _ = np.polyfit(x, y, 1)
            divider = float(slope)

        return H, divider

    # ---------- recurrence plot ----------
    def log_recurrence_plot(
        self,
        series,
        epoch,
        m: int = 2,                 # embedding dimension (matches screenshot)
        tau: int = 1,               # delay (matches screenshot)
        percent: float = 2.5,       # % of points set to 1; ~2–5% gives sparse yellow speckles
        metric: str = "euclidean",
        resample_to: Optional[int] = 256,   # resample for stable visuals
        tag: str = "Recurrence/plot",
        cmap: str = "plasma"        # purple->yellow like your image
    ):
        """
        Log a time-delay recurrence plot like the screenshot (m=2, τ=1).
        """
        # --- prepare 1D continuous series ---
        s = np.asarray(series, dtype=np.float64).ravel()
        if s.size < 16:
            self.log_text("Recurrence/warn", f"Series too short at epoch {epoch}", epoch)
            return

        # optional resample so images are comparable across epochs
        if resample_to is not None and s.size != resample_to:
            idx = np.linspace(0, s.size - 1, resample_to)
            s = np.interp(idx, np.arange(s.size), s)

        # z-score to stabilize distances
        s = (s - s.mean()) / (s.std() + 1e-8)

        # --- build recurrence image ---
        rp = RecurrencePlot(
            dimension=m,
            time_delay=tau,
            threshold="point",     # choose a threshold so that a fixed % are 1s
            percentage=float(percent)
        )
        R = rp.fit_transform([s])[0]   # binary image {0,1}

        # --- plot with purple background + yellow diagonal/points ---
        fig = plt.figure(figsize=(3.2, 3.2), dpi=160)
        ax = fig.add_subplot(111)
        ax.imshow(R, origin="lower", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"Recurrence plot (m={m}, τ={tau})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")

        self.writer.add_figure(tag, fig, epoch)
        plt.close(fig)


    # ---------- DFA convenience ----------
    def log_dfa_value(self, dfa_value, epoch):
        self.log_scalar("Fractal/DFA", dfa_value, epoch)

    # ---------- Hurst + Divider dimension ----------
    @staticmethod
    

    def log_fractal_metrics(self, series, epoch):
        """
        Convenience wrapper: computes Hurst + Divider and logs them.
        Returns (H, Divider) so you can also print/store if desired.
        """
        try:
            H, divider = self.compute_fractal_metrics(series)
            self.log_scalar("Fractal/Hurst", H, epoch)
            self.log_scalar("Fractal/DividerDim", divider, epoch)
            return H, divider
        except Exception as e:
            self.log_text("Errors", f"Fractal metrics failed at epoch {epoch}: {e}", epoch)
            return np.nan, np.nan
    

    # ---------- close ----------
    def close(self):
        self.writer.close()
