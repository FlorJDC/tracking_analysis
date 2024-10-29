# -*- coding: utf-8 -*-
"""
Stabilizer

XY and Z are managed independently.
"""

import numpy as _np
import scipy as _sp
import threading as _th
import logging as _lgn
import os as _os
from concurrent.futures import ProcessPoolExecutor as _PPE
import warnings as _warnings


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


def _gaussian2D(grid, amplitude, x0, y0, sigma, offset, ravel=True):
    """Generate a 2D gaussian.

    The amplitude is not normalized, as the function result is meant to be used
    just for fitting centers.

    Parameters
    ----------
    grid: numpy.ndarray
        X, Y coordinates grid to generate the gaussian over
    amplitude: float
        Non-normalized amplitude
    x0, y0: float
        position of gaussian center
    sigma: float
        FWHM
    offset:
        uniform background value
    ravel: bool, default True
        If True, returns the raveled values, otherwise return a 2D array
    """
    x, y = grid
    x0 = float(x0)
    y0 = float(y0)
    a = 1.0 / (2 * sigma**2)
    G = offset + amplitude * _np.exp(
        -(a * ((x - x0) ** 2) + a * ((y - y0) ** 2))
    )
    if ravel:
        G = G.ravel()
    return G


def _gaussian_fit(
    data: _np.ndarray, x_max: float, y_max: float, sigma: float
) -> tuple[float, float, float]:
    """Fit a gaussian to an image.

    All data is in PIXEL units.

    Parameters
    ----------
    data : numpy.ndarray
        image as a 2D array
    x_max : float
        initial estimator of X position of the maximum
    y_max : float
        initial estimator of Y position of the maximum
    sigma : float
        initial estimator of spread of the gaussian

    Returns
    -------
    x : float
        X position of the maximum
    y : float
        y position of the maximum
    sigma : float
        FWHM

    If case of an error, returns numpy.nan for every value

    Raises
    ------
        Should not raise
    """
    try:
        xdata = _np.meshgrid(
            _np.arange(data.shape[0]), _np.arange(data.shape[1]), indexing="ij"
        )
        v_min = data.min()
        v_max = data.max()
        args = (v_max - v_min, x_max, y_max, sigma, v_min)
        popt, pcov = _sp.optimize.curve_fit(
            _gaussian2D, xdata, data.ravel(), p0=args
        )
    except Exception as e:
        _lgr.warning("Error fiting: %s, %s", e, type(e))
        return _np.nan, _np.nan, _np.nan
    return popt[1:4]


class AjustaNPs(_th.Thread):
    """Trackea NPs

    As usual, functions names beggining with an underscore do not belong to the
    public interface.

    Functions that *do* belong to the public interface communicate with the
    running thread relying on the GIL (like setting a value) or in events.
    """

    # Status flags

    # ROIS from the user
    _xy_rois: _np.ndarray = None  # [ [min, max]_x, [min, max]_y] * n_rois
    _last_image: _np.ndarray = _np.empty((50, 50))

    def __init__(
        self,
        nmppx: float,
        *args,
        **kwargs,
    ):
        """Init stabilization thread.

        Parameters
        ----------
        camera:
            Camera. Must implement a method called `get_image`, that returns
            a 2d numpy.ndarray representing the image
        piezo:
            Piezo controller. Must implement a method called `set_position` that
            accepts x, y and z positions
        camera_info: info_types.CameraInfo
            Holds information about camera and (x,y) and z marks relation
        corrector:
            object that provides a response
        callback: Callable
            Callable to report measured shifts. Will receive a `PointInfo`
            object as the only parameter
        """
        super().__init__(*args, **kwargs)
        self._nmpp_xy = nmppx

    def set_xy_rois(self, rois) -> bool:
        """Set ROIs for xy stabilization.

        Can not be used while XY tracking is active.

        Parameters
        ----------
        rois: list
            list of XY rois

        Return
        ------
        True if successful, False otherwise
        """
        self._xy_rois = rois
        return True

    def start_loop(self) -> bool:
        """Start."""
        self._executor = _PPE()
        # prime pool for responsiveness (a _must_ on windows).
        nproc = _os.cpu_count()
        params = [
            [_np.eye(3)] * nproc,
            [1.0] * nproc,
            [1.0] * nproc,
            [1.0] * nproc,
        ]
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            _ = tuple(self._executor.map(_gaussian_fit, *params))
        self.start()
        return True

    def stop_loop(self):
        """Stop tracking and stabilization loop and release resources.

        Must be called from another thread to avoid deadlocks.
        """
        self.join()
        self._executor.shutdown()
        _lgr.debug("Loop ended")


    def _locate_xy_centers(self, image: _np.ndarray) -> _np.ndarray:
        """Locate centers in XY ROIS.

        Returns values in pixels

        Parameter
        ---------
        image: numpy.ndarray
            2D array with the image to process

        Return
        ------
            numpy.ndarray of shape (NROIS, 2) with x,y center in nm
        """
        trimmeds = [
            image[roi[0, 0]: roi[0, 1], roi[1, 0]: roi[1, 1]]
            for roi in self._xy_rois
        ]
        x = self._last_params["x"]
        y = self._last_params["y"]
        s = self._last_params["s"]
        locs = _np.array(
            tuple(self._executor.map(_gaussian_fit, trimmeds, x, y, s))
        )
        self._last_params["x"] = locs[:, 0]
        nanloc = _np.isnan(locs[:, 0])  # if x is nan, y also is nan
        self._last_params["x"][nanloc] = x[nanloc]
        self._last_params["y"] = locs[:, 1]
        self._last_params["y"][nanloc] = y[nanloc]
        self._last_params["s"] = locs[:, 2]
        self._last_params["s"][nanloc] = s[nanloc]
        rv = locs[:, :2] + self._xy_rois[:, 0, :]
        rv *= self._nmpp_xy
        return rv

    def _initialize_last_params(self, image):
        """Initialize fitting parameters.

        All values are *in pixels* inside each ROI.

        TODO: Protect against errors (image must exist, ROIS must fit into
        image, etc.)
        """
        trimmeds = [
            image[roi[0, 0]: roi[0, 1], roi[1, 0]: roi[1, 1]]
            for roi in self._xy_rois
        ]
        pos_max = [
            _np.unravel_index(_np.argmax(data), data.shape) for data in trimmeds
        ]
        sigmas = [data.shape[0] / 3 for data in trimmeds]
        self._last_params = {
            "x": _np.array([p[0] for p in pos_max], dtype=float),
            "y": _np.array([p[1] for p in pos_max], dtype=float),
            "s": _np.array(sigmas, dtype=float),
        }

    def cargar_imagenes(self):
        rv = _np.zeros((30, 1024, 1980), dtype=_np.uint8)
        for c, img in enumerate(rv):
            img[512-4+c: 512+4+c, 990-4+c: 990+4+c] = 120
            img[212-4+c: 212+4+c, 490-4-c: 490+4-c] = 20
        return rv

    def run(self):
        """Run loop."""
        images = self.cargar_imagenes()  # array[n_imagenes, size_x, size_y]
        self._initialize_last_params(images[0])
        self.results = []
        for img in images:
            xy_positions = self._locate_xy_centers(img)
            self.results.append(xy_positions)
        _lgr.debug("Ending loop.")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lup = AjustaNPs(23.5)
    rois = _np.array([
        [[512-20, 512+40],[990-20, 990+40]],
        [[212-20, 212+40],[490-20, 490+40]],
        ])
    lup.set_xy_rois(rois)
    lup.start_loop()
    lup.stop_loop()
    n_rois = lup.results[0].shape[0]
    plt.figure("taki")
    for _ in range(n_rois):
        x = [p[_, 0] for p in lup.results]
        y = [p[_, 1] for p in lup.results]
        plt.plot(x, marker='x', label=f" x roi ={_}")
        plt.plot(y, marker="o", label=f" y roi ={_}")
    plt.legend()
