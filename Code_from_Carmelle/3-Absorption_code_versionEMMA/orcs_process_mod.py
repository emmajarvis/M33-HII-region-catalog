import logging

import numpy as np
import numpy.typing as npt
import orb
from orcs import process
from scipy.stats import binned_statistic


class SpectralCube(process.SpectralCube):
    """
    The child class of ORCS spectral cube general processing class
    (`orcs.process.SpectralCube`).
    """

    def __init__(self, cube_path: str, debug: bool = False, **kwargs) -> None:
        """
        Class initialization.

        Parameters
        ----------
        cube_path : str
            Path to the HDF5 cube.
        debug : bool
            ... ?
        **kwargs
            Kwargs are `orb.cube.SpectralCube` properties.
        """
        super().__init__(cube_path, debug, **kwargs)

    def _get_data_from_region(self, region: npt.NDArray[np.float64]) -> npt.NDArray:
        """
        Return a list of vectors extracted along the 3rd axis at the pixel
        positions defined by a list of pixels.

        Parameters
        ----------
        region : npt.NDArray[np.float64]
            Array of pixels' positions, i.e., (x_positions_1d_array, y_positions_1d_array).

        Returns
        -------
        Data along the 3rd axis.
        """
        # Corners of the quadrant
        xmin = np.nanmin(region[0])
        xmax = np.nanmax(region[0]) + 1
        ymin = np.nanmin(region[1])
        ymax = np.nanmax(region[1]) + 1

        # Extract data from the quadrant of cube
        quadrant = self.get_data(xmin, xmax, ymin, ymax, 0, self.dimz)

        return quadrant[region[0] - xmin, region[1] - ymin, :]

    def get_data_from_region(self, region: str | npt.ArrayLike) -> npt.NDArray:
        """
        Wrapper around the ``_get_data_from_region`` method to automatically
        divide the region into smaller regions if it's too large.

        Parameters
        ----------
        region : str | npt.ArrayLike
            A ds9-like region file or an ``ArrayLike`` object of pixels' positions,
            i.e., (x_positions_1d_array, y_positions_1d_array).

        Returns
        -------
        Data along the 3rd axis.
        """
        # Maximal number of spectra to extract
        SPECTRA_LIMIT = 200_000

        if isinstance(region, str):
            region = self.get_region(region)

        region = np.array(region)

        # Checking region's format
        if len(region) != 2:
            raise TypeError("badly formatted region.")
        if not region[0].size == region[1].size:
            raise TypeError("badly formatted region.")
        if not region.dtype.type is np.int_:
            raise TypeError("Pixels' positions should be an integer")

        # Checking the number of spectra to extract
        if region[0].size > SPECTRA_LIMIT:
            raise Exception(
                f"Maximal number of spectra to extract (i.e. {SPECTRA_LIMIT}) have been exceeded; try a smaller region"
            )

        # ...
        if region[0].size == 1:
            return np.atleast_2d(self[int(region[0]), int(region[1]), :])

        # Corners of the quadrant
        xmin = self.validate_x_index(np.nanmin(region[0]), clip=False)
        xmax = self.validate_y_index(np.nanmax(region[0]), clip=False) + 1
        ymin = self.validate_x_index(np.nanmin(region[1]), clip=False)
        ymax = self.validate_y_index(np.nanmax(region[1]), clip=False) + 1

        # Number of splits
        n = 2 * ((xmax - xmin) // 400 * (ymax - ymin) // 400)

        if n == 0:
            return self._get_data_from_region(region)

        # Arrays of pixels along the x axis within the quadrant
        x = np.arange(xmin, xmax + 1, 1)

        # Split the arrays of pixels along the x axis
        # within the quadrant into `n` sub-arrays
        x_splited = np.array_split(x, n)

        # List of subregions
        subregions = list()

        for split in x_splited:
            # Verifying if the pixels within the region are
            # present in the sub-array of quadrant pixels.
            idx = np.isin(region[0], split)

            # Add subregions to a list
            if np.count_nonzero(idx) > 0:
                subregions.append((region[0][idx], region[1][idx]))

        # Extracting spectra from each subregion
        spectra = np.empty((0, self.dimz), np.ndarray)

        for subregion in subregions:
            if subregion[0].size == 1:
                spectra = np.vstack(
                    (spectra, self[int(subregion[0]), int(subregion[1]), :])
                )
            else:
                spectra = np.vstack((spectra, self._get_data_from_region(subregion)))

        return spectra

    def get_corr_spectrum_from_region(
        self,
        region: str | list | tuple,
        vel: npt.NDArray[np.float64],
        median: bool = False,
        mean_flux: bool = False,
        subtract_spectrum: npt.NDArray[np.float64] | None = None,
    ) -> orb.fft.RealSpectrum:  # type: ignore --> HACK: To ignore an error
        """
        Return the integrated spectrum of a given region corrected for Doppler shift.

        Parameters
        ----------
        region : str | list | tuple
            A ds9-like region file or an ``ArrayLike`` object of pixels' positions,
            i.e., (x_positions_1d_array, y_positions_1d_array).
        vel : npt.NDArray[np.float64]
            Velocity in km/s at each pixel's position given in ``region`` (observed
            line-of-sight velocity; i.e., it should not be corrected for OH lines and
            it should not be a heliocentric velocity map).
        median : bool
            If ``True``, a median is used instead of a mean to combine spectra. As the
            resulting spectrum is integrated, the median value of the combined spectra
            is then scaled to the number of integrated pixels.
        mean_flux : bool
            If ``True``, the mean spectrum (i.e., per pixel flux) is returned.
        subtract_spectrum : npt.NDArray[np.float64] | None
            Spectrum (sky or background) to subtract the data.

        Returns
        -------
        spectrum : orb.fft.RealSpectrum
            Integrated spectrum corrected for Doppler shift.
        """
        if not self.has_wavenumber_calibration():
            logging.warning(
                "Spectral cube is not calibrated in wavenumber, a large region may result in a deformation of the ILS."
            )

        # Extracting the spectra from region
        spectra = self.get_data_from_region(region)
        # region = np.array(region)

        # Apply flux calibration if needed (when the real data is in counts).
        if self.has_flux_calibration() and self.get_level() >= 3:
            spectra *= self.params.flambda / self.dimz / self.params.exposure_time

        # Subtracting a spectrum (background or sky)
        if not subtract_spectrum is None:
            spectra -= subtract_spectrum

        # Compute axis
        calib_coeff = np.nanmean(self.get_calibration_coeff_map()[region])
        calib_coeff_orig = np.nanmean(self.get_calibration_coeff_map_orig()[region])
        axis_orig = orb.utils.spectrum.create_cm1_axis(  # type: ignore --> HACK: To ignore an error
            self.dimz, self.params.step, self.params.order, corr=calib_coeff
        )

        # Calculate the number of integrated pixels
        params = dict(self.params)
        params["pixels"] = len(spectra)

        # Compute counts and error
        counts = np.nansum(self.get_deep_frame().data[region])
        err = np.ones(self.dimz, dtype=float) * np.sqrt(counts * self.get_gain())
        err = orb.core.Cm1Vector1d(err, axis_orig, params=params)  # type: ignore --> HACK: To ignore an error

        # Compute counts and error
        counts = np.nansum(self.get_deep_frame().data[region])
        err = np.ones(self.dimz, dtype=float) * np.sqrt(counts * self.get_gain())
        err = orb.core.Cm1Vector1d(err, axis_orig, params=params)  # type: ignore --> HACK: To ignore an error

        if isinstance(self.params.flambda, float):
            flambda = np.ones(self.dimz, dtype=float) * self.params.flambda

        else:
            flambda = np.copy(self.params.flambda)

        flambda = orb.core.Cm1Vector1d(flambda, self.get_base_axis(), params=params)  # type: ignore --> HACK: To ignore an error

        err = err.multiply(flambda)

        params["source_counts"] = counts
        params["calib_coeff"] = calib_coeff
        params["calib_coeff_orig"] = calib_coeff_orig

        # Corrected axis for Doppler shift
        axis_corr = axis_orig - orb.utils.spectrum.line_shift(
            vel.reshape((-1, 1)),
            axis_orig,
            wavenumber=True,
        )

        # Flattening the matrices of the corrected axis for Doppler shift and spectra.
        axis_corr = axis_corr.flatten().astype(np.float64)
        spectra = spectra.flatten()

        # Combinning the spectra by computing the binned statistic (sum or median)
        axis = binned_statistic(axis_corr, axis_corr, bins=self.dimz).statistic

        if not median:
            spectrum = binned_statistic(axis_corr, spectra, np.nansum, self.dimz).statistic  # type: ignore --> HACK: To ignore an error

        else:
            spectrum = binned_statistic(axis_corr, spectra, np.nanmedian, self.dimz).statistic  # type: ignore --> HACK: To ignore an error
            spectrum *= params["pixels"]

        if mean_flux:
            spectrum /= params["pixels"]
            err.data /= params["pixels"]
            params["pixels"] = 1

        # Modifying the filter range
        idx1, idx2 = self.get_filter_range_pix().astype(int)

        params["filter_cm1_min"] = axis[idx1]
        params["filter_cm1_max"] = axis[idx2]
        params["filter_range"] = np.array([axis[idx1], axis[idx2]])

        # Create RealSpectrum object (Note: the axis's sampling should be uniform and for that use
        # the original axis; i.e., not corrected for Doppler shift. It will be replaced later)
        spectrum = orb.fft.RealSpectrum(  # type: ignore --> HACK: To ignore an error
            spectrum, err=err.data, axis=axis_orig, params=params
        )

        # Replacing the axis by the one corrected for Doppler shift
        spectrum.axis.data = axis

        return spectrum
