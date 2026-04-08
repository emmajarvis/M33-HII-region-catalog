import numpy as np
import numpy.typing as npt
import orb
from IPython.utils.capture import capture_output

from orcs_process_mod import SpectralCube
from util import transform_position


class AbsSpectrum:
    """
    Builds the absorption spectrum corrected for velocity.
    """

    def __init__(
        self,
        mask: npt.NDArray[np.bool_],
        flux_ha_map: npt.NDArray[np.float64],
        cont_snr: npt.NDArray[np.float64],
        flux_ha_thr: float = 1e-19,
        cont_snr_thr: float = 3,
    ) -> None:
        """
        Initialization of the class.

        Parameters
        ----------
        mask : npt.NDArray[np.bool_]
            Mask map to select a region where we want to extract the absorption spectra.
        flux_ha_map : npt.NDArray[np.float64]
            Flux H-alpha map.
        cont_snr : npt.NDArray[np.float64]
            Continuum S/N in the SN3 filter.
        flux_ha_thr : float
            Threshold (upper limit) of H-alpha emission.
        cont_snr_thr : float
            Threshold (lower limit) of the continuum S/N in SN3 filter.
        """
        # Pixels to use to build the absorption spectra
        self.__px2use = (
            (mask == 1) & (flux_ha_map < flux_ha_thr) & (cont_snr > cont_snr_thr)
        )

    def set_px2use(
        self,
        mask: npt.NDArray[np.bool_],
        flux_ha_map: npt.NDArray[np.float64],
        cont_snr: npt.NDArray[np.float64],
        flux_ha_thr: float = 1e-19,
        cont_snr_thr: float = 3,
    ) -> None:
        """
        Modifies the array of pixels (True/False) used to build the absorption spectrum.

        Parameters
        ----------
        mask : npt.NDArray[np.bool_]
            Mask map to select a region where we want to extract the absorption spectra.
        flux_ha_map : npt.NDArray[np.float64]
            Flux H-alpha map.
        cont_snr : npt.NDArray[np.float64]
            Continuum S/N in SN3 filter.
        flux_ha_thr : float
            Threshold (upper limit) of H-alpha emission.
        cont_snr_thr : float
            Threshold (lower limit) of the continuum S/N in SN3 filter.
        """
        self.__px2use = (
            (mask == 1) & (flux_ha_map < flux_ha_thr) & (cont_snr > cont_snr_thr)
        )

    def get_px2use(self) -> np.ndarray:
        """
        Returns the array of pixels (True/False) used to build the absorption spectrum.
        """
        return self.__px2use

    def get_abs_spectrum(
        self,
        cube: SpectralCube,
        vel_map: npt.NDArray[np.float64],
        noise_map: npt.NDArray[np.float64],
        sky: npt.NDArray[np.float64],
        fwhm_sps: float = 0.0,
        transform: dict | None = None,
        **kwargs
    ) -> tuple[orb.fft.RealSpectrum, float]:  # type: ignore --> HACK: ignore an error
        """
        Returns the absorption spectrum corrected for Doppler shift and an estimate of the noise.

        Parameters
        ----------
        cube : SpectralCube
            Data cube.
        vel_map : npt.NDArray[np.float64]
            Velocity map in km/s (observed line-of-sight velocity; i.e., it should
            not be corrected for OH lines and it should not be a heliocentric
            velocity map).
        noise_map: npt.NDArray[np.float64]
            Noise map.
        sky : npt.NDArray[np.float64]
            Sky spectrum to subtract.
        fwhm_sps : float
            Spectral resolution (FWHM) of a stellar population synthesis model to
            give to PPXF (units: Å).
        transform : dict | None
            Tanslation and rotation parameters (for SN1 and SN2 filters).
        **kwargs
            All other keyword arguments are passed on to the
            ``orb.fft.RealSpectrum.get_corr_spectrum_from_region`` call.

        Returns
        -------
        spectrum : orb.fft.RealSpectrum
            Absorption spectrum corrected for Doppler shift.
        noise : float
            Noise estimate in spectrum.
        """
        # Region where to extract the absorption spectrum (in SN3 filter)
        y, x = np.where(self.__px2use)

        # Velocity at each pixel
        vel = vel_map[y, x]

        # Region where to extract the absorption spectrum (in SN1 or SN1 filters)
        if transform is not None:
            x, y = transform_position(x, y, transform)

        # Extraction the absorption spectrum
        with capture_output() as cap:
            spectrum = cube.get_corr_spectrum_from_region(
                (x, y), vel, subtract_spectrum=sky, **kwargs
            )

        # Mean wavelength (in cm-1) in the data cube
        meanlambda = (cube.params.axis_min + cube.params.axis_max) / 2

        # Mean FWHM of the data cube (in Å)
        fwhm = 10 * orb.utils.spectrum.fwhm_cm12nm(spectrum.params.line_fwhm, meanlambda)  # type: ignore --> HACK: ignore an error

        # Change the spectrum's resolution if its mean FWHM is
        # lower than that of the stellar population synthesis library
        if fwhm < fwhm_sps:
            # To change the spectrum's resolution, the axis of
            # wavenumbers must be regularly sampled
            axis = spectrum.axis.data
            spectrum.axis.data = cube.params.base_axis

            spectrum = spectrum.change_resolution(1e8 / meanlambda / fwhm_sps)

            spectrum.axis.data = axis

        # Noise estimate
        noise = np.sqrt( np.sum(noise_map[y, x] ** 2) )

        if "mean_flux" in kwargs and kwargs["mean_flux"]:
            noise /= spectrum.params.pixels

        return spectrum, noise
