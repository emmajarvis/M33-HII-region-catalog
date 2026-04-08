from glob import glob

import numpy as np
import numpy.typing as npt
import orb
import ppxf.miles_util as lib
import ppxf.ppxf_util as util
from astropy.io import fits
from orb.fft import RealSpectrum
from ppxf.ppxf import ppxf
from scipy.interpolate import interp1d


class PpxfFit:
    """
    Performs the pPXF fit on the absorption spectrum.
    """

    def __init__(
        self,
        spectra: list[RealSpectrum],
        sps_dir: str,
        fwhm_sps: float,
        noise: npt.ArrayLike | None = None,
    ) -> None:
        """
        Class initialization.

        Parameters
        ----------
        spectra : list[RealSpectrum]
            List of absorption spectra (``orb.fft.RealSpectrum`` objects)
        sps_dir : str
            Directory path containing the Stellar Population Synthesis (SPS) templates.
        fwhm_sps : float
            FWHM of the SPS templates
        noise : npt.ArrayLike | None
            Mean noise in absorption spectra.
        """
        self.__spectra = spectra
        self.__noise = noise

        self.__fwhm_sps = fwhm_sps
        self.__sps_path = sps_dir + "/*.fits"

        # Convert wavenumbers to wavelengths (in Å) and combine
        # absorption spectrum in each filter into one spectrum
        self.__lam: list = list()
        self.__flux: list = list()
        filter_ranges: list = list()
        for spectr in self.__spectra:
            # Filter bandpass in channels to crop absorption spectra to filter range
            idx1, idx2 = spectr.get_filter_bandpass_pix()

            self.__lam.extend(np.flip(1e8 / spectr.get_axis()[idx1:idx2]))
            self.__flux.extend(np.flip(spectr.get_real()[idx1:idx2]))

            filter_ranges.append(np.flip(1e8 / spectr.params.filter_range))  # type: ignore --> HACK: To ignore an error

        # Gaps' range betweem filters
        self.__gaps = [
            (filter_ranges[i - 1][1], filter_ranges[i][0])
            for i in range(1, len(filter_ranges))
        ]

        self.__set_fwhm_spectra()  # Sets FWHM for each spectrum
        self.__set_velscale()  # Sets velocity scale per pixel
        self.__set_fwhm_gal()  # Sets FWHM over sps templates' wavelength range

    def __set_fwhm_spectra(self) -> None:
        """
        Measures the mean FWHM for each spectra (units: Å).
        """
        self.__fwhm_spectra = list()

        for spectr in self.__spectra:
            # Mean wavenumber in the spectrum
            mean_wave = (spectr.params.axis_min + spectr.params.axis_max) / 2  # type: ignore

            # Mean FWHM in the absorption spectrum
            fwhm = 10 * orb.utils.spectrum.fwhm_cm12nm(spectr.params.line_fwhm, mean_wave)  # type: ignore --> HACK: To ignore an error

            # Appends FWHM to a list. If the spectral resolution of the absorption spectrum
            # is greater than that of the stellar population synthesis temlates, then it is
            # equal to the spectral resolution of the stellar population synthesis temlates.
            if fwhm < self.__fwhm_sps:
                self.__fwhm_spectra.append(self.__fwhm_sps)
            else:
                self.__fwhm_spectra.append(fwhm)

    def __set_velscale(self) -> None:
        """
        Sets velocity scale per pixel value.
        """
        # Spectrum of a stellar template
        spectrum_temp, header = fits.getdata(glob(self.__sps_path)[0], header=True)  # type: ignore --> HACK: To ignore an error

        # Wavelength axis of stellar spslates
        lam_temp = header["CRVAL1"] + np.arange(header["NAXIS1"]) * header["CDELT1"]

        self.__velscale = float(util.log_rebin(lam_temp, spectrum_temp)[2])

    def __set_fwhm_gal(self) -> None:
        """
        Computes the FWHM over sps templates' wavelength range (units: Å).
        """
        # Import a stellar population synthesis library just to have
        # the correct shape for the FWHM array
        sps = lib.miles(self.__sps_path, self.__velscale, self.__fwhm_sps)

        # Create FWHM array to perform templates convolution to math
        # spectral resolution data cubes wavelength range
        self.__fwhm_gal = np.full(sps.templates[:, 0, 0].shape, self.__fwhm_sps)

        for i in range(len(self.__spectra)):
            lam_min = 1e8 / self.__spectra[i].get_axis().max()
            lam_max = 1e8 / self.__spectra[i].get_axis().min()

            # Wavelengths' indexes inside cube's wavelength range
            idx_range = np.where((sps.lam_temp >= lam_min) & (sps.lam_temp <= lam_max))

            # Assign cube's spectral resolution in units of
            # wavelength over its range
            self.__fwhm_gal[idx_range] = self.__fwhm_spectra[i]

    def __set_noise(
        self, noise: npt.NDArray[np.float64], lam: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes the noise over the entire wavelength range (with all filters combined).

        Parameters
        ----------
        noise : npt.NDArray[np.float64]
            Mean noise in absorption spectra.
        lam : npt.NDArray[np.float64]
            Wavelength axis of all combined filters, including gaps between them (units: Å).

        Returns
        -------
        noise_gal : npt.NDArray[np.float64]
            noise over the entire wavelength range.
        """
        noise_gal = np.ones(lam.shape)

        for i in range(len(self.__spectra)):
            lam_min = 1e8 / self.__spectra[i].get_axis().max()
            lam_max = 1e8 / self.__spectra[i].get_axis().min()

            # Wavelengths' indexes inside cube's wavelength range
            idx_range = np.where((lam >= lam_min) & (lam <= lam_max))

            # Assign cube's mean noise of over its range
            noise_gal[idx_range] = noise[i]

        return noise_gal

    def __interpolate(self) -> tuple:
        """
        Interpolates the absorption spectrum to achieve a uniform increment in the data

        Returns
        -------
        new_lam : npt.NDArray[np.float64]
            Interpolated wavelength axis (units: Å).
        new_flux : npt.NDArray[np.float64]
            Interpolated and normalized flux axis.
        norm_factor : float
            Normalization factor (useful to get the PPXF's best-fit spectrum in flux units).
        """
        # Header of a stellar population synthesis template
        header = fits.getdata(glob(self.__sps_path)[0], header=True)[1]  # type: ignore --> HACK: To ignore an error

        # Create the interpolated arrays
        new_lam = np.arange(min(self.__lam), max(self.__lam), header["CDELT1"])
        new_flux = interp1d(self.__lam, self.__flux, kind="linear")(new_lam)

        # Normalize spectrum to avoid numerical issues
        norm_factor = np.median(new_flux)
        new_flux = new_flux / norm_factor

        # Gaps between the filters are set to 0
        for gap in self.__gaps:
            range_gap = np.where((new_lam > gap[0]) & (new_lam < gap[1]))
            new_flux[range_gap] = 0

        return new_lam, new_flux, norm_factor

    def get_noise(self) -> npt.ArrayLike | None:
        """
        Returns the mean noise
        """
        return self.__noise

    def get_fwhm_spectra(self) -> list:
        """
        Returns the list of the mean FWHM for each spectrum (units: Å).
        """
        return self.__fwhm_spectra

    def get_velscale(self) -> float:
        """
        Returns velocity scale per pixel (units: km/s)
        """
        return self.__velscale

    def get_ppxf_fit(self) -> tuple:
        """
        Performs a PPXF fit.

        Returns
        -------
        pp : ppxf.ppxf.ppxf
            Results of the PPXF fit stored as attributes of the PPXF class.
        norm_factor : float:
            Normalization factor (useful to get the PPXF's best-fit spectrum in flux units).
        """
        # Interpolate spectrum (to get uniform increment in data)
        new_lam, new_flux, norm_factor = self.__interpolate()

        # Importing stellar templates
        sps = lib.miles(self.__sps_path, self.__velscale, self.__fwhm_gal)

        # The stellar templates are reshaped below into a 2-dim array with each
        # spectrum as a column, however we save the original array dimensions,
        # which are needed to specify the regularization dimensions
        reg_dim = sps.templates.shape[1:]
        stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

        # See the pPXF documentation for the keyword REGUL,
        regul = 0  # Desired regularization

        # Wavelength fitted range
        lam_range = (new_lam.min(), new_lam.max())

        # Construct a set of Sinc emission line templates
        gas_templates, gas_names = util.emission_lines(
            sps.ln_lam_temp, lam_range, self.__fwhm_gal, use_sinc=True
        )[:2]

        # Combines the stellar and gaseous templates into a single array
        # During the PPXF fit they will be assigned a different kinematic
        templates = np.column_stack([stars_templates, gas_templates])

        # Rebin the spectrum on an exponential scale
        flux, ln_lam_gal = util.log_rebin(
            lam_range, new_flux, velscale=self.__velscale
        )[:2]

        # If noise is not given (None), then assume a constant noise per pixel.
        if self.__noise is None:
            noise_gal = np.full(flux.shape, 1)

        else:
            noise_gal = self.__set_noise(
                np.asarray(self.__noise) / norm_factor, np.exp(ln_lam_gal)
            )

        # Fit two kinematics components, one for the stars and one for
        # the gas. Assign component=0 to the stellar templates,
        # component=1 to the gas.
        n_stars = stars_templates.shape[1]
        n_gas = len(gas_names)
        component = [0] * n_stars + [1] * n_gas
        gas_component = np.array(component) > 0  # gas_component=True for gas templates

        moments = [2, 2]  # Fit (V, sig) moments=2 for both the stars and the gas
        start = [0, 200]  # (km/s), starting guess for [V, sigma]

        # Adopt the same starting value for both the stars and the gas components
        start = [start, start]

        # Integer vector containing the indices of the good pixels in the 'spectrum'
        # (in increasing order). Only these spectral pixels are included in the fit.
        goodpixels = np.where(flux != 0)[0]

        # return np.exp(ln_lam_gal)

        # Here the actual fit starts
        chi2 = 0  # Expected chi2 from the fit
        while not np.isclose(1, chi2, rtol=0.001):
            pp = ppxf(
                templates,
                flux,
                noise_gal,
                self.__velscale,
                start,
                moments=moments,  # type: ignore --> HACK: To ignore an error
                degree=-1,
                mdegree=8,
                regul=regul,  # type: ignore --> HACK: To ignore an error
                reg_dim=reg_dim,
                component=component,  # type: ignore --> HACK: To ignore an error
                gas_component=gas_component,
                gas_names=gas_names,
                reddening=0,
                gas_reddening=0,
                lam=np.exp(ln_lam_gal),
                lam_temp=sps.lam_temp,
                goodpixels=goodpixels,
            )

            # Rescaling the input noise spectrum so that
            # Chi^2/DOF = Chi^2/goodPixels.size = 1
            chi2 = pp.chi2
            noise_gal *= np.sqrt(chi2)

            # Increase regul and iteratively redo the pPXF until chi2 = 1
            regul += 0.5

        return pp, norm_factor  # type: ignore --> HACK: To ignore an error
