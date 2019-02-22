# third party imports
import numpy as np

# stdlib imports
from openquake.hazardlib.imt import PGA, PGV, SA, IA,PGD,IH
from shakelib.gmice.gmice import GMICE


class tiberi18(GMICE):
    """
    Implements the ground motion intensity conversion equations (GMICE) of
    Tiberi et al. (2018).

    References:
     """

    # -----------------------------------------------------------------------
    #
    # MMI = c2->C1 + c2->C2 * log(Y)  for log(Y) <= c2->T1
    # MMI = C1 + C2 * log(Y)          for c2->T1 < log(Y) <= T1
    # MMI = C3 + C4 * log(Y)          for log(Y) > T1
    #
    #
    # Limit the distance residuals to between 10 and 300 km.
    # Limit the magnitude residuals to between M3.0 and M7.3.
    #
    # -----------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.min_max = (1.0, 10.0)
        self.name = 'Tiberi et al. (2018)'
        self.scale = 'scale_tiberietal18.ps'
        self._constants = {
            self._pga:  {'C1':  2.39, 'C2':  1.84, 'SMMI': 0.5, 'SPGM': 0.02, 'T1' : 0.00},
            self._pgv:  {'C1':  2.47, 'C2':  5.13, 'SMMI': 0.5, 'SPGM': 0.11,  'T1' : 0.00},
            self._ia:   {'C1':  0.99, 'C2':  5.17, 'SMMI': 0.5, 'SPGM': 0.11,  'T1' : 0.00},
            self._ih:   {'C1':  1.69, 'C2':  3.69, 'SMMI': 0.5, 'SPGM': 0.17,  'T1' : 0.00},
            self._pgd:  {'C1':  2.21, 'C2':  7.24, 'SMMI': 0.5, 'SPGM': 0.09,  'T1' : 0.00},
            self._sa03: {'C1':  1.74, 'C2':  1.94, 'SMMI': 0.5, 'SPGM': 0.14, 'T1' : 0.00},
            self._sa10: {'C1':  1.62, 'C2':  3.20, 'SMMI': 0.5, 'SPGM': 0.26, 'T1' : 0.00},
            self._sa30: {'C1':  1.54, 'C2':  4.68, 'SMMI': 0.5, 'SPGM': 0.15, 'T1' : 0.00}
        }

        self.DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
            PGA,
            PGV,
            SA,
            IA,
            PGD,
            IH
        ])

        self.DEFINED_FOR_SA_PERIODS = set([0.3, 1.0, 3.0])

    def getMIfromGM(self, amps, imt, dists=None, mag=None):
        """
        Function to compute macroseismic intensity from ground-motion
        intensity. Supported ground-motion IMTs are PGA, PGV and PSA
        at 0.3, 1.0, and 3.0 sec periods.

        Args:
            amps (ndarray):
                Ground motion amplitude; natural log units; g for PGA and
                PSA, cm/s for PGV.
            imt (OpenQuake IMT):
                Type the input amps (must be one of PGA, PGV, or SA).
                Supported SA periods are 0.3, 1.0, and 3.0 sec.
                `[link] <http://docs.openquake.org/oq-hazardlib/master/imt.html>`
            dists (ndarray):
                Numpy array of distances from rupture (km).
            mag (float):
                Earthquake magnitude.

        Returns:
            ndarray of Modified Mercalli Intensity and ndarray of
            dMMI / dln(amp) (i.e., the slope of the relationship at the
            point in question).
        """  # noqa
        lfact = np.log10(np.e)
        c = self._getConsts(imt)

        #
        # Convert (for accelerations) from ln(g) to cm/s^2
        # then take the log10
        #
        if imt == self._pgv: 
            units = 1.0
        elif imt == self._ia:
            units = 1.0
        elif imt == self._pgd:
            units = 1.0
        elif imt == self._ih:
            units = 1.0
        else:
            units = 981.0

        #
        # Math: log10(981 * exp(amps)) = log10(981) + log10(exp(amps))
        # = log10(981) + amps * log10(e)
        # For PGV, just convert ln(amp) to log10(amp) by multiplying
        # by log10(e)
        #
        lamps = np.log10(units) + amps * lfact
        mmi = np.zeros_like(amps)
        dmmi_damp = np.zeros_like(amps)
 
        idx = amps < c['T1']
        mmi[idx] = c['C2'] + c['C1'] * lamps[idx]
        dmmi_damp[idx] = c['C1'] * lfact

        idx = amps > c['T1']
        mmi[idx] = c['C2'] + c['C1'] * lamps[idx]
        dmmi_damp[idx] = c['C1'] * lfact

        mmi = np.clip(mmi, 1.0, 10.0)
        return mmi, dmmi_damp

    def getGMfromMI(self, mmi, imt, dists=None, mag=None):
        """
        Function to tcompute ground-motion intensity from macroseismic
        intensity. Supported IMTs are PGA, PGV and PSA for 0.3, 1.0, and
        3.0 sec periods.

        Args:
            mmi (ndarray):
                Macroseismic intensity.
            imt (OpenQuake IMT):
                IMT of the requested ground-motions intensities (must be
                one of PGA, PGV, or SA).
                `[link] <http://docs.openquake.org/oq-hazardlib/master/imt.html>`
            dists (ndarray):
                Rupture distances (km) to the corresponding MMIs.
            mag (float):
                Earthquake magnitude.

        Returns:
            Ndarray of ground motion intensity in natural log of g for PGA
            and PSA, and natural log cm/s for PGV; ndarray of dln(amp) / dMMI
            (i.e., the slope of the relationship at the point in question).
        """  # noqa
        lfact = np.log10(np.e)
        c = self._getConsts(imt)
        mmi = mmi.copy()
        ix_nan = np.isnan(mmi)
        mmi[ix_nan] = 1.0

        pgm = np.zeros_like(mmi)
        dpgm_dmmi = np.zeros_like(mmi)

        #
        # MMI 1 to 10
        #
        idx = np.zeros_like(mmi)  == np.zeros_like(mmi)
        pgm[idx] = np.power(10, (mmi[idx] - c['C2']) / c['C1'])
        dpgm_dmmi[idx] = 1.0 / (c['C1'] * lfact)


        if imt == self._pgv: 
            units = 1.0
        elif imt == self._ia:
            units = 1.0
        elif imt == self._pgd:
            units = 1.0
        elif imt == self._ih:
            units = 1.0
        else:
            units = 981.0

        pgm /= units
        pgm = np.log(pgm)
        pgm[ix_nan] = np.nan
        dpgm_dmmi[ix_nan] = np.nan

        return pgm, dpgm_dmmi

    def getGM2MIsd(self):
        """
        Return a dictionary of standard deviations for the ground-motion
        to MMI conversion. The keys are the ground motion types.

        Returns:
            Dictionary of GM to MI sigmas (in MMI units).
        """
        return {self._pga: self._constants[self._pga]['SMMI'],
                self._pgv: self._constants[self._pgv]['SMMI'],
                self._ia: self._constants[self._ia]['SMMI'],
                self._ih: self._constants[self._ih]['SMMI'],
                self._pgd: self._constants[self._pgd]['SMMI'],
                self._sa03: self._constants[self._sa03]['SMMI'],
                self._sa10: self._constants[self._sa10]['SMMI'],
                self._sa30: self._constants[self._sa30]['SMMI']}

    def getMI2GMsd(self):
        """
        Return a dictionary of standard deviations for the MMI
        to ground-motion conversion. The keys are the ground motion
        types.

        Returns:
            Dictionary of MI to GM sigmas (ln(PGM) units).
        """
        #
        # Need to convert log10 to ln units
        #
        lfact = np.log(10.0)
        return {self._pga: lfact * self._constants[self._pga]['SPGM'],
                self._pgv: lfact * self._constants[self._pgv]['SPGM'],
                self._ia: lfact * self._constants[self._ia]['SPGM'],
                self._ih: lfact * self._constants[self._ih]['SPGM'],
               self._pgd: lfact * self._constants[self._pgd]['SPGM'],
                self._sa03: lfact * self._constants[self._sa03]['SPGM'],
                self._sa10: lfact * self._constants[self._sa10]['SPGM'],
                self._sa30: lfact * self._constants[self._sa30]['SPGM']}

    def _getConsts(self, imt):
        """
        Helper function to get the constants.
        """

        if (imt != self._pga and imt != self._pgv and imt != self._sa03 and
                imt != self._sa10 and imt != self._sa30 and imt != self._ia and imt != self._pgd and imt != self._ih):
            raise ValueError("Invalid IMT " + str(imt))
        c = self._constants[imt]
        return (c)
