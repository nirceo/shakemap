# The Hazard Library
# Copyright (C) 2012-2014, GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module exports :class:`BooreEtAl2014`
"""
from __future__ import division

import numpy as np

from scipy.constants import g
from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA,IA,PGD,IH

class TiberiEtAl2018(GMPE):
    """
     Modified by Lara in May 2018
    """
    #: Supported tectonic region type is active shallow crust
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration,
    #: peak ground velocity and peak ground acceleration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        PGV,
        SA,
        IA,
        PGD,
        IH
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GREATER_OF_TWO_HORIZONTAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total, see equation 2, pag 106.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters is Vs30
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameters are magnitude, and rake.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', 'rake'))

    #: Required distance measure is R_epicentral
    REQUIRES_DISTANCES = set(('rjb', 'repi',))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extracting dictionary of coefficients specific to required
        # intensity measure type.
        C = self.COEFFS[imt]
        #if isinstance(imt, (PGA, PGV)):
        #    imt_per = 0.0
        #else:
        #    imt_per = imt.period

        imean = (self._get_magnitude_scaling_term(C, rup) +
                self._get_path_scaling(C, dists, rup.mag) +
                self._get_site_scaling(C, sites))
        istddevs = self._get_stddevs(C, stddev_types, num_sites=len(sites.vs30))

        # Convert units to g,
        # but only for PGA and SA (not PGV):
        if isinstance(imt, (PGA, SA)):
            mean = np.log((10.0 ** (imean - 2.0)) / g)
        else:
            mean = np.log(10.0 ** imean)       
	# Return stddevs in terms of natural log scaling
        stddevs = np.log(10.0 ** np.array(istddevs))
        #mean_LogNaturale = np.log((10 ** mean) * 1e-2 / g)

        return mean, stddevs

#----------modified-------------------------------------------------------------------
    def _get_magnitude_scaling_term(self, C, rup):
        """
        Compute the second term of the equation:
	a + b Mw + e Mw^2 
        """
        mag_term = C["a"] + C["b"]*rup.mag# + C["e"]*(rup.mag**2)
        
        return mag_term
 
    def _get_path_scaling(self, C, dists, mag):
        """
        Compute the third term of the equation:

        `c*log10((R2 + d2)1/2) ``
        """
        rval = np.sqrt(dists.repi ** 2 + C['d'] ** 2)
        return (C['c'] * np.log10(rval))

    def _get_site_scaling(self, C, sites):
        """
        Compute the fourth term of the equation 1 described on paragraph :
        The functional form Fs in Eq. (1) represents the site amplification and
        it is given by FS = sj Cj , for j = 1,...,5, where sj are the
        coefficients to be determined through the regression analysis,
        while Cj are dummy variables used to denote the five different EC8
        site classes
        """
        ssa, ssb, ssc, ssd, sse = self._get_site_type_dummy_variables(sites)

        return (C['sA'] * ssa) + (C['sB'] * ssb) + (C['sC'] * ssc) + \
            (C['sD'] * ssd) + (C['sE'] * sse)

 
    def _get_site_type_dummy_variables(self, sites):
        """
        Get site type dummy variables, five different EC8 site classes
        he recording sites are classified into 5 classes,
        based on the shear wave velocity intervals in the uppermost 30 m, Vs30,
        according to the EC8 (CEN 2003):
        class A: Vs30 > 800 m/s
        class B: Vs30 = 360 − 800 m/s
        class C: Vs30 = 180 - 360 m/s
        class D: Vs30 < 180 m/s
        class E: 5 to 20 m of C- or D-type alluvium underlain by
        stiffer material with Vs30 > 800 m/s.
        """
        ssa = np.zeros(len(sites.vs30))
        ssb = np.zeros(len(sites.vs30))
        ssc = np.zeros(len(sites.vs30))
        ssd = np.zeros(len(sites.vs30))
        sse = np.zeros(len(sites.vs30))

        # Class E Vs30 = 0 m/s. We fixed this value to define class E
        idx = (np.fabs(sites.vs30) < 1E-10)
        sse[idx] = 1.0
        # Class D;  Vs30 < 180 m/s.
        idx = (sites.vs30 >= 1E-10) & (sites.vs30 < 180.0)
        ssd[idx] = 1.0
        # SClass C; 180 m/s <= Vs30 <= 360 m/s.
        idx = (sites.vs30 >= 180.0) & (sites.vs30 < 360.0)
        ssc[idx] = 1.0
        # Class B; 360 m/s <= Vs30 <= 800 m/s.
        idx = (sites.vs30 >= 360.0) & (sites.vs30 < 800)
        ssb[idx] = 1.0
        # Class A; Vs30 > 800 m/s.
        idx = (sites.vs30 >= 800.0)
        ssa[idx] = 1.0
        return ssa, ssb, ssc, ssd, sse

    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        Return standard deviations as defined in table 1.
        """
        assert all(stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            for stddev_type in stddev_types)
        stddevs = [np.zeros(num_sites) + C['sigmatot'] for _ in stddev_types]
        return stddevs
#------------------------------------------------------------------------------------
 
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT             a           b            c            d           sA           sB           sC           sD           sE     sigmatot
    pgv     -1.679506     0.865751   -2.074698     8.478378     0.000000     0.101583     0.350503     0.132607     0.240249     0.406312
    pga      1.132003     0.717163   -2.523451    11.001275     0.000000     0.118430     0.283016    -0.024014     0.339207     0.464516
    ia      -3.476919     1.373298   -3.189469     6.631724     0.000000     0.177052     0.457079    -0.038237     0.540782     0.743363
    ih      -1.553305     0.890095   -1.874366     7.496964     0.000000     0.104991     0.367493     0.184603     0.210600     0.386165
    pgd     -4.487469     1.092399   -1.582077     4.746987     0.000000     0.107645     0.424684     0.300167     0.093353     0.393106
    0.300    0.321608     0.817793   -2.188081    12.245043     0.000000     0.134044     0.354773     0.142426     0.167675     0.417201
    1.000   -2.609323     0.984937   -1.324107     4.097013     0.000000     0.131329     0.427318     0.550000     0.050678     0.377312
    3.000   -3.924920     1.061718   -1.237628     2.630822     0.000000     0.074432     0.390749     0.182565     0.045685     0.368224
    """)                                                                                        

