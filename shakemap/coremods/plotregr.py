# stdlib
import os.path
from collections import OrderedDict

# third party

import matplotlib.pyplot as plt
import numpy as np

# neic imports
from impactutils.io.smcontainers import ShakeMapOutputContainer

# local imports
from shakemap.utils.config import get_config_paths
from .base import CoreModule
from shakelib.utils.imt_string import oq_to_file


class PlotRegr(CoreModule):
    """
    plotregr -- Plot the regression curves from a data file
    """

    command_name = 'plotregr'
    targets = [r'products/.*_regr\.png']
    dependencies = [('products/shake_result.hdf', True)]

    # supply here a data structure with information about files that
    # can be created by this module.
    regr_page = {'title': 'Regression Plots', 'slug': 'regression'}
    contents = OrderedDict.fromkeys(['miRegr',
                                     'pgaRegr',
                                     'pgvRegr',
                                     'iaRegr',
                                     'pgdRegr',
                                     'ihRegr',
                                     'psa[PERIOD]Regr'])
    contents['miRegr'] = {
        'title': 'Intensity Regression',
        'caption': 'Regression plot of macroseismic intensity.',
        'page': regr_page,
        'formats': [{'filename': 'MMI_regr.png',
                     'type': 'image/png'}]
    }

    contents['pgaRegr'] = {
        'title': 'PGA Regression',
        'caption': 'Regression plot of [COMPONENT] peak ground '
                   'acceleration (%g).',
        'page': regr_page,
        'formats': [{'filename': 'pga_regr.png',
                     'type': 'image/png'}]
    }
    contents['pgvRegr'] = {
        'title': 'PGV Regression',
        'caption': 'Regression plot of [COMPONENT] peak ground '
                   'velocity (cm/s).',
        'page': regr_page,
        'formats': [{'filename': 'pgv_regr.png',
                     'type': 'image/png'}]
    }
    contents['iaRegr'] = {
	'title': 'IA Regression',
        'caption': 'Regression plot of [COMPONENT] Arias Intensity '\
                   '(cm/s).',
        'page': regr_page,
        'formats': [{'filename':'IA_regr.png',
                     'type':'image/png'}
        ]
    }
    contents['pgdRegr'] = {
	'title': 'PGD Regression',
        'caption': 'Regression plot of [COMPONENT] peak ground '\
                   'displacement (cm).',
        'page': regr_page,
        'formats': [{'filename':'PGD_regr.png',
                     'type':'image/png'}
        ]
    }
    contents['ihRegr'] = {
	'title': 'IH Regression',
        'caption': 'Regression plot of [COMPONENT] Housner '\
                   'intensity (cm).',
        'page': regr_page,
        'formats': [{'filename':'IH_regr.png',
                     'type':'image/png'}
        ]
    }
    psacap = 'Regression plot of [COMPONENT] [FPERIOD] sec 5% damped ' \
             'pseudo-spectral acceleration(%g).'
    contents['psa[PERIOD]Regr'] = {
        'title': 'PSA[PERIOD] Regression',
        'page': regr_page,
        'caption': psacap,
        'formats': [{'filename': 'psa[0-9]p[0-9]_regr.png',
                     'type': 'image/png'}]
    }

    def execute(self):
        """
        Raises:
            NotADirectoryError: When the event data directory does not exist.
            FileNotFoundError: When the the shake_result HDF file does not
                exist.
        """
        _, data_path = get_config_paths()
        datadir = os.path.join(data_path, self._eventid, 'current', 'products')
        if not os.path.isdir(datadir):
            raise NotADirectoryError('%s is not a valid directory.' % datadir)
        datafile = os.path.join(datadir, 'shake_result.hdf')
        if not os.path.isfile(datafile):
            raise FileNotFoundError('%s does not exist.' % datafile)

        # Open the ShakeMapOutputContainer and extract the data
        oc = ShakeMapOutputContainer.load(datafile)
        if oc.getDataType() != 'grid':
            raise NotImplementedError('plotregr module can only operate on '
                                      'gridded data not sets of points')

        #
        # Cheating here a bit by assuming that the IMTs are the same
        # as the regression IMTs
        #
        rockgrid = {}
        soilgrid = {}
        rocksd = {}
        soilsd = {}
        imtlist = oc.getIMTs('GREATER_OF_TWO_HORIZONTAL')
        for myimt in imtlist:
            rockgrid[myimt], _ = oc.getArray(['regression', 'rock', myimt],
                                             'mean')
            soilgrid[myimt], _ = oc.getArray(['regression', 'soil', myimt],
                                             'mean')
            rocksd[myimt], _ = oc.getArray(['regression', 'rock', myimt],
                                           'std')
            soilsd[myimt], _ = oc.getArray(['regression', 'soil', myimt],
                                           'std')
        distances, _ = oc.getArray(['regression'], 'distances')

        stations = oc.getStationDict()
        oc.close()

        #
        # Make plots
        #
        for myimt in imtlist:
            plt.figure(figsize=(10, 10))

            plt.semilogx(distances, rockgrid[myimt], 'r', label='rock')
            plt.semilogx(distances, soilgrid[myimt], 'g', label='soil')
            plt.semilogx(distances, rockgrid[myimt] + rocksd[myimt], 'r--',
                         label='rock +/- stddev')
            plt.semilogx(distances, rockgrid[myimt] - rocksd[myimt], 'r--')
            plt.semilogx(distances, soilgrid[myimt] + soilsd[myimt], 'g--',
                         label='soil +/- stddev')
            plt.semilogx(distances, soilgrid[myimt] - soilsd[myimt], 'g--')

            for station in stations['features']:
                dist = station['properties']['distance']
                if dist > distances[-1]:
                    continue
                if station['properties']['station_type'] == 'seismic':
                    symbol = '^'
                    if myimt == 'MMI':
                        value = station['properties']['intensity']
                        if value != 'null':
                            plt.semilogx(dist, value, symbol + 'k', mfc='none')
                    else:
                        imtstr = myimt.lower()
                        value = np.nan
                        for chan in station['properties']['channels']:
                            if chan['name'].endswith('Z') or \
                               chan['name'].endswith('U'):
                                continue
                            for amp in chan['amplitudes']:
                                if amp['name'] != imtstr:
                                    continue
                                if amp['flag'] != '' and amp['flag'] != '0':
                                    break
                                if amp['value'] is None or \
                                        amp['value'] == 'null':
                                    break
                                if isinstance(amp['value'], str):
                                    thisamp = float(amp['value'])
                                else:
                                    thisamp = amp['value']
                                if thisamp <= 0:
                                    break
                                if myimt == 'PGV'or myimt == 'IA' or myimt == 'PGD' or myimt == 'IH':
                                    tmpval = np.log(thisamp)
                                else:
                                    tmpval = np.log(thisamp / 100.)
                                if np.isnan(value) or tmpval > value:
                                    value = tmpval
                                break
                        if not np.isnan(value):
                            plt.semilogx(dist, value, symbol + 'k', mfc='none')
                else:
                    symbol = 'o'
                    if myimt == 'MMI':
                        amp = station['properties']['intensity']
                        flag = station['properties']['intensity_flag']
                        if flag == '' or flag == '0':
                            if amp is not None and amp != 'null':
                                if isinstance(amp, str):
                                    value = float(amp)
                                else:
                                    value = amp
                                plt.semilogx(dist, value, symbol + 'k',
                                             mfc='none')
                    else:
                        imtstr = myimt.lower()
                        for thing in station['properties']['pgm_from_mmi']:
                            if thing['name'] != imtstr:
                                continue
                            amp = thing['value']
                            if amp is not None and amp != 'null' and amp != 0:
                                if myimt == 'PGV'or myimt == 'IA' or myimt == 'PGD' or myimt == 'IH':
                                    amp = np.log(amp)
                                else:
                                    amp = np.log(amp / 100.)
                                plt.semilogx(dist, amp, symbol + 'k',
                                             mfc='none')
                            break

            plt.title(self._eventid + ': ' + myimt + ' mean')
            plt.xlabel('Rrup (km)')
            if myimt == 'MMI':
                plt.ylabel('MMI')
            elif myimt == 'PGV':
                plt.ylabel('PGV ln(cm/s)')
            elif myimt == 'IA':
                plt.ylabel('IA ln(cm/s)')
            elif myimt == 'PGD':
                plt.ylabel('PGD ln(cm)')
            elif myimt == 'IH':
                plt.ylabel('IH ln(cm)')
            else:
                plt.ylabel(myimt + ' ln(g)')
            plt.legend()

            fileimt = oq_to_file(myimt)
            pfile = os.path.join(datadir, fileimt + '_regr.png')
            plt.savefig(pfile)
            plt.close()
