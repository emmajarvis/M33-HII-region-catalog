import os
os.environ["OMP_NUM_THREADS"] = "16"
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


from IPython.utils import io
from astropy.io import fits
from astropy.table import QTable
from astropy.wcs import WCS
from orcs.process import SpectralCube


plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)


EMI_LINES = np.array([3727, 4861, 4959, 5007, 6562, 6548, 6583, 6716, 6731])  # Angstrom

LINE1 = ['[OII]3726']
RANGE1 = [26405, 27195]

LINE2 = ['Hbeta', '[OIII]5007', '[OIII]4959']
RANGE2 = [19493, 20746]

LINE3 = ['Halpha', '[NII]6583', '[NII]6548', '[SII]6716', '[SII]6731']
RANGE3 = [14750, 15495]

LIN = [
    r'[OII]3726',
    r'H$\beta$',
    r'[OIII]5007',
    r'[OIII]4959',
    r'H$\alpha$',
    r'[NII]6583',
    r'[NII]6548',
    r'[SII]6716',
    r'[SII]6731'
]


@dataclass
class IntegratedSpectraContext:
    field: str
    fnum: int
    fprefix: str

    cube1: object
    cube2: object
    cube3: object

    wcs1: object
    wcs2: object
    wcs3: object

    imdeep1: np.ndarray
    imdeep2: np.ndarray
    imdeep3: np.ndarray

    fluxo2: np.ndarray
    fluxo3: np.ndarray
    fluxha: np.ndarray

    height1: np.ndarray
    height2: np.ndarray
    height3: np.ndarray

    velocity_map: np.ndarray
    sigma_map: np.ndarray

    dom1: np.ndarray
    dom2: np.ndarray
    dom3: np.ndarray

    xpic: np.ndarray
    ypic: np.ndarray
    ipic: np.ndarray
    npic: int

    gal_vel: float
    wlines: np.ndarray
    folder: str

    colors: np.ndarray


def readdata(file):
    data, hdr = fits.getdata(file, header=True)
    print('Reading : ', file, data.shape)
    return data, hdr


def get_field_config(field):
    field = str(field)

    if field == 'NW':
        return {
            'cube3': '/arc/projects/signals/M33/M33-NW_SN3.merged.cm1.1.0.hdf5',
            'cube2': '/arc/projects/signals/M33/M33-NW_SN2.merged.cm1.1.0.hdf5',
            'cube1': '/arc/projects/signals/M33/M33-NW.2190905z.SN1.hdf5',
            'fnum': 2,
        }

    elif field == 'NE':
        return {
            'cube3': '/arc/home/emmajarvis/M33/M33-NE_SN3.merged.cm1.1.0.hdf5',
            'cube2': '/arc/projects/signals/M33/M33-NE_SN2.merged.cm1.1.0.hdf5',
            'cube1': '/arc/projects/signals/M33/M33-NE_SN1.merged.cm1.1.0.hdf5',
            'fnum': 1,
        }

    elif field == 'SE':
        return {
            'cube3': '/arc/projects/signals/M33/M33-SE_SN3.merged.cm1.1.0.hdf5',
            'cube2': '/arc/projects/signals/M33/M33-SE_SN2.merged.cm1.1.0.hdf5',
            'cube1': '/arc/projects/signals/M33/M33-SE_SN1.merged.cm1.1.0.hdf5',
            'fnum': 3,
        }

    elif field == 'SW':
        return {
            'cube3': '/arc/projects/signals/M33/M33-SW_SN3.merged.cm1.1.0.hdf5',
            'cube2': '/arc/projects/signals/M33/M33-SW_SN2.merged.cm1.1.0.hdf5',
            'cube1': '/arc/projects/signals/M33/M33-SW_SN1.merged.cm1.1.0.hdf5',
            'fnum': 4,
            
        }

    elif field == '7':
        return {
            'cube3': '/arc/projects/signals/M33_FIELD7_SN3_18BP41.hdf5',
            'cube2': '/arc/projects/signals/M33_FIELD7_SN2_18BP41.hdf5',
            'cube1': '/arc/projects/signals/M33_FIELD7_SN1_18BP41.hdf5',
            'fnum': 7,
        }

    elif field == '8':
        return {
            'cube3': '/arc/projects/signals/M33_FIELD8_SN3_20BP41.hdf5',
            'cube2': '/arc/projects/signals/M33_FIELD8_SN2_20BP41.hdf5',
            'cube1': '/arc/projects/signals/M33_FIELD8_SN1_20BP41.hdf5',
            'fnum': 8,
        }

    elif field == '9':
        return {
            'cube3': '/arc/projects/signals/M33_FIELD9_SN3_19BP41.hdf5',
            'cube2': '/arc/projects/signals/M33_FIELD9_SN2_20BP41.hdf5',
            'cube1': '/arc/projects/signals/M33_FIELD9_SN1_20BP41.hdf5', 
            'fnum': 9,
        }

    else:
        raise ValueError(f"Missing cubes for field '{field}'")


def get_field_prefix(field):
    return 'F' if str(field) in {'5', '6', '7', '8', '9'} else ''


def get_maps_base(field, fprefix):
    return f'/arc/home/emmajarvis/M33/M33-Maps/M33-{fprefix}{field}'


def make_output_folder(field, base_folder=None, timestamp=None):
    """
    Make an output folder automatically dated by the run date.

    Example output:
        5-Intspec_Updated_20260507/Intspec_fig_SE_20260507_143210
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_date = timestamp.split("_")[0]

    if base_folder is None:
        base_folder = f"5-Intspec_Updated_{run_date}"

    folder = f"{base_folder}/Intspec_fig_{field}_{timestamp}"
    os.makedirs(folder, exist_ok=True)

    print("Output folder:", folder)
    return folder


def build_empty_flux_table():
    return QTable(names=[
        'id', 'x', 'y', 'FIELD',
        'F_[OII]3727', 'F_[OII]3727_e', 'SNR_[OII]3727',
        'F_Hbeta', 'F_Hbeta_e', 'SNR_Hbeta',
        'F_[OIII]4959', 'F_[OIII]4959_e', 'SNR_[OIII]4959',
        'F_[OIII]5007', 'F_[OIII]5007_e', 'SNR_[OIII]5007',
        'F_[NII]6548', 'F_[NII]6548_e', 'SNR_[NII]6548',
        'F_Halpha', 'F_Halpha_e', 'SNR_Halpha',
        'F_[NII]6583', 'F_[NII]6583_e', 'SNR_[NII]6583',
        'F_[SII]6716', 'F_[SII]6716_e', 'SNR_[SII]6716',
        'F_[SII]6731', 'F_[SII]6731_e', 'SNR_[SII]6731'
    ])


def load_context(field, base_output_folder=None, output_timestamp=None):
    
    cfg = get_field_config(field)
    fprefix = get_field_prefix(field)
    maps_base = get_maps_base(field, fprefix)
    print(cfg['cube3'])
    cube3 = SpectralCube(cfg['cube3'])
    
    cube2 = SpectralCube(cfg['cube2'])
    cube1 = SpectralCube(cfg['cube1'])
    
    wcs3 = cube3.get_wcs()
    imdeep3, _ = readdata(f'{maps_base}/M33{fprefix}{field}SN3Deep.fits')
    fluxha, _ = readdata(f'{maps_base}/M33{fprefix}{field}-Haflux.fits')
    height3, _ = readdata(f'{maps_base}/M33{fprefix}{field}-SN3Continuum.fits')

    print()
    wcs2 = cube2.get_wcs()
    imdeep2, _ = readdata(f'{maps_base}/M33{fprefix}{field}SN2Deep.fits')
    fluxo3, _ = readdata(f'{maps_base}/M33{fprefix}{field}-OIII5007flux.fits')
    height2, _ = readdata(f'{maps_base}/M33{fprefix}{field}-SN2Continuum.fits')

    print()
    wcs1 = cube1.get_wcs()
    imdeep1, _ = readdata(f'{maps_base}/M33{fprefix}{field}-OII3727flux.fits')
    fluxo2, _ = readdata(f'{maps_base}/M33{fprefix}{field}-OII3727flux.fits')
    height1, _ = readdata(f'{maps_base}/M33{fprefix}{field}-SN1Continuum.fits')

    print()
    velocity_map, _ = readdata(f'{maps_base}/M33{fprefix}{field}-velocity.fits')
    sigma_map, _ = readdata(f'{maps_base}/M33{fprefix}{field}-sigma.fits')

    print()
    peaks = pd.read_csv(f'peak_maps/final_peaks_{fprefix}{field}.csv')
    xpic = peaks['x'].to_numpy()
    ypic = peaks['y'].to_numpy()
    ipic = np.arange(len(xpic), dtype=int)
    npic = len(xpic)

    dom3, _ = readdata(f'DOMAIN_MAPS/Boundary_map_{fprefix}{field}.fits')
    dom2 = dom3
    dom1 = dom3

    print()
    gal_vel = np.nanmean(velocity_map[100:2000, 100:2000])
    print('Galaxy velocity (km/s): ', gal_vel)
    print(r'Main emission lines centroid (A):', EMI_LINES)

    wlines = 1.e8 / (EMI_LINES * (1 + (gal_vel / 3e5)))

    print()
    folder = make_output_folder(
        field,
        base_folder=base_output_folder,
        timestamp=output_timestamp
    )

    cmap = mpl.colormaps['magma']
    colors = cmap(np.linspace(0, 1, 20))

    return IntegratedSpectraContext(
        field=str(field),
        fnum=cfg['fnum'],
        fprefix=fprefix,
        cube1=cube1,
        cube2=cube2,
        cube3=cube3,
        wcs1=wcs1,
        wcs2=wcs2,
        wcs3=wcs3,
        imdeep1=imdeep1,
        imdeep2=imdeep2,
        imdeep3=imdeep3,
        fluxo2=fluxo2,
        fluxo3=fluxo3,
        fluxha=fluxha,
        height1=height1,
        height2=height2,
        height3=height3,
        velocity_map=velocity_map,
        sigma_map=sigma_map,
        dom1=dom1,
        dom2=dom2,
        dom3=dom3,
        xpic=xpic,
        ypic=ypic,
        ipic=ipic,
        npic=npic,
        gal_vel=gal_vel,
        wlines=wlines,
        folder=folder,
        colors=colors,
    )


def get_line_fit_constraints(lines):
    """
    Return amplitude constraints for line groups that should have fixed ratios.

    Current tied ratios:
        [OIII]5007 / [OIII]4959 = 3.0582752761564054
        [NII]6583 / [NII]6548 = 2.967824020860382

        - same amp_def label means amplitudes are tied
        - amp_guess values set the relative ratio within that tied group
    """
    lines = list(lines)

    if lines == ['Hbeta', '[OIII]5007', '[OIII]4959']:
        return {
            # Hbeta independent; [OIII]5007 and [OIII]4959 tied together
            # Line order is: Hbeta, [OIII]5007, [OIII]4959
            'amp_def': ['1', '2', '2'],

            # This makes [OIII]5007 / [OIII]4959 = 3.0582752761564054
            'amp_guess': [1, 3.0582752761564054, 1],
        }

    if lines == ['Halpha', '[NII]6583', '[NII]6548', '[SII]6716', '[SII]6731']:
        return {
            # Halpha independent; [NII]6583 and [NII]6548 tied together;
            # each [SII] line independent
            # Line order is: Halpha, [NII]6583, [NII]6548, [SII]6716, [SII]6731
            'amp_def': ['1', '2', '2', '3', '4'],
            # This makes [NII]6583 / [NII]6548 = 2.967824020860382
            'amp_guess': [1, 2.967824020860382, 1, 1, 1],
        }

    return {}


def fit_spectrum(ireg, x0, y0, lines, signal_range, dom, cube, velocity_map, sigma_map):
    y, x = np.where(dom - 1 == ireg)

    fit_constraints = get_line_fit_constraints(lines)

    with io.capture_output():
        wavenb, spec, fit = cube.fit_lines_in_integrated_region(
            (x, y),
            lines,
            fmodel='sincgauss',
            pos_def=['1'],
            pos_cov=velocity_map[y0, x0],
            sigma_def=['1'],
            sigma_cov=sigma_map[y0, x0],
            mean_flux=False,
            nofilter=False,
            signal_range=signal_range,
            max_iter=5000,
            **fit_constraints
        )

    if fit.__class__.__name__ != 'OutputParams':
        print(ireg, 'no fit')
        return (
            wavenb.astype(float),
            spec.real.astype(float),
            0 * spec.real.astype(float),
            np.nan * np.zeros(len(lines)),
            np.nan * np.zeros(len(lines)),
            np.nan * np.zeros(len(lines)),
            False
        )

    return (
        wavenb.astype(float),
        spec.real.astype(float),
        fit['fitted_vector'].astype(float),
        fit['flux'],
        fit['flux_err'],
        fit['snr'],
        True
    )


def makeplot(codex, codey, codelog, codecmap, alp, codebar,
             imagein, imagevvi, x1, x2, y1, y2, vvimin, vvimax):
    plt.rcParams['axes.linewidth'] = 2
    plt.tick_params(axis='both', direction='out', labelsize=15, length=6, width=2)
    plt.axis([x1, x2, y1, y2])

    cmapp = 'gray'
    if codecmap == 1:
        cmapp = 'rainbow'
    if codecmap == 2:
        cmapp = 'Greys'
    if codecmap == 3:
        cmapp = 'hsv'

    if codelog == 0:
        vvi = imagevvi[y1:y2, x1:x2]
        plt.imshow(
            imagein,
            cmap=cmapp,
            origin='lower',
            vmin=np.nanpercentile(vvi, vvimin),
            vmax=np.nanpercentile(vvi, vvimax),
            alpha=alp
        )

    if codelog == 1:
        vvi = np.log10(imagevvi[y1:y2, x1:x2])
        plt.imshow(
            np.log10(imagein),
            cmap=cmapp,
            origin='lower',
            vmin=np.nanpercentile(vvi, vvimin),
            vmax=np.nanpercentile(vvi, vvimax),
            alpha=alp
        )

    if codebar == 1:
        cbar = plt.colorbar(pad=0.03)
        cbar.ax.tick_params(labelsize=15)

    if codex == 1:
        plt.xlabel('X', size=15)
    if codey == 1:
        plt.ylabel('Y', size=15)


def makeplotspec(ireg, x0, y0,
                 wn1, sp1, fi1,
                 wn2, sp2, fi2,
                 wn3, sp3, fi3,
                 flu, err, snr, lin, plotshow, ctx):
    f = np.nanmax(sp3).real
    facplot = np.floor(np.log10(f)).astype(int)

    sp3p = sp3 * 10 ** (-1 * facplot)
    sp2p = sp2 * 10 ** (-1 * facplot)
    sp1p = sp1 * 10 ** (-1 * facplot)
    fi3p = fi3 * 10 ** (-1 * facplot)
    fi2p = fi2 * 10 ** (-1 * facplot)
    fi1p = fi1 * 10 ** (-1 * facplot)

    ymin, ymax = np.nanmin(sp3p), np.nanmax(sp3p)
    dely = (ymax - ymin) / 20.
    ymin, ymax = ymin - dely, ymax + dely

    imasiz = 30
    x3, y3 = x0, y0
    r3, d3 = ctx.wcs3.wcs_pix2world(x3, y3, 0)
    x2, y2 = ctx.wcs2.wcs_world2pix(r3, d3, 0)
    x2, y2 = int(x2), int(y2)
    x1, y1 = ctx.wcs1.wcs_world2pix(r3, d3, 0)
    x1, y1 = int(x1), int(y1)

    nyy, nxx, ip = 5, 3, 0
    fig = plt.figure(figsize=(6 * nxx, 4.5 * nyy))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)

    xmi, xma, ymi, yma = int(x3 - imasiz), int(x3 + imasiz), int(y3 - imasiz), int(y3 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(0, 1, 1, 2, 1, 1, ctx.imdeep3, ctx.imdeep3, xmi, xma, ymi, yma, 1, 99.5)
    plt.plot(ctx.xpic, ctx.ypic, '+', color='red', ms=8)
    for j in range(ctx.npic):
        if ctx.xpic[j] < xma - 1 and ctx.xpic[j] > xmi + 1 and ctx.ypic[j] < yma - 1 and ctx.ypic[j] > ymi + 1:
            plt.text(ctx.xpic[j], ctx.ypic[j], s=str(j), size=8, color='red')
    ax.set_title(str(ireg) + '    Log(SN3 Deepimage)', size=20)

    xmi, xma, ymi, yma = int(x2 - imasiz), int(x2 + imasiz), int(y2 - imasiz), int(y2 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(0, 0, 1, 2, 1, 1, ctx.imdeep2, ctx.imdeep2, xmi, xma, ymi, yma, 1, 99.5)
    ax.set_title('Log(SN2 Deepimage)', size=20)

    xmi, xma, ymi, yma = int(x1 - imasiz), int(x1 + imasiz), int(y1 - imasiz), int(y1 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(0, 0, 1, 2, 1, 1, ctx.imdeep1, ctx.imdeep1, xmi, xma, ymi, yma, 1, 99.5)
    ax.set_title('Log(SN1 Deepimage)', size=20)

    xmi, xma, ymi, yma = int(x3 - imasiz), int(x3 + imasiz), int(y3 - imasiz), int(y3 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(0, 1, 1, 2, 1, 1, ctx.fluxha, ctx.fluxha, xmi, xma, ymi, yma, 1, 99.5)
    ax.set_title(r'Log(H$\alpha$)', size=20)

    xmi, xma, ymi, yma = int(x2 - imasiz), int(x2 + imasiz), int(y2 - imasiz), int(y2 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(0, 0, 1, 2, 1, 1, ctx.fluxo3, ctx.fluxo3, xmi, xma, ymi, yma, 1, 99.5)
    ax.set_title(r'Log([OIII]$\lambda$5007)', size=20)

    xmi, xma, ymi, yma = int(x1 - imasiz), int(x1 + imasiz), int(y1 - imasiz), int(y1 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(0, 0, 1, 2, 1, 1, ctx.fluxo2, ctx.fluxo2, xmi, xma, ymi, yma, 1, 99.5)
    ax.set_title(r'Log([OII]$\lambda$3727)', size=20)

    xmi, xma, ymi, yma = int(x3 - imasiz), int(x3 + imasiz), int(y3 - imasiz), int(y3 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(1, 1, 1, 2, 1, 1, ctx.height3, ctx.height3, xmi, xma, ymi, yma, 10, 99.9)
    ax.set_title(r'Log(Height SN3)', size=20)

    xmi, xma, ymi, yma = int(x2 - imasiz), int(x2 + imasiz), int(y2 - imasiz), int(y2 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(1, 0, 1, 2, 1, 1, ctx.height2, ctx.height2, xmi, xma, ymi, yma, 10, 99.9)
    ax.set_title(r'Log(Height SN2)', size=20)

    xmi, xma, ymi, yma = int(x1 - imasiz), int(x1 + imasiz), int(y1 - imasiz), int(y1 + imasiz)
    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    makeplot(1, 0, 1, 2, 1, 1, ctx.height1, ctx.height1, xmi, xma, ymi, yma, 10, 99.9)
    ax.set_title(r'Log(Height SN1)', size=20)

    sizw = 14

    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axis([0, 10, 0, 10])
    for i in range(4, 9):
        ax.text(0.5, 7 - (i - 4), lin[i], size=sizw)
        ax.text(3.0, 7 - (i - 4), ': ' + '{:.2e}'.format(flu[i]), size=sizw)
        ax.text(5.5, 7 - (i - 4), '+/- ' + '{:.2e}'.format(err[i]), size=sizw)
        ax.text(8.7, 7 - (i - 4), 'SNR: ' + '{:.1f}'.format(snr[i]), size=sizw)

    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axis([0, 10, 0, 10])
    for i in range(1, 4):
        ax.text(0.5, 7 - (i - 1), lin[i], size=sizw)
        ax.text(3.0, 7 - (i - 1), ': ' + '{:.2e}'.format(flu[i]), size=sizw)
        ax.text(5.5, 7 - (i - 1), '+/- ' + '{:.2e}'.format(err[i]), size=sizw)
        ax.text(8.7, 7 - (i - 1), 'SNR: ' + '{:.1f}'.format(snr[i]), size=sizw)

    ip += 1
    ax = plt.subplot(nyy, nxx, ip)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axis([0, 10, 0, 10])
    ax.text(0.5, 7, lin[0], size=sizw)
    ax.text(3.0, 7, ': ' + '{:.2e}'.format(flu[0]), size=sizw)
    ax.text(5.5, 7, '+/- ' + '{:.2e}'.format(err[0]), size=sizw)
    ax.text(8.7, 7, 'SNR: ' + '{:.1f}'.format(snr[0]), size=sizw)

    wlines = EMI_LINES
    wn3 = 1.e8 / (wn3 * (1 + (ctx.gal_vel / 3e5)))
    wn2 = 1.e8 / (wn2 * (1 + (ctx.gal_vel / 3e5)))
    wn1 = 1.e8 / (wn1 * (1 + (ctx.gal_vel / 3e5)))

    color_spec = ctx.colors[4]
    color_fit = ctx.colors[12]

    ax = plt.subplot(nyy, nxx, 15)
    ax.yaxis.tick_left()
    ax.tick_params(axis='both', direction='in', left=False, top=False, right=False,
                   bottom=True, labelleft=False, labelbottom=True, labelsize=25,
                   length=7, width=2)
    ax.set_xlim(6500, 6750)
    ax.set_ylim(ymin, ymax)
    ax.plot(wn3, sp3p, '-', color=color_spec, lw=2)
    ax.plot(wn3, fi3p, '-', color=color_fit, lw=1)
    for i, c in zip([4, 5, 6, 7, 8], [ctx.colors[15], 'k', 'black', ctx.colors[15], 'k']):
        ax.plot([wlines[i], wlines[i]], [ymin, ymax], '--', color=c)

    ax = plt.subplot(nyy, nxx, 14)
    ax.tick_params(axis='both', direction='in', left=False, top=False, right=False,
                   bottom=True, labelleft=False, labelbottom=True, labelsize=25,
                   length=7, width=2)
    ax.set_xlim(4800, 5070)
    ax.set_ylim(ymin, ymax)
    ax.plot(wn2, sp2p, '-', color=color_spec, lw=1)
    ax.plot(wn2, fi2p, '-', color=color_fit, lw=1)
    ax.set_xlabel(r'Wavelength ($\AA$)', size=25)
    for i, c in zip([1, 2, 3], [ctx.colors[15], 'black', ctx.colors[15]]):
        ax.plot([wlines[i], wlines[i]], [ymin, ymax], '--', color=c)

    ax = plt.subplot(nyy, nxx, 13)
    ax.tick_params(axis='both', direction='in', left=True, top=False, right=False,
                   bottom=True, labelleft=True, labelbottom=True, labelsize=25,
                   length=7, width=2)
    ax.set_ylim(ymin, ymax)
    ax.plot(wn1, sp1p, '-', color=color_spec, lw=1, label='Observed')
    ax.plot(wn1, fi1p, '-', color=color_fit, lw=1, label='Fit')
    ax.plot([wlines[0], wlines[0]], [ymin, ymax], '--', color='black')
    ax.set_ylabel(
        r'Flux [10$^{' + str(facplot) + r'}$erg s$^{-1}$cm$^{-2}\AA^{-1}$]',
        size=25
    )
    ax.legend()

    if plotshow:
        plt.show()

    fig.savefig(f'{ctx.folder}/Spectre_{ireg}.jpeg', bbox_inches='tight')
    plt.close(fig)


def format_result_row(ireg, x0, y0, fnum, flu, err, snr):
    return [
        ireg, x0, y0, fnum,
        '{:.3e}'.format(flu[0]), '{:.3e}'.format(err[0]), '{:.1f}'.format(snr[0]),
        '{:.3e}'.format(flu[1]), '{:.3e}'.format(err[1]), '{:.1f}'.format(snr[1]),
        '{:.3e}'.format(flu[3]), '{:.3e}'.format(err[3]), '{:.1f}'.format(snr[3]),
        '{:.3e}'.format(flu[2]), '{:.3e}'.format(err[2]), '{:.1f}'.format(snr[2]),
        '{:.3e}'.format(flu[6]), '{:.3e}'.format(err[6]), '{:.1f}'.format(snr[6]),
        '{:.3e}'.format(flu[4]), '{:.3e}'.format(err[4]), '{:.1f}'.format(snr[4]),
        '{:.3e}'.format(flu[5]), '{:.3e}'.format(err[5]), '{:.1f}'.format(snr[5]),
        '{:.3e}'.format(flu[7]), '{:.3e}'.format(err[7]), '{:.1f}'.format(snr[7]),
        '{:.3e}'.format(flu[8]), '{:.3e}'.format(err[8]), '{:.1f}'.format(snr[8]),
    ]


def run_region_fit(ctx, ireg, plotshow=True):
    x0, y0 = int(ctx.xpic[ireg]), int(ctx.ypic[ireg])

    wn1, sp1, fi1, flu1, err1, snr1, ok1 = fit_spectrum(
        ireg, x0, y0, LINE1, RANGE1, ctx.dom1, ctx.cube1, ctx.velocity_map, ctx.sigma_map
    )
    wn2, sp2, fi2, flu2, err2, snr2, ok2 = fit_spectrum(
        ireg, x0, y0, LINE2, RANGE2, ctx.dom2, ctx.cube2, ctx.velocity_map, ctx.sigma_map
    )
    wn3, sp3, fi3, flu3, err3, snr3, ok3 = fit_spectrum(
        ireg, x0, y0, LINE3, RANGE3, ctx.dom3, ctx.cube3, ctx.velocity_map, ctx.sigma_map
    )

    flu = [*flu1, *flu2, *flu3]
    err = [*err1, *err2, *err3]
    snr = [*snr1, *snr2, *snr3]

    # makeplotspec(
    #     ireg, x0, y0,
    #     wn1, sp1, fi1,
    #     wn2, sp2, fi2,
    #     wn3, sp3, fi3,
    #     flu, err, snr, LIN, plotshow, ctx
    # )

    row = format_result_row(ireg, x0, y0, ctx.fnum, flu, err, snr)

    return {
        'ireg': ireg,
        'x0': x0,
        'y0': y0,
        'row': row,
        'fit_success': bool(ok1 and ok2 and ok3),
    }