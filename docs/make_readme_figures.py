"""Draw the figures of the toolkit README (docs/images/): the workflow, the data layout, and example plots made
with the explore functions from a real build.

usage: python docs/make_readme_figures.py /path/to/Urban-CAMELS"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ucamels_prep as up

up._plt()                                        # Times New Roman (or its metric clone Nimbus Roman) everywhere
MONO = ['Courier New', 'Nimbus Mono PS', 'DejaVu Sans Mono']
IMG = HERE / 'images'
IMG.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, GREY = '#0072B2', '#D55E00', '#009E73', '#7B3294', '#555555'
DPI = 160


def box(ax, x, y, w, h, title, body, color, fun=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.025', fc=color, ec='none', alpha=0.13))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.12), w, 0.12, boxstyle='round,pad=0.012,rounding_size=0.025', fc=color, ec='none'))
    ax.text(x + w / 2, y + h - 0.06, title, ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    ax.text(x + w / 2, y + h - 0.16, body, ha='center', va='top', fontsize=12, linespacing=1.4, color='#222222')
    if fun:
        ax.text(x + w / 2, y + 0.03, fun, ha='center', va='bottom', fontsize=10.5, family=MONO, color=color)


def workflow():
    fig, ax = plt.subplots(figsize=(15, 3.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    steps = [
        ('The data', '995 urban basins\nWY1990-2025\nforcing, streamflow,\nPET, attributes\n(Zenodo archives)', GREY, None),
        ('1  Choose', 'basins (preset, IDs,\nstates, completeness,\narea, imperviousness)\ntime step and period\ninputs, target, attributes\ngaps, split, output', BLUE, 'CONFIG = dict(...)'),
        ('2  Check', 'impossible combinations\nleakage warnings\nMRMS gap, UTC days\nsize estimate', BLUE, 'up.validate()'),
        ('3  Build', 'every basin in parallel\none time axis\nshort input gaps filled\nquality rules\nstatic attributes', GREEN, 'up.build()'),
        ('4  Explore', 'maps, time series\nmonthly regime\nwater-year trends\ncorrelations at\nany time scale', ORANGE, 'up.plot_*()'),
        ('5  Split and write', 'temporal / spatial /\nk-fold split\nNeuralHydrology, numpy\narrays or CSV\nnorm. stats, manifest', PURPLE, 'up.write_package()'),
    ]
    w, gap, y, h = 0.1375, 0.032, 0.04, 0.92
    for k, (t, b, c, f) in enumerate(steps):
        x = 0.008 + k * (w + gap)
        box(ax, x, y, w, h, t, b, c, f)
        if k:
            ax.add_patch(FancyArrowPatch((x - gap + 0.003, y + h / 2), (x - 0.003, y + h / 2), arrowstyle='-|>',
                                         mutation_scale=18, color='#444444', lw=1.6))
    fig.savefig(IMG / 'workflow.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def layout():
    fig, ax = plt.subplots(figsize=(15, 7.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    chips = {'daily': BLUE, 'hourly': GREEN, 'sub-hourly': ORANGE, 'annual': PURPLE, 'static': GREY}
    rows = [
        (0, 'Urban-CAMELS/', '', []),
        (1, 'attributes/', 'static attributes, basin polygons, dams and levees (one row per structure)', ['static']),
        (2, 'annual/', 'imperviousness, 6 land-cover classes, population, housing, climate indices, signatures', ['annual']),
        (1, 'timeseries/', '', []),
        (2, 'forcing/', '', []),
        (3, 'daymet/  gridmet/  nldas_daily/', 'precipitation, temperature, radiation, humidity, wind ...', ['daily']),
        (3, 'aorc/  nldas_hourly/', 'the same variables every hour (UTC, end-of-hour stamps)', ['hourly']),
        (3, 'mrms/<YYYY-MM>/', 'radar precipitation every 2-5 minutes, from 2002', ['sub-hourly']),
        (2, 'streamflow_observed/', '', []),
        (3, 'daily/  daily_with_quality_code/', 'USGS NWIS daily mean, local calendar day', ['daily']),
        (3, 'hourly/', 'mean of the readings in each UTC hour', ['hourly']),
        (3, 'subhourly/', 'instantaneous readings (5-60 min), with quality codes', ['sub-hourly']),
        (2, 'PET/', 'Priestley-Taylor and Penman-Monteith from each forcing product', ['daily', 'hourly']),
        (2, 'complmentery_data/', 'actual ET, snow, groundwater, water quality, withdrawal, NWM v3.0 (benchmark)',
         ['daily', 'hourly', 'sub-hourly']),
    ]
    y = 0.95; dy = 0.064
    for lvl, name, desc, ch in rows:
        x = 0.02 + lvl * 0.04
        if lvl:
            ax.plot([x - 0.03, x - 0.03, x - 0.008], [y + dy * 0.75, y, y], color='#b0b0b0', lw=1.2)
        ax.text(x, y, name, va='center', fontsize=13.5, family=MONO, fontweight='bold',
                color='#111111' if not desc else '#1f3b57')
        cx = 0.47
        for c in ch:
            ax.add_patch(FancyBboxPatch((cx, y - 0.018), 0.08, 0.036, boxstyle='round,pad=0.003,rounding_size=0.012',
                                        fc=chips[c], ec='none'))
            ax.text(cx + 0.04, y, c, ha='center', va='center', fontsize=11, color='white', fontweight='bold')
            cx += 0.087
        if desc:
            ax.text(0.47 + 0.087 * max(len(ch), 1) + 0.005, y, desc, va='center', fontsize=13, color='#333333')
        y -= dy
    ax.text(0.02, 0.0, 'Files are named by the USGS station number (STAID). Hourly and sub-hourly series are in UTC.',
            fontsize=12.5, style='italic', color='#444444')
    fig.savefig(IMG / 'data_layout.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def examples(root):
    cfg = dict(data_root=root, resolution='daily', period=('1989-10-01', '2025-09-30'), basins=dict(preset='benchmark_247'),
               forcing={'daymet': ['prcp', 'tmax', 'srad', 'vp']}, pet={'daymet': ['pet_pt']}, dynamic_attributes=['imperv'],
               target=dict(unit='mm'), static_attributes=dict(groups=['location_topography'], categorical='drop'),
               gap_fill=dict(max_steps=2, precip='zero'), qc=dict(max_target_missing_pct=100, drop_if_inputs_missing=True),
               split=dict(method='temporal', train=('1989-10-01', '2005-09-30'), validation=('2005-10-01', '2010-09-30'),
                          test=('2010-10-01', '2025-09-30')), output=dict(dir='unused', format='csv'))
    basins = up.select_basins(cfg['basins'], verbose=False)
    tables, _, _ = up.build(cfg, basins)
    ax = up.plot_map(basins, title='The 247 benchmark basins (complete daily record), coloured by imperviousness')
    ax.figure.savefig(IMG / 'example_map.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    up.plot_timeseries(tables, '02336300', period=('2020-10-01', '2021-09-30')).savefig(
        IMG / 'example_timeseries.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    up.plot_monthly(tables, ['daymet__prcp', 'pet__daymet__pet_pt', 'streamflow_mm']).savefig(
        IMG / 'example_monthly.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    up.plot_annual(tables, ['daymet__prcp', 'streamflow_mm', 'dyn__imperv']).savefig(
        IMG / 'example_annual.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    up.plot_correlation(tables, ['daymet__prcp', 'daymet__tmax', 'daymet__srad', 'daymet__vp', 'pet__daymet__pet_pt',
                                 'streamflow_mm'], freq='MS').savefig(IMG / 'example_correlation.png', dpi=DPI,
                                                                     bbox_inches='tight', facecolor='white')
    cfg_h = dict(cfg, resolution='hourly', period=('2021-08-12', '2021-08-16'), basins=dict(ids=['07165565']),
                 forcing={'mrms': ['prcp'], 'aorc': ['tair']}, pet={}, dynamic_attributes=[], gap_fill=dict(max_steps=0),
                 qc=dict(max_target_missing_pct=100, drop_if_inputs_missing=False))
    th, _, _ = up.build(cfg_h, ['07165565'], workers=1)
    up.plot_timeseries(th, '07165565').savefig(IMG / 'example_hourly_storm.png', dpi=DPI, bbox_inches='tight',
                                               facecolor='white')
    plt.close('all')


if __name__ == '__main__':
    workflow(); layout()
    if len(sys.argv) > 1:
        examples(sys.argv[1])
    print('written:', ', '.join(sorted(p.name for p in IMG.glob('*.png'))))
