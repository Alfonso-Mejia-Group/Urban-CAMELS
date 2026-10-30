"""
ucamels_prep.py -- build model-ready datasets from Urban-CAMELS (995 basins).

The user describes what they need in one CONFIG dictionary (see the notebook Urban_CAMELS_data_prep.ipynb):
where the data are (data_root: the folder holding timeseries/ and attributes/, made by unzipping the Urban-CAMELS
archives into one place), which basins, which period and time step, which forcing variables, which PET, which static
and dynamic attributes, how to split the data, and which output format. The functions below then

    select_basins()   -> the basin list (presets, completeness thresholds, area / imperviousness filters)
    load_basin()      -> one basin as a table on a regular time axis: forcing, PET, dynamic attributes, target
    build()           -> all basins in parallel, gap handling, a quality report, static attributes
    make_split()      -> temporal, spatial (random / stratified / k-fold) or combined train / validation / test split
    write_package()   -> NeuralHydrology layout, numpy arrays, or per-basin CSV, plus the split, normalisation
                         statistics computed on the training data only, and a manifest of every choice

Conventions (match the dataset):
- daily step: rows are calendar dates. NWIS daily streamflow is the local calendar day; hourly products are aggregated
  to the basin's local standard-time day; Daymet, gridMET and NLDAS-2 daily are used as given (one value per date).
- hourly step: UTC, each row describes the hour that STARTS at its label. Hourly streamflow is the mean of all
  instantaneous readings in that hour; MRMS rates are averaged over it. AORC and NLDAS-2 (and the hourly PET computed
  from them) stamp each value at the END of its hour, so the toolkit moves them back one hour on reading.
- sub-hourly step (CONFIG['subhourly_step'], e.g. '15min', must divide one hour): UTC, label = start of the step. The
  target is the mean of the instantaneous streamflow readings inside each step; MRMS rates are averaged to the step;
  hourly forcing and PET (rates) are held constant across the steps of their hour.
- streamflow target in mm per time step by default (cfs x 0.0283168 / area).
- annual attributes are held constant within each calendar year; housing density (census years) is held or
  interpolated between census years.
"""
from __future__ import annotations

import json, os, re, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

TOOLKIT = Path(__file__).resolve().parent
WINDOW = ('1989-10-01', '2025-09-30')
CFS_TO_CMS = 0.0283168

# ------------------------------------------------------------------------------------------------------------------
# catalogue of time-series products: file template, time column, step, variables (name -> unit)
# ------------------------------------------------------------------------------------------------------------------
PRODUCTS = {
    'daymet': dict(step='D', tcol='date', file='timeseries/forcing/daymet/{s}/daymet_daily_1989_2025.csv',
                   vars={'prcp': 'mm/day', 'tmax': 'degC', 'tmin': 'degC', 'tmean': 'degC', 'srad': 'W/m2',
                         'vp': 'Pa', 'dayl': 's', 'swe': 'mm'}),
    'gridmet': dict(step='D', tcol='date', file='timeseries/forcing/gridmet/{s}/gridmet_daily_1989_2025.csv',
                    vars={'pr': 'mm/day', 'tmmn_C': 'degC', 'tmmx_C': 'degC', 'tmean_C': 'degC', 'srad': 'W/m2',
                          'rmin': '%', 'rmax': '%', 'rhmean': '%', 'vs': 'm/s', 'vpd': 'kPa', 'sph': 'kg/kg',
                          'pet': 'mm/day', 'etr': 'mm/day'}),
    'nldas_daily': dict(step='D', tcol='date', file='timeseries/forcing/nldas_daily/{s}/nldas_daily_1989_2025.csv',
                        vars={'prcp': 'mm/day', 'tair': 'degC', 'tmax': 'degC', 'tmin': 'degC', 'srad': 'W/m2',
                              'lrad': 'W/m2', 'vp': 'Pa', 'wind': 'm/s', 'pres': 'Pa', 'potevap': 'mm/day'}),
    'aorc': dict(step='H', tcol='datetime', stamp='end', file='timeseries/forcing/aorc/{s}/aorc_hourly_1989_2025.csv',
                 vars={'prcp': 'mm/h', 'tair': 'degC', 'srad': 'W/m2', 'lrad': 'W/m2', 'vp': 'Pa', 'wind': 'm/s',
                       'pres': 'Pa'}),
    'nldas_hourly': dict(step='H', tcol='datetime_utc', stamp='end', file='timeseries/forcing/nldas_hourly/{s}/nldas_hourly_1989_2025.csv',
                         vars={'prcp': 'mm/h', 'tair': 'degC', 'srad': 'W/m2', 'lrad': 'W/m2', 'vp': 'Pa',
                               'wind': 'm/s', 'pres': 'Pa', 'potevap': 'mm/h'}),
    'mrms': dict(step='H', tcol='datetime', file='timeseries/forcing/mrms/{m}/{s}.csv',
                 vars={'prcp': 'mm/h', 'coverage_frac': '-'}),
}
PET = {  # (step, product) -> file, time column; methods pet_pt (Priestley-Taylor) and pet_pm (Penman-Monteith)
    ('D', 'daymet'): ('timeseries/PET/daily/daymet/{s}.csv', 'date', ['pet_pt']),
    ('D', 'gridmet'): ('timeseries/PET/daily/gridmet/{s}.csv', 'date', ['pet_pt', 'pet_pm']),
    ('D', 'nldas_daily'): ('timeseries/PET/daily/nldas_daily/{s}.csv', 'date', ['pet_pt', 'pet_pm']),
    ('D', 'nldas_hourly'): ('timeseries/PET/daily/nldas_hourly/{s}.csv', 'date', ['pet_pt', 'pet_pm']),
    ('D', 'aorc'): ('timeseries/PET/daily/aorc/{s}.csv', 'date', ['pet_pt', 'pet_pm']),
    ('H', 'aorc'): ('timeseries/PET/hourly/aorc/{s}.csv', 'datetime', ['pet_pt', 'pet_pm']),
    ('H', 'nldas_hourly'): ('timeseries/PET/hourly/nldas_hourly/{s}.csv', 'datetime_utc', ['pet_pt', 'pet_pm']),
}
TARGET = {'D': ('timeseries/streamflow_observed/daily/{s}.csv', 'date', 'discharge_daily_cfs'),
          'H': ('timeseries/streamflow_observed/hourly/{s}.csv', 'datetime', 'discharge_hourly_cfs'),
          'S': ('timeseries/streamflow_observed/subhourly/{s}.csv', 'datetime_utc', 'discharge_subhourly_cfs')}
STEP = {'daily': 'D', 'hourly': 'H', 'subhourly': 'S'}
DYNAMIC = {  # annual (calendar-year) attributes: name -> (file, column, unit)
    'imperv': ('impervious_1989_2025.csv', 'imperv', '%'),
    'class1_urban': ('landcover_6class_1989_2025.csv', 'class1_urban', '%'),
    'class2_agriculture': ('landcover_6class_1989_2025.csv', 'class2_agriculture', '%'),
    'class3_forest': ('landcover_6class_1989_2025.csv', 'class3_forest', '%'),
    'class4_grassland': ('landcover_6class_1989_2025.csv', 'class4_grassland', '%'),
    'class5_bare': ('landcover_6class_1989_2025.csv', 'class5_bare', '%'),
    'class6_water': ('landcover_6class_1989_2025.csv', 'class6_water', '%'),
    'pop_density': ('population_1989_2025.csv', 'pop_density', 'persons/km2'),
    'housing_density': ('housing_1990_2020.csv', 'housing_density', 'units/km2'),   # census years only
}
# static attribute groups (columns of attributes/static_attributes.csv)
STATIC_GROUPS = {
    'location_topography': ['lat', 'lon', 'elev_mean', 'slope_mean', 'area_gages2'],
    'soil': ['soil_depth_statsgo', 'soil_depth_pelletier', 'soil_porosity', 'soil_conductivity', 'max_water_content',
             'sand_frac', 'silt_frac', 'clay_frac'],
    'geology': ['carbonate_rocks_frac', 'geol_porosity', 'geol_permeability', 'glim_1st_class_frac',
                'glim_2nd_class_frac', 'geol_carbonate_dom'],
    'geology_categorical': ['glim_1st_code', 'geol_2nd_class', 'geol_lithology_class'],
    'vegetation': ['frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff', 'root_depth_50', 'root_depth_99'],
    'climate': ['p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 'high_prec_freq', 'high_prec_dur',
                'low_prec_freq', 'low_prec_dur'],
    'climate_categorical': ['high_prec_timing', 'low_prec_timing'],
    'hydrologic_signatures': ['q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc', 'baseflow_index', 'hfd_mean', 'Q5',
                              'Q95', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur', 'zero_q_freq'],
    'dams': ['nid_dam_n', 'nid_storage_mm', 'nid_normal_storage_mm', 'nid_stor_frac_pre1990', 'nid_storage_change_mm'],
    'levees': ['levee_area_frac', 'levee_length_km', 'levee_density_km_per_km2'],
}
# attributes derived from streamflow: using them while testing on the same basins leaks target information
FLOW_DERIVED = set(STATIC_GROUPS['hydrologic_signatures'])
# climate attributes that can be recomputed over the training water years only (from the annual files)
CLIMATE_ANNUAL = ['p_mean', 'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
                  'p_seasonality']
SIG_ANNUAL = {'q_mean': 'q_mean', 'runoff_ratio': 'runoff_ratio', 'slope_fdc': 'slope_fdc', 'baseflow_index':
              'baseflow_index', 'hfd_mean': 'hfd', 'Q5': 'Q5', 'Q95': 'Q95', 'high_q_freq': 'high_q_freq',
              'high_q_dur': 'high_q_dur', 'low_q_freq': 'low_q_freq', 'low_q_dur': 'low_q_dur',
              'zero_q_freq': 'zero_q_freq'}


# ------------------------------------------------------------------------------------------------------------------
# paths and catalogue
# ------------------------------------------------------------------------------------------------------------------
def data_root(cfg: dict | None = None) -> Path:
    """The Urban-CAMELS data folder: CONFIG['data_root'], or the UCAMELS_DATA environment variable."""
    r = (cfg or {}).get('data_root') or os.environ.get('UCAMELS_DATA')
    if not r:
        raise ValueError("set CONFIG['data_root'] (or the UCAMELS_DATA environment variable) to the folder where the "
                         'Urban-CAMELS archives were unzipped; it holds timeseries/ and attributes/')
    r = Path(r).expanduser()
    if not (r / 'attributes' / 'static_attributes.csv').exists():
        raise FileNotFoundError(f'{r} is not the Urban-CAMELS data folder: attributes/static_attributes.csv is missing')
    return r


def load_catalog() -> pd.DataFrame:
    """The 995 basins with name, location, area, imperviousness, megaregion, preset and data flags
    (catalog/basins_995.csv)."""
    return pd.read_csv(TOOLKIT / 'catalog' / 'basins_995.csv', dtype={'STAID': str})


def coverage(resolution: str = 'daily') -> pd.DataFrame:
    """Percent completeness of observed streamflow per basin and water-year window (daily, hourly, subhourly)."""
    return pd.read_csv(TOOLKIT / 'catalog' / f'coverage_{resolution}.csv', dtype={'STAID': str}).set_index('STAID')


# ------------------------------------------------------------------------------------------------------------------
# 1. basin selection
# ------------------------------------------------------------------------------------------------------------------
def select_basins(sel: dict, verbose: bool = True) -> list[str]:
    """Basin list from the CONFIG['basins'] block. Keys (all optional):
       preset: 'all' | 'benchmark_247'          start set
       ids / id_file: explicit STAIDs (list, or a text/CSV file with one STAID per line or a STAID column)
       completeness: {'resolution': 'daily'|'hourly'|'subhourly', 'window': 'WY2004-2025', 'min_pct': 90}
       states: two-letter codes, e.g. ['PA'] or ['PA', 'NJ', 'NY']
       area_km2: (min, max); impervious_pct: (min, max); megaregions: [...]; exclude_flags: [...]; exclude: [...]"""
    known = {'preset', 'ids', 'id_file', 'completeness', 'states', 'area_km2', 'impervious_pct', 'megaregions',
             'exclude_flags', 'exclude'}
    if set(sel) - known:
        raise ValueError(f'unknown basin selection keys {sorted(set(sel) - known)}; available {sorted(known)}')
    cat = load_catalog().set_index('STAID')
    keep = pd.Series(True, index=cat.index)
    steps = [('start: 995 basins', int(keep.sum()))]
    p = sel.get('preset', 'all')
    if p == 'benchmark_247':
        keep &= cat.preset_benchmark_247
    elif p != 'all':
        raise ValueError(f'unknown preset {p!r}: use all or benchmark_247')
    steps.append((f'preset {p}', int(keep.sum())))
    ids = list(sel.get('ids') or [])
    if sel.get('id_file'):
        f = Path(sel['id_file'])
        t = pd.read_csv(f, dtype=str, header=None) if f.suffix != '.csv' else pd.read_csv(f, dtype=str)
        ids += (t['STAID'] if 'STAID' in t else t.iloc[:, 0]).str.strip().tolist()
    if ids:
        unknown = sorted(set(ids) - set(cat.index))
        if unknown:
            warnings.warn(f'{len(unknown)} STAIDs are not Urban-CAMELS basins and are ignored: {unknown[:10]}')
        keep &= cat.index.isin(ids); steps.append(('explicit ids', int(keep.sum())))
    c = sel.get('completeness')
    if c:
        cv = coverage(c.get('resolution', 'daily'))
        if c['window'] not in cv:
            raise ValueError(f'window {c["window"]!r} not tabulated; available: {list(cv.columns)}')
        keep &= cv[c['window']].reindex(cat.index).fillna(0) >= c.get('min_pct', 100) - 1e-9
        steps.append((f'{c.get("resolution", "daily")} completeness >= {c.get("min_pct", 100)} % over {c["window"]}', int(keep.sum())))
    if sel.get('states'):
        st = [x.strip().upper() for x in ([sel['states']] if isinstance(sel['states'], str) else sel['states'])]
        unknown = sorted(set(st) - set(cat.state))
        if unknown:
            warnings.warn(f'no Urban-CAMELS basins in {unknown}; states with basins: {sorted(cat.state.unique())}')
        keep &= cat.state.isin(st); steps.append((f'states {", ".join(st)}', int(keep.sum())))
    if sel.get('area_km2'):
        lo, hi = sel['area_km2']; keep &= cat.area_km2.between(lo, hi); steps.append((f'area {lo}-{hi} km2', int(keep.sum())))
    if sel.get('impervious_pct'):
        lo, hi = sel['impervious_pct']; keep &= cat.IMPNLCD06.between(lo, hi)
        steps.append((f'IMPNLCD06 {lo}-{hi} %', int(keep.sum())))
    if sel.get('megaregions'):
        keep &= cat.megaregion.isin(sel['megaregions']); steps.append(('megaregions', int(keep.sum())))
    for flag in sel.get('exclude_flags') or []:
        col = f'flag_{flag}'
        if col not in cat:
            raise ValueError(f'unknown flag {flag!r}; available {[c[5:] for c in cat.columns if c.startswith("flag_")]}')
        keep &= ~cat[col].astype(bool); steps.append((f'exclude flag {flag}', int(keep.sum())))
    if sel.get('exclude'):
        keep &= ~cat.index.isin(sel['exclude']); steps.append(('exclude list', int(keep.sum())))
    out = sorted(cat.index[keep])
    if verbose:
        for s, n in steps:
            print(f'  {s:<60s} -> {n:4d} basins')
    return out


# ------------------------------------------------------------------------------------------------------------------
# 2. loading one basin
# ------------------------------------------------------------------------------------------------------------------
def step_seconds(cfg: dict) -> int:
    r = cfg['resolution']
    if r == 'daily':
        return 86400
    if r == 'hourly':
        return 3600
    sec = int(pd.Timedelta(cfg.get('subhourly_step', '15min')).total_seconds())
    if 3600 % sec:
        raise ValueError('subhourly_step must divide one hour (e.g. 5min, 10min, 15min, 30min)')
    return sec


def time_axis(cfg: dict) -> pd.DatetimeIndex:
    a, b = cfg['period']
    if cfg['resolution'] not in STEP:
        raise ValueError("resolution must be 'daily', 'hourly' or 'subhourly'")
    if cfg['resolution'] == 'daily':
        return pd.date_range(a, b, freq='D')
    end = pd.Timestamp(b) + pd.Timedelta(days=1) - pd.Timedelta(seconds=step_seconds(cfg))
    return pd.date_range(pd.Timestamp(a), end, freq=f'{step_seconds(cfg)}s')


def _std_offset(lon: float) -> int:
    """Hours behind UTC of the basin's local standard time (EST/CST/MST/PST by longitude)."""
    return 5 if lon > -87.5 else 6 if lon > -102 else 7 if lon > -115 else 8


def _read(f: Path, tcol: str, cols: list[str]) -> pd.DataFrame:
    d = pd.read_csv(f, usecols=[tcol] + cols)
    t = pd.to_datetime(d.pop(tcol), utc=True) if tcol != 'date' else pd.to_datetime(d.pop(tcol))
    d.index = t.dt.tz_convert(None) if getattr(t.dt, 'tz', None) is not None else t    # naive UTC, as the axis
    return d[~d.index.duplicated()].sort_index()


def _mrms_hourly(sid: str, root: Path, ax: pd.DatetimeIndex, vars_: list[str], step: str = '1h') -> pd.DataFrame:
    """MRMS 2-5 min rates -> mean rate (mm/h) and mean coverage per model step, over the months touched by the axis."""
    parts = []
    for m in pd.period_range(ax[0], ax[-1], freq='M'):
        f = root / 'timeseries/forcing/mrms' / str(m) / f'{sid}.csv'
        if f.exists():
            parts.append(_read(f, 'datetime', ['rate_mmhr', 'coverage_frac']))
    if not parts:
        return pd.DataFrame(index=ax, columns=vars_, dtype='float32')
    d = pd.concat(parts)
    h = pd.DataFrame({'prcp': d.rate_mmhr.resample(step).mean(), 'coverage_frac': d.coverage_frac.resample(step).mean()})
    return h[vars_].reindex(ax)


def load_basin(sid: str, cfg: dict, cat: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One basin on the regular time axis of the config. Returns (table, info).
    Columns: <product>__<variable> for forcing, pet__<product>__<method>, dyn__<attribute>, and the target column."""
    ax = time_axis(cfg); step = STEP[cfg['resolution']]; sec = step_seconds(cfg)
    root = data_root(cfg); lon = float(cat.loc[sid, 'lon'])
    area = float(cat.loc[sid, 'area_km2'])
    cols, info = {}, {'STAID': sid}
    # forcing
    for prod, vars_ in cfg['forcing'].items():
        spec = PRODUCTS[prod]
        if spec['step'] == 'D' and step != 'D':
            raise ValueError(f'{prod} is daily and cannot drive an {cfg["resolution"]} model; use aorc, nldas_hourly or mrms')
        if prod == 'mrms':
            if step == 'D':
                raise ValueError('mrms is offered at the hourly and sub-hourly steps only')
            d = _mrms_hourly(sid, root, ax, vars_, f'{sec}s')
        else:
            d = _read(root / spec['file'].format(s=sid), spec['tcol'], vars_)
            if spec.get('stamp') == 'end':              # source stamps the end of the hour -> label = start of the hour
                d.index = d.index - pd.Timedelta(hours=1)
            if spec['step'] == 'H' and step == 'D':   # hourly product -> local standard-time day
                off = pd.Timedelta(hours=_std_offset(lon))
                g = d.groupby((d.index - off).normalize())
                n = g.count()                                       # a day needs >= 20 of its 24 hours
                d = pd.DataFrame({v: (g[v].sum() if v in ('prcp', 'potevap') else g[v].mean()).where(n[v] >= 20)
                                  for v in vars_})
            d = _to_axis(d, ax, step)
        for v in vars_:
            cols[f'{prod}__{v}'] = d[v].astype('float32').values
    # PET
    for pr, methods in (cfg.get('pet') or {}).items():
        key = ('H' if step == 'S' else step, pr)
        if key not in PET:
            raise ValueError(f'no {cfg["resolution"]} PET for product {pr!r}; available: {[k[1] for k in PET if k[0] == key[0]]}')
        f, tcol, ok = PET[key]
        bad = set(methods) - set(ok)
        if bad:
            raise ValueError(f'PET {pr}: methods {sorted(bad)} not available (available {ok})')
        d = _read(root / f.format(s=sid), tcol, methods)
        if key[0] == 'H' and PRODUCTS[pr].get('stamp') == 'end':    # hourly PET follows its forcing's stamps
            d.index = d.index - pd.Timedelta(hours=1)
        d = _to_axis(d, ax, step)
        for m in methods:
            cols[f'pet__{pr}__{m}'] = d[m].astype('float32').values
    # dynamic attributes (annual / census) -> time step
    for name in cfg.get('dynamic_attributes') or []:
        fname, col, _ = DYNAMIC[name]
        a = _annual(root / 'attributes/annual' / fname, col).get(sid)
        if a is None:
            cols[f'dyn__{name}'] = np.full(len(ax), np.nan, 'float32'); continue
        if name == 'housing_density':
            yrs = np.arange(ax[0].year, ax[-1].year + 1)
            s = a.reindex(sorted(set(a.index) | set(yrs)))
            s = s.interpolate('index', limit_area='inside') if cfg.get('housing_interp', 'step') == 'linear' else s.ffill()
            a = s.ffill().bfill()
        cols[f'dyn__{name}'] = a.reindex(ax.year).astype('float32').values
    # target
    f, tcol, qcol = TARGET[step]
    fq = root / f.format(s=sid)
    if fq.exists():
        q = _read(fq, tcol, [qcol])[qcol]
        q = q.where(q > -999990)
        q = q.resample(f'{sec}s').mean().reindex(ax) if step == 'S' else q.reindex(ax)
    else:
        q = pd.Series(np.nan, index=ax); info['note'] = 'no streamflow file'
    unit = cfg['target'].get('unit', 'mm')
    if unit == 'mm':
        q = q * CFS_TO_CMS * sec / (area * 1e6) * 1000          # mm per time step
    elif unit == 'mm/h':
        q = q * CFS_TO_CMS * 3600 / (area * 1e6) * 1000
    elif unit == 'cms':
        q = q * CFS_TO_CMS
    q = q.where(q >= 0)                                         # negative / sentinel -> NaN
    tname = 'streamflow_' + unit.replace('/', 'per')
    cols[tname] = q.astype('float32').values
    df = pd.DataFrame(cols, index=ax); df.index.name = 'date'
    return df, info


def _to_axis(d: pd.DataFrame, ax: pd.DatetimeIndex, step: str) -> pd.DataFrame:
    """Place a series on the model axis. Hourly rates on a sub-hourly axis are held across the steps of their hour."""
    if step == 'S':
        per_hour = int(round(pd.Timedelta(hours=1) / (ax[1] - ax[0])))
        return d.reindex(ax, method='ffill', limit=per_hour - 1, tolerance=pd.Timedelta(hours=1))
    return d.reindex(ax)


_ANNUAL_CACHE: dict = {}


def _annual(f: Path, col: str) -> dict:
    key = (str(f), col)
    if key not in _ANNUAL_CACHE:
        d = pd.read_csv(f, dtype={'STAID': str})
        _ANNUAL_CACHE[key] = {s: g.set_index('year')[col].astype(float) for s, g in d.groupby('STAID')}
    return _ANNUAL_CACHE[key]


# ------------------------------------------------------------------------------------------------------------------
# 3. gap handling and quality report
# ------------------------------------------------------------------------------------------------------------------
def fill_gaps(df: pd.DataFrame, target: str, gf: dict) -> tuple[pd.DataFrame, dict]:
    """Fill short gaps in the INPUT columns only (never the target). Precipitation-like columns (prcp, pr, potevap,
    PET) follow gf['precip'] ('zero' | 'linear' | 'none'); other variables are interpolated linearly. Only gaps of at
    most gf['max_steps'] consecutive steps are filled."""
    n = int(gf.get('max_steps', 0)); filled = {}
    if n <= 0:
        return df, filled
    for c in df.columns:
        if c == target or not df[c].isna().any():
            continue
        s = df[c]; isna = s.isna()
        run = isna.groupby((~isna).cumsum()).transform('sum')       # length of the NaN run each NaN belongs to
        short = isna & (run <= n)
        if not short.any():
            continue
        wet = c.split('__')[-1] in ('prcp', 'pr')                    # precipitation only; PET is interpolated
        how = gf.get('precip', 'zero') if wet else 'linear'
        if how == 'none':
            continue
        new = s.fillna(0.0) if how == 'zero' else s.interpolate(limit_direction='both')
        df[c] = s.where(~short, new); filled[c] = int(short.sum())
    return df, filled


def _work(args):
    sid, cfg, cat = args
    try:
        df, info = load_basin(sid, cfg, cat)
    except Exception as e:                                       # keep going; the report lists the failure
        return sid, None, {'STAID': sid, 'error': f'{type(e).__name__}: {e}'[:300]}
    target = [c for c in df.columns if c.startswith('streamflow_')][0]
    raw_miss = {c: float(df[c].isna().mean() * 100) for c in df.columns if c != target}
    df, filled = fill_gaps(df, target, cfg.get('gap_fill', {}))
    info.update(target_missing_pct=round(float(df[target].isna().mean() * 100), 3),
                inputs_missing_pct_before_fill=round(max(raw_miss.values(), default=0.0), 4),
                inputs_missing_pct_after_fill=round(float(df.drop(columns=target).isna().mean().max() * 100) if
                                                    df.shape[1] > 1 else 0.0, 4),
                steps_filled=int(sum(filled.values())))
    return sid, df, info


def build(cfg: dict, basins: list[str], workers: int | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load every basin in parallel, fill short input gaps, apply the QC rules, and assemble static attributes.
    Returns (tables {STAID: DataFrame}, static attributes DataFrame, quality report DataFrame)."""
    cat = load_catalog().set_index('STAID')
    workers = workers or int(os.environ.get('SLURM_CPUS_PER_TASK', min(8, os.cpu_count() or 1)))
    t0 = time.time(); tables, rows = {}, []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (sid, df, info) in enumerate(ex.map(_work, [(s, cfg, cat) for s in basins], chunksize=4)):
            rows.append(info)
            if df is not None:
                tables[sid] = df
            if (k + 1) % 50 == 0 or k + 1 == len(basins):
                print(f'  loaded {k + 1}/{len(basins)} basins  ({time.time() - t0:.0f} s)', flush=True)
    rep = pd.DataFrame(rows).set_index('STAID')
    qc = cfg.get('qc', {})
    rep['kept'] = True; rep['reason'] = ''
    if 'error' in rep:
        bad = rep['error'].notna(); rep.loc[bad, ['kept', 'reason']] = [False, 'load error']
    m = rep.kept & (rep.target_missing_pct > qc.get('max_target_missing_pct', 100))
    rep.loc[m, ['kept', 'reason']] = [False, f'target missing > {qc.get("max_target_missing_pct")} %']
    if qc.get('drop_if_inputs_missing', True):
        m = rep.kept & (rep.inputs_missing_pct_after_fill > 0)
        rep.loc[m, ['kept', 'reason']] = [False, 'input gaps remain after filling']
    for sid in rep.index[~rep.kept]:
        tables.pop(sid, None)
    static = static_attributes(cfg, list(tables))
    print(f'  kept {len(tables)} of {len(basins)} basins; dropped {int((~rep.kept).sum())}')
    return tables, static, rep


# ------------------------------------------------------------------------------------------------------------------
# 4. static attributes
# ------------------------------------------------------------------------------------------------------------------
_STATIC: dict = {}


def _static_table(root: Path) -> pd.DataFrame:
    """Static attributes of all 995 basins, with the dam and levee attributes joined."""
    if root not in _STATIC:
        s = pd.read_csv(root / 'attributes/static_attributes.csv', dtype={'STAID': str}).set_index('STAID')
        for sub in ('dam_attributes.csv', 'levee_attributes.csv'):
            d = pd.read_csv(root / 'attributes' / sub, dtype={'STAID': str}).set_index('STAID')
            s = s.join(d.drop(columns=[c for c in d.columns if c in s or c == 'basin_area_km2']), how='left')
        _STATIC[root] = s
    return _STATIC[root].copy()


def static_attributes(cfg: dict, basins: list[str]) -> pd.DataFrame:
    """Static attributes for the basins, from CONFIG['static_attributes']:
       groups / columns: which attributes; from_training_period: recompute climate attributes (and signatures, if
       requested) over the training water years only; categorical: 'codes' | 'onehot' | 'drop'; fill: 'median' | 'none'."""
    sa = cfg.get('static_attributes') or {}
    cols = list(sa.get('columns') or [])
    for g in sa.get('groups') or []:
        cols += STATIC_GROUPS[g]
    cols = list(dict.fromkeys(cols))
    if not cols:
        return pd.DataFrame(index=pd.Index(basins, name='STAID'))
    root = data_root(cfg)
    S = _static_table(root).reindex(basins)
    unknown = [c for c in cols if c not in S]
    if unknown:
        raise ValueError(f'unknown static attributes: {unknown}')
    S = S[cols].copy()
    flow = sorted(set(cols) & FLOW_DERIVED)
    if flow:
        warnings.warn(f'{len(flow)} attributes are derived from observed streamflow ({flow[:4]}...). They are '
                      'computed over the training period when from_training_period is set; never use them for basins '
                      'held out in a spatial split.')
    if sa.get('from_training_period') and cfg.get('split', {}).get('train'):
        a, b = (pd.Timestamp(x) for x in cfg['split']['train'])
        wy = (a.year + (a.month >= 10), b.year + (b.month >= 10))          # water years spanned by the training period
        _recompute_training(S, wy, root)
    for c in [c for c in S.columns if S[c].dtype == object]:
        how = sa.get('categorical', 'codes')
        if how == 'drop':
            S = S.drop(columns=c)
        elif how == 'onehot':
            S = S.drop(columns=c).join(pd.get_dummies(S[c], prefix=c, dtype='float32'))
        else:                                                   # stable integer codes over all 995 basins
            levels = sorted(_static_table(root)[c].dropna().astype(str).unique())
            S[c] = S[c].astype(str).map({v: i + 1 for i, v in enumerate(levels)}).astype('float32')
    if sa.get('fill', 'median') == 'median':
        S = S.fillna(S.median(numeric_only=True))
    S.index.name = 'STAID'
    return S.astype('float32')


def _recompute_training(S: pd.DataFrame, wy: tuple[int, int], root: Path):
    """Replace climate attributes (and streamflow signatures) by their mean over the training water years."""
    lo, hi = wy
    ci = pd.read_csv(root / 'attributes/annual/climate_indices_annual.csv', dtype={'STAID': str})
    ci = ci[(ci.water_year >= lo) & (ci.water_year <= hi) & (ci.status == 'ok')].groupby('STAID').mean(numeric_only=True)
    for c in set(CLIMATE_ANNUAL) & set(S.columns):
        S.loc[S.index.isin(ci.index), c] = ci[c].reindex(S.index[S.index.isin(ci.index)]).values
    pa = pd.read_csv(root / 'attributes/annual/pet_aridity_annual.csv', dtype={'STAID': str})
    pa = pa[(pa.water_year >= lo) & (pa.water_year <= hi) & (pa.status == 'ok')].groupby('STAID').mean(numeric_only=True)
    for c in {'pet_mean', 'aridity'} & set(S.columns):
        S.loc[S.index.isin(pa.index), c] = pa[c].reindex(S.index[S.index.isin(pa.index)]).values
    hs = pd.read_csv(root / 'attributes/annual/hydrologic_signatures_annual.csv', dtype={'STAID': str})
    hs = hs[(hs.water_year >= lo) & (hs.water_year <= hi) & (hs.status == 'ok')].groupby('STAID').mean(numeric_only=True)
    for c in set(SIG_ANNUAL) & set(S.columns):
        S.loc[S.index.isin(hs.index), c] = hs[SIG_ANNUAL[c]].reindex(S.index[S.index.isin(hs.index)]).values


# ------------------------------------------------------------------------------------------------------------------
# 5. split
# ------------------------------------------------------------------------------------------------------------------
def make_split(basins: list[str], cfg: dict) -> dict:
    """Train / validation / test split from CONFIG['split'].
       method 'temporal'          all basins in every set; periods train / validation / test
       method 'spatial'           basins divided (fractions, seed, optional stratify by megaregion /
                                  impervious_tercile / area_tercile); every set covers the full period
       method 'spatial_kfold'     k folds of basins (each fold is a test set once); full period
       method 'temporal_spatial'  train basins in the train period, validation basins in the validation period,
                                  test basins in the test period (prediction in ungauged basins and in time)"""
    sp = cfg['split']; meth = sp['method']; rng = np.random.default_rng(sp.get('seed', 42))
    full = tuple(cfg['period'])
    per = {k: tuple(sp[k]) if sp.get(k) else full for k in ('train', 'validation', 'test')}
    out = {'method': meth, 'seed': sp.get('seed', 42)}
    if meth == 'temporal':
        out['sets'] = {k: {'basins': list(basins), 'period': per[k]} for k in per}
        return _check_periods(out, cfg)
    cat = load_catalog().set_index('STAID').reindex(basins)
    strat = sp.get('stratify')
    if strat == 'impervious_tercile':
        key = pd.qcut(cat.IMPNLCD06, 3, labels=False, duplicates='drop')
    elif strat == 'area_tercile':
        key = pd.qcut(cat.area_km2, 3, labels=False, duplicates='drop')
    elif strat == 'megaregion':
        key = cat[strat]
    elif strat:
        raise ValueError(f"unknown stratify {strat!r}: use 'megaregion', 'impervious_tercile' or 'area_tercile'")
    else:
        key = pd.Series(0, index=cat.index)
    if meth == 'spatial_kfold':
        k = int(sp.get('k', 5)); folds = {i: [] for i in range(k)}
        for _, g in key.groupby(key):
            ids = list(g.index); rng.shuffle(ids)
            for j, s in enumerate(ids):
                folds[j % k].append(s)
        out['folds'] = {f'fold_{i + 1}': {'test': sorted(v), 'train': sorted(set(basins) - set(v))} for i, v in folds.items()}
        out['period'] = full
        return out
    ft, fv = float(sp.get('test_fraction', 0.2)), float(sp.get('validation_fraction', 0.1))
    sets = {'train': [], 'validation': [], 'test': []}
    for _, g in key.groupby(key):
        ids = list(g.index); rng.shuffle(ids); n = len(ids)
        nt, nv = int(round(n * ft)), int(round(n * fv))
        sets['test'] += ids[:nt]; sets['validation'] += ids[nt:nt + nv]; sets['train'] += ids[nt + nv:]
    if meth == 'spatial':
        out['sets'] = {k: {'basins': sorted(v), 'period': full} for k, v in sets.items()}
    elif meth == 'temporal_spatial':
        out['sets'] = {k: {'basins': sorted(v), 'period': per[k]} for k, v in sets.items()}
        out = _check_periods(out, cfg)
    else:
        raise ValueError(f'unknown split method {meth!r}')
    return out


def _check_periods(out, cfg):
    p = {k: v['period'] for k, v in out['sets'].items()}
    a, b = pd.Timestamp(cfg['period'][0]), pd.Timestamp(cfg['period'][1])
    for k, (s, e) in p.items():
        if pd.Timestamp(s) < a or pd.Timestamp(e) > b:
            raise ValueError(f'{k} period {s}..{e} lies outside the data period {cfg["period"]}')
    order = sorted(p.items(), key=lambda kv: pd.Timestamp(kv[1][0]))
    for (k1, (s1, e1)), (k2, (s2, e2)) in zip(order, order[1:]):
        if pd.Timestamp(s2) <= pd.Timestamp(e1):
            warnings.warn(f'{k1} and {k2} periods overlap ({e1} >= {s2})')
    return out


# ------------------------------------------------------------------------------------------------------------------
# 6. normalisation statistics (training data only)
# ------------------------------------------------------------------------------------------------------------------
def norm_stats(tables: dict, static: pd.DataFrame, split: dict) -> dict:
    """Mean and standard deviation of every input and the target over the TRAINING basins and period only."""
    if 'folds' in split:
        train_b, (a, b) = split['folds']['fold_1']['train'], split['period']
    else:
        train_b, (a, b) = split['sets']['train']['basins'], split['sets']['train']['period']
    parts = [tables[s].loc[a:b] for s in train_b if s in tables]
    d = pd.concat(parts) if parts else pd.DataFrame()
    st = {'dynamic': {c: {'mean': float(d[c].mean()), 'std': float(d[c].std())} for c in d.columns},
          'static': {c: {'mean': float(static.loc[static.index.isin(train_b), c].mean()),
                         'std': float(static.loc[static.index.isin(train_b), c].std())} for c in static.columns},
          'computed_on': {'basins': len(parts), 'period': [str(a), str(b)]}}
    return st


# ------------------------------------------------------------------------------------------------------------------
# 7. writing
# ------------------------------------------------------------------------------------------------------------------
def estimate_size(cfg: dict, n_basins: int) -> str:
    n_t = len(time_axis(cfg))
    n_f = sum(len(v) for v in cfg['forcing'].values()) + sum(len(v) for v in (cfg.get('pet') or {}).values()) + \
        len(cfg.get('dynamic_attributes') or []) + 1
    gb = n_basins * n_t * n_f * 4 / 1e9
    return f'{n_basins} basins x {n_t:,} steps x {n_f} variables = {gb:.2f} GB in memory as float32'


def write_package(tables: dict, static: pd.DataFrame, report: pd.DataFrame, split: dict, cfg: dict) -> Path:
    """Write the package: data in the chosen format, splits, normalisation statistics, quality report, manifest."""
    out = Path(cfg['output']['dir']); fmt = cfg['output'].get('format', 'neuralhydrology')
    out.mkdir(parents=True, exist_ok=True)
    basins = sorted(tables)
    target = [c for c in tables[basins[0]].columns if c.startswith('streamflow_')][0]
    inputs = [c for c in tables[basins[0]].columns if c != target]
    # splits
    (out / 'splits').mkdir(exist_ok=True)
    if 'folds' in split:
        for f, v in split['folds'].items():
            for k in ('train', 'test'):
                (out / 'splits' / f'{f}_{k}_basins.txt').write_text('\n'.join(s for s in v[k] if s in tables) + '\n')
    else:
        for k, v in split['sets'].items():
            (out / 'splits' / f'{k}_basins.txt').write_text('\n'.join(s for s in v['basins'] if s in tables) + '\n')
    json.dump(split, open(out / 'splits' / 'split.json', 'w'), indent=1, default=str)
    ns = norm_stats(tables, static, split)
    json.dump(ns, open(out / 'normalization_stats.json', 'w'), indent=1)
    report.to_csv(out / 'quality_report.csv')
    static.to_csv(out / 'static_attributes.csv')
    if fmt == 'neuralhydrology':
        import xarray as xr
        (out / 'time_series').mkdir(exist_ok=True); (out / 'attributes').mkdir(exist_ok=True)
        for s in basins:
            ds = xr.Dataset({c: ('date', tables[s][c].values.astype('float32')) for c in tables[s].columns},
                            coords={'date': tables[s].index.values})
            ds.to_netcdf(out / 'time_series' / f'{s}.nc')
        a = static.copy(); a.index.name = 'gauge_id'; a.to_csv(out / 'attributes' / 'attributes.csv')
        (out / 'basins_all.txt').write_text('\n'.join(basins) + '\n')
        _nh_config(out, cfg, split, inputs, target, list(static.columns))
    elif fmt == 'arrays':
        T = len(next(iter(tables.values())))
        x = np.lib.format.open_memmap(out / 'x_dynamic.npy', mode='w+', dtype='float32', shape=(len(basins), T, len(inputs)))
        y = np.lib.format.open_memmap(out / 'y_target.npy', mode='w+', dtype='float32', shape=(len(basins), T, 1))
        for i, s in enumerate(basins):
            x[i] = tables[s][inputs].values; y[i, :, 0] = tables[s][target].values
        x.flush(); y.flush()
        np.save(out / 'c_static.npy', static.reindex(basins).values.astype('float32'))
        json.dump({'basins': basins, 'time': [str(tables[basins[0]].index[0]), str(tables[basins[0]].index[-1])],
                   'n_steps': T, 'step': cfg['resolution'], 'dynamic_features': inputs, 'target': target,
                   'static_features': list(static.columns)}, open(out / 'arrays_index.json', 'w'), indent=1)
    elif fmt == 'csv':
        (out / 'time_series').mkdir(exist_ok=True)
        for s in basins:
            tables[s].to_csv(out / 'time_series' / f'{s}.csv', float_format='%.6g')
    else:
        raise ValueError(f'unknown output format {fmt!r}: neuralhydrology, arrays or csv')
    _manifest(out, cfg, basins, inputs, target, static, report)
    return out


def _nh_config(out, cfg, split, inputs, target, statics):
    """A starting NeuralHydrology config with the right variable names, basin files and dates."""
    def d(x):
        return pd.Timestamp(x).strftime('%d/%m/%Y')
    sets = split.get('sets')
    lines = ['# NeuralHydrology starting config written by ucamels_prep.py; edit model settings before training',
             'experiment_name: urban_camels', f'data_dir: {out}', 'dataset: generic',
             f'train_basin_file: {out / "splits/train_basins.txt"}',
             f'validation_basin_file: {out / "splits/validation_basins.txt"}',
             f'test_basin_file: {out / "splits/test_basins.txt"}']
    if sets:
        for k, nm in (('train', 'train'), ('validation', 'validation'), ('test', 'test')):
            a, b = sets[k]['period']
            lines += [f'{nm}_start_date: {d(a)}', f'{nm}_end_date: {d(b)}']
    lines += ['dynamic_inputs:'] + [f'  - {c}' for c in inputs]
    lines += ['target_variables:', f'  - {target}', 'static_attributes:'] + [f'  - {c}' for c in statics]
    lines += ['model: cudalstm', 'hidden_size: 256', 'initial_forget_bias: 3', 'output_dropout: 0.4',
              'loss: NSE', 'epochs: 30', 'batch_size: 256', f'seq_length: {365 if cfg["resolution"] == "daily" else int(14 * 86400 / step_seconds(cfg))}',
              f'predict_last_n: 1', 'clip_targets_to_zero:', f'  - {target}']
    (out / 'nh_config_template.yml').write_text('\n'.join(lines) + '\n')


def _manifest(out, cfg, basins, inputs, target, static, report):
    units = {}
    for c in inputs:
        p = c.split('__')
        if p[0] == 'pet':
            units[c] = 'mm/day' if cfg['resolution'] == 'daily' else 'mm/h'
        elif p[0] == 'dyn':
            units[c] = DYNAMIC[p[1]][2]
        else:
            u = PRODUCTS[p[0]]['vars'][p[1]]
            units[c] = u.replace('mm/h', 'mm/day') if (cfg['resolution'] == 'daily' and p[0] in ('aorc', 'nldas_hourly')) else u
    unit = cfg['target'].get('unit', 'mm')
    units[target] = {'mm': f'mm per {cfg["resolution"] if cfg["resolution"] != "subhourly" else cfg.get("subhourly_step", "15min")} step',
                     'mm/h': 'mm/h', 'cms': 'm3/s', 'cfs': 'ft3/s'}[unit]
    m = {'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), 'toolkit': 'ucamels_prep.py',
         'config': cfg, 'n_basins': len(basins), 'dropped': report.index[~report.kept].tolist(),
         'time_convention': ('calendar date; streamflow = local calendar day, hourly products = local standard-time day'
                             if cfg['resolution'] == 'daily' else
                             'UTC; each row describes the step that starts at its label (AORC / NLDAS-2 moved back one '
                             'hour from their end-of-hour stamps)'),
         'inputs': inputs, 'target': target, 'units': units, 'static_attributes': list(static.columns)}
    json.dump(m, open(out / 'manifest.json', 'w'), indent=1, default=str)


def validate(cfg: dict) -> None:
    """Check a CONFIG before any data is read; raise with a clear message, warn about risky choices."""
    need = ['resolution', 'period', 'basins', 'forcing', 'target', 'split', 'output']
    miss = [k for k in need if k not in cfg]
    if miss:
        raise ValueError(f'CONFIG is missing {miss}')
    data_root(cfg)
    step = STEP.get(cfg['resolution'])
    if step is None:
        raise ValueError("resolution must be 'daily', 'hourly' or 'subhourly'")
    step_seconds(cfg)
    a, b = (pd.Timestamp(x) for x in cfg['period'])
    if a < pd.Timestamp(WINDOW[0]) or b > pd.Timestamp(WINDOW[1]) or a >= b:
        raise ValueError(f'period {cfg["period"]} must lie inside {WINDOW}')
    if not cfg['forcing']:
        raise ValueError('choose at least one forcing product')
    for prod, vars_ in cfg['forcing'].items():
        if prod not in PRODUCTS:
            raise ValueError(f'unknown forcing product {prod!r}; available {list(PRODUCTS)}')
        bad = set(vars_) - set(PRODUCTS[prod]['vars'])
        if bad:
            raise ValueError(f'{prod}: unknown variables {sorted(bad)}; available {list(PRODUCTS[prod]["vars"])}')
        if PRODUCTS[prod]['step'] == 'D' and step != 'D':
            raise ValueError(f'{prod} is daily; an {cfg["resolution"]} model needs aorc, nldas_hourly or mrms')
        if prod == 'mrms':
            if step == 'D':
                raise ValueError('mrms is offered at the hourly and sub-hourly steps only')
            if a < pd.Timestamp('2014-11-01') and b > pd.Timestamp('2012-01-01'):
                warnings.warn('MRMS has no data from January 2012 to October 2014; with drop_if_inputs_missing=True '
                              'every basin would be dropped. Start the period in November 2014 or later, or keep gaps.')
            if a < pd.Timestamp('2001-10-01'):
                warnings.warn('MRMS starts in 2002; earlier steps will be empty.')
    for pr, ms in (cfg.get('pet') or {}).items():
        key = ('H' if step == 'S' else step, pr)
        if key not in PET or set(ms) - set(PET[key][2]):
            raise ValueError(f'PET {pr} {ms} is not available at the {cfg["resolution"]} step')
    for d in cfg.get('dynamic_attributes') or []:
        if d not in DYNAMIC:
            raise ValueError(f'unknown dynamic attribute {d!r}; available {list(DYNAMIC)}')
    if cfg['target'].get('unit', 'mm') not in ('mm', 'mm/h', 'cms', 'cfs'):
        raise ValueError("target unit must be 'mm', 'mm/h', 'cms' or 'cfs'")
    for g in (cfg.get('static_attributes') or {}).get('groups') or []:
        if g not in STATIC_GROUPS:
            raise ValueError(f'unknown static group {g!r}; available {list(STATIC_GROUPS)}')
    sp = cfg['split']
    if sp.get('method') not in ('temporal', 'spatial', 'spatial_kfold', 'temporal_spatial'):
        raise ValueError("split method must be 'temporal', 'spatial', 'spatial_kfold' or 'temporal_spatial'")
    if sp['method'] in ('temporal', 'temporal_spatial'):
        for k in ('train', 'validation', 'test'):
            if not sp.get(k):
                raise ValueError(f'split method {sp["method"]} needs a {k} period')
            s0, s1 = (pd.Timestamp(x) for x in sp[k])
            if s0 < a or s1 > b or s0 >= s1:
                raise ValueError(f'{k} period {sp[k]} must lie inside the data period {cfg["period"]}')
    if cfg['output'].get('format', 'neuralhydrology') not in ('neuralhydrology', 'arrays', 'csv'):
        raise ValueError("output format must be 'neuralhydrology', 'arrays' or 'csv'")
    flow = set((cfg.get('static_attributes') or {}).get('columns') or []) | {
        c for g in (cfg.get('static_attributes') or {}).get('groups') or [] for c in STATIC_GROUPS[g]}
    if flow & FLOW_DERIVED and sp['method'] != 'temporal':
        warnings.warn('streamflow signatures are static inputs in a spatial split: the held-out basins\' signatures '
                      'come from their observed flow, which is target leakage for ungauged prediction.')
    if flow & FLOW_DERIVED and sp['method'] == 'temporal' and not (cfg.get('static_attributes') or {}).get('from_training_period'):
        warnings.warn('streamflow signatures are computed over the full record, including the test period; set '
                      'static_attributes.from_training_period=True to use the training years only.')
    print('CONFIG is valid.')


def summary(cfg: dict, basins: list[str]) -> None:
    """One-screen summary of what will be built."""
    n_in = sum(len(v) for v in cfg['forcing'].values()) + sum(len(v) for v in (cfg.get('pet') or {}).values()) + \
        len(cfg.get('dynamic_attributes') or [])
    print(f'time step     : {cfg["resolution"]}' + (f' ({cfg.get("subhourly_step", "15min")})' if cfg['resolution'] == 'subhourly' else ''))
    print(f'period        : {cfg["period"][0]} to {cfg["period"][1]}  ({len(time_axis(cfg)):,} steps)')
    print(f'basins        : {len(basins)}')
    print(f'inputs        : {n_in} time-varying ({", ".join(f"{k}: {v}" for k, v in cfg["forcing"].items())})')
    print(f'target        : streamflow in {cfg["target"].get("unit", "mm")}')
    print(f'split         : {cfg["split"]["method"]}')
    print(f'output        : {cfg["output"].get("format")} -> {cfg["output"]["dir"]}')
    print(f'size          : {estimate_size(cfg, len(basins))}')


def describe_products():
    """Print every product, step and variable that CONFIG can use."""
    for p, s in PRODUCTS.items():
        print(f'{p:13s} {"daily" if s["step"] == "D" else "hourly":6s}  ' + ', '.join(f'{k} [{u}]' for k, u in s['vars'].items()))
    print('PET (daily): ' + ', '.join(f'{k[1]}: {v[2]}' for k, v in PET.items() if k[0] == 'D'))
    print('PET (hourly): ' + ', '.join(f'{k[1]}: {v[2]}' for k, v in PET.items() if k[0] == 'H'))
    print('dynamic attributes: ' + ', '.join(f'{k} [{v[2]}]' for k, v in DYNAMIC.items()))
    for g, c in STATIC_GROUPS.items():
        print(f'static {g}: {", ".join(c)}')


# ------------------------------------------------------------------------------------------------------------------
# what can be chosen: one table of every variable, its unit, source and the time steps it can be used at
# ------------------------------------------------------------------------------------------------------------------
SOURCE = {'daymet': ('Daymet v4', '1 km', '1989-10 to 2025-09'),
          'gridmet': ('gridMET', '4 km', '1989-10 to 2025-09'),
          'nldas_daily': ('NLDAS-2', '12.5 km', '1989-10 to 2025-09'),
          'aorc': ('AORC v1.1', '1 km', '1989-10 to 2025-09'),
          'nldas_hourly': ('NLDAS-2', '12.5 km', '1989-10 to 2025-09'),
          'mrms': ('MRMS radar (2-5 min, averaged to the step)', '1 km', '2002 to 2025-09, gap 2012-01 to 2014-10')}
VAR_DESC = {'prcp': 'precipitation', 'pr': 'precipitation', 'tmax': 'maximum air temperature',
            'tmin': 'minimum air temperature', 'tmean': 'mean air temperature', 'tmmx_C': 'maximum air temperature',
            'tmmn_C': 'minimum air temperature', 'tmean_C': 'mean air temperature', 'tair': 'air temperature',
            'srad': 'shortwave radiation', 'lrad': 'longwave radiation', 'vp': 'vapour pressure',
            'dayl': 'day length', 'swe': 'snow water equivalent', 'rmin': 'minimum relative humidity',
            'rmax': 'maximum relative humidity', 'rhmean': 'mean relative humidity', 'vs': 'wind speed',
            'wind': 'wind speed', 'vpd': 'vapour pressure deficit', 'sph': 'specific humidity',
            'pet': 'reference evapotranspiration, grass', 'etr': 'reference evapotranspiration, alfalfa',
            'pres': 'surface pressure', 'potevap': 'potential evaporation (NLDAS-2)',
            'coverage_frac': 'share of the basin covered by radar data'}
USABLE = {'D': 'daily', 'H': 'daily, hourly, sub-hourly'}
DYN_DESC = {'imperv': 'imperviousness', 'class1_urban': 'urban land cover', 'class2_agriculture': 'agricultural land cover',
            'class3_forest': 'forest land cover', 'class4_grassland': 'grassland land cover',
            'class5_bare': 'bare land cover', 'class6_water': 'water and wetland land cover',
            'pop_density': 'population density', 'housing_density': 'housing density'}


def variables() -> pd.DataFrame:
    """Every time-varying input and the target: product, variable, description, unit, native step, the time steps
    (RESOLUTION) it can be used at, source and period. Use the product and variable names in FORCING / PET."""
    rows = []
    for p, s in PRODUCTS.items():
        src, res, per = SOURCE[p]
        usable = 'hourly, sub-hourly' if p == 'mrms' else USABLE[s['step']]
        for v, u in s['vars'].items():
            rows.append(dict(group='forcing', product=p, variable=v, description=VAR_DESC.get(v, v), unit=u,
                             native_step='daily' if s['step'] == 'D' else ('2-5 min' if p == 'mrms' else 'hourly'),
                             usable_at=usable, source=f'{src}, {res}', period=per))
    for (step, p), (_, _, methods) in PET.items():
        for m in methods:
            rows.append(dict(group='PET', product=p, variable=m,
                             description='potential evapotranspiration, ' + ('Priestley-Taylor' if m == 'pet_pt' else 'Penman-Monteith'),
                             unit='mm/day' if step == 'D' else 'mm/h', native_step='daily' if step == 'D' else 'hourly',
                             usable_at='daily' if step == 'D' else 'hourly, sub-hourly',
                             source=f'computed from {p}', period=SOURCE[p][2]))
    for k, (f, _, u) in DYNAMIC.items():
        rows.append(dict(group='annual attribute', product='attributes/annual', variable=k,
                         description=DYN_DESC[k], unit=u,
                         native_step='census years' if k == 'housing_density' else 'annual',
                         usable_at='daily, hourly, sub-hourly', source=f, period='1990-2020' if k == 'housing_density' else '1989-2025'))
    for step, (f, _, c) in TARGET.items():
        rows.append(dict(group='target', product='streamflow', variable=c,
                         description='observed streamflow (USGS NWIS)', unit='mm per step, mm/h, m3/s or ft3/s',
                         native_step={'D': 'daily', 'H': 'hourly', 'S': '5-60 min'}[step],
                         usable_at={'D': 'daily', 'H': 'hourly', 'S': 'sub-hourly'}[step], source='USGS NWIS',
                         period='1989-10 to 2025-09'))
    return pd.DataFrame(rows)


def static_groups() -> pd.DataFrame:
    """The static attribute groups and their columns (STATIC['groups'] in the notebook)."""
    return pd.DataFrame([dict(group=g, n=len(c), columns=', '.join(c)) for g, c in STATIC_GROUPS.items()]).set_index('group')


# ------------------------------------------------------------------------------------------------------------------
# explore: maps, time series, monthly regime, annual trends and correlations (needs matplotlib)
# ------------------------------------------------------------------------------------------------------------------
SERIF = ['Times New Roman', 'Times', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif']     # first one installed is used
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000', '#F0E442']


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('the explore functions need matplotlib: pip install matplotlib') from e
    plt.rcParams.update({'font.family': 'serif', 'font.serif': SERIF, 'mathtext.fontset': 'stix', 'font.size': 12.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
                         'axes.titlesize': 13.5, 'axes.titlelocation': 'left', 'axes.labelsize': 12.5})
    import logging
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)     # quiet when Times New Roman is absent
    return plt


UNIT_TEXT = {'degC': '°C', 'W/m2': 'W/m²', 'persons/km2': 'persons/km²', 'units/km2': 'units/km²'}


def _unit(u: str) -> str:
    return UNIT_TEXT.get(u, u)


def _step_sec(tables: dict) -> int:
    """Time step of the built tables in seconds."""
    ix = next(iter(tables.values())).index
    return int((ix[1] - ix[0]).total_seconds()) if len(ix) > 1 else 86400


def _short(col: str, sec: int = 86400) -> str:
    """Two-line axis label: 'daymet__tmax' -> 'daymet tmax\n(°C)'."""
    full = _label(col, sec)
    name, unit = (full.rsplit(' (', 1) + [''])[:2]
    if col.startswith('streamflow_'):
        name = 'Streamflow'
    elif not col.startswith(('pet__', 'dyn__')):
        name = col.replace('__', ' ')
    return f'{name}\n({unit}' if unit else name


def _basin_name(sid: str) -> str:
    """'PEACHTREE CREEK AT ATLANTA, GA' -> 'Peachtree Creek at Atlanta, GA'."""
    cat = load_catalog().set_index('STAID')
    n = str(cat.loc[sid, 'name']).title()
    for w in (' At ', ' Near ', ' Nr ', ' Of ', ' And ', ' In ', ' On ', ' Above ', ' Below ', ' The '):
        n = n.replace(w, w.lower())
    n = re.sub(r'(\d)(St|Nd|Rd|Th)\b', lambda m: m.group(1) + m.group(2).lower(), n)
    head, _, tail = n.rpartition(', ')
    return f'{head}, {tail.upper()}' if head and len(tail) == 2 else n


def _label(col: str, sec: int = 86400) -> str:
    """Readable name of a table column at a time step of `sec` seconds: 'daymet__prcp' -> 'daymet precipitation
    (mm/day)'. Per-step depths are written as mm/day, mm/h or mm per N min."""
    per_step = 'mm/day' if sec >= 86400 else ('mm/h' if sec == 3600 else f'mm per {sec // 60} min')
    if col.startswith('streamflow_'):
        u = col[len('streamflow_'):].replace('per', '/')
        return f'Streamflow ({per_step if u == "mm" else u})'
    parts = col.split('__')
    if parts[0] == 'dyn':
        return f'{DYN_DESC[parts[1]]} ({_unit(DYNAMIC[parts[1]][2])})'
    if parts[0] == 'pet':
        return f'PET {parts[1]} {"PT" if parts[2] == "pet_pt" else "PM"} ({"mm/day" if sec >= 86400 else "mm/h"})'
    p, v = parts
    return f'{p} {VAR_DESC.get(v, v)} ({_unit(PRODUCTS[p]["vars"].get(v, ""))})'


def _aggregate(t: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    """Mean of every column over each period of `freq` (e.g. 'D', 'W', 'MS'); means keep fluxes and states comparable
    across periods with missing steps. None keeps the built time step."""
    if not freq:
        return t
    g = t.resample(freq)
    return g.mean().where(g.count() >= 0.8 * g.size().max())


def _water_year(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.year + (idx.month >= 10)


def plot_map(basins: list[str] | None = None, color: str = 'IMPNLCD06', ax=None, title: str | None = None):
    """Map of the basins (all 995 if None), coloured by a catalog column: 'IMPNLCD06' (imperviousness 2006, %),
    'area_km2', or 'megaregion'. State outlines from catalog/us_states_outline.csv."""
    plt = _plt()
    cat = load_catalog().set_index('STAID')
    sel = cat.loc[basins] if basins is not None else cat
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    _basemap(ax, cat)
    if color == 'megaregion':
        regs = sel.megaregion.value_counts().index
        cmap = plt.get_cmap('tab20')
        for k, r in enumerate(regs):
            m = sel.megaregion == r
            ax.scatter(sel.lon[m], sel.lat[m], s=34, color=cmap(k % 20), edgecolor='black', lw=0.3, zorder=3,
                       label=f'{r} ({int(m.sum())})')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=11.5, title='Megaregion')
    else:
        from matplotlib.colors import LogNorm
        norm = LogNorm() if color == 'area_km2' else None
        sc = ax.scatter(sel.lon, sel.lat, c=sel[color], s=34, cmap='Reds' if color != 'area_km2' else 'viridis',
                        norm=norm, edgecolor='black', lw=0.3, zorder=3)
        cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
        cb.set_label({'IMPNLCD06': 'Imperviousness, 2006 (%)', 'area_km2': 'Area (km²)'}.get(color, color))
    ax.set_title(title or f'{len(sel)} Urban-CAMELS basins' + (' (grey: all 995)' if len(sel) < len(cat) else ''))
    return ax


def _basemap(ax, cat: pd.DataFrame):
    """State outlines and all 995 gauges in grey, on a lon/lat frame without axes."""
    o = pd.read_csv(TOOLKIT / 'catalog' / 'us_states_outline.csv')
    ax.plot(o.lon, o.lat, color='#9a9a9a', lw=0.6, zorder=1)
    ax.scatter(cat.lon, cat.lat, s=10, color='#d9d9d9', zorder=2)
    ax.set_xlim(-125.5, -66.5); ax.set_ylim(24, 50); ax.set_aspect(1 / np.cos(np.radians(38)))
    ax.set_axis_off()


def plot_imperviousness(basins: list[str], root: str | None = None):
    """Compare basins (two work best): where they are, with arrows on the map, and their annual imperviousness from
    1989 to 2025. root: the data folder (default: UCAMELS_DATA)."""
    plt = _plt()
    imp = pd.read_csv(data_root({'data_root': root}) / 'attributes/annual/impervious_1989_2025.csv', dtype={'STAID': str})
    w = imp[imp.STAID.isin(basins)].pivot(index='year', columns='STAID', values='imperv')
    cat = load_catalog().set_index('STAID')
    colors = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']
    offsets = [(-150, 105), (-290, -45), (-290, 60), (-150, -105)]     # labels to the upper and lower left (points)
    fig = plt.figure(figsize=(14, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0]); ts = fig.add_subplot(gs[1])
    _basemap(ax, cat)
    for k, sid in enumerate(basins):
        c = colors[k % len(colors)]; lon, lat = cat.loc[sid, 'lon'], cat.loc[sid, 'lat']
        a, b = w[sid].iloc[0], w[sid].iloc[-1]
        ax.scatter([lon], [lat], s=120, color=c, edgecolor='black', lw=0.8, zorder=4)
        ax.annotate(f'{_basin_name(sid)}\n{sid} · {a:.0f} % → {b:.0f} %', xy=(lon, lat),
                    xytext=offsets[k % len(offsets)], textcoords='offset points', fontsize=11.5, color=c,
                    fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=c, lw=1.2),
                    arrowprops=dict(arrowstyle='-|>', color=c, lw=1.8, mutation_scale=16, shrinkB=6,
                                    connectionstyle='arc3,rad=0.15'))
        ts.plot(w.index, w[sid], color=c, lw=2.6, marker='o', ms=3.5, label=f'{sid} ({b - a:+.1f} points)')
        ts.annotate(f'{b:.1f} %', xy=(w.index[-1], b), xytext=(6, 0), textcoords='offset points', va='center',
                    color=c, fontsize=11.5, fontweight='bold')
    ax.set_title('Where the basins are')
    ts.set_title('Imperviousness, 1989-2025'); ts.set_xlabel('Year'); ts.set_ylabel('Imperviousness (%)')
    ts.set_xlim(w.index[0] - 1, w.index[-1] + 4); ts.grid(alpha=0.25, lw=0.5)
    ts.legend(frameon=False, loc='best')
    return fig


def plot_timeseries(tables: dict, sid: str | None = None, variables: list[str] | None = None,
                    period: tuple[str, str] | None = None, freq: str | None = None):
    """One basin: precipitation (bars, top), the target, and up to four other inputs, each in its own panel.
    variables: table columns (default: every input except annual attributes); period: ('2020-10-01', '2021-09-30');
    freq: aggregate first ('D', 'W', 'MS'), e.g. to see a long hourly record."""
    plt = _plt()
    sid = sid or next(iter(tables))
    t = tables[sid]
    t = t.loc[period[0]:period[1]] if period else t
    t = _aggregate(t, freq)
    target = [c for c in t.columns if c.startswith('streamflow_')][0]
    sec = _step_sec({sid: tables[sid]})
    cols = variables or [c for c in t.columns if c != target and not c.startswith('dyn__')]
    rain = [c for c in cols if c.split('__')[-1] in ('prcp', 'pr')][:1]
    other = [c for c in cols if c not in rain][:4]
    n = 1 + len(rain) + len(other)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.0 + 1.6 * n), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0] * len(rain) + [1.7] + [0.85] * len(other)))
    axes = np.atleast_1d(axes); k = 0
    width = (t.index[1] - t.index[0]) if len(t) > 1 else pd.Timedelta(days=1)
    if rain:
        axes[0].bar(t.index, t[rain[0]], width=width, color='#56B4E9', align='edge', lw=0)
        axes[0].invert_yaxis(); axes[0].set_ylabel(_short(rain[0], sec))
        k = 1
    axes[k].plot(t.index, t[target], color='#0072B2', lw=0.9)
    axes[k].fill_between(t.index, t[target], color='#0072B2', alpha=0.12, lw=0)
    axes[k].set_ylabel(_short(target, sec))
    for j, c in enumerate(other):
        axes[k + 1 + j].plot(t.index, t[c], color=PALETTE[(j + 1) % len(PALETTE)], lw=0.8)
        axes[k + 1 + j].set_ylabel(_short(c, sec))
    cat = load_catalog().set_index('STAID')
    axes[0].set_title(f'{sid} · {_basin_name(sid)} · {cat.loc[sid, "area_km2"]:,.0f} km²')
    for a in axes:
        a.grid(alpha=0.25, lw=0.5)
    axes[-1].set_xlim(t.index[0], t.index[-1] + width)
    fig.align_ylabels(axes); fig.tight_layout()
    return fig


def plot_monthly(tables: dict, variables: list[str] | None = None):
    """Mean monthly regime over all built basins: for each variable the median of the basins' monthly means
    (line) and the 25-75 % range across basins (band), October to September."""
    plt = _plt()
    first = next(iter(tables.values()))
    cols = variables or [c for c in first.columns if not c.startswith('dyn__')]
    sec = _step_sec(tables)
    per = {c: pd.DataFrame({s: t[c].groupby(t.index.month).mean() for s, t in tables.items()}) for c in cols}
    order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    names = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    n = len(cols); nc = min(3, n); nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.0 * nr), squeeze=False)
    for k, c in enumerate(cols):
        a = axes.flat[k]; d = per[c].reindex(order)
        x = np.arange(12)
        a.fill_between(x, d.quantile(0.25, axis=1), d.quantile(0.75, axis=1), color=PALETTE[k % 8], alpha=0.2, lw=0)
        a.plot(x, d.median(axis=1), color=PALETTE[k % 8], lw=2.2, marker='o', ms=4)
        a.set_xticks(x, names, fontsize=10.5); a.set_title(_label(c, sec), fontsize=12.5); a.grid(alpha=0.25, lw=0.5)
    for a in axes.flat[n:]:
        a.set_visible(False)
    fig.suptitle(f'Mean monthly regime, {len(tables)} basins (line: median, band: 25-75 % of basins)', x=0.01,
                 ha='left', fontweight='bold')
    fig.tight_layout()
    return fig


def plot_annual(tables: dict, variables: list[str] | None = None):
    """Water-year means over all built basins: median across basins (line), 25-75 % range (band), and the
    least-squares trend of the median, in units per decade."""
    plt = _plt()
    first = next(iter(tables.values()))
    cols = variables or list(first.columns)
    sec = _step_sec(tables)
    n = len(cols); nc = min(3, n); nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.0 * nr), squeeze=False)
    for k, c in enumerate(cols):
        a = axes.flat[k]
        d = pd.DataFrame({s: t[c].groupby(_water_year(t.index)).mean().where(
            t[c].groupby(_water_year(t.index)).count() >= 0.9 * t[c].groupby(_water_year(t.index)).size()) for s, t in tables.items()})
        med = d.median(axis=1).dropna()
        a.fill_between(d.index, d.quantile(0.25, axis=1), d.quantile(0.75, axis=1), color=PALETTE[k % 8], alpha=0.2, lw=0)
        a.plot(med.index, med.values, color=PALETTE[k % 8], lw=2)
        if len(med) > 5:
            b = np.polyfit(med.index, med.values, 1)
            a.plot(med.index, np.polyval(b, med.index), color='black', lw=1, ls='--')
            a.text(0.02, 0.95, f'trend {10 * b[0]:+.3g} per decade', transform=a.transAxes, va='top', fontsize=11)
        a.set_title(_label(c, sec), fontsize=12.5); a.grid(alpha=0.25, lw=0.5); a.set_xlabel('Water year')
    for a in axes.flat[n:]:
        a.set_visible(False)
    fig.suptitle(f'Water-year means, {len(tables)} basins (line: median, band: 25-75 % of basins, dashed: trend)',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    return fig


def correlation(tables: dict, variables: list[str] | None = None, freq: str | None = None,
                method: str = 'spearman') -> pd.DataFrame:
    """Correlation between variables at the built time step, or after aggregating to `freq` ('D', 'W', 'MS',
    'YS'). Computed inside each basin, then the median over basins, so differences between basins (climate, size)
    do not create correlation."""
    first = next(iter(tables.values()))
    cols = variables or [c for c in first.columns if not c.startswith('dyn__')]
    mats = [_aggregate(t[cols], freq).corr(method=method, min_periods=30) for t in tables.values()]
    return pd.concat(mats).groupby(level=0).median().loc[cols, cols]


def plot_correlation(tables: dict, variables: list[str] | None = None, freq: str | None = None,
                     method: str = 'spearman'):
    """Heat map of correlation() with the values written in the cells."""
    plt = _plt()
    c = correlation(tables, variables, freq, method)
    sec = _step_sec(tables)                    # means over freq keep the unit of the built step
    labels = [_label(x, sec).replace(' (', '\n(') for x in c.columns]
    fig, ax = plt.subplots(figsize=(1.2 + 1.05 * len(c), 0.9 + 0.95 * len(c)))
    im = ax.imshow(c.values, cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=11, color='white' if abs(v) > 0.6 else 'black')
    ax.set_xticks(range(len(c)), labels, rotation=45, ha='right', fontsize=10.5)
    ax.set_yticks(range(len(c)), labels, fontsize=10.5)
    ax.spines[:].set_visible(False)
    plt.colorbar(im, ax=ax, shrink=0.8, label=f'{method.title()} correlation')
    step = {'D': 'daily', 'W': 'weekly', 'MS': 'monthly', 'YS': 'annual'}.get(freq, freq) if freq else 'built time step'
    ax.set_title(f'{method.title()} correlation at the {step} scale\n(within each basin, median of {len(tables)} basins)')
    fig.tight_layout()
    return fig


def plot_scatter(tables: dict, x: str, y: str, freq: str | None = None, log: bool = False):
    """Density of y against x over all basins and steps (hexagonal bins), after optional aggregation to `freq`."""
    plt = _plt()
    d = pd.concat([_aggregate(t[[x, y]], freq) for t in tables.values()]).dropna()
    if log:
        d = d[(d[x] > 0) & (d[y] > 0)]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    hb = ax.hexbin(d[x], d[y], gridsize=60, bins='log', cmap='viridis', mincnt=1,
                   xscale='log' if log else 'linear', yscale='log' if log else 'linear')
    plt.colorbar(hb, ax=ax, label='steps (log scale)')
    sec = _step_sec(tables)
    ax.set_xlabel(_label(x, sec)); ax.set_ylabel(_label(y, sec))
    r = d[x].corr(d[y], method='spearman')
    ax.set_title(f'{len(d):,} values, {len(tables)} basins · Spearman {r:.2f}')
    fig.tight_layout()
    return fig


def animate_urbanization(highlight: list[str], root: str | None = None, basins: list[str] | None = None,
                         path: str | Path | None = None, fps: int = 4) -> bytes:
    """Animated GIF, one frame per year 1989-2025. Left: the basins (all 995 if None) coloured by that year's
    imperviousness, with arrows to the `highlight` basins (two work best). Right: for the highlighted basins, the
    annual imperviousness, the water-year precipitation (Daymet) and the water-year streamflow, drawn up to that year.
    Returns the GIF bytes and also writes them to `path` if given. root: the data folder (default: UCAMELS_DATA)."""
    plt = _plt()
    from matplotlib import animation
    import tempfile
    r = data_root({'data_root': root})
    cat = load_catalog().set_index('STAID')
    sel = list(cat.index) if basins is None else list(basins)
    imp = pd.read_csv(r / 'attributes/annual/impervious_1989_2025.csv', dtype={'STAID': str})
    w = imp[imp.STAID.isin(set(sel) | set(highlight))].pivot(index='STAID', columns='year', values='imperv')
    years = list(w.columns)
    P, Q = {}, {}
    for sid in highlight:                                       # water-year totals, mm; streamflow needs 90 % of days
        p = _read(r / PRODUCTS['daymet']['file'].format(s=sid), 'date', ['prcp'])['prcp']
        P[sid] = p.groupby(_water_year(p.index)).sum(min_count=330)
        fq = r / TARGET['D'][0].format(s=sid)
        if fq.exists():
            q = _read(fq, 'date', [TARGET['D'][2]])[TARGET['D'][2]]
            q = q.where(q >= 0) * CFS_TO_CMS * 86400 / (cat.loc[sid, 'area_km2'] * 1e6) * 1000
            g = q.groupby(_water_year(q.index))
            Q[sid] = g.sum().where(g.count() >= 329)
        else:
            Q[sid] = pd.Series(dtype=float)
    colors = ['#D55E00', '#0072B2', '#009E73', '#CC79A7']
    offsets = [(-150, 105), (-290, -45), (-290, 60), (-150, -105)]
    fig = plt.figure(figsize=(15, 6.0), dpi=80)
    ax = fig.add_axes([0.0, 0.03, 0.56, 0.86]); cax = fig.add_axes([0.565, 0.17, 0.011, 0.58])
    tsx = [fig.add_axes([0.675, y, 0.305, 0.235]) for y in (0.69, 0.40, 0.11)]
    _basemap(ax, cat)
    pts = w.loc[sel]
    vmax = float(w.max().max())
    sc = ax.scatter(cat.loc[pts.index, 'lon'], cat.loc[pts.index, 'lat'], c=pts[years[0]], s=34, cmap='Reds', vmin=0,
                    vmax=vmax, edgecolor='black', lw=0.3, zorder=3)
    hl = ax.scatter(cat.loc[highlight, 'lon'], cat.loc[highlight, 'lat'], c=w.loc[highlight, years[0]], s=170,
                    cmap='Reds', vmin=0, vmax=vmax, edgecolors=[colors[k % 4] for k in range(len(highlight))], lw=2.6,
                    zorder=4)
    fig.colorbar(sc, cax=cax).set_label('Imperviousness (%)')
    for k, sid in enumerate(highlight):
        ax.annotate(f'{_basin_name(sid)}\n{sid}', xy=(cat.loc[sid, 'lon'], cat.loc[sid, 'lat']),
                    xytext=offsets[k % 4], textcoords='offset points', fontsize=11.5, color=colors[k % 4],
                    fontweight='bold', zorder=5, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=colors[k % 4], lw=1.2),
                    arrowprops=dict(arrowstyle='-|>', color=colors[k % 4], lw=1.8, mutation_scale=16, shrinkB=9,
                                    connectionstyle='arc3,rad=0.15'))
    ax.set_title(f'Imperviousness of {len(pts)} basins')
    yr = ax.text(0.02, 0.05, '', transform=ax.transAxes, fontsize=30, fontweight='bold', color='#7F0000')
    series = [('Imperviousness (%)', {s: w.loc[s] for s in highlight}),
              ('Precipitation (mm/yr)', P), ('Streamflow (mm/yr)', Q)]
    lines = []
    for a, (lab, d) in zip(tsx, series):
        vals = pd.concat([v for v in d.values() if len(v)]) if any(len(v) for v in d.values()) else pd.Series([0.0, 1.0])
        a.set_xlim(years[0] - 0.5, years[-1] + 0.5); lo, hi = float(vals.min()), float(vals.max())
        a.set_ylim(lo - 0.08 * (hi - lo + 1e-9), hi + 0.12 * (hi - lo + 1e-9))
        a.set_ylabel(lab.replace(' (', '\n('), fontsize=11.5); a.grid(alpha=0.25, lw=0.5)
        row = []
        for k, sid in enumerate(highlight):
            (ln,) = a.plot([], [], color=colors[k % 4], lw=2.4)
            (dt,) = a.plot([], [], 'o', color=colors[k % 4], ms=6)
            row.append((ln, dt, d.get(sid, pd.Series(dtype=float))))
        lines.append(row)
    for a in tsx[:-1]:
        a.tick_params(labelbottom=False)
    tsx[-1].set_xlabel('Year (water year for precipitation and streamflow)')
    tsx[0].set_title('The highlighted basins, year by year')
    tsx[0].legend([row[0] for row in lines[0]], highlight, frameon=False, fontsize=10.5, loc='lower right', ncol=1)

    def frame(i):
        y = years[i]
        sc.set_array(pts[y].values); hl.set_array(w.loc[highlight, y].values); yr.set_text(str(y))
        for row in lines:
            for ln, dt, s in row:
                s = s[s.index <= y].dropna()
                ln.set_data(s.index, s.values)
                dt.set_data(s.index[-1:], s.values[-1:])
        return [sc, hl, yr]

    anim = animation.FuncAnimation(fig, frame, frames=len(years), interval=1000 // fps)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / 'urbanization.gif'
        anim.save(f, writer=animation.PillowWriter(fps=fps))
        data = f.read_bytes()
    plt.close(fig)
    if path:
        Path(path).write_bytes(data)
    return data
