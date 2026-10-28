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
  to the basin's local standard-time day; Daymet and gridMET are used as distributed (their daily precipitation matches
  the 24 h starting near 12 UTC of the date best, see README); nldas_daily is the UTC day.
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

import json, os, time, warnings
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
       area_km2: (min, max); impervious_pct: (min, max); megaregions: [...]; exclude_flags: [...]; exclude: [...]"""
    known = {'preset', 'ids', 'id_file', 'completeness', 'area_km2', 'impervious_pct', 'megaregions', 'exclude_flags',
             'exclude'}
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
    if step == 'D' and 'nldas_daily' in cfg['forcing']:
        warnings.warn('nldas_daily is aggregated over the UTC day (00-24 UTC), while the daily streamflow is the local '
                      'calendar day (a 5-8 h shift). For a daily model prefer nldas_hourly, which the toolkit aggregates '
                      "to each basin's local standard-time day.")
    if step == 'D' and set(cfg.get('pet') or {}) & {'nldas_daily', 'nldas_hourly', 'aorc'}:
        warnings.warn('daily PET from nldas_daily, nldas_hourly or aorc is summed over UTC days; daymet and gridmet PET '
                      'follow local days.')
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
