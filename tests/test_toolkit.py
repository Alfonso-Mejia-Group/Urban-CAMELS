"""Toolkit test. Checks, against the data files:
  1. every forcing product and PET at the daily step for 12 random basins (values unchanged, NaNs handled)
  2. hourly AORC / NLDAS / MRMS alignment and target conversion for 8 basins, including a tidal basin (negatives -> NaN)
  3. the sub-hourly step at 5 and 30 min (target = mean of readings in the step; hourly inputs held)
  4. all basin selectors (presets, completeness rule, area, imperviousness, megaregion, flags, exclude, id_file)
  5. static attributes: every group loads, the climate indices have no NaN, from_training_period recompute,
     categorical codes / onehot / drop, leakage warning for streamflow-derived attributes
  6. all four split methods (membership, no leakage, folds), normalisation stats from training only
  7. all three output formats round-trip (netCDF, arrays, csv) and the manifest records the config
  8. gap filling respects max_steps and never touches the target
  9. a full 995-basin daily build with the toolkit's QC (nothing crashes, dropped basins are the expected ones)
 10. the state selector, the variables() table and every explore plot

usage: python tests/test_toolkit.py /path/to/Urban-CAMELS        (or set UCAMELS_DATA)
Takes about 3 minutes with 4 to 8 CPU cores and 16 GB of memory; output goes to a temporary folder."""
import json, os, sys, shutil, tempfile, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ucamels_prep as up

ROOT = up.data_root({'data_root': sys.argv[1] if len(sys.argv) > 1 else None})
TMP = Path(tempfile.mkdtemp(prefix='ucamels_test_'))
ok = True; n_pass = n_fail = 0
def check(cond, msg):
    global ok, n_pass, n_fail
    cond = bool(cond); ok &= cond; n_pass += cond; n_fail += (not cond)
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)

cat = up.load_catalog().set_index('STAID')
hourly_ok = (up.coverage('hourly')['WY2004-2025'] >= 90).reindex(cat.index).fillna(False)
rng = np.random.default_rng(1)
CFS = 0.0283168466

# ---------------------------------------------------------------- 1. daily, every product, 12 basins
basins = list(rng.choice(cat.index[cat.preset_benchmark_247], 12, replace=False))
cfg = dict(data_root=str(ROOT), resolution='daily', period=('1989-10-01', '2025-09-30'), basins=dict(ids=basins),
           forcing={'daymet': ['prcp', 'tmax', 'tmin', 'srad', 'vp', 'dayl', 'swe'],
                    'gridmet': ['pr', 'tmmx_C', 'tmmn_C', 'srad', 'vs', 'sph', 'pet'],
                    'nldas_daily': ['prcp', 'tair', 'wind', 'pres'], 'aorc': ['prcp', 'tair', 'srad'], 'nldas_hourly': ['prcp', 'tair']},
           pet={'daymet': ['pet_pt'], 'gridmet': ['pet_pt', 'pet_pm'], 'nldas_daily': ['pet_pm'], 'aorc': ['pet_pt', 'pet_pm'], 'nldas_hourly': ['pet_pm']},
           dynamic_attributes=['imperv', 'class1_urban', 'class3_forest', 'pop_density', 'housing_density'], target=dict(unit='mm'),
           static_attributes=dict(groups=list(up.STATIC_GROUPS), categorical='codes', fill='none'),
           gap_fill=dict(max_steps=0), qc=dict(max_target_missing_pct=100, drop_if_inputs_missing=False),
           split=dict(method='temporal', train=('1989-10-01', '2009-09-30'), validation=('2009-10-01', '2015-09-30'), test=('2015-10-01', '2025-09-30')),
           output=dict(dir=str(TMP / 't1'), format='neuralhydrology'))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    up.validate(cfg)
tabs, st, rep = up.build(cfg, basins, workers=4)
check(len(tabs) == 12 and rep.kept.all(), '1. twelve basins built at the daily step with all five forcing products')
hd = pd.read_csv(ROOT / 'attributes/annual/housing_1990_2020.csv', dtype={'STAID': str})
bad = []
for s in basins:
    t = tabs[s]
    for prod, cols in (('daymet', ['prcp', 'tmax', 'srad']), ('gridmet', ['pr', 'vs', 'pet']), ('nldas_daily', ['prcp', 'pres'])):
        spec = up.PRODUCTS[prod]
        src = pd.read_csv(ROOT / spec['file'].format(s=s), parse_dates=['date']).set_index('date').reindex(t.index)
        for c in cols:
            a, b = t[f'{prod}__{c}'].values, src[c].values.astype(float)
            m = np.isfinite(b)
            if not np.allclose(a[m], b[m], rtol=1e-5, atol=1e-4) or np.isfinite(a[~m]).any():
                bad.append((s, prod, c))
    # AORC daily = local-standard-day sum of prcp / mean of tair, from end-of-hour stamps
    ar = pd.read_csv(ROOT / f'timeseries/forcing/aorc/{s}/aorc_hourly_1989_2025.csv', parse_dates=['datetime']).set_index('datetime')
    off = up._std_offset(float(cat.loc[s, 'lon']))
    day = pd.Timestamp('2012-06-15')
    lo = day + pd.Timedelta(hours=off + 1); hi = day + pd.Timedelta(hours=off + 24)         # stamps hour-end
    psum = ar.prcp.loc[lo:hi].sum(); tmean = ar.tair.loc[lo:hi].mean()
    if abs(t.loc[day, 'aorc__prcp'] - psum) > 1e-3 or abs(t.loc[day, 'aorc__tair'] - tmean) > 1e-3:
        bad.append((s, 'aorc-day', round(float(t.loc[day, 'aorc__prcp'] - psum), 4)))
    # target
    q = pd.read_csv(ROOT / f'timeseries/streamflow_observed/daily/{s}.csv', parse_dates=['date']).set_index('date').discharge_daily_cfs
    mm = (q * CFS * 86400 / (float(cat.loc[s, 'area_km2']) * 1e6) * 1000).reindex(t.index)
    if not np.allclose(t['streamflow_mm'].values, mm.values, rtol=1e-5, equal_nan=True):
        bad.append((s, 'target', ''))
    # PET
    pet = pd.read_csv(ROOT / f'timeseries/PET/daily/gridmet/{s}.csv', parse_dates=['date']).set_index('date').reindex(t.index)
    if not np.allclose(t['pet__gridmet__pet_pm'].values, pet.pet_pm.values.astype(float), rtol=1e-5, equal_nan=True):
        bad.append((s, 'pet', ''))
    # dynamic: housing density step-held within the census decade
    v2010 = float(hd[(hd.STAID == s) & (hd.year == 2010)].housing_density.iloc[0])
    if not np.allclose(t.loc['2010-01-01':'2019-12-31', 'dyn__housing_density'].unique(), [v2010]):
        bad.append((s, 'housing', ''))
check(not bad, f'1. daily values equal the source files for every product, PET, target and dynamic attributes (bad: {bad[:3]})')
check(st.shape[0] == 12 and st[up.STATIC_GROUPS['climate']].notna().all().all(), '5. static table has all climate indices with no NaN')
nstatic = sum(len(v) for v in up.STATIC_GROUPS.values())
check(st.shape[1] >= nstatic - 5, f'5. every static group loaded ({st.shape[1]} columns for {nstatic} listed)')
# from_training_period recompute: p_mean over WY1990-2009 differs from the full-record value and equals the annual mean
cfg_tr = dict(cfg, static_attributes=dict(groups=['climate', 'hydrologic_signatures'], from_training_period=True, categorical='drop'))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    st_tr = up.static_attributes(cfg_tr, basins)
ca = pd.read_csv(ROOT / 'attributes/annual/climate_indices_annual.csv', dtype={'STAID': str})
ca = ca[(ca.status == 'ok') & ca.water_year.between(1990, 2009)].groupby('STAID').p_mean.mean()
check(np.allclose(st_tr['p_mean'].values, ca.reindex(basins).values, rtol=1e-3), '5. from_training_period: p_mean recomputed over the training water years')
full = up.static_attributes(dict(cfg, static_attributes=dict(groups=['climate'], categorical='drop')), basins)
check((np.abs(st_tr['p_mean'].values - full['p_mean'].values) > 1e-6).any(), '5. training-period climate differs from the full-record static value')
oh = up.static_attributes(dict(cfg, static_attributes=dict(groups=['climate_categorical'], categorical='onehot')), basins)
check(oh.shape[1] >= 4 and set(oh.values.ravel()) <= {0.0, 1.0}, '5. one-hot encoding of categorical attributes')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    up.static_attributes(dict(cfg, static_attributes=dict(groups=['hydrologic_signatures'])), basins)
    check(any('observed streamflow' in str(x.message) for x in w), '5. leakage warning for streamflow-derived attributes')
# split + write + readback
sp = up.make_split(list(tabs), cfg); out = up.write_package(tabs, st, rep, sp, cfg)
ns = json.load(open(out / 'normalization_stats.json')); man = json.load(open(out / 'manifest.json'))
check(ns['computed_on']['period'] == ['1989-10-01', '2009-09-30'], '6. normalisation stats from the training period only')
import xarray as xr
ds = xr.open_dataset(out / 'time_series' / f'{basins[0]}.nc')
check(set(ds.data_vars) == set(tabs[basins[0]].columns) and np.allclose(ds['daymet__prcp'].values, tabs[basins[0]]['daymet__prcp'].values, equal_nan=True),
      '7. netCDF round-trip equals the built table')
check(man['config']['resolution'] == 'daily' and (out / 'quality_report.csv').exists() and (out / 'static_attributes.csv').exists(),
      '7. manifest, quality report and static table written')

# ---------------------------------------------------------------- 8. gap filling
cfg_g = dict(cfg, basins=dict(ids=basins[:2]), forcing={'daymet': ['prcp', 'tmax']}, pet={}, dynamic_attributes=[], gap_fill=dict(max_steps=2, precip='zero'))
tg, _, rg = up.build(cfg_g, basins[:2], workers=2)
t0 = tg[basins[0]]
check(t0[['daymet__prcp', 'daymet__tmax']].isna().sum().sum() == 0, '8. Daymet leap-day gaps (1 day) filled with max_steps=2')
raw_q = tabs[basins[0]]['streamflow_mm']
check(t0['streamflow_mm'].isna().sum() == raw_q.isna().sum(), '8. gap filling never touches the target')
check(t0.loc[pd.Timestamp('2000-12-31'), 'daymet__prcp'] == 0.0, '8. precipitation gaps filled with zero when precip="zero"')

# ---------------------------------------------------------------- 4. selectors
sel = up.select_basins(dict(preset='all'), verbose=False); check(len(sel) == 995, '4. preset all -> 995')
sel = up.select_basins(dict(preset='benchmark_247'), verbose=False); check(len(sel) == 247, '4. preset benchmark_247 -> 247')
sel = up.select_basins(dict(preset='all', completeness=dict(resolution='daily', window='WY1990-2025', min_pct=90)), verbose=False)
check(len(sel) == 340, f'4. completeness rule daily WY1990-2025 >= 90 % -> 340 ({len(sel)})')
sel = up.select_basins(dict(completeness=dict(resolution='hourly', window='WY2004-2025', min_pct=90)), verbose=False)
check(len(sel) == int(hourly_ok.sum()), f'4. completeness rule hourly WY2004-2025 >= 90 % matches the coverage table ({len(sel)})')
sel = up.select_basins(dict(preset='all', area_km2=(0, 10)), verbose=False); check(len(sel) == int((cat.area_km2 <= 10).sum()), f'4. area filter (0, 10), inclusive bounds -> {len(sel)}')
sel = up.select_basins(dict(preset='all', impervious_pct=(20, 100)), verbose=False)
check(len(sel) == int((cat.IMPNLCD06 >= 20).sum()), '4. imperviousness filter matches the catalogue')
sel = up.select_basins(dict(preset='all', megaregions=['Florida']), verbose=False); check(len(sel) == 61, f'4. megaregion Florida -> 61 ({len(sel)})')
sel = up.select_basins(dict(preset='all', exclude_flags=['negative_hourly_q']), verbose=False)
check(len(sel) == 995 - int(cat.flag_negative_hourly_q.sum()), '4. exclude_flags removes the flagged basins')
sel = up.select_basins(dict(preset='benchmark_247', exclude=basins[:5]), verbose=False); check(len(sel) == 242, '4. exclude list')
idf = TMP / 'ids.txt'; idf.write_text('\n'.join(basins[:7]))
sel = up.select_basins(dict(id_file=str(idf)), verbose=False); check(sorted(sel) == sorted(basins[:7]), '4. id_file')
try:
    up.select_basins(dict(groups=['x']), verbose=False); check(False, '4. unknown selection key is rejected')
except ValueError:
    check(True, '4. unknown selection key is rejected')

# ---------------------------------------------------------------- 2. hourly, 8 basins incl. a tidal one
tidal = cat.index[cat.flag_negative_hourly_q & hourly_ok][:1].tolist()
hb = tidal + list(rng.choice(cat.index[hourly_ok & ~cat.flag_negative_hourly_q], 7, replace=False))
cfg_h = dict(data_root=str(ROOT), resolution='hourly', period=('2017-10-01', '2018-09-30'), basins=dict(ids=hb),
             forcing={'aorc': ['prcp', 'tair', 'pres'], 'nldas_hourly': ['prcp', 'tair'], 'mrms': ['prcp', 'coverage_frac']},
             pet={'aorc': ['pet_pt'], 'nldas_hourly': ['pet_pm']}, dynamic_attributes=['imperv'], target=dict(unit='cms'),
             static_attributes=dict(groups=['location_topography', 'climate'], categorical='drop'),
             gap_fill=dict(max_steps=6, precip='zero'), qc=dict(max_target_missing_pct=50, drop_if_inputs_missing=False),
             split=dict(method='spatial', test_fraction=0.25, validation_fraction=0.25, seed=3, stratify='impervious_tercile'),
             output=dict(dir=str(TMP / 't2'), format='arrays'))
th, sth, rh = up.build(cfg_h, hb, workers=4)
check(len(th) == 8 and all(len(t) == 8760 for t in th.values()), '2. eight basins built at the hourly step (8760 h)')
bad = []
for s in hb:
    t = th[s]
    ar = pd.read_csv(ROOT / f'timeseries/forcing/aorc/{s}/aorc_hourly_1989_2025.csv', parse_dates=['datetime']).set_index('datetime')
    nl = pd.read_csv(ROOT / f'timeseries/forcing/nldas_hourly/{s}/nldas_hourly_1989_2025.csv', parse_dates=['datetime_utc']).set_index('datetime_utc')
    nl.index = nl.index.tz_convert(None) if nl.index.tz is not None else nl.index
    sh = t.index + pd.Timedelta(hours=1)
    if not np.allclose(t['aorc__tair'].values, ar.tair.reindex(sh).values, atol=1e-4, equal_nan=True): bad.append((s, 'aorc shift'))
    if not np.allclose(t['nldas_hourly__tair'].values, nl.tair.reindex(sh).values, atol=1e-4, equal_nan=True): bad.append((s, 'nldas shift'))
    r0 = t['aorc__prcp'].corr(t['mrms__prcp']); r1 = t['aorc__prcp'].corr(t['mrms__prcp'].shift(-1)); rm1 = t['aorc__prcp'].corr(t['mrms__prcp'].shift(1))
    if not (r0 > r1 and r0 > rm1): bad.append((s, f'mrms lag r0={r0:.2f} r+1={r1:.2f} r-1={rm1:.2f}'))
    q = pd.read_csv(ROOT / f'timeseries/streamflow_observed/hourly/{s}.csv', parse_dates=['datetime']).set_index('datetime').discharge_hourly_cfs
    q = q[~q.index.duplicated()].reindex(t.index) * CFS
    exp = q.where(q >= 0)
    if not np.allclose(t['streamflow_cms'].values, exp.values, rtol=1e-5, equal_nan=True): bad.append((s, 'target cms'))
check(not bad, f'2. hourly: AORC/NLDAS moved to hour-start, MRMS aligned, target in m3/s (bad: {bad[:3]})')
qt = pd.read_csv(ROOT / f'timeseries/streamflow_observed/hourly/{tidal[0]}.csv', parse_dates=['datetime']).set_index('datetime').discharge_hourly_cfs
nneg = int((qt.loc['2017-10-01':'2018-09-30'] < 0).sum())
check(nneg > 0 and th[tidal[0]]['streamflow_cms'].min() >= 0, f'2. tidal basin {tidal[0]}: {nneg} negative source hours became NaN, none remain')
sp_h = up.make_split(list(th), cfg_h)
sets = {k: set(v['basins']) for k, v in sp_h['sets'].items()}
check(sum(len(v) for v in sets.values()) == 8 and not (sets['train'] & sets['test']) and not (sets['train'] & sets['validation']), '6. stratified spatial split: disjoint sets covering every basin')
out_h = up.write_package(th, sth, rh, sp_h, cfg_h)
x = np.load(out_h / 'x_dynamic.npy', mmap_mode='r'); y = np.load(out_h / 'y_target.npy', mmap_mode='r'); idx = json.load(open(out_h / 'arrays_index.json'))
check(x.shape == (8, 8760, len(idx['dynamic_features'])) and y.shape[:2] == (8, 8760), f'7. arrays written {x.shape}, {y.shape}')
b0 = idx['basins'][0]; j = idx['dynamic_features'].index('aorc__prcp')
check(np.allclose(x[0, :, j], th[b0]['aorc__prcp'].values, equal_nan=True), '7. arrays round-trip equals the built table')

# ---------------------------------------------------------------- 3. sub-hourly at 5 and 30 min
sb = hb[1:3]
for step, per_hour in (('5min', 12), ('30min', 2)):
    cfg_s = dict(data_root=str(ROOT), resolution='subhourly', subhourly_step=step, period=('2019-06-01', '2019-06-30'), basins=dict(ids=sb),
                 forcing={'aorc': ['prcp', 'tair'], 'mrms': ['prcp']}, pet={'aorc': ['pet_pt']}, target=dict(unit='mm/h'),
                 static_attributes=dict(groups=['location_topography']), gap_fill=dict(max_steps=per_hour, precip='zero'),
                 qc=dict(max_target_missing_pct=80, drop_if_inputs_missing=False),
                 split=dict(method='temporal', train=('2019-06-01', '2019-06-20'), validation=('2019-06-21', '2019-06-25'), test=('2019-06-26', '2019-06-30')),
                 output=dict(dir=str(TMP / f't3_{step}'), format='csv'))
    ts, sts, rs = up.build(cfg_s, sb, workers=2)
    t = ts[sb[0]]
    check(len(t) == 30 * 24 * per_hour, f'3. {step} axis has {len(t)} steps')
    sub = pd.read_csv(ROOT / f'timeseries/streamflow_observed/subhourly/{sb[0]}.csv', parse_dates=['datetime_utc']).set_index('datetime_utc').discharge_subhourly_cfs
    sub.index = sub.index.tz_convert(None)
    t_at = t.index[len(t) // 2]; win = sub.loc[t_at:t_at + pd.Timedelta(step) - pd.Timedelta(seconds=1)]
    if len(win):
        exp = win.mean() * CFS * 3600 / (float(cat.loc[sb[0], 'area_km2']) * 1e6) * 1000
        check(abs(t.loc[t_at, 'streamflow_mmperh'] - exp) < 1e-6, f'3. {step} target = mean of readings in the step, mm/h')
    hour0 = t_at.floor('h'); blk = t.loc[hour0:hour0 + pd.Timedelta(minutes=59), 'aorc__tair']
    check(blk.nunique() == 1, f'3. {step}: hourly AORC held constant across the {per_hour} steps of the hour')
    sp_s = up.make_split(list(ts), cfg_s); out_s = up.write_package(ts, sts, rs, sp_s, cfg_s)
    rt = pd.read_csv(next((out_s / 'time_series').glob('*.csv')), index_col=0, parse_dates=True)
    check(len(rt) == len(t), f'7. csv round-trip for the {step} build')

# ---------------------------------------------------------------- 6. k-fold and temporal_spatial
cfg_k = dict(cfg, basins=dict(ids=basins), split=dict(method='spatial_kfold', k=4, seed=0))
folds = up.make_split(basins, cfg_k)['folds']
check(len(folds) == 4 and sorted(sum((f['test'] for f in folds.values()), [])) == sorted(basins), '6. spatial k-fold: every basin tested exactly once')
cfg_ts = dict(cfg, split=dict(method='temporal_spatial', train=('1989-10-01', '2009-09-30'), validation=('2009-10-01', '2015-09-30'),
                              test=('2015-10-01', '2025-09-30'), test_fraction=0.25, seed=0))
sp_ts = up.make_split(basins, cfg_ts)
check('sets' in sp_ts and sorted(sum((v['basins'] for v in sp_ts['sets'].values()), [])) == sorted(basins), '6. temporal_spatial split covers every basin once')

# ---------------------------------------------------------------- 9. full 995 daily build
cfg_all = dict(data_root=str(ROOT), resolution='daily', period=('1989-10-01', '2025-09-30'), basins=dict(preset='all'),
               forcing={'daymet': ['prcp', 'tmax', 'tmin']}, pet={'daymet': ['pet_pt']}, dynamic_attributes=['imperv'], target=dict(unit='mm'),
               static_attributes=dict(groups=['location_topography', 'climate'], categorical='drop'),
               gap_fill=dict(max_steps=2, precip='zero'), qc=dict(max_target_missing_pct=100, drop_if_inputs_missing=True),
               split=dict(method='temporal', train=('1989-10-01', '2009-09-30'), validation=('2009-10-01', '2015-09-30'), test=('2015-10-01', '2025-09-30')),
               output=dict(dir=str(TMP / 't9'), format='neuralhydrology'))
t0 = time.time()
ta, sta, ra = up.build(cfg_all, up.select_basins(cfg_all['basins'], verbose=False))
check(len(ta) == 995 and ra.kept.all(), f'9. all 995 basins built at the daily step in {time.time()-t0:.0f} s, none dropped (inputs complete after filling)')
check(sta.shape[0] == 995 and sta.isna().sum().sum() == 0, '9. static table for 995 basins has no NaN (climate + topography)')
all_nan_target = sum(int(t['streamflow_mm'].isna().all()) for t in ta.values())
check(all_nan_target == 86, f'9. {all_nan_target} basins have no daily streamflow (expected 86: 53 without a daily file, 33 whose daily file holds no values)')
cfg_drop = dict(cfg_all, qc=dict(max_target_missing_pct=0, drop_if_inputs_missing=True))
tb, _, rb = up.build(cfg_drop, up.select_basins(dict(preset='all'), verbose=False))
check(len(tb) == 247 and (~rb.kept).sum() == 748, f'9. max_target_missing_pct=0 keeps exactly the 247 complete basins ({len(tb)})')

# ---------------------------------------------------------------- 10. states, variables table, explore plots
pa = up.select_basins(dict(states=['PA']), verbose=False)
check(len(pa) == int((cat.state == 'PA').sum()) and len(pa) > 0 and all(cat.loc[pa, 'state'] == 'PA'),
      f'10. states=[PA] selects the {len(pa)} Pennsylvania basins')
two = up.select_basins(dict(states='pa, wa'.split(', '), preset='benchmark_247'), verbose=False)
check(set(cat.loc[two, 'state']) <= {'PA', 'WA'} and len(two) == int((cat.state.isin(['PA', 'WA']) & cat.preset_benchmark_247).sum()),
      '10. states combine with the other selectors (lower case accepted)')
vt = up.variables()
check(sum(len(s['vars']) for s in up.PRODUCTS.values()) == int((vt.group == 'forcing').sum()) and
      int((vt.group == 'PET').sum()) == sum(len(v[2]) for v in up.PET.values()) and set(vt.group) == {'forcing', 'PET', 'annual attribute', 'target'},
      f'10. variables() lists every product variable, PET method, annual attribute and target ({len(vt)} rows)')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sub = dict(list(tabs.items())[:6])
try:
    figs = [up.plot_map(basins).figure, up.plot_map(color='megaregion').figure, up.plot_timeseries(sub, period=('2020-10-01', '2021-09-30')),
            up.plot_monthly(sub), up.plot_annual(sub, ['daymet__prcp', 'streamflow_mm', 'dyn__imperv']),
            up.plot_correlation(sub, ['daymet__prcp', 'daymet__tmax', 'pet__daymet__pet_pt', 'streamflow_mm'], freq='MS'),
            up.plot_scatter(sub, 'daymet__prcp', 'streamflow_mm', freq='W', log=True),
            up.plot_imperviousness(['02334885', '02458450'], str(ROOT))]
    check(len(figs) == 8, '10. all explore plots draw (map, time series, monthly, annual, correlation, scatter, imperviousness comparison)')
except Exception as e:
    check(False, f'10. explore plots: {type(e).__name__}: {e}')
plt.close('all')
c = up.correlation(sub, ['daymet__prcp', 'streamflow_mm'], freq='MS')
check(c.shape == (2, 2) and np.allclose(np.diag(c), 1) and -1 <= c.iloc[0, 1] <= 1, f'10. monthly correlation P-Q = {c.iloc[0, 1]:.2f}')

shutil.rmtree(TMP, ignore_errors=True)
print(f'\n{n_pass} passed, {n_fail} failed')
print('ALL PASSED' if ok else 'SOME TESTS FAILED')
sys.exit(0 if ok else 1)
