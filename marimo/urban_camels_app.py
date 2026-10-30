import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Urban-CAMELS data preparation")


@app.cell(hide_code=True)
def _():
    import contextlib, io, json, os, sys, tempfile, warnings
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import animation

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))    # ucamels_prep.py sits one folder up
    import ucamels_prep as up

    up._plt()
    alt.data_transformers.disable_max_rows()
    SERIF = "Times New Roman, Times, Nimbus Roman, serif"
    return Path, SERIF, alt, animation, contextlib, io, json, mo, np, os, pd, plt, tempfile, up, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Urban-CAMELS: build a model-ready dataset
    **995 urban basins in the conterminous United States, water years 1990 to 2025.** Make your choices with the
    controls below; the map, the checks and the summary update as you go. Press **Build** to assemble the data,
    explore it in the tabs, then press **Write** to save the package.
    """)
    return


@app.cell(hide_code=True)
def _(mo, os):
    data_root = mo.ui.text(value=os.environ.get("UCAMELS_DATA", ""), placeholder="/path/to/Urban-CAMELS", full_width=True,
                           label="Your Urban-CAMELS data folder (the unzipped archives: it holds timeseries/ and attributes/)")
    data_root
    return (data_root,)


@app.cell(hide_code=True)
def _(Path, data_root, mo):
    data_ok = bool(data_root.value) and (Path(data_root.value) / "attributes").is_dir()
    _msg = (mo.callout(mo.md("Data folder found."), kind="success") if data_ok else
            mo.callout(mo.md("**Set the data folder above** to build and to see the animations. Download the archives "
                             "from Zenodo and unzip them into one folder. Choosing basins works without the data."),
                       kind="info"))
    _msg
    return (data_ok,)


@app.cell(hide_code=True)
def _(mo, up):
    catalog = up.load_catalog()
    preset = mo.ui.dropdown({"All 995 basins": "all", "247 benchmark basins (complete daily record)": "benchmark_247"},
                            value="All 995 basins", label="Start from")
    states = mo.ui.multiselect(sorted(catalog.state.unique()), label="States (empty = all)")
    megaregions = mo.ui.multiselect(sorted(catalog.megaregion.dropna().unique()), label="Megaregions (empty = all)")
    imperv = mo.ui.range_slider(start=0, stop=70, step=1, value=[0, 70], label="Imperviousness 2006 (%)", show_value=True)
    area = mo.ui.range_slider(start=0, stop=int(catalog.area_km2.max()) + 1, step=10, value=[0, int(catalog.area_km2.max()) + 1],
                              label="Area (km²)", show_value=True)
    comp_res = mo.ui.dropdown({"no rule": "", "daily": "daily", "hourly": "hourly", "sub-hourly": "subhourly"},
                              value="no rule", label="Streamflow completeness rule")
    return area, catalog, comp_res, imperv, megaregions, preset, states


@app.cell(hide_code=True)
def _(comp_res, mo, up):
    _cols = list(up.coverage(comp_res.value).columns) if comp_res.value else ["WY2004-2025"]
    comp_window = mo.ui.dropdown(_cols, value="WY2004-2025" if "WY2004-2025" in _cols else _cols[0], label="over")
    comp_pct = mo.ui.slider(start=50, stop=100, step=5, value=90, label="at least (%)", show_value=True)
    return comp_pct, comp_window


@app.cell(hide_code=True)
def _(area, catalog, comp_pct, comp_res, comp_window, contextlib, imperv, io, megaregions, preset, states, up):
    BASINS = dict(preset=preset.value)
    if states.value:
        BASINS["states"] = list(states.value)
    if megaregions.value:
        BASINS["megaregions"] = list(megaregions.value)
    if list(imperv.value) != [0, 70]:
        BASINS["impervious_pct"] = tuple(imperv.value)
    if area.value[0] > 0 or area.value[1] < catalog.area_km2.max():
        BASINS["area_km2"] = tuple(area.value)
    if comp_res.value:
        BASINS["completeness"] = dict(resolution=comp_res.value, window=comp_window.value, min_pct=comp_pct.value)
    _log = io.StringIO()
    with contextlib.redirect_stdout(_log):
        basins = up.select_basins(BASINS)
    selection_log = _log.getvalue()
    return BASINS, basins, selection_log


@app.cell(hide_code=True)
def _(basins, comp_pct, comp_res, comp_window, imperv, area, megaregions, mo, plt, preset, selection_log, states, up):
    _fig, _ax = plt.subplots(figsize=(9, 4.6))
    up.plot_map(basins, ax=_ax, title=f"{len(basins)} basins selected (grey: all 995)")
    _controls = mo.vstack([preset, states, megaregions, imperv, area,
                           mo.hstack([comp_res, comp_window, comp_pct], justify="start") if comp_res.value else comp_res])
    mo.vstack([mo.md("## 1 · Basins"), mo.hstack([_controls, _fig], widths=[1, 1.6], align="center"),
               mo.accordion({"How the selection was made": mo.md(f"```\n{selection_log}```")})])
    return


@app.cell(hide_code=True)
def _(Path, alt, animation, basins, catalog, data_ok, data_root, mo, pd, plt, SERIF, tempfile, up):
    mo.stop(not data_ok or not basins, mo.md(""))
    _imp = pd.read_csv(Path(data_root.value) / "attributes/annual/impervious_1989_2025.csv", dtype={"STAID": str})
    _w = _imp[_imp.STAID.isin(basins)].pivot(index="STAID", columns="year", values="imperv")
    _cat = catalog.set_index("STAID").loc[_w.index]
    _years = list(_w.columns)
    _vmax = float(_w.max().max())

    def _gif():
        _fig, _ax = plt.subplots(figsize=(9, 4.3), dpi=90)
        up.plot_map(list(_w.index), ax=_ax)
        for _c in list(_ax.collections)[1:]:
            _c.remove()
        for _cb in _fig.axes[1:]:
            _cb.remove()
        _ax.set_position([0.0, 0.0, 0.87, 0.91])
        _sc = _ax.scatter(_cat.lon, _cat.lat, c=_w[_years[0]], s=46, cmap="Reds", vmin=0, vmax=_vmax,
                          edgecolor="black", lw=0.3, zorder=3)
        _cbar = _fig.colorbar(_sc, cax=_fig.add_axes([0.88, 0.12, 0.022, 0.72]))
        _cbar.set_label("Imperviousness (%)")
        _title = _ax.set_title("")
        _yr = _ax.text(0.02, 0.06, "", transform=_ax.transAxes, fontsize=28, fontweight="bold", color="#7F0000")

        def _frame(i):
            _sc.set_array(_w[_years[i]].values)
            _title.set_text(f"Imperviousness of {len(_w)} basins, 1989-2025")
            _yr.set_text(str(_years[i]))
            return _sc, _yr

        _anim = animation.FuncAnimation(_fig, _frame, frames=len(_years), interval=250)
        _f = Path(tempfile.mkstemp(suffix=".gif")[1])
        _anim.save(_f, writer=animation.PillowWriter(fps=5))
        plt.close(_fig)
        _data = _f.read_bytes(); _f.unlink()
        return _data

    urban_gif = _gif()
    _q = _w.quantile([0.25, 0.5, 0.75]).T.reset_index().rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})
    _band = alt.Chart(_q).mark_area(opacity=0.25, color="#C0392B").encode(
        x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d")), y=alt.Y("q25:Q", title="Imperviousness (%)"), y2="q75:Q")
    _line = alt.Chart(_q).mark_line(color="#7F0000", strokeWidth=3, point=True).encode(
        x="year:Q", y="median:Q", tooltip=[alt.Tooltip("year:Q", format="d"), alt.Tooltip("median:Q", format=".2f", title="median (%)"),
                                           alt.Tooltip("q25:Q", format=".2f", title="25 %"), alt.Tooltip("q75:Q", format=".2f", title="75 %")])
    _trend = (_band + _line).properties(width=330, height=300, title="Median and middle half of the basins").configure(
        font=SERIF).configure_title(fontSize=15, anchor="start")
    _grew = int((_w[_years[-1]] - _w[_years[0]] > 1).sum())
    mo.vstack([mo.md(f"## Urbanization, 1989 to 2025\nImperviousness rose by more than 1 percentage point in "
                     f"**{_grew} of {len(_w)}** selected basins; median {_w[_years[0]].median():.1f} % in 1989, "
                     f"{_w[_years[-1]].median():.1f} % in 2025."),
               mo.hstack([mo.image(urban_gif), _trend], align="center", widths=[1.6, 1])])
    return (urban_gif,)


@app.cell(hide_code=True)
def _(catalog, mo):
    _labels = {f"{r.STAID} · {r.name.title()}": r.STAID for r in catalog.itertuples()}
    _key = {v: k for k, v in _labels.items()}
    compare_a = mo.ui.dropdown(_labels, value=_key.get("02334885"), searchable=True, label="Basin 1")
    compare_b = mo.ui.dropdown(_labels, value=_key.get("02458450"), searchable=True, label="Basin 2")
    return compare_a, compare_b


@app.cell(hide_code=True)
def _(compare_a, compare_b, data_ok, data_root, mo, plt, up):
    mo.stop(not data_ok, mo.md(""))
    plt.close("all")
    _ids = [x for x in (compare_a.value, compare_b.value) if x]
    mo.vstack([mo.md("## Compare two basins\nWhere they are, and how their imperviousness changed from 1989 to 2025. "
                     "The defaults are a basin that urbanized fast and one that was already built out."),
               mo.hstack([compare_a, compare_b], justify="start"),
               up.plot_imperviousness(_ids, data_root.value) if _ids else mo.md("")])
    return


@app.cell(hide_code=True)
def _(mo):
    resolution = mo.ui.radio({"daily": "daily", "hourly": "hourly", "sub-hourly": "subhourly"}, value="daily",
                             label="Time step", inline=True)
    sub_step = mo.ui.dropdown(["5min", "10min", "15min", "30min"], value="15min", label="Sub-hourly step")
    start = mo.ui.date(start="1989-10-01", stop="2025-09-30", value="1989-10-01", label="From")
    end = mo.ui.date(start="1989-10-01", stop="2025-09-30", value="2025-09-30", label="To")
    return end, resolution, start, sub_step


@app.cell(hide_code=True)
def _(end, mo, resolution, start, sub_step):
    mo.vstack([mo.md("## 2 · Time step and period"),
               mo.hstack([resolution, sub_step if resolution.value == "subhourly" else mo.md(""), start, end],
                         justify="start", gap=2),
               mo.md("_Daily rows are dates. Hourly and sub-hourly rows are UTC "
                     "and describe the step that starts at the label. Hourly records are dense from water year 2008; "
                     "MRMS radar starts in 2002 and is missing from January 2012 to October 2014._")])
    return


@app.cell(hide_code=True)
def _(mo, resolution, up):
    _daily = resolution.value == "daily"
    _prods = [p for p, s in up.PRODUCTS.items() if (p != "mrms" if _daily else s["step"] == "H")]
    _default = {"daymet": ["prcp", "tmax", "tmin", "srad", "vp"]} if _daily else {"aorc": ["prcp", "tair", "srad", "vp", "wind"]}
    forcing = mo.ui.dictionary({p: mo.ui.multiselect(list(up.PRODUCTS[p]["vars"]), value=_default.get(p, []),
                                                     label=f"{p} ({up.SOURCE[p][0]})") for p in _prods})
    _key = "D" if _daily else "H"
    _pet = [f"{p}: {m}" for (st, p), (_, _, ms) in up.PET.items() if st == _key for m in ms]
    pet = mo.ui.multiselect(_pet, value=["daymet: pet_pt"] if _daily else ["aorc: pet_pm"], label="PET (product: method)")
    dynamic = mo.ui.multiselect(list(up.DYNAMIC), value=["imperv"], label="Annual attributes")
    target_unit = mo.ui.dropdown({"mm per step": "mm", "mm/h": "mm/h", "m³/s": "cms", "ft³/s": "cfs"},
                                 value="mm per step", label="Streamflow unit")
    return dynamic, forcing, pet, target_unit


@app.cell(hide_code=True)
def _(dynamic, forcing, mo, pet, target_unit, up):
    mo.vstack([mo.md("## 3 · Inputs and target"),
               mo.md("Variables for each forcing product (leave a product empty to skip it):"), forcing,
               mo.hstack([pet, dynamic, target_unit], justify="start", gap=2),
               mo.accordion({"Every variable, its unit, source and the time steps it can be used at": up.variables()})])
    return


@app.cell(hide_code=True)
def _(mo, up):
    static_groups = mo.ui.multiselect(list(up.STATIC_GROUPS), value=["location_topography", "soil", "climate"],
                                      label="Static attribute groups")
    from_training = mo.ui.checkbox(value=True, label="Recompute climate attributes over the training years only")
    max_steps = mo.ui.number(start=0, stop=200, value=2, label="Fill input gaps up to (steps)")
    precip_fill = mo.ui.dropdown(["zero", "linear", "none"], value="zero", label="Precipitation gaps")
    max_missing = mo.ui.slider(start=0, stop=100, step=5, value=100, label="Drop basins with target missing above (%)",
                               show_value=True)
    mo.vstack([mo.md("## 4 · Attributes, gaps and quality"),
               mo.hstack([static_groups, from_training], justify="start", gap=2),
               mo.hstack([max_steps, precip_fill, max_missing], justify="start", gap=2)])
    return from_training, max_missing, max_steps, precip_fill, static_groups


@app.cell(hide_code=True)
def _(mo):
    split_method = mo.ui.dropdown({"temporal (periods)": "temporal", "spatial (basins)": "spatial",
                                   "spatial k-fold": "spatial_kfold", "temporal and spatial": "temporal_spatial"},
                                  value="temporal (periods)", label="Split")
    train_end = mo.ui.date(start="1990-09-30", stop="2025-09-29", value="2005-09-30", label="Training ends")
    val_end = mo.ui.date(start="1990-09-30", stop="2025-09-29", value="2010-09-30", label="Validation ends")
    test_frac = mo.ui.slider(start=0.05, stop=0.5, step=0.05, value=0.2, label="Test share of basins", show_value=True)
    kfold = mo.ui.number(start=2, stop=20, value=5, label="Folds")
    out_format = mo.ui.dropdown(["neuralhydrology", "arrays", "csv"], value="neuralhydrology", label="Output format")
    out_dir = mo.ui.text(value="urban_camels_dataset", label="Output folder")
    return kfold, out_dir, out_format, split_method, test_frac, train_end, val_end


@app.cell(hide_code=True)
def _(kfold, mo, out_dir, out_format, split_method, test_frac, train_end, val_end):
    _m = split_method.value
    _parts = [split_method]
    if _m in ("temporal", "temporal_spatial"):
        _parts += [train_end, val_end]
    if _m in ("spatial", "temporal_spatial"):
        _parts += [test_frac]
    if _m == "spatial_kfold":
        _parts += [kfold]
    mo.vstack([mo.md("## 5 · Split and output"), mo.hstack(_parts, justify="start", gap=2),
               mo.hstack([out_format, out_dir], justify="start", gap=2)])
    return


@app.cell(hide_code=True)
def _(BASINS, data_root, dynamic, end, forcing, from_training, kfold, max_missing, max_steps, out_dir, out_format, pd,
      pet, precip_fill, resolution, split_method, start, static_groups, sub_step, target_unit, test_frac, train_end,
      val_end):
    _p0, _p1 = str(start.value), str(end.value)
    _t1, _v1 = pd.Timestamp(train_end.value), pd.Timestamp(val_end.value)
    _pet = {}
    for _s in pet.value:
        _pr, _me = _s.split(": ")
        _pet.setdefault(_pr, []).append(_me)
    CONFIG = dict(
        data_root=data_root.value, resolution=resolution.value, subhourly_step=sub_step.value, period=(_p0, _p1),
        basins=BASINS, forcing={p: list(v) for p, v in forcing.value.items() if v}, pet=_pet,
        dynamic_attributes=list(dynamic.value), housing_interp="step", target=dict(unit=target_unit.value),
        static_attributes=dict(groups=list(static_groups.value), columns=[], from_training_period=from_training.value,
                               categorical="codes", fill="median"),
        gap_fill=dict(max_steps=int(max_steps.value), precip=precip_fill.value),
        qc=dict(max_target_missing_pct=max_missing.value, drop_if_inputs_missing=True),
        split=dict(method=split_method.value, train=(_p0, str(_t1.date())),
                   validation=(str((_t1 + pd.Timedelta(days=1)).date()), str(_v1.date())),
                   test=(str((_v1 + pd.Timedelta(days=1)).date()), _p1),
                   test_fraction=test_frac.value, validation_fraction=0.1, k=int(kfold.value), seed=42,
                   stratify="megaregion"),
        output=dict(dir=out_dir.value, format=out_format.value))
    return (CONFIG,)


@app.cell(hide_code=True)
def _(CONFIG, basins, contextlib, io, mo, up, warnings):
    _out = io.StringIO()
    config_ok = False
    with warnings.catch_warnings(record=True) as _w, contextlib.redirect_stdout(_out):
        warnings.simplefilter("always")
        try:
            up.validate(CONFIG)
            up.summary(CONFIG, basins)
            config_ok = len(basins) > 0
            _err = "" if config_ok else "No basins are selected."
        except Exception as _e:
            _err = str(_e)
    _msgs = [mo.md("## Check")]
    if _err:
        _msgs.append(mo.callout(mo.md(f"**Cannot build yet:** {_err}"), kind="danger"))
    for _x in _w:
        _msgs.append(mo.callout(mo.md(str(_x.message)), kind="warn"))
    if config_ok:
        _msgs.append(mo.md(f"```\n{_out.getvalue()}```"))
    build_button = mo.ui.run_button(label=f"Build {len(basins)} basins", disabled=not config_ok, kind="success")
    _msgs.append(build_button)
    mo.vstack(_msgs)
    return build_button, config_ok


@app.cell(hide_code=True)
def _(CONFIG, basins, json, mo, pd):
    _cfg = json.dumps(CONFIG, indent=4).replace("true", "True").replace("false", "False").replace("null", "None")
    _code = f"""import ucamels_prep as up

CONFIG = {_cfg}

up.validate(CONFIG)
basins = up.select_basins(CONFIG['basins'])
tables, static, report = up.build(CONFIG, basins)
split = up.make_split(list(tables), CONFIG)
up.write_package(tables, static, report, split, CONFIG)
"""
    mo.accordion({"Reproduce these choices in Python (Jupyter notebook or script)": mo.vstack([
        mo.md("The same build from code: paste it next to `ucamels_prep.py`, or download the configuration and the basin list."),
        mo.hstack([mo.download(data=json.dumps(CONFIG, indent=2).encode(), filename="urban_camels_config.json",
                               label="Configuration (JSON)"),
                   mo.download(data=pd.Series(basins, name="STAID").to_csv(index=False).encode(),
                               filename="urban_camels_basins.csv", label=f"Basin list ({len(basins)} STAIDs)"),
                   mo.download(data=_code.encode(), filename="build_urban_camels.py", label="Python script")],
                  justify="start"),
        mo.md(f"```python\n{_code}```")])})
    return


@app.cell(hide_code=True)
def _(CONFIG, basins, build_button, config_ok, contextlib, io, mo, up):
    mo.stop(not (build_button.value and config_ok), mo.md("_Press **Build** to assemble the data._"))
    _log = io.StringIO()
    with mo.status.spinner(title=f"Building {len(basins)} basins ..."), contextlib.redirect_stdout(_log):
        tables, static, report = up.build(CONFIG, basins)
    mo.vstack([mo.md(f"## Built: {len(tables)} of {len(basins)} basins kept"),
               mo.accordion({"Quality report": report, "Build log": mo.md(f"```\n{_log.getvalue()}```")})])
    return report, static, tables


@app.cell(hide_code=True)
def _(mo, tables):
    _cols = list(next(iter(tables.values())).columns)
    _target = [c for c in _cols if c.startswith("streamflow_")][0]
    _rain = next((c for c in _cols if c.split("__")[-1] in ("prcp", "pr")), _cols[0])
    ex_basin = mo.ui.dropdown(list(tables), value=list(tables)[0], label="Basin", searchable=True)
    ex_from = mo.ui.date(value=str(next(iter(tables.values())).index[-1].date().replace(month=1, day=1)), label="From")
    ex_to = mo.ui.date(value=str(next(iter(tables.values())).index[-1].date()), label="To")
    ex_freq = mo.ui.dropdown({"built step": "", "weekly": "W", "monthly": "MS", "annual": "YS"}, value="monthly",
                             label="Time scale")
    ex_vars = mo.ui.multiselect([c for c in _cols if not c.startswith("dyn__")], label="Variables (empty = all)")
    ex_x = mo.ui.dropdown(_cols, value=_rain, label="x")
    ex_y = mo.ui.dropdown(_cols, value=_target, label="y")
    ex_log = mo.ui.checkbox(value=True, label="log axes")
    storm_rank = mo.ui.dropdown({"largest event": 0, "2nd largest": 1, "3rd largest": 2, "4th largest": 3,
                                 "5th largest": 4}, value="largest event", label="Event")
    return ex_basin, ex_freq, ex_from, ex_log, ex_to, ex_vars, ex_x, ex_y, storm_rank


@app.cell(hide_code=True)
def _(SERIF, alt, ex_basin, ex_from, ex_to, pd, tables, up, warnings):
    _t = tables[ex_basin.value].loc[str(ex_from.value):str(ex_to.value)]
    _target = [c for c in _t.columns if c.startswith("streamflow_")][0]
    _rain = next((c for c in _t.columns if c.split("__")[-1] in ("prcp", "pr")), None)
    _sec = up._step_sec({ex_basin.value: tables[ex_basin.value]})
    _d = pd.DataFrame({"time": _t.index, "Q": _t[_target].values})
    warnings.filterwarnings("ignore", message="Automatically deduplicated selection parameter")
    _zoom = alt.selection_interval(bind="scales", encodings=["x"])
    _x = alt.X("time:T", title=None)
    _q = alt.Chart(_d).mark_area(line={"color": "#0072B2", "strokeWidth": 1.4}, color="#0072B2", opacity=0.15).encode(
        x=_x, y=alt.Y("Q:Q", title=up._label(_target, _sec)),
        tooltip=[alt.Tooltip("time:T", format="%Y-%m-%d %H:%M"), alt.Tooltip("Q:Q", format=".3f", title="streamflow")]
    ).properties(width=760, height=240).add_params(_zoom)
    _charts = [_q]
    if _rain:
        _d["P"] = _t[_rain].values
        _p = alt.Chart(_d).mark_bar(color="#56B4E9").encode(
            x=_x.axis(None), y=alt.Y("P:Q", title=up._label(_rain, _sec), scale=alt.Scale(reverse=True)),
            tooltip=[alt.Tooltip("time:T", format="%Y-%m-%d %H:%M"), alt.Tooltip("P:Q", format=".2f", title="precipitation")]
        ).properties(width=760, height=110).add_params(_zoom)
        _charts = [_p, _q]
    interactive_ts = alt.vconcat(*_charts, spacing=4).resolve_scale(x="shared").configure(font=SERIF).configure_axis(
        labelFontSize=12, titleFontSize=13)
    return (interactive_ts,)


@app.cell(hide_code=True)
def _(Path, animation, ex_basin, plt, storm_rank, tables, tempfile, up):
    _t = tables[ex_basin.value]
    _target = [c for c in _t.columns if c.startswith("streamflow_")][0]
    _rain = next((c for c in _t.columns if c.split("__")[-1] in ("prcp", "pr")), None)
    _sec = up._step_sec({ex_basin.value: _t})
    _half = {86400: 12, 3600: 48}.get(_sec, int(48 * 3600 / _sec))
    _q = _t[_target].dropna()
    _peaks, _used = [], set()
    for _ts in _q.sort_values(ascending=False).index:              # distinct events: peaks at least 2 windows apart
        if all(abs((_ts - _u).total_seconds()) > 2 * _half * _sec for _u in _used):
            _peaks.append(_ts); _used.add(_ts)
        if len(_peaks) > storm_rank.value:
            break
    storm_gif = None
    storm_info = "No streamflow in this basin."
    if len(_peaks) > storm_rank.value:
        _pk = _peaks[storm_rank.value]
        _w = _t.loc[_pk - _half * (_t.index[1] - _t.index[0]):_pk + _half * (_t.index[1] - _t.index[0])]
        _n = len(_w); _stride = max(1, _n // 60)
        _fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(8.5, 5.2), dpi=80, sharex=True,
                                        gridspec_kw=dict(height_ratios=[1, 1.6]))
        _width = (_w.index[1] - _w.index[0])
        _bars = _a1.bar(_w.index, _w[_rain] if _rain else 0 * _w[_target], width=_width, align="edge", color="#56B4E9")
        _a1.invert_yaxis(); _a1.set_ylabel(up._short(_rain, _sec) if _rain else "")
        _a2.plot(_w.index, _w[_target], color="#bbbbbb", lw=1)
        (_line,) = _a2.plot([], [], color="#0072B2", lw=2.4)
        (_dot,) = _a2.plot([], [], "o", color="#D55E00", ms=8)
        _a2.set_ylabel(up._short(_target, _sec))
        _a1.set_title(f"{ex_basin.value} · {up._basin_name(ex_basin.value)}: event of {_pk:%d %b %Y}")
        _fig.autofmt_xdate(); _fig.tight_layout()

        def _frame(i):
            _k = min(_n - 1, i * _stride)
            for _j, _b in enumerate(_bars):
                _b.set_alpha(1.0 if _j <= _k else 0.15)
            _line.set_data(_w.index[:_k + 1], _w[_target].values[:_k + 1])
            _dot.set_data([_w.index[_k]], [_w[_target].values[_k]])
            return (_line, _dot, *_bars)

        _anim = animation.FuncAnimation(_fig, _frame, frames=_n // _stride + 1, interval=120)
        _f = Path(tempfile.mkstemp(suffix=".gif")[1])
        _anim.save(_f, writer=animation.PillowWriter(fps=10))
        plt.close(_fig)
        storm_gif = _f.read_bytes(); _f.unlink()
        storm_info = (f"Peak {_w[_target].max():.2f} at {_pk:%Y-%m-%d %H:%M}; "
                      + (f"rain over the window {_w[_rain].sum():.1f} mm." if _rain and _sec >= 3600 else ""))
    return storm_gif, storm_info


@app.cell(hide_code=True)
def _(ex_basin, ex_freq, ex_from, ex_log, ex_to, ex_vars, ex_x, ex_y, interactive_ts, mo, plt, storm_gif, storm_info,
      storm_rank, tables, up):
    _v = list(ex_vars.value) or None
    _f = ex_freq.value or None
    plt.close("all")
    _ts = up.plot_timeseries(tables, ex_basin.value, period=(str(ex_from.value), str(ex_to.value)))
    _map, _ax = plt.subplots(figsize=(9, 4.6)); up.plot_map(list(tables), ax=_ax)
    mo.vstack([mo.md("## Explore"), mo.ui.tabs({
        "Map": _map,
        "Time series": mo.vstack([mo.hstack([ex_basin, ex_from, ex_to], justify="start"), _ts]),
        "Interactive time series": mo.vstack([mo.hstack([ex_basin, ex_from, ex_to], justify="start"),
                                              mo.md("_Scroll to zoom, drag to pan, hover for values; double-click to reset._"),
                                              interactive_ts]),
        "Storm replay": mo.vstack([mo.hstack([ex_basin, storm_rank], justify="start"), mo.md(storm_info),
                                   mo.image(storm_gif) if storm_gif else mo.md("")]),
        "Monthly regime": mo.vstack([ex_vars, up.plot_monthly(tables, _v)]),
        "Water-year trends": mo.vstack([ex_vars, up.plot_annual(tables, _v)]),
        "Correlation": mo.vstack([mo.hstack([ex_freq, ex_vars], justify="start"), up.plot_correlation(tables, _v, freq=_f)]),
        "Scatter": mo.vstack([mo.hstack([ex_x, ex_y, ex_freq, ex_log], justify="start"),
                              up.plot_scatter(tables, ex_x.value, ex_y.value, freq=_f, log=ex_log.value)]),
    })])
    return


@app.cell(hide_code=True)
def _(mo):
    write_button = mo.ui.run_button(label="Write the package", kind="success")
    mo.vstack([mo.md("## Split and write"), write_button])
    return (write_button,)


@app.cell(hide_code=True)
def _(CONFIG, mo, report, static, tables, up, write_button):
    mo.stop(not write_button.value)
    _split = up.make_split(list(tables), CONFIG)
    out = up.write_package(tables, static, report, _split, CONFIG)
    _rows = [f"| `{p.name}` | " + (f"{len(list(p.iterdir()))} files" if p.is_dir() else f"{p.stat().st_size / 1e6:.2f} MB") + " |"
             for p in sorted(out.iterdir())]
    mo.md(f"**Written to** `{out}`\n\n| | |\n|---|---|\n" + "\n".join(_rows))
    return (out,)


if __name__ == "__main__":
    app.run()
