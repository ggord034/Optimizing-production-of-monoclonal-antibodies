import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import root_mean_squared_error

from src.constants import *


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def progress_bar(c_iter, n_iters, length_bar=20):
    progress = c_iter / n_iters
    length_fill = round(length_bar * progress)
    sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + (length_bar - length_fill) * " " + "]")
    if c_iter == n_iters:
        print()


def load_dataset():
    # Every batch_####.csv file in the datasets folder is one batch run (i.e., one independent time series)
    records = []
    for file in sorted(glob.glob(os.path.join(DIR_DATASETS, "batch_*.csv"))):
        df_batch = pd.read_csv(file)
        df_batch.insert(0, COL_BATCH, int(os.path.basename(file)[6:10]))
        records.append(df_batch)
    df = pd.concat(records, ignore_index=True)

    # Only keep the initial conditions of the batches that were run
    df_ic = pd.read_csv(FILE_INITIAL_CONDITIONS)
    batch_ids = np.unique(df.loc[:, COL_BATCH])
    df_ic = df_ic.loc[df_ic.loc[:, COL_BATCH].isin(batch_ids)].reset_index(drop=True)

    # Replicate batches share the same initial conditions, so they are given the same Condition ID
    df_ic[COL_CONDITION] = df_ic.groupby(IC_FEATURES).ngroup()

    # Process variables modelled by VAR/NVAR are all columns except Batch ID and Time
    features = df.columns[2:]

    print(f"Number of batches:     {batch_ids.shape[0]}")
    print(f"Unique conditions:     {df_ic[COL_CONDITION].nunique()}")
    print(f"Time points per batch: {df.shape[0] // batch_ids.shape[0]}")
    print(f"Process variables:     {list(features)}")
    print()
    print("Initial conditions of the batch runs:")
    print(df_ic.loc[:, IC_FEATURES].describe().loc[["min", "max"]].to_string())

    return df, df_ic, features, batch_ids


def generate_x_lag_cur_var(p, batch_ids, df, features):
    # Lagged values are generated separately for each batch so that they never cross batch boundaries.
    # Each row of x_lag is [x(t-1), x(t-2), ..., x(t-p)], where every lag contains all process variables.
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    for batch_id in batch_ids:
        mask = df.loc[:, COL_BATCH] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, COL_TIME].to_numpy()

        lagged_values = []
        for i in range(1, p + 1):
            lagged_values.append(x_batch[p - i:-i])

        x_lag.append(np.concatenate(lagged_values, axis=1))
        x_cur.append(x_batch[p:])
        t_cur.append(t_batch[p:])
        b_cur.append(np.full(x_batch.shape[0] - p, batch_id))

    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur


def calculate_normalized_rmse(y_true, y_pred):
    # The process variables have very different scales (VCD ~10^6 cells/mL, mAb ~1 mg/L), so the RMSE of each
    # variable is divided by its standard deviation before averaging. Otherwise, VCD and TCD would dominate the score.
    rmse = root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
    return np.mean(rmse / np.std(y_true, axis=0))


def calculate_cost(glucose, glutamine, vcd):
    return COST_GLUCOSE * glucose + COST_GLUTAMINE * glutamine + COST_VCD * vcd


def calculate_space_time_profit(mab, cost, time_h):
    profit = VALUE_MAB * mab - cost
    return profit / (time_h / HOURS_PER_DAY + DOWNTIME)


def calculate_space_time_yield(mab, time_h):
    return mab / (time_h / HOURS_PER_DAY + DOWNTIME)


def calculate_interval(values):
    # Lower bound, median, and upper bound of the prediction interval
    alpha = 1 - PREDICTION_INTERVAL
    return np.quantile(values, q=[alpha / 2, 0.5, 1 - alpha / 2], axis=0)


def create_screen_grid(df_ic):
    # Candidate initial conditions are evenly spaced within the range of the training data, since the model is not
    # reliable when extrapolating to initial conditions it has never seen.
    levels = []
    for feature in IC_FEATURES:
        levels.append(np.unique(np.linspace(df_ic[feature].min(), df_ic[feature].max(), num=SCREEN_N_LEVELS)))

    grid = [(glucose, glutamine, vcd) for glucose in levels[0] for glutamine in levels[1] for vcd in levels[2]]
    return grid


# ============================================
#           Figures and Tables
# ===========================================
def plot_lobocv_rmse(lobocv_results):
    colors = {"var": "tab:blue", "poly": "tab:orange", "knn": "tab:red"}
    labels = {"var": "VAR", "poly": "NVAR (Polynomial)", "knn": "NVAR (kNN)"}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=200, layout="constrained")
    for model_type in MODEL_TYPES:
        results = [lobocv for lobocv in lobocv_results if lobocv.model_type == model_type]
        p = [lobocv.p for lobocv in results]
        rmse_mean = [np.mean(lobocv.rmse_mab) for lobocv in results]
        rmse_sd = [np.std(lobocv.rmse_mab) for lobocv in results]
        ax.errorbar(p, rmse_mean, yerr=rmse_sd, label=labels[model_type], color=colors[model_type], marker="o",
                    markersize=4, capsize=3)

    ax.set_xlabel("p (number of lagged values)", fontsize=10)
    ax.set_ylabel(f"Forecast RMSE, {COL_MAB}", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    fig.savefig(os.path.join("figures", "figure_01_lobocv_rmse.png"))
    plt.close(fig)


def plot_lobocv_forecasts(lobocv, df, df_ic):
    # One subplot per test batch, showing the forecast made from its initial conditions only
    ncols = 6
    nrows = int(np.ceil(len(lobocv.batch_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 1.8 * nrows), dpi=200, layout="constrained",
                             sharex=True, sharey=True)

    for ax, batch_id in zip(axes.flatten(), lobocv.batch_ids):
        mask = df.loc[:, COL_BATCH] == batch_id
        t_test = df.loc[mask, COL_TIME].to_numpy() / HOURS_PER_DAY
        mab_test = df.loc[mask, COL_MAB].to_numpy()
        mab_lower, mab_median, mab_upper = lobocv.forecasts[batch_id]

        ax.scatter(t_test, mab_test, label="Test", s=4, color="tab:orange", edgecolors="none", zorder=3)
        ax.plot(t_test, mab_median, label=lobocv.name, color="black", linewidth=1, zorder=4)
        ax.fill_between(t_test, mab_lower, mab_upper, label=f"{PREDICTION_INTERVAL:.0%} PI", color="tab:blue",
                        edgecolors="none", alpha=0.3, zorder=2)

        ic = df_ic.loc[df_ic.loc[:, COL_BATCH] == batch_id].iloc[0]
        message = (f"ID {batch_id}: GLN {ic[COL_GLUTAMINE]:.0f} mM\n"
                   f"VCD {ic[COL_VCD] / 1e6:.1f}e6 cells/mL")
        ax.text(x=0.03, y=0.96, s=message, va='top', ha='left', transform=ax.transAxes, fontsize=6)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [d]", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(COL_MAB, fontsize=9)
    axes[0, 0].legend(loc="lower right", fontsize=6)

    fig.savefig(os.path.join("figures", "figure_02_lobocv_forecasts.png"))
    plt.close(fig)


def plot_white_noise(model, df, batch_ids, features):
    # Note: The residuals are calculated on the training data because we are checking whether the model has learned
    # the patterns in the time series, not how well it generalizes to new batches.
    _, _, t_cur, _ = generate_x_lag_cur_var(model.p, batch_ids, df, features)
    idx_mab = list(features).index(COL_MAB)
    res_mab = model.residuals[:, idx_mab]

    t_unique = np.unique(t_cur)
    res_intervals = np.array([calculate_interval(res_mab[t_cur == t]) for t in t_unique])
    y_max = 1.05 * np.max(np.abs(res_mab))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=200, layout="constrained")
    ax.scatter(t_cur / HOURS_PER_DAY, res_mab, label="Residuals", color="tab:orange", s=6, edgecolors="none",
               alpha=0.5, zorder=3)
    ax.plot(t_unique / HOURS_PER_DAY, res_intervals[:, 1], label="Median", color="black", zorder=4)
    ax.fill_between(t_unique / HOURS_PER_DAY, res_intervals[:, 0], res_intervals[:, 2],
                    label=f"{PREDICTION_INTERVAL:.0%} PI", color="tab:blue", edgecolors="none", alpha=0.3, zorder=2)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.5, zorder=1)

    ax.set_xlabel("Time [d]", fontsize=10)
    ax.set_ylabel(f"{COL_MAB} (Residuals)", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(-y_max, y_max)

    fig.savefig(os.path.join("figures", "figure_03_white_noise.png"))
    plt.close(fig)


def plot_response_curves(batches, best_batch):
    # Each initial condition is varied while the other two are held at the recommended values
    attributes = ["glucose", "glutamine", "vcd"]
    labels = [COL_GLUCOSE, COL_GLUTAMINE, COL_VCD]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7), dpi=200, layout="constrained", sharey="row")
    for j, (attribute, label) in enumerate(zip(attributes, labels)):
        others = [a for a in ["glucose", "glutamine", "vcd"] if a != attribute]
        curve = [b for b in batches if all(getattr(b, a) == getattr(best_batch, a) for a in others)]
        curve.sort(key=lambda b: getattr(b, attribute))
        x = [getattr(b, attribute) for b in curve]

        stp = np.array([b.get_harvest_interval(b.stp) for b in curve])
        sty = np.array([b.get_harvest_interval(b.sty) for b in curve])

        for i, (values, ylabel) in enumerate(zip([stp, sty], ["Space-time profit [$/(L d)]",
                                                              "Space-time yield [mg/(L d)]"])):
            ax = axes[i, j]
            ax.plot(x, values[:, 1], color="black", marker="o", markersize=3, label="Median")
            ax.fill_between(x, values[:, 0], values[:, 2], color="tab:blue", edgecolors="none", alpha=0.3,
                            label=f"{PREDICTION_INTERVAL:.0%} PI")
            ax.axvline(x=getattr(best_batch, attribute), color="tab:red", linestyle="--", linewidth=1,
                       label="Recommended")
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)

    axes[0, 0].legend(fontsize=9)

    fig.savefig(os.path.join("figures", "figure_04_response_curves.png"))
    plt.close(fig)


def plot_recommended_batch(batch):
    t = batch.t / HOURS_PER_DAY
    series = [(batch.mab, COL_MAB), (batch.stp, "Space-time profit [$/(L d)]")]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=200, layout="constrained")
    for ax, (values, ylabel) in zip(axes, series):
        lower, median, upper = calculate_interval(values)
        ax.plot(t, median, color="black", label="Median")
        ax.fill_between(t, lower, upper, color="tab:blue", edgecolors="none", alpha=0.3,
                        label=f"{PREDICTION_INTERVAL:.0%} PI")
        ax.axvline(x=batch.harvest_time, color="tab:red", linestyle="--", linewidth=1,
                   label=f"Harvest ({batch.harvest_time:.2f} d)")
        ax.set_xlabel("Time [d]", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=9)

    fig.savefig(os.path.join("figures", "figure_05_recommended_batch.png"))
    plt.close(fig)


def export_batch_runs(df, df_ic):
    # Observed space-time profit of each batch run, if it had been harvested at its best time point
    records = []
    for _, ic in df_ic.iterrows():
        mask = df.loc[:, COL_BATCH] == ic[COL_BATCH]
        t = df.loc[mask, COL_TIME].to_numpy()
        mab = df.loc[mask, COL_MAB].to_numpy()
        cost = calculate_cost(ic[COL_GLUCOSE], ic[COL_GLUTAMINE], ic[COL_VCD])
        stp = calculate_space_time_profit(mab, cost, t)
        i = np.argmax(stp)
        record = (int(ic[COL_BATCH]), ic[COL_GLUCOSE], ic[COL_GLUTAMINE], ic[COL_VCD], cost, t[i] / HOURS_PER_DAY,
                  mab[i], stp[i])
        records.append(record)

    column_names = ["Batch ID", COL_GLUCOSE, COL_GLUTAMINE, COL_VCD, "Cost [$/L]", "Best Harvest Time [d]",
                    "mAb at Harvest [mg/L]", "Space-Time Profit [$/(L d)]"]
    df_export = pd.DataFrame(records, columns=column_names).round(4)
    df_export.to_csv(os.path.join("tables", "table_01_batch_runs.csv"), index=False)


def export_lobocv_rmse(lobocv_results):
    records = []
    for lobocv in lobocv_results:
        record = (lobocv.name, lobocv.p, np.mean(lobocv.rmse_mab), np.std(lobocv.rmse_mab), np.mean(lobocv.coverage))
        records.append(record)

    column_names = ["Model", "p", f"RMSE (mean) {COL_MAB}", f"RMSE (SD) {COL_MAB}",
                    f"{PREDICTION_INTERVAL:.0%} PI Coverage"]
    df_export = pd.DataFrame(records, columns=column_names).round(4)
    df_export.to_csv(os.path.join("tables", "table_02_lobocv_rmse.csv"), index=False)


def export_screen(batches, n_rows=10):
    # Candidate initial conditions ranked by their median space-time profit at harvest
    records = []
    for batch in batches[:n_rows]:
        stp = batch.get_harvest_interval(batch.stp)
        sty = batch.get_harvest_interval(batch.sty)
        record = (batch.glucose, batch.glutamine, batch.vcd, batch.cost, batch.harvest_time, stp[1], sty[1])
        records.append(record)

    column_names = [COL_GLUCOSE, COL_GLUTAMINE, COL_VCD, "Cost [$/L]", "Harvest Time [d]",
                    "Space-Time Profit [$/(L d)]", "Space-Time Yield [mg/(L d)]"]
    df_export = pd.DataFrame(records, columns=column_names).round(4)
    df_export.to_csv(os.path.join("tables", "table_03_screen.csv"), index=False)


def export_recommendation(batch):
    value = batch.get_harvest_interval(batch.value)
    stp = batch.get_harvest_interval(batch.stp)
    sty = batch.get_harvest_interval(batch.sty)

    records = [
        ("Initial glucose [mM]", batch.glucose, None, None),
        ("Initial glutamine [mM]", batch.glutamine, None, None),
        ("Initial VCD [cells/mL]", batch.vcd, None, None),
        ("Harvest time [d]", batch.harvest_time, None, None),
        ("Cell culture cost [$/L]", batch.cost, None, None),
        ("Cell culture value [$/L]", value[1], value[0], value[2]),
        ("Space-time profit [$/(L d)]", stp[1], stp[0], stp[2]),
        ("Space-time yield [mg/(L d)]", sty[1], sty[0], sty[2]),
    ]

    column_names = ["Quantity", "Estimate", f"{PREDICTION_INTERVAL:.0%} PI Lower", f"{PREDICTION_INTERVAL:.0%} PI Upper"]
    df_export = pd.DataFrame(records, columns=column_names).round(4)
    df_export.to_csv(os.path.join("tables", "table_04_recommendation.csv"), index=False)
