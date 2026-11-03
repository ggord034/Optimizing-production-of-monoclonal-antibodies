import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import src.functions as sf
from src.constants import *


class StandardScalerVAR(TransformerMixin, BaseEstimator):
    # The lagged feature matrix [x(t-1), x(t-2), ..., x(t-p)] contains p copies of the same process variables.
    # sklearn's StandardScaler would scale every column separately, so x_1(t-1) and x_1(t-2) would be scaled
    # differently. Instead, one mean and SD is calculated for each process variable using all of its lags.
    def __init__(self, p=1):
        self.p = p

    def fit(self, x, y=None):
        n_variables = x.shape[1] // self.p

        # Stack the lag blocks vertically so each column holds every lag of one process variable
        x_stacked = np.concatenate([x[:, i * n_variables:(i + 1) * n_variables] for i in range(self.p)], axis=0)

        # Tile the statistics so every lag of a process variable uses the same mean and SD
        self.mean_ = np.tile(np.mean(x_stacked, axis=0), self.p)
        self.scale_ = np.tile(np.std(x_stacked, axis=0), self.p)
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_


class ForecastModel:
    def __init__(self, model_type, p):
        self.model_type = model_type
        self.p = p
        self.model_gscv = None  # Will hold the GridSearchCV object once fit() is called
        self.residuals = None  # 1-step-ahead residuals of the training data, sampled as white noise
        self.x_min = None
        self.x_max = None

        if model_type == "var":
            self.name = f"VAR({p})"
            self.pipeline = make_pipeline(StandardScalerVAR(p=p), LinearRegression())
            self.param_grid = {}
        elif model_type == "poly":
            self.name = f"NVAR({p}) Poly"
            self.pipeline = make_pipeline(StandardScalerVAR(p=p), PolynomialFeatures(include_bias=False),
                                          LinearRegression())
            self.param_grid = dict(polynomialfeatures__degree=POLY_DEGREES)
        elif model_type == "knn":
            self.name = f"NVAR({p}) kNN"
            self.pipeline = make_pipeline(StandardScalerVAR(p=p), KNeighborsRegressor())
            self.param_grid = dict(kneighborsregressor__n_neighbors=KNN_NEIGHBORS,
                                   kneighborsregressor__weights=["uniform"])

    def fit(self, df, df_ic, batch_ids, features):
        x_lag, x_cur, _, b_cur = sf.generate_x_lag_cur_var(self.p, batch_ids, df, features)

        # GroupKFold keeps every batch, and its replicates, within a single fold. Otherwise, a replicate in the training
        # set would make the validation score look better than it would be for a new set of initial conditions.
        condition_ids = df_ic.set_index(COL_BATCH).loc[b_cur, COL_CONDITION].to_numpy()
        self.model_gscv = GridSearchCV(self.pipeline, param_grid=self.param_grid, cv=GroupKFold(n_splits=CV_SPLITS),
                                       scoring=make_scorer(sf.calculate_normalized_rmse, greater_is_better=False))
        self.model_gscv.fit(x_lag, x_cur, groups=condition_ids)

        # Residual errors of the 1-step-ahead predictions: x(t) = prediction + white noise
        self.residuals = x_cur - self.model_gscv.predict(x_lag)

        # Range of each process variable in the training data
        self.x_min = np.min(x_cur, axis=0)
        self.x_max = np.max(x_cur, axis=0)

    def forecast(self, x_initial, n_steps, rng):
        # Each row of x_initial is a separate simulation, so all simulations are forecast in parallel
        x_forecast = [x_initial]

        while len(x_forecast) <= n_steps:
            # Reverse the list and take the first p states: x_lag[0] --> x(t-1), x_lag[1] --> x(t-2), etc.
            # At the start there may be fewer than p states, so the most recent state is repeated.
            x_lag = x_forecast[::-1][:self.p]
            while len(x_lag) < self.p:
                x_lag.append(x_lag[-1])
            x_lag = np.concatenate(x_lag, axis=1)

            # Don't let the model predict values outside the range it has been trained on (polynomials can diverge)
            x_cur = np.clip(self.model_gscv.predict(x_lag), a_min=self.x_min, a_max=self.x_max)

            # Add white noise sampled from the residuals of each process variable separately
            idx = rng.integers(0, self.residuals.shape[0], size=x_cur.shape)
            x_cur = x_cur + np.take_along_axis(self.residuals, idx, axis=0)

            # Prevent negative concentration predictions
            x_cur = np.clip(x_cur, a_min=0, a_max=None)

            x_forecast.append(x_cur)

        return np.stack(x_forecast, axis=1)  # Shape: (simulations, time points, process variables)

    def display_hyperparameters(self):
        message = f"{self.name:<16} | "
        if self.model_type == "poly":
            message += f"Degree: {self.model_gscv.best_params_['polynomialfeatures__degree']}"
        elif self.model_type == "knn":
            knn = self.model_gscv.best_estimator_['kneighborsregressor']
            message += f"k (n_neighbors): {knn.n_neighbors} | Weights: {knn.weights}"
        else:
            message += "No hyperparameters"
        print(message)


class Batch:
    def __init__(self, batch_id, glucose, glutamine, vcd):
        self.batch_id = batch_id
        self.glucose = glucose
        self.glutamine = glutamine
        self.vcd = vcd
        self.cost = sf.calculate_cost(glucose, glutamine, vcd)

        # Will hold the simulated forecasts once simulate() is called. Shape: (simulations, time points)
        self.t = None
        self.mab = None
        self.value = None
        self.stp = None
        self.sty = None
        self.idx_harvest = None
        self.harvest_time = None

    def get_initial_state(self, features):
        # Lactate, ammonia, and mAb start at zero, and the initial TCD is equal to the initial VCD
        state = {COL_GLUCOSE: self.glucose, COL_GLUTAMINE: self.glutamine, COL_LACTATE: 0.0, COL_AMMONIA: 0.0,
                 COL_VCD: self.vcd, COL_TCD: self.vcd, COL_MAB: 0.0}
        return np.array([state[feature] for feature in features])

    def simulate(self, model, features, t, n_simulations, rng):
        x_initial = np.tile(self.get_initial_state(features), (n_simulations, 1))
        x_forecasts = model.forecast(x_initial, n_steps=t.shape[0] - 1, rng=rng)

        self.t = t
        self.mab = x_forecasts[:, :, list(features).index(COL_MAB)]
        self.value = VALUE_MAB * self.mab
        self.stp = sf.calculate_space_time_profit(self.mab, self.cost, t)
        self.sty = sf.calculate_space_time_yield(self.mab, t)

        # Harvest at the time point with the highest median space-time profit
        self.idx_harvest = np.argmax(np.median(self.stp, axis=0))
        self.harvest_time = t[self.idx_harvest] / HOURS_PER_DAY

    def get_harvest_interval(self, values):
        # Lower bound, median, and upper bound at the harvest time
        return sf.calculate_interval(values[:, self.idx_harvest])

    def display_info(self):
        value = self.get_harvest_interval(self.value)
        stp = self.get_harvest_interval(self.stp)
        sty = self.get_harvest_interval(self.sty)
        print(f"Initial glucose:   {self.glucose:10.2f} mM")
        print(f"Initial glutamine: {self.glutamine:10.2f} mM")
        print(f"Initial VCD:       {self.vcd:10.0f} cells/mL")
        print(f"Harvest time:      {self.harvest_time:10.2f} d")
        print(f"Cost:              {self.cost:10.2f} $/L")
        print(f"Value:             {value[1]:10.2f} $/L       ({PREDICTION_INTERVAL:.0%} PI: {value[0]:.2f} to {value[2]:.2f})")
        print(f"Space-time profit: {stp[1]:10.2f} $/(L d)   ({PREDICTION_INTERVAL:.0%} PI: {stp[0]:.2f} to {stp[2]:.2f})")
        print(f"Space-time yield:  {sty[1]:10.4f} mg/(L d)  ({PREDICTION_INTERVAL:.0%} PI: {sty[0]:.4f} to {sty[2]:.4f})")


class LeaveOneBatchOut:
    def __init__(self, model_type, p, df, df_ic, features, batch_ids):
        self.model_type = model_type
        self.p = p
        self.name = ForecastModel(model_type, p).name
        self.df = df
        self.df_ic = df_ic
        self.features = features
        self.batch_ids = batch_ids
        self.rmse_mab = []
        self.coverage = []
        self.forecasts = {}  # Lower bound, median, and upper bound of the mAb forecast for each test batch

    def run(self, rng):
        for batch_id_test in self.batch_ids:
            # Train on every batch except the test batch and its replicates, so the model is always tested on
            # initial conditions it has never seen
            ic = self.df_ic.loc[self.df_ic.loc[:, COL_BATCH] == batch_id_test].iloc[0]
            mask_train = self.df_ic.loc[:, COL_CONDITION] != ic[COL_CONDITION]
            batch_ids_train = self.df_ic.loc[mask_train, COL_BATCH].to_numpy()
            model = ForecastModel(self.model_type, self.p)
            model.fit(self.df, self.df_ic, batch_ids_train, self.features)

            # Forecast the entire test batch from its initial conditions only
            batch = Batch(batch_id_test, ic[COL_GLUCOSE], ic[COL_GLUTAMINE], ic[COL_VCD])
            mask = self.df.loc[:, COL_BATCH] == batch_id_test
            t_test = self.df.loc[mask, COL_TIME].to_numpy()
            mab_test = self.df.loc[mask, COL_MAB].to_numpy()
            batch.simulate(model, self.features, t_test, N_SIMULATIONS_LOBOCV, rng)

            mab_lower, mab_median, mab_upper = sf.calculate_interval(batch.mab)
            self.forecasts[batch_id_test] = (mab_lower, mab_median, mab_upper)
            self.rmse_mab.append(root_mean_squared_error(mab_test, mab_median))

            # Fraction of the test values that fall within the prediction interval
            self.coverage.append(np.mean((mab_test >= mab_lower) & (mab_test <= mab_upper)))

    def display_results(self):
        print(f"{self.name:<16} | RMSE (Forecast): {np.mean(self.rmse_mab):.4f} +/- {np.std(self.rmse_mab):.4f} "
              f"mg/L | {PREDICTION_INTERVAL:.0%} PI Coverage: {np.mean(self.coverage):.1%}")
