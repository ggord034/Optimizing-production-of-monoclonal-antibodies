import numpy as np

import src.functions as sf
from src.classes import Batch, ForecastModel, LeaveOneBatchOut
from src.constants import *

rng = np.random.default_rng(RANDOM_STATE)

# 1. Load the batch runs and their initial conditions
sf.print_header("1. Dataset")
df, df_ic, features, batch_ids = sf.load_dataset()
t = np.unique(df.loc[:, COL_TIME])

# 2. Leave-one-batch-out cross-validation: forecast each batch from its initial conditions using a model trained on
# the remaining batches, for every model type and number of lagged values
sf.print_header("2. Leave-one-batch-out cross-validation")
lobocv_results = []
for i, (model_type, p) in enumerate([(m, p) for m in MODEL_TYPES for p in P_VALUES]):
    lobocv = LeaveOneBatchOut(model_type, p, df, df_ic, features, batch_ids)
    lobocv.run(rng)
    lobocv_results.append(lobocv)
    sf.progress_bar(i + 1, len(MODEL_TYPES) * len(P_VALUES))

for lobocv in lobocv_results:
    lobocv.display_results()

# 3. Train the model with the lowest forecast RMSE on all batches
sf.print_header("3. Selected model")
best_lobocv = min(lobocv_results, key=lambda lobocv: np.mean(lobocv.rmse_mab))
model = ForecastModel(best_lobocv.model_type, best_lobocv.p)
model.fit(df, df_ic, batch_ids, features)
model.display_hyperparameters()

# 4. Screen candidate initial conditions within the range of the training data
sf.print_header("4. Initial condition screen")
batches = []
for i, (glucose, glutamine, vcd) in enumerate(sf.create_screen_grid(df_ic)):
    batch = Batch(batch_id=i + 1, glucose=glucose, glutamine=glutamine, vcd=vcd)
    batch.simulate(model, features, t, N_SIMULATIONS_SCREEN, rng)
    batches.append(batch)

# Rank the candidates by their median space-time profit at harvest
batches.sort(key=lambda b: b.get_harvest_interval(b.stp)[1], reverse=True)
print(f"Candidates screened: {len(batches)}")
for batch in batches[:5]:
    print(f"GLC: {batch.glucose:5.1f} mM | GLN: {batch.glutamine:5.2f} mM | VCD: {batch.vcd:9.0f} cells/mL | "
          f"Harvest: {batch.harvest_time:5.2f} d | STP: {batch.get_harvest_interval(batch.stp)[1]:7.2f} $/(L d)")

# 5. Simulate the best candidate again with more simulations for its prediction intervals
sf.print_header("5. Recommended batch")
best_batch = Batch(batch_id=0, glucose=batches[0].glucose, glutamine=batches[0].glutamine, vcd=batches[0].vcd)
best_batch.simulate(model, features, t, N_SIMULATIONS, rng)
best_batch.display_info()

# 6. Plots and tables
sf.plot_lobocv_rmse(lobocv_results)
sf.plot_lobocv_forecasts(best_lobocv, df, df_ic)
sf.plot_white_noise(model, df, batch_ids, features)
sf.plot_response_curves(batches, batches[0])
sf.plot_recommended_batch(best_batch)

sf.export_batch_runs(df, df_ic)
sf.export_lobocv_rmse(lobocv_results)
sf.export_screen(batches)
sf.export_recommendation(best_batch)
