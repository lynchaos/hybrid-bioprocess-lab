# Batch Data Contract

`hybridbio.ingestion.load_batches_csv()` converts a scientist-provided CSV
export into validated `Batch` objects. Modeling modules do not consume raw CSV
rows directly.

## Required columns and units

| Column | Meaning | Unit |
|---|---|---|
| `batch_id` | Unique batch identifier | text |
| `time_h` | Elapsed process time | h |
| `Xv_1e6_cells_per_mL` | Viable cell density | $10^6$ cells/mL |
| `S_mM` | Substrate concentration | mM |
| `L_mM` | Lactate concentration | mM |
| `P_mg_per_L` | Product titre | mg/L |
| `V_L` | Culture volume | L |
| `feed_rate_L_per_h` | Continuous feed rate | L/h |
| `feed_start_h` | Feed start time | h |
| `feed_S_mM` | Substrate concentration in feed | mM |

## Validation rules

- Every required column must be present and finite.
- Each batch has at least two measurements with strictly increasing `time_h`.
- Measured states are non-negative; volume is strictly positive.
- Feed settings are constant inside a batch.
- Unit conversion happens before this boundary. A non-unit-explicit export is
  rejected by convention rather than guessed at by model code.

The contract intentionally does not impute missing measurements. Imputation is
a scientific decision that must be configured and recorded by the study layer.