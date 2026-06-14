# Radius-Weather Analysis

This analyzes `radius_*km/radius_summary.json` outputs from your radius search and produces:

- Tidy per-fold CSV
- Radius-level aggregate CSV (including skip rate and utility)
- Weather x radius summary CSV
- Figures for feasibility and performance trends

## Script

- `scripts/07_analyze_radius_weather.py`

## Default fold-to-session mapping

The script defaults to:

- fold 3 -> Session 1
- fold 1 -> Session 2
- fold 0 -> Session 3
- fold 2 -> Session 4
- fold 4 -> Session 5

## Optional custom mappings

You can override with JSON files.

Example `fold_session_map.json`:

```json
{
  "0": "Session 3",
  "1": "Session 2",
  "2": "Session 4",
  "3": "Session 1",
  "4": "Session 5"
}
```

Example `session_weather_map.json`:

```json
{
  "Session 1": "Clear skies",
  "Session 2": "Snow-covered ground",
  "Session 3": "Windy",
  "Session 4": "Rainy/Windy",
  "Session 5": "Clear skies"
}
```

## Run

```powershell
python scripts/07_analyze_radius_weather.py --results-root "D:/results/20260511-080325Z(experiment norwegian_only with unfrozen backbone)"
```

With custom weather/session mappings:

```powershell
python scripts/07_analyze_radius_weather.py `
  --results-root "D:/results/20260511-080325Z(experiment norwegian_only with unfrozen backbone)" `
  --fold-session-map-json "D:/results/fold_session_map.json" `
  --session-weather-map-json "D:/results/session_weather_map.json" `
  --lambda-skip-penalty 0.2
```

## Outputs

Default output folder:

- `outputs/radius_weather_analysis/`

Key files:

- `radius_fold_tidy.csv`
- `radius_summary_aggregated.csv`
- `weather_radius_summary.csv`
- `01_trained_folds_by_radius.png`
- `02_skip_rate_by_radius.png`
- `03_test_pos_rate_by_radius.png`
- `04_heatmap_auc_weather_radius.png`
- `05_lines_auc_by_weather.png`
- `06_box_auc_by_radius.png`
- `07_scatter_auc_vs_test_pos_rate.png`
