# events_analyzer
python -m src.model build src/data/Показатели.xlsx 100   # → enriched.csv
python -m src.model risk  data/enriched.csv 0 100       # → risk_vector.csv