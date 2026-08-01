from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


def main() -> None: 
    reference = pd.read_csv("data/monitoring/reference.csv") 
    current_path = Path("data/monitoring/predictions.csv") 
    if not current_path.exists(): 
        raise FileNotFoundError("Generate API predictions before monitoring.") 
    current = pd.read_csv(current_path) 
 
    shared_columns = [column for column in reference.columns if column in current.columns] 
    report = Report(metrics=[DataSummaryPreset(), DataDriftPreset()])
    snapshot = report.run( 
        reference_data=reference[shared_columns], 
        current_data=current[shared_columns], 
    ) 
    Path("reports").mkdir(exist_ok=True)
    snapshot.save_html("reports/monitoring.html")
    print("Saved reports/monitoring.html")

if __name__ == "__main__":
    main()