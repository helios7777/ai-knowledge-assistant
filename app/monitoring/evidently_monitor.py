from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

class EvidentlyMonitor:
    """Evidently AI monitoring for model drift"""
    
    def __init__(self):
        self.reference_data = []
        self.current_data = []
    
    def log_prediction(self, query: str, answer: str, confidence: float):
        self.current_data.append({
            "query_length": len(query.split()),
            "answer_length": len(answer.split()),
            "confidence": confidence
        })
    
    def generate_report(self):
        if len(self.current_data) < 10:
            return None
        
        current_df = pd.DataFrame(self.current_data[-50:])
        reference_df = pd.DataFrame(self.current_data[:50]) if len(self.current_data) > 50 else current_df
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        
        return report

evidently_monitor = EvidentlyMonitor()