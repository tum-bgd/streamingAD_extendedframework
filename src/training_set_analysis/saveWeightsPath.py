

def save_weights_paths(out_base_path: str, date_id: str, model_id: str, r_id: str, filename: str, anomaly_score_id=None):
    if anomaly_score_id is None:
        anomaly_score_ids = ['avg_of_window', 'anomaly_likelihood', 'confidence_levels']
        return [
            f'{out_base_path}/{model_id}-{r_id}-{anomaly_score_id}/{date_id}/{filename}'
            for anomaly_score_id in anomaly_score_ids
        ]
    else:
        return [
            f'{out_base_path}/{model_id}-{r_id}-{anomaly_score_id}/{date_id}/{filename}'
        ]