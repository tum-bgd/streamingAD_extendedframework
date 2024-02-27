

def save_paths(out_base_path: str, date_id: str, model_id: str, reservoir_ids: "list[str]", anomaly_score_ids: "list[str]", filename='anomaly_scores.csv'):
    # reservoir_ids = ['sw_mu_sig', 'ures_mu_sig', 'sw_ks', 'ures_ks', 'ares_al_mu_sig', 'ares_cl_mu_sig', 'ares_al_ks', 'ares_cl_ks']
    # anomaly_score_ids = ['avg_of_window', 'anomaly_likelihood', 'confidence_levels']

    return [
        f'{out_base_path}/{model_id}-{r_id}-{anomaly_score_id}/{date_id}/{filename}'
        for r_id in reservoir_ids
        for anomaly_score_id in anomaly_score_ids
    ]