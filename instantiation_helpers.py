import numpy as np
from itertools import product

from src.tsWindowPublisher import TsWindowPublisher
from src.dataRepresentation import WindowStreamVectors
from src.modelWrapper import ModelWrapper
from src.models.simpleRegressionModel import get_simple_regression_model
from src.models.usad import get_usad
from src.models.onlineARIMA import get_online_arima
from src.models.nbeats import get_nbeats
from src.models.onlineVAR import OnlineVAR
from src.nonconformity_scores.nonconformity_wrapper import NonConformityWrapper
from src.anomaly_scores.anomalyLikelihood import AnomalyLikelihood
from src.anomaly_scores.confidenceLevels import ConfidenceLevels
from src.anomaly_scores.averageOfWindow import AverageOfWindow


def instantiate_model_wrappers(models_list: list, publisher: WindowStreamVectors, input_shape: "tuple[int]", debug=False):
    new_models = [
        ModelWrapper(tf_model=get_simple_regression_model(input_shape=input_shape),
                     publisher=publisher, subscribers=[], model_id='simple_regression',
                     model_type='reconstruction', debug=debug),
        ModelWrapper(tf_model=get_usad(input_shape=input_shape, latent_size=input_shape[0]//10),
                     publisher=publisher, subscribers=[], model_id='usad',
                     model_type='reconstruction', debug=debug),
        ModelWrapper(tf_model=get_online_arima(input_shape=(input_shape[0]-1, *input_shape[1:]), d=input_shape[0]//5),
                     publisher=publisher, subscribers=[], model_id='online_arima',
                     model_type='forecasting', debug=debug),
        ModelWrapper(tf_model=get_nbeats(input_shape=(input_shape[0]-1, *input_shape[1:])),
                     publisher=publisher, subscribers=[], model_id='nbeats',
                     model_type='forecasting', debug=debug),
    ]
    if input_shape[1] > 1:
        new_models.append(
            OnlineVAR(lag_order=10, publisher=publisher, subscribers=[], model_id='online_var',
                      model_type='forecasting', debug=debug))
    models_list.extend(new_models)
    publisher.add_subscribers(new_models)
    

def instantiate_anomaly_scores(ts_window_publisher: TsWindowPublisher, nonconformity_score_wrapper: NonConformityWrapper,
                               first_nonconformity_scores: np.ndarray, out_base_path: str, model_id: str,
                               date_id: str, training_set_length: int, anomaly_score_length: int, debug=False):
    return [
        AverageOfWindow(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['sw_mu_sig', 'ures_mu_sig', 'sw_ks', 'ures_ks'], anomaly_score_ids=['avg_of_window']),
            initial_nonconformity_scores=first_nonconformity_scores,
            window_length=anomaly_score_length,
            debug=debug),
        AnomalyLikelihood(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['sw_mu_sig', 'ures_mu_sig', 'sw_ks', 'ures_ks', 'ares_al_mu_sig', 'ares_al_ks'], anomaly_score_ids=['anomaly_likelihood']),
            initial_nonconformity_scores=first_nonconformity_scores,
            short_term_length=anomaly_score_length//5,
            long_term_length=anomaly_score_length,
            debug=debug),
        ConfidenceLevels(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            training_set_length=training_set_length,
            training_set_publisher_id='sw_mu_sig',
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['sw_mu_sig'], anomaly_score_ids=['confidence_levels']),
            initial_nonconformity_scores=first_nonconformity_scores,
            confidence_window_length=anomaly_score_length*5,
            debug=debug),
        ConfidenceLevels(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            training_set_length=training_set_length,
            training_set_publisher_id='sw_ks',
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['sw_ks'], anomaly_score_ids=['confidence_levels']),
            initial_nonconformity_scores=first_nonconformity_scores,
            confidence_window_length=anomaly_score_length*5,
            debug=debug),
        ConfidenceLevels(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            training_set_length=training_set_length,
            training_set_publisher_id='ures_mu_sig',
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['ures_mu_sig'], anomaly_score_ids=['confidence_levels']),
            initial_nonconformity_scores=first_nonconformity_scores,
            confidence_window_length=anomaly_score_length*5,
            debug=debug),
        ConfidenceLevels(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            training_set_length=training_set_length,
            training_set_publisher_id='ures_ks',
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['ures_ks'], anomaly_score_ids=['confidence_levels']),
            initial_nonconformity_scores=first_nonconformity_scores,
            confidence_window_length=anomaly_score_length*5,
            debug=debug),
        ConfidenceLevels(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            training_set_length=training_set_length,
            training_set_publisher_id='ares_cl_mu_sig',
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['ares_cl_mu_sig'], anomaly_score_ids=['confidence_levels']),
            initial_nonconformity_scores=first_nonconformity_scores,
            confidence_window_length=anomaly_score_length*5,
            debug=debug),
        ConfidenceLevels(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            training_set_length=training_set_length,
            training_set_publisher_id='ares_cl_ks',
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['ares_cl_ks'], anomaly_score_ids=['confidence_levels']),
            initial_nonconformity_scores=first_nonconformity_scores,
            confidence_window_length=anomaly_score_length*5,
            debug=debug),
    ]

def save_paths(out_base_path: str, date_id: str, model_id: str, reservoir_ids: "list[str]", anomaly_score_ids: "list[str]", filename='anomaly_scores.csv'):
    # reservoir_ids = ['sw_mu_sig', 'ures_mu_sig', 'sw_ks', 'ures_ks', 'ares_al_mu_sig', 'ares_cl_mu_sig', 'ares_al_ks', 'ares_cl_ks']
    # anomaly_score_ids = ['avg_of_window', 'anomaly_likelihood', 'confidence_levels']

    return [
        f'{out_base_path}/{model_id}-{r_id}-{anomaly_score_id}/{date_id}/{filename}'
        for r_id in reservoir_ids
        for anomaly_score_id in anomaly_score_ids
    ]
    
def initial_nonconformity_save_paths(out_base_path: str, date_id: str, model_id: str, filename='anomaly_scores.csv'):
    reservoir_ids = ['sw_mu_sig', 'ures_mu_sig', 'sw_ks', 'ures_ks']
    anomaly_score_ids = ['avg_of_window', 'anomaly_likelihood', 'confidence_levels']
    save_paths = [
        f'{out_base_path}/{model_id}-{r_id}-{anomaly_score_id}/{date_id}/{filename}'
        for r_id in reservoir_ids
        for anomaly_score_id in anomaly_score_ids
    ]
    save_paths.extend([
        f'{out_base_path}/{model_id}-ares_al_mu_sig-anomaly_likelihood/{date_id}/{filename}',
        f'{out_base_path}/{model_id}-ares_al_ks-anomaly_likelihood/{date_id}/{filename}',
        f'{out_base_path}/{model_id}-ares_cl_mu_sig-confidence_levels/{date_id}/{filename}',
        f'{out_base_path}/{model_id}-ares_cl_ks-confidence_levels/{date_id}/{filename}',
    ])
    return save_paths


