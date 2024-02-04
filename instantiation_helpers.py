import numpy as np
from itertools import product

from src.tsWindowPublisher import TsWindowPublisher
from src.dataRepresentation import WindowStreamVectors
from src.modelWrapper import ModelWrapper
from src.models.simpleRegressionModel import SimpleRegressionModel, get_simple_regression_model
from src.nonconformity_scores.nonconformity_wrapper import NonConformityWrapper
from src.anomaly_scores.anomalyLikelihood import AnomalyLikelihood
from src.anomaly_scores.confidenceLevels import ConfidenceLevels
from src.anomaly_scores.averageOfWindow import AverageOfWindow


def instantiate_model_wrappers(models_list: list, publisher: WindowStreamVectors, input_shape: tuple[int], initial_weights_base_path: str = None, debug=False):
    if initial_weights_base_path == None:
        simple_regression = get_simple_regression_model(input_shape=input_shape)
    else:
        simple_regression = get_simple_regression_model(input_shape=input_shape)
        simple_regression(np.zeros((32, *input_shape)))
        simple_regression.load_weights(
            f'{initial_weights_base_path}/simple_regression.h5')
    new_models = [
        ModelWrapper(tf_model=simple_regression,
                     publisher=publisher, subscribers=[], model_id='simple_regression',
                     model_type='reconstruction', debug=debug),
        ModelWrapper(tf_model=simple_regression,
                     publisher=publisher, subscribers=[], model_id='simple_regression2',
                     model_type='reconstruction', debug=debug)
    ]
    models_list.extend(
        [model_wrapper for model_wrapper in new_models])
    publisher.add_subscribers(new_models)
    

def instantiate_anomaly_scores(ts_window_publisher: TsWindowPublisher, nonconformity_score_wrapper: NonConformityWrapper,
                               first_nonconformity_scores: np.ndarray, out_base_path: str, model_id: str,
                               date_id: str, training_set_length: int, anomaly_score_length: int, debug=False):
    all_reservoir_ids = ['sw', 'ures', 'ares_al_mu_sig', 'ares_cl_mu_sig', 'ares_al_ks', 'ares_cl_ks']
    return [
        AverageOfWindow(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['sw', 'ures'], anomaly_score_ids=['avg_of_window']),
            initial_nonconformity_scores=first_nonconformity_scores,
            window_length=anomaly_score_length,
            debug=debug),
        AnomalyLikelihood(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['sw', 'ures', 'ares_al_mu_sig', 'ares_al_ks'], anomaly_score_ids=['anomaly_likelihood']),
            initial_nonconformity_scores=first_nonconformity_scores,
            short_term_length=anomaly_score_length//5,
            long_term_length=anomaly_score_length,
            debug=debug),
        ConfidenceLevels(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            training_set_length=training_set_length,
            training_set_publisher_id='sw',
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['sw'], anomaly_score_ids=['confidence_levels']),
            initial_nonconformity_scores=first_nonconformity_scores,
            confidence_window_length=anomaly_score_length*5,
            debug=debug),
        ConfidenceLevels(
            publisher=nonconformity_score_wrapper,
            ts_window_publisher=ts_window_publisher,
            training_set_length=training_set_length,
            training_set_publisher_id='ures',
            subscribers=[],
            save_paths=save_paths(out_base_path, date_id, model_id, reservoir_ids=['ures'], anomaly_score_ids=['confidence_levels']),
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

def save_paths(out_base_path: str, date_id: str, model_id: str, reservoir_ids: list[str], anomaly_score_ids: list[str], filename='anomaly_scores.csv'):
    # reservoir_ids = ['sw', 'ures', 'ares_al_mu_sig', 'ares_cl_mu_sig', 'ares_al_ks', 'ares_cl_ks']
    # anomaly_score_ids = ['avg_of_window', 'anomaly_likelihood', 'confidence_levels']

    return [
        f'{out_base_path}/{model_id}-{r_id}-{anomaly_score_id}/{date_id}/{filename}'
        for r_id in reservoir_ids
        for anomaly_score_id in anomaly_score_ids
    ]
    
def initial_nonconformity_save_paths(out_base_path: str, date_id: str, model_id: str, filename='anomaly_scores.csv'):
    reservoir_ids = ['sw', 'ures']
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


