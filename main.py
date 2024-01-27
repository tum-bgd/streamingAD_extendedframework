import numpy as np
import pandas as pd
from datetime import datetime

from src.tsWindowPublisher import TsWindowPublisher
from src.dataRepresentation import WindowStreamVectors
from src.modelWrapper import ModelWrapper
from src.training_set_update.slidingWindow import SlidingWindow, sliding_window as sw_helper
from src.training_set_update.uniformReservoir import UniformReservoir
from src.training_set_update.anomalyAwareReservoir import AnomalyAwareReservoir
from src.training_set_analysis.muSigmaChange import MuSigmaChange
from src.training_set_analysis.ksTest import KsTest
from src.nonconformity_scores.euclidean_distance import EuclideanDistanceNonConformity, calc_nonconformity_score
from src.anomaly_scores.averageOfWindow import AverageOfWindow
from src.anomaly_scores.anomalyLikelihood import AnomalyLikelihood
from src.anomaly_scores.confidenceLevels import ConfidenceLevels
from models.simpleRegressionModel import SimpleRegressionModel


def main() -> None:
    # 0) Results location path structure: out / {collection_id} / {dataset_id} / {model_id}-{training_set_update}-{training_set_analysis}-{anomaly_score} / {datetime}
    datetime_this_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_id, dataset_id = 'KDD-TSAD', '001_UCR_Anomaly_DISTORTED1sddb40'
    dataset_base_path = f'./data/univariate/{collection_id}/{dataset_id}'
    out_base_path = f'out/{collection_id}/{dataset_id}'
    initial_weights_path = f'{out_base_path}/initial_weights'

    # 1) Create tsWindowPublisher with test data
    dataset_test = pd.read_csv(f'{dataset_base_path}.test.csv').values[:, 1]
    dataset_train = pd.read_csv(f'{dataset_base_path}.train.csv').values[:, 1]
    ts_memory_length, data_representation_length, training_set_length, anomaly_score_length = 1000, 100, 500, 100
    ts_window_subscribers = []
    ts_window_publisher = TsWindowPublisher(
        dataset=dataset_test, window_length=ts_memory_length, subscribers=ts_window_subscribers)

    # 2) Set up data representation
    data_representation_subscribers = []
    data_representation = WindowStreamVectors(publisher=ts_window_publisher,
                                              window_length=data_representation_length, subscribers=data_representation_subscribers)
    ts_window_subscribers.append(data_representation)

    # 3) Model Instantiation & Training
    vars = {}
    temp = {'training_set_update': None, 'training_set_analysis': None,
            'models': [], 'nonconformity_scores': [], 'anomaly_scores': []}
    vars['sw_mu_sig'], vars['sw_ks'], vars['ures_mu_sig'], vars['ures_ks'] = temp.copy(
    ), temp.copy(), temp.copy(), temp.copy()
    vars['ares_al_mu_sig'], vars['ares_al_ks'], vars['ares_ks_mu_sig'], vars['ares_ks_ks'] = temp.copy(
    ), temp.copy(), temp.copy(), temp.copy()
    del temp
    instantiate_model_wrappers(
        models_list=vars['sw_mu_sig']['models'], publisher=data_representation)
    x = np.stack([dataset_train[i:i+data_representation_length]
                 for i in range(len(dataset_train) - data_representation_length + 1)], axis=0)
    for model_wrapper in vars['sw_mu_sig']['models']:
        model_wrapper.train(x=x, epochs=3)
        model_wrapper.tf_model.save_weights(
            f"{initial_weights_path}/{model_wrapper.model_id}.h5")
    del x
    for var_key in ['sw_ks', 'ures_mu_sig', 'ures_ks', 'ares_al_mu_sig', 'ares_al_ks', 'ares_ks_mu_sig', 'ares_ks_ks']:
        instantiate_model_wrappers(models_list=vars[var_key]['models'],
                                   publisher=data_representation, initial_weights_base_path=initial_weights_path)

    # 4) Set up training set update methods
    first_training_set = sw_helper(outer_window=ts_window_publisher.get_window(
    ), feature_vector_length=data_representation_length)
    sliding_window = SlidingWindow(publisher=data_representation, reservoir_length=training_set_length,
                                   first_reservoir=first_training_set, subscribers=[])
    vars['sw_mu_sig']['training_set_update'], vars['sw_ks']['training_set_update'] = sliding_window, sliding_window
    ures = UniformReservoir(publisher=data_representation, reservoir_length=training_set_length,
                            first_reservoir=first_training_set, subscribers=[])
    vars['ures_mu_sig']['training_set_update'], vars['ures_ks']['training_set_update'] = ures, ures

    # 5) Set up µ/sig - change, KS tests for SW, URES and add models as subscribers to trigger retraining
    mu_sigma_change_sw = MuSigmaChange(
        publisher=sliding_window, models=vars['sw_mu_sig']['models'])
    ks_test_sw = KsTest(publisher=sliding_window,
                        models=vars['sw_ks']['models'], alpha=0.05)
    vars['sw_mu_sig']['training_set_analysis'], vars['sw_ks']['training_set_analysis'] = mu_sigma_change_sw, ks_test_sw

    mu_sigma_change_ures = MuSigmaChange(
        publisher=ures, models=vars['ures_mu_sig']['models'])
    ks_test_ures = KsTest(
        publisher=ures, models=vars['ures_ks']['models'], alpha=0.05)
    vars['ures_mu_sig']['training_set_analysis'], vars['ures_ks']['training_set_analysis'] = mu_sigma_change_ures, ks_test_ures

    # 6) Set up nonconformity score methods
    for var_key in ['sw_ks', 'ures_mu_sig', 'ures_ks', 'ares_al_mu_sig', 'ares_al_ks', 'ares_ks_mu_sig', 'ares_ks_ks']:
        instantiate_nonconformity_scores(
            nonconformity_scores_list=vars[var_key]['nonconformity_scores'], models_list=vars[var_key]['models'])

    # 7) Initial nonconformity scores, all models share the same weights
    first_nonconformity_scores = [
        calc_nonconformity_score(
            first_training_set, model_wrapper.tfModel.predict(first_training_set).numpy())
        for model_wrapper in vars['sw_mu_sig']['models']
    ]

    # 8) Set up anomaly score methods
    instantiate_anomaly_scores(variables=vars, first_nonconformity_scores=first_nonconformity_scores,
                               out_base_path=out_base_path, date_id=datetime_this_run, anomaly_score_length=anomaly_score_length)

    # 9) Initial anomaly scores
    first_anomaly_scores_al = [np.ones_like(
        first_nonconformity_scores[0]) for i in range(len(first_nonconformity_scores))]
    first_anomaly_scores_ks = [np.random.uniform(size=anomaly_score_length)]

    # 10) Set up anomaly-aware reservoirs
    for var_key in ['ares_al_mu_sig', 'ares_al_ks']:
        vars[var_key]['training_set_update'] = \
            AnomalyAwareReservoir(publisher=data_representation, 
                                  anomalyScorePublisher=vars[var_key]['anomaly_scores'][1], 
                                  reservoir_length=training_set_length,
                                  first_reservoir=first_training_set, 
                                  first_anomaly_scores=first_anomaly_scores_al, 
                                  subscribers=[])
    for var_key in ['ares_ks_mu_sig', 'ares_ks_ks']:
        vars[var_key]['training_set_update'] = \
            AnomalyAwareReservoir(publisher=data_representation, 
                                  anomalyScorePublisher=vars[var_key]['anomaly_scores'][2], 
                                  reservoir_length=training_set_length,
                                  first_reservoir=first_training_set, 
                                  first_anomaly_scores=first_anomaly_scores_ks, 
                                  subscribers=[])

    # 11) Instantiate remaining µ/sig - change, KS tests for training sets \
    #    and call model retraining
    for var_key in ['ares_al_mu_sig', 'ares_ks_mu_sig']:
        vars[var_key]['training_set_analysis'] = MuSigmaChange(
            publisher=vars[var_key]['training_set_update'], models=vars[var_key]['models'])
    for var_key in ['ares_al_ks', 'ares_ks_ks']:
        vars[var_key]['training_set_analysis'] = KsTest(
            publisher=vars[var_key]['training_set_update'], models=vars[var_key]['models'], alpha=0.05)
        
    # 12) Add training_set_analysis methods as subscribers of training_set_update methods
    for var_key in ['sw_ks', 'ures_mu_sig', 'ures_ks', 'ares_al_mu_sig', 'ares_al_ks', 'ares_ks_mu_sig', 'ares_ks_ks']:
        vars[var_key]['training_set_update'].add_subscriber(vars[var_key]['training_set_analysis'])


# --- Instantiation helper functions ---
def instantiate_model_wrappers(models_list: list, publisher: WindowStreamVectors, initial_weights_base_path: str = None):
    models_list.extend([
        ModelWrapper(tfModel=SimpleRegressionModel() if initial_weights_base_path == None
                     else SimpleRegressionModel().load_weights(f'{initial_weights_base_path}/simple_regression_model.h5'),
                     publisher=publisher, subscribers=[], model_id='simple_regression_model')
    ])

def instantiate_nonconformity_scores(nonconformity_scores_list: list, models_list: list[ModelWrapper]):
    new_nonconformity_scores = [
        EuclideanDistanceNonConformity(publisher=model, subscribers=[])
        for model in models_list]
    for i, model in enumerate(models_list):
        model.add_subscriber(new_nonconformity_scores[i])
    nonconformity_scores_list.extend(new_nonconformity_scores)

def instantiate_anomaly_scores(variables: dict, first_nonconformity_scores: list[np.ndarray], out_base_path: str, date_id: str, anomaly_score_length: int):
    for var_key in ['sw_ks', 'ures_mu_sig', 'ures_ks', 'ares_al_mu_sig', 'ares_al_ks', 'ares_ks_mu_sig', 'ares_ks_ks']:
        def save_path(
            model_id, anomaly_score_id): return f'{out_base_path}/{model_id}-{variables[var_key]["training_set_update"]}-' + \
                f'{variables[var_key]["training_set_analysis"]}-{anomaly_score_id}/{date_id}/anomaly_scores.csv'
        variables[var_key]['anomaly_scores'].extend([[
            AverageOfWindow(
                publisher=variables[var_key]['nonconformity_scores'][i],
                subscribers=[],
                save_path=save_path(model_wrapper.model_id, 'avg_of_window'),
                initial_nonconformity_scores=first_nonconformity_scores,
                window_length=anomaly_score_length),
            AnomalyLikelihood(
                publisher=variables[var_key]['nonconformity_scores'][i],
                subscribers=[],
                save_path=save_path(model_wrapper.model_id,
                                    'anomaly_likelihood'),
                initial_nonconformity_scores=first_nonconformity_scores,
                short_term_length=anomaly_score_length//5,
                long_term_length=anomaly_score_length),
            ConfidenceLevels(
                publisher=variables[var_key]['nonconformity_scores'][i],
                training_set_publisher=variables[var_key]['training_set_update'],
                subscribers=[],
                save_path=save_path(model_wrapper.model_id,
                                    'confidence_levels'),
                initial_nonconformity_scores=first_nonconformity_scores,
                confidence_window_length=anomaly_score_length)
        ]
            for i, model_wrapper in enumerate(variables[var_key]['models'])
        ])
    for i, nonconformity_score in enumerate(variables[var_key]['nonconformity_scores']):
        for anomaly_score in variables[var_key]['anomaly_scores'][i]:
            nonconformity_score.add_subscriber(anomaly_score)


if __name__ == '__main__':
    main()
