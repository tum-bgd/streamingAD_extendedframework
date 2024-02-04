import os
import numpy as np
import pandas as pd
import copy
from datetime import datetime

from src.tsWindowPublisher import TsWindowPublisher
from src.dataRepresentation import WindowStreamVectors
from src.training_set_update.slidingWindow import SlidingWindow, sliding_window as sw_helper
from src.training_set_update.uniformReservoir import UniformReservoir
from src.training_set_update.anomalyAwareReservoir import AnomalyAwareReservoir
from src.training_set_analysis.muSigmaChange import MuSigmaChange
from src.training_set_analysis.ksTest import KsTest
from src.nonconformity_scores.nonconformity_wrapper import calc_nonconformity_scores, NonConformityWrapper
from src.anomaly_scores.confidenceLevels import ConfidenceLevels
from src.anomaly_scores.anomalyLikelihood import AnomalyLikelihood
from instantiation_helpers import instantiate_model_wrappers, instantiate_anomaly_scores, initial_nonconformity_save_paths

DEBUG = True

# def main() -> None:
# 0) Results location path structure: out / {collection_id} / {dataset_id} / {model_id}-{training_set_update}-{training_set_analysis}-{anomaly_score} / {datetime}
datetime_this_run = datetime.now().strftime("%Y%m%d_%H%M%S")
collection_id, dataset_id = 'KDD-TSAD', '001_UCR_Anomaly_DISTORTED1sddb40'
dataset_base_path = f'./data/univariate/{collection_id}/{dataset_id}'
out_base_path = f'out/{collection_id}/{dataset_id}'
initial_weights_path = f'{out_base_path}/initial_weights/{datetime_this_run}'

# 1) Create tsWindowPublisher with test data
dataset_test = pd.read_csv(f'{dataset_base_path}.test.csv').values[:, 1:2]
dataset_train = pd.read_csv(f'{dataset_base_path}.train.csv').values[:, 1:2]
ts_memory_length, data_representation_length, number_of_channels, training_set_length, anomaly_score_length = \
    1000, 100, dataset_train.shape[1], 500, 250
input_shape = (data_representation_length, number_of_channels)
ts_window_subscribers = []
ts_window_publisher = TsWindowPublisher(
    dataset=dataset_test, window_length=ts_memory_length, subscribers=ts_window_subscribers, debug=DEBUG)

# 2) Set up data representation
data_representation_subscribers = []
data_representation = WindowStreamVectors(
    publisher=ts_window_publisher,
    window_length=data_representation_length,
    subscribers=data_representation_subscribers, debug=DEBUG)
ts_window_subscribers.append(data_representation)

# 3) Model Instantiation & Training
variables = {
    "ts_window_publisher": {
        "object": ts_window_publisher,
        "data_representation": {
            "object": data_representation,
            "subscribers": {
                "models": [],
                "training_set_update_methods": []
            }
        }
    }
}
variables_data_rep = variables['ts_window_publisher']['data_representation']['subscribers']
models = []
instantiate_model_wrappers(
    models_list=models, publisher=data_representation, input_shape=input_shape, debug=DEBUG)
x = np.stack([dataset_train[i:i+data_representation_length]
              for i in range(len(dataset_train) - data_representation_length + 1)], axis=0)
for model_wrapper in models:
    model_wrapper.train(x=x, epochs=3)
    if not os.path.exists(initial_weights_path):
        os.makedirs(initial_weights_path)
    model_wrapper.tf_model.save_weights(
        f"{initial_weights_path}/{model_wrapper.model_id}.h5")
del x
variables_data_rep['models'].extend([
    {
        "object": model,
        "version": "constant_weights",
        "nonconformity_score": {
            "object": NonConformityWrapper(
                publisher=model,
                subscribers=[],
                save_paths=initial_nonconformity_save_paths(
                    out_base_path=out_base_path,
                    date_id=datetime_this_run,
                    model_id=model.model_id,
                    filename='nonconformity_scores.csv'),
                debug=DEBUG),
            "subscribers": []
        }
    }
    for model in models])
for entry in variables_data_rep['models']:
    entry['object'].add_subscriber(entry['nonconformity_score']['object'])

# 4) Set up training set update methods and add them as subscribers to data representation
first_training_set = \
    sw_helper(outer_window=ts_window_publisher.get_window(),
              feature_vector_length=data_representation_length,
              training_set_length=training_set_length)
variables_data_rep['training_set_update_methods'].extend([
    {
        "id": "sw",
        "model_id": "all",
        "object": SlidingWindow(publisher=data_representation, window_length=training_set_length,
                                first_reservoir=first_training_set, subscribers=[], id="sw", model_id='all', debug=DEBUG),
        "subscribers": []
    },
    {
        "id": "ures",
        "model_id": "all",
        "object": UniformReservoir(publisher=data_representation, reservoir_length=training_set_length,
                                   first_reservoir=first_training_set, subscribers=[], id="ures", model_id='all', debug=DEBUG),
        "subscribers": []
    }
])
variables_data_rep['training_set_update_methods'].extend([
    {
        "id": f"ares_{anomaly_score_id}_{training_set_analysis_id}",
        "model_id": model['object'].model_id,
        "object": AnomalyAwareReservoir(
            publisher=data_representation,
            reservoir_length=training_set_length,
            first_reservoir=first_training_set,
            subscribers=[], id=f"ares_{anomaly_score_id}_{training_set_analysis_id}",
            model_id=model['object'].model_id,
            debug=DEBUG),
        "subscribers": []
    }
    for training_set_analysis_id in ['mu_sig', 'ks']
    for anomaly_score_id in ['al', 'cl']
    for model in variables_data_rep['models']
])
data_representation.add_subscribers([
    training_set_update_method['object']
    for training_set_update_method in variables_data_rep['training_set_update_methods']
    if training_set_update_method['id'] in ["sw", "ures"]
])

# 5) Set up µ/sig - change, KS tests for SW, URES and add models as subscribers to trigger retraining
# def change_tree_after_retraining_hook(variables, out_base_path, date_id, model_id, reservoir_id, training_set_analysis_id, anomaly_score_id):
for entry in variables_data_rep['training_set_update_methods']:
    new_subscribers = []
    if entry['id'] in ['sw', 'ures', 'ares_al_mu_sig', 'ares_cl_mu_sig']:
        new_subscribers.append(
            MuSigmaChange(publisher=entry['object'],
                        models=variables_data_rep['models'],
                        out_base_path=out_base_path,
                        date_id=datetime_this_run,
                        check_every=100, debug=DEBUG),
    )
    if entry['id'] in ['sw', 'ures', 'ares_al_ks', 'ares_cl_ks']:
        new_subscribers.append(
            KsTest(publisher=entry['object'],
                models=variables_data_rep['models'], alpha=0.05,
                out_base_path=out_base_path,
                date_id=datetime_this_run,
                check_every=1000, debug=DEBUG)
        )
    entry['subscribers'].extend(new_subscribers)
    entry['object'].add_subscribers(new_subscribers)

# 6) Initial nonconformity scores, all models share the same weights
first_nonconformity_scores = [
    calc_nonconformity_scores(
        first_training_set, model['object'].tf_model(first_training_set).numpy())
    for model in variables_data_rep['models']
]

# 7) Set up anomaly score methods and add them as subscribers to the nonconformity score wrappers
for i, model in enumerate(variables_data_rep['models']):
    new_anomaly_scores = instantiate_anomaly_scores(
        ts_window_publisher=ts_window_publisher,
        nonconformity_score_wrapper=model['nonconformity_score']['object'],
        first_nonconformity_scores=first_nonconformity_scores[i],
        out_base_path=out_base_path,
        model_id=model['object'].model_id,
        date_id=datetime_this_run,
        training_set_length=training_set_length,
        anomaly_score_length=anomaly_score_length,
        debug=DEBUG)
    model['nonconformity_score']['subscribers'].extend(new_anomaly_scores)
    model['nonconformity_score']['object'].add_subscribers(new_anomaly_scores)

# 8) Connect anomaly scores with reservoirs
first_anomaly_scores = np.random.uniform(0, 1, size=anomaly_score_length)
for i, model in enumerate(variables_data_rep['models']):
    for anomaly_score in model['nonconformity_score']['subscribers']:
        if isinstance(anomaly_score, ConfidenceLevels):
            for training_set_update_method in variables_data_rep['training_set_update_methods']:
                if (training_set_update_method['model_id'] == model['object'].model_id or
                    training_set_update_method['model_id'] == 'all') and \
                    training_set_update_method['id'] == anomaly_score.training_set_publisher_id:
                    anomaly_score.set_training_set_publisher(
                        training_set_update_method['object'])
                    if isinstance(training_set_update_method['object'], AnomalyAwareReservoir):
                        anomaly_score.add_subscriber(
                            training_set_update_method['object'])
                        anomaly_score.update_parameters_with_notify = False
                        training_set_update_method['object'].set_anomaly_score_publisher(
                            anomaly_score)
                        training_set_update_method['object'].set_first_anomaly_scores(
                            first_anomaly_scores)
                        training_set_update_method['object'].update_anomaly_score_params_with_reservoir = True
        elif isinstance(anomaly_score, AnomalyLikelihood):
            for training_set_update_method in variables_data_rep['training_set_update_methods']:
                if isinstance(training_set_update_method['object'], AnomalyAwareReservoir) and \
                    training_set_update_method['id'].startswith('ares_al') and \
                        training_set_update_method['model_id'] == model['object'].model_id:
                        anomaly_score.add_subscriber(
                            training_set_update_method['object'])
                        training_set_update_method['object'].set_anomaly_score_publisher(
                            anomaly_score)
                        training_set_update_method['object'].set_first_anomaly_scores(
                            first_anomaly_scores)

# 9) Iteration over test set - update once all anomaly_scores returned a notification
for i in range(len(ts_window_publisher.dataset) - ts_memory_length + 1):
    if i % 1000 == 0 or DEBUG:
        print(f'At index {i}')
    ts_window_publisher.update_window()


# if __name__ == '__main__':
#     main()
