import os
import numpy as np
import pandas as pd
import copy
from time import time
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
from instantiation_helpers import instantiate_model_wrappers, instantiate_anomaly_scores, initial_nonconformity_save_paths, save_paths
from src.models.onlineIsolationForest import PCBIForest


DEBUG = False
BATCH = 50
CHECK_TRAINING_SET_EVERY = 50       # if batch is set, this refers to number of batches before check is performed
TRAINING_SET_LENGTH = 5000
ANOMALY_SCORE_LENGTH = 50


TS_MEMORY_LENGTH = 6000           # has to be at least TRAINING_SET_LENGTH + DATA_REPRESENTATION_LENGTH
DATA_REPRESENTATION_LENGTH = 100

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
number_of_channels = dataset_train.shape[1]
input_shape = (DATA_REPRESENTATION_LENGTH, number_of_channels)
ts_window_subscribers = []
ts_window_publisher = TsWindowPublisher(
    dataset=dataset_test, window_length=TS_MEMORY_LENGTH, subscribers=ts_window_subscribers, debug=DEBUG)

# 2) Set up data representation
data_representation_subscribers = []
data_representation = WindowStreamVectors(
    publisher=ts_window_publisher,
    window_length=DATA_REPRESENTATION_LENGTH,
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
training_dataset = np.stack([dataset_train[i:i+DATA_REPRESENTATION_LENGTH]
              for i in range(len(dataset_train) - DATA_REPRESENTATION_LENGTH + 1)], axis=0)
for model_wrapper in models:
    model_wrapper.train(x=training_dataset, epochs=5)
    if not os.path.exists(initial_weights_path):
        os.makedirs(initial_weights_path)
    model_wrapper.save_model(
        save_path=f"{initial_weights_path}/{model_wrapper.model_id}.h5")

for model in models:
    if model.model_type in ['forecasting', 'reconstruction']:
        new_entry = {
            "object": model,
            "version": "constant_weights",
            "nonconformity_score": {
                "object": NonConformityWrapper(
                    publisher=model,
                    subscribers=[],
                    measure='cosine_sim',
                    save_paths=initial_nonconformity_save_paths(
                        out_base_path=out_base_path,
                        date_id=datetime_this_run,
                        model_id=model.model_id,
                        filename='nonconformity_scores.csv'),
                    debug=DEBUG),
                "subscribers": []
            }
        }
        new_entry['object'].add_subscriber(
            new_entry['nonconformity_score']['object'])
        variables_data_rep['models'].append(new_entry)

# 4) Set up training set update methods and add them as subscribers to data representation
first_training_set = \
    sw_helper(outer_window=ts_window_publisher.get_window(),
              feature_vector_length=DATA_REPRESENTATION_LENGTH,
              training_set_length=TRAINING_SET_LENGTH)
variables_data_rep['training_set_update_methods'].extend([
    {
        "id": "sw_mu_sig",
        "model_id": "all",
        "object": SlidingWindow(publisher=data_representation, reservoir_length=TRAINING_SET_LENGTH,
                                first_reservoir=first_training_set, subscribers=[], id="sw_mu_sig", 
                                model_id='all', debug=DEBUG),
        "subscribers": []
    },
    {
        "id": "sw_ks",
        "model_id": "all",
        "object": SlidingWindow(publisher=data_representation, reservoir_length=TRAINING_SET_LENGTH,
                                first_reservoir=first_training_set, subscribers=[], id="sw_ks", 
                                model_id='all', debug=DEBUG),
        "subscribers": []
    },
    {
        "id": "ures_mu_sig",
        "model_id": "all",
        "object": UniformReservoir(publisher=data_representation, reservoir_length=TRAINING_SET_LENGTH,
                                   first_reservoir=first_training_set, subscribers=[], id="ures_mu_sig", 
                                   model_id='all', debug=DEBUG),
        "subscribers": []
    },
    {
        "id": "ures_ks",
        "model_id": "all",
        "object": UniformReservoir(publisher=data_representation, reservoir_length=TRAINING_SET_LENGTH,
                                   first_reservoir=first_training_set, subscribers=[], id="ures_ks", 
                                   model_id='all', debug=DEBUG),
        "subscribers": []
    }
])
variables_data_rep['training_set_update_methods'].extend([
    {
        "id": f"ares_{anomaly_score_id}_{training_set_analysis_id}",
        "model_id": model['object'].model_id,
        "object": AnomalyAwareReservoir(
            publisher=data_representation,
            reservoir_length=TRAINING_SET_LENGTH,
            first_reservoir=first_training_set,
            subscribers=[
            ], id=f"ares_{anomaly_score_id}_{training_set_analysis_id}",
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
    if training_set_update_method['id'] in ["sw_mu_sig", "sw_ks", "ures_mu_sig", "ures_ks"]
])

# 5) Set up µ/sig - change, KS tests for SW, URES and add models as subscribers to trigger retraining
# def change_tree_after_retraining_hook(variables, out_base_path, date_id, model_id, reservoir_id, training_set_analysis_id, anomaly_score_id):
for entry in variables_data_rep['training_set_update_methods']:
    new_subscribers = []
    if entry['id'] in ['sw_mu_sig', 'ures_mu_sig', 'ares_al_mu_sig', 'ares_cl_mu_sig']:
        new_subscribers.append(
            MuSigmaChange(publisher=entry['object'],
                          models=variables_data_rep['models'],
                          out_base_path=out_base_path,
                          date_id=datetime_this_run,
                          check_every=CHECK_TRAINING_SET_EVERY, debug=DEBUG),
        )
    if entry['id'] in ['sw_ks', 'ures_ks', 'ares_al_ks', 'ares_cl_ks']:
        new_subscribers.append(
            KsTest(publisher=entry['object'],
                   models=variables_data_rep['models'], alpha=0.05,
                   out_base_path=out_base_path,
                   date_id=datetime_this_run,
                   check_every=CHECK_TRAINING_SET_EVERY, debug=DEBUG)
        )
    entry['subscribers'].extend(new_subscribers)
    entry['object'].add_subscribers(new_subscribers)

# 6) Initial nonconformity scores, all models share the same weights
first_nonconformity_scores = []
for model in variables_data_rep['models']:
    if model['object'].model_type == 'reconstruction':
        first_nonconformity_scores.append(calc_nonconformity_scores(
            first_training_set, model['object'].predict(first_training_set), measure='cosine_sim'))
    else:
        nc_scores = calc_nonconformity_scores(
            first_training_set[:, -1:], model['object'].predict(first_training_set), measure='mean_abs_diff')
        nc_scores /= np.max(nc_scores)
        first_nonconformity_scores.append(nc_scores)


# 7) Set up anomaly score methods and add them as subscribers to the nonconformity score wrappers
for i, model in enumerate(variables_data_rep['models']):
    new_anomaly_scores = instantiate_anomaly_scores(
        ts_window_publisher=ts_window_publisher,
        nonconformity_score_wrapper=model['nonconformity_score']['object'],
        first_nonconformity_scores=first_nonconformity_scores[i],
        out_base_path=out_base_path,
        model_id=model['object'].model_id,
        date_id=datetime_this_run,
        training_set_length=TRAINING_SET_LENGTH,
        anomaly_score_length=ANOMALY_SCORE_LENGTH,
        debug=DEBUG)
    model['nonconformity_score']['subscribers'].extend(new_anomaly_scores)
    model['nonconformity_score']['object'].add_subscribers(new_anomaly_scores)

# 8) Connect anomaly scores with reservoirs
first_anomaly_scores = np.random.uniform(0, 1, size=TRAINING_SET_LENGTH)
for i, model in enumerate([x for x in variables_data_rep['models'] if x['object'].model_type in ['forecasting', 'reconstruction']]):
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

# 9) IForest model + training set update methods + anomaly likelihood + KS
model_id, model_type = 'pcb_iforest', 'iforest'
for version in []: #['sw_ks', 'ares_al_ks']:
    model_wrapper = PCBIForest(publisher=data_representation, subscribers=[], 
                              model_id=model_id, model_type=model_type, debug=DEBUG)
    print(f'Training of pcb_iforest model for version {version}')
    model_wrapper.train(x=training_dataset, epochs=1)
    model_wrapper.save_model(
        save_path=f"{initial_weights_path}/{model_wrapper.model_id}.h5")
    data_representation.add_subscriber(model_wrapper)
    new_entry = {
        "object": model_wrapper,
        "version": version,
    }
    new_entry["nonconformity_score"] = {
        "object": NonConformityWrapper(
            publisher=new_entry['object'],
            subscribers=[],
            save_paths=save_paths(out_base_path, datetime_this_run, model_id, reservoir_ids=[
                version], anomaly_score_ids=['anomaly_likelihood'], filename='nonconformity_scores.csv'),
            measure='iforest',
            debug=DEBUG),
        "subscribers": []
    }
    new_entry['object'].add_subscriber(
        new_entry["nonconformity_score"]['object'])
    kwargs = dict(
        publisher=data_representation,
        reservoir_length=TRAINING_SET_LENGTH,
        first_reservoir=first_training_set,
        id=version, model_id=model_id, debug=DEBUG)
    training_set_update_method = \
        SlidingWindow(subscribers=[], **kwargs) if version in ['sw_mu_sig', 'sw_ks'] \
        else UniformReservoir(subscribers=[], **kwargs) if version in ['ures_mu_sig', 'ures_ks'] \
        else AnomalyAwareReservoir(subscribers=[], **kwargs)
    variables_data_rep['training_set_update_methods'].append(
        {
            "id": version,
            "model_id": model_id,
            "object": training_set_update_method,
            "subscribers": []
        })
    if 'ares' not in version:
        data_representation.add_subscriber(training_set_update_method)
    ks_analysis = KsTest(
        publisher=training_set_update_method,
        models=variables_data_rep['models'], alpha=0.05,
        out_base_path=out_base_path,
        date_id=datetime_this_run,
        check_every=CHECK_TRAINING_SET_EVERY, debug=DEBUG)
    training_set_update_method.add_subscriber(ks_analysis)
    variables_data_rep['training_set_update_methods'][-1]['subscribers'].append(
        ks_analysis)
    anomaly_likelihood = AnomalyLikelihood(
        publisher=new_entry["nonconformity_score"]['object'],
        ts_window_publisher=ts_window_publisher,
        subscribers=[],
        save_paths=save_paths(out_base_path, datetime_this_run, model_id, reservoir_ids=[
            version], anomaly_score_ids=['anomaly_likelihood']),
        initial_nonconformity_scores=new_entry['object'].predict(
            first_training_set),
        short_term_length=ANOMALY_SCORE_LENGTH//5,
        long_term_length=ANOMALY_SCORE_LENGTH,
        update_at_notify=[],
        debug=DEBUG
    )
    new_entry["nonconformity_score"]['object'].add_subscriber(
        anomaly_likelihood)
    new_entry["nonconformity_score"]['subscribers'].append(
        anomaly_likelihood)
    anomaly_likelihood.update_at_notify.append((new_entry['object'].update_performance_counters, (
        anomaly_likelihood.calculate_anomaly_scores, anomaly_likelihood.threshold)))
    if isinstance(training_set_update_method, AnomalyAwareReservoir):
        anomaly_likelihood.add_subscriber(training_set_update_method)
        training_set_update_method.set_anomaly_score_publisher(
            anomaly_score_publisher=anomaly_likelihood)
        training_set_update_method.set_first_anomaly_scores(
                        first_anomaly_scores)

    variables_data_rep['models'].append(new_entry)

# 10) Iteration over test set - update once all anomaly_scores returned a notification
dataset_len = len(ts_window_publisher.dataset)
t = time()
for i in range(TS_MEMORY_LENGTH - 1, dataset_len + 1, BATCH):
    msg = f'At index {i} / {dataset_len}'
    time_remaining = int((time() - t) * ((dataset_len - i) / BATCH))
    hours, rest = time_remaining // 3600, time_remaining % 3600
    minutes, seconds = rest // 60, rest % 60
    msg += f' (Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d})'
    t = time()
    print(msg, end='\n')
    ts_window_publisher.update_window(step_size=BATCH)


# if __name__ == '__main__':
#     main()
