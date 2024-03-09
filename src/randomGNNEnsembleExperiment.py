import os
import numpy as np
import pandas as pd
import copy
from time import time
from datetime import datetime

from tsWindowPublisher import TsWindowPublisher
from dataRepresentation import WindowStreamVectors
from training_set_update.slidingWindow import SlidingWindow, sliding_window as sw_helper
from training_set_update.uniformReservoir import UniformReservoir
from training_set_update.anomalyAwareReservoir import AnomalyAwareReservoir
from training_set_analysis.muSigmaChange import MuSigmaChange
from training_set_analysis.ksTest import KsTest
from nonconformity_scores.nonconformity_wrapper import calc_nonconformity_scores, NonConformityWrapper
from anomaly_scores.anomalyLikelihood import AnomalyLikelihood
from utils import save_paths
from models.gnn_ensembles import EnsembleGNN, EnsembleGNNWrapper


# TODO:
#   1) Insert x artificial anomalies at beginning of stream
#       --> also record component nonconformity / anomaly scores
#       --> investigate how large the difference in nonconformity / anomaly scores is for components vs ensemble
#           --> adapt performance counters for them to make sense given this knowledge
#       --> check on actual anomalies


DEBUG = False
# if batch_size is set, this refers to number of batches before check is performed
CHECK_TRAINING_SET_EVERY = 50
ANOMALY_SCORE_LENGTH = 50
# has to be at least TRAINING_SET_LENGTH + DATA_REPRESENTATION_LENGTH
TS_MEMORY_LENGTH = 6000


def main(data_folder_path: str, out_folder_path: str, dataset_category: str, collection_id: str, dataset_id: str, batch_size=50, training_set_length=5000, data_representation_length=100, date_id=None) -> None:
    # 0) Results location path structure: out / {collection_id} / {dataset_id} / {model_id}-{training_set_update}-{training_set_analysis}-{anomaly_score} / {datetime}
    datetime_this_run = datetime.now().strftime("%Y%m%d_%H%M%S") if date_id is None else date_id
    dataset_base_path = f'{data_folder_path}/{dataset_category}/{collection_id}/{dataset_id}'
    out_base_path = f'{out_folder_path}/{collection_id}/{dataset_id}'
    initial_weights_path = f'{out_base_path}/initial_weights/{datetime_this_run}'

    # 1) Create tsWindowPublisher with test data
    dataset_test = pd.read_csv(
        f'{dataset_base_path}.test.csv').values[:, 1:-1].astype(float)
    if os.path.exists(f'{dataset_base_path}.train.csv'):
        dataset_train = pd.read_csv(
            f'{dataset_base_path}.train.csv').values[:, 1:-1].astype(float)
    else:
        dataset_train = dataset_test[:training_set_length]
    number_of_channels = dataset_test.shape[1]
    input_shape = (data_representation_length, number_of_channels)
    ts_window_subscribers = []
    ts_window_publisher = TsWindowPublisher(
        dataset=dataset_test, 
        window_length=TS_MEMORY_LENGTH, 
        subscribers=ts_window_subscribers, 
        base_anomaly_save_path=f'{out_base_path}/artificial_anomalies/{datetime_this_run}',
        debug=DEBUG)

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
    first_training_set = \
        sw_helper(outer_window=ts_window_publisher.get_window(),
                  feature_vector_length=data_representation_length,
                  training_set_length=training_set_length)

    # 4) Ensemble GNN + training set update methods + KS + Anomaly Likelihood
    model_id, model_type = 'ensemble_gnn', 'reconstruction'
    for version in [
        # 'sw_mu_sig',
        'sw_ks',
        # 'ures_mu_sig',
        # 'ures_ks',
        # 'ares_al_mu_sig',
        # 'ares_al_ks'
        ]:
        ensemble_length = 2
        model_wrapper = EnsembleGNNWrapper(
            publisher=data_representation,
            subscribers=[],
            model_id=model_id,
            model_type=model_type,
            window_length=data_representation_length,
            number_of_channels=number_of_channels,
            ensemble_length=ensemble_length,
            batch_size=batch_size,
            save_paths=save_paths(out_base_path, datetime_this_run, model_id, reservoir_ids=[
                    version], anomaly_score_ids=['anomaly_likelihood'], filename='component_anomaly_scores.csv'),
            debug=DEBUG
        )
        print(f'Training of ensemble_gnn model for version {version}')
        model_wrapper.train(x=first_training_set, epochs=1)
        if not os.path.exists(initial_weights_path):
            os.makedirs(initial_weights_path)
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
                measure='cosine_sim',
                debug=DEBUG),
            "subscribers": []
        }
        new_entry['object'].add_subscriber(
            new_entry["nonconformity_score"]['object'])
        kwargs = dict(
            publisher=data_representation,
            reservoir_length=training_set_length,
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
            initial_nonconformity_scores=calc_nonconformity_scores(
                first_training_set, new_entry['object'].predict(first_training_set), measure='cosine_sim'),
            short_term_length=ANOMALY_SCORE_LENGTH//5,
            long_term_length=ANOMALY_SCORE_LENGTH,
            update_at_notify=[],
            threshold=0.6,
            debug=DEBUG
        )
        new_entry["nonconformity_score"]['object'].add_subscriber(
            anomaly_likelihood)
        new_entry["nonconformity_score"]['subscribers'].append(
            anomaly_likelihood)
        # anomaly_likelihood.update_at_notify.append((new_entry['object'].update_performance_counters_with_anomaly_scores, (
        #     anomaly_likelihood.calculate_anomaly_scores, anomaly_likelihood.threshold)))

        first_anomaly_scores = np.random.uniform(
            0, 1, size=training_set_length)

        if isinstance(training_set_update_method, AnomalyAwareReservoir):
            anomaly_likelihood.add_subscriber(training_set_update_method)
            training_set_update_method.set_anomaly_score_publisher(
                anomaly_score_publisher=anomaly_likelihood)
            training_set_update_method.set_first_anomaly_scores(
                first_anomaly_scores)

        variables_data_rep['models'].append(new_entry)

    # 4.5) Insert artificial anomalies
    offsets = [500, 1000, 1500, 2000, 2500, 3000]
    for offset in offsets:
        ts_window_publisher.insert_artificial_anomaly(offset=offset)

    # 5) Iteration over test set - update once all anomaly_scores returned a notification
    dataset_len = len(ts_window_publisher.dataset)
    t = time()
    for i in range(TS_MEMORY_LENGTH - 1, dataset_len + 1, batch_size):
        msg = f'At index {i} / {dataset_len}'
        time_remaining = int((time() - t) * ((dataset_len - i) / batch_size))
        hours, rest = time_remaining // 3600, time_remaining % 3600
        minutes, seconds = rest // 60, rest % 60
        msg += f' (Estimated time remaining: {hours:02d}:{minutes:02d}:{seconds:02d})'
        t = time()
        print(msg, end='\n')
        ts_window_publisher.update_window(step_size=batch_size)


if __name__ == '__main__':
    datetime_this_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_category = 'multivariate'
    collection_id = 'SMD'
    data_folder_path = 'data'
    out_folder_path = f'out'
    for dataset_id in sorted(list(set([x.split('.')[0] for x in 
                                       os.listdir(f'data/{dataset_category}/{collection_id}') if not x.startswith('.')]))):
        print(f'Dataset id: {dataset_id}')
        main(
            data_folder_path,
            out_folder_path,
            dataset_category,
            collection_id,
            dataset_id,
            data_representation_length=100,
            date_id=datetime_this_run)
