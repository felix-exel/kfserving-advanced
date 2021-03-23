from typing import Dict
import logging
import requests
import kfserving
import os
import json
import numpy as np
import threading
from influxdb import InfluxDBClient
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

PREDICTOR_URL_FORMAT = "http://{0}/v1/models/{1}:predict"
PREDICTOR_V2_URL_FORMAT = "http://{0}/v2/models/{1}/infer"
EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)


class RecommenderTransformer(kfserving.KFModel):
    """ A class object for the Transformer for the Recommender Model.

    Args:
        kfserving (class object): The KFModel class from the KFServing
        modeule is passed here.
    """

    def __init__(self, name: str, predictor_host: str, influx_host: str, influx_port: int, database: str):
        """Initialize the model name, predictor host and the explainer host

        Args:
            name (str): Name of the model.
            predictor_host (str): The host in which the predictor runs.
        """
        super().__init__(name)
        self.predictor_host = predictor_host
        self.explainer_host = predictor_host
        logging.info("MODEL NAME %s", name)
        logging.info("PREDICTOR URL %s", self.predictor_host)
        logging.info("EXPLAINER URL %s", self.explainer_host)
        self.timeout = 999999999999
        logging.info("TIMEOUT URL %s", self.timeout)
        self.item_ground_truth = None
        # influxDB Client
        self.database = database
        self.client = InfluxDBClient(host=influx_host, port=influx_port)
        self.create_influx_database()
        self.client.switch_database(database)
        # Grafana Datasource and Dashboard Import
        self.create_grafana_datasource_dashboard(influx_host, influx_port)

    def preprocess(self, inputs: Dict) -> Dict:
        """Pre-process activity of the Session Input data.

        Args:
            inputs (Dict): KFServing http request

        Returns:
            Dict: Returns the request input after converting it into a tensor
        """
        self.item_ground_truth = inputs['id']
        return inputs

    def postprocess(self, request: Dict) -> Dict:
        """Post process function of TFServing on the KFServing side is
        written here.

        Args:
            request (Dict): KFServing http request

        Returns:
            :param request: Dict: Returns the request input after converting it into a tensor
        """
        y_pred = np.array(request['predictions'])  # (batch_size, n_classes)
        item_pred = y_pred.argsort()[:, ::-1][:, 0]  # (batches)

        thread = threading.Thread(target=self.calculate_push_metrics, args=[item_pred])
        thread.start()
        request['id'] = self.item_ground_truth
        return request

    def calculate_push_metrics(self, item_pred: np.ndarray):
        for batch in range(item_pred.shape[0]):
            self.push_prediction_2_influx(int(item_pred[batch]), int(self.item_ground_truth[batch]))
            # get all predictions and ground truths
            query = "select * from model_predictions"
            res = self.client.query(query)
            list_res = list(res)[0]
            pred_list = [list_res[i]['prediction'] for i in range(len(list_res))]
            y_true_list = [list_res[i]['ground_truth'] for i in range(len(list_res))]
            logging.info("Predictions: %s", pred_list)
            logging.info("Ground Truth: %s", y_true_list)
            self.push_metrics_2_influx(float(accuracy_score(y_true_list, pred_list, normalize=True)),
                                       float(precision_score(y_true_list, pred_list, average='macro', zero_division=1)),
                                       float(recall_score(y_true_list, pred_list, average='macro', zero_division=1)),
                                       float(f1_score(y_true_list, pred_list, average='macro', zero_division=1)))

    def push_prediction_2_influx(self, prediction: int, ground_truth: int):
        json_body = [{
            "measurement": "model_predictions",
            "tags": {
            },
            "fields": {
                "prediction": prediction,
                "ground_truth": ground_truth
            },
            "time": datetime.now()
        }]
        if not self.client.write_points(json_body):
            logging.error("COULD NOT WRITE DATA INTO INFLUXDB!")

    def push_metrics_2_influx(self, accuracy: float, precision: float, recall: float, f1: float):
        json_body = [{
            "measurement": "model_metrics",
            "tags": {
            },
            "fields": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "time": datetime.now()
        }]
        if not self.client.write_points(json_body):
            logging.error("COULD NOT WRITE DATA INTO INFLUXDB!")

    def create_index_es(self, index_name='model_metrics'):
        if not self.es.ping():
            logging.error("CANNOT PING ELASTICSEARCH CLIENT")
        if self.es.indices.exists(index_name):
            self.es_indeces[index_name] = index_name
        else:
            response = self.es.indices.create(index=index_name)
            if response:
                self.es_indeces[index_name] = index_name
                logging.info("Created Index %s", index_name)
            else:
                logging.error("Could not create Index %s", index_name)

    def create_influx_database(self):
        databases = self.client.get_list_database()
        for db in databases:
            if db['name'] == self.database:
                return
        self.client.create_database(self.database)
        logging.info(f"Created InfluxDB {self.database} Database!")

    def create_grafana_datasource_dashboard(self, influx_host, influx_port):
        grafana_user = 'admin'
        grafana_password = 'admin'
        grafana_host = 'grafana.knative-monitoring.svc.cluster.local'
        grafana_port = 30802
        grafana_url = os.path.join('http://', '%s:%u' % (grafana_host, grafana_port))
        session = requests.Session()
        login_post = session.post(
            os.path.join(grafana_url, 'login'),
            data=json.dumps({
                'user': grafana_user,
                'email': '',
                'password': grafana_password}),
            headers={'content-type': 'application/json'})

        # Get list of datasources
        datasources_get = session.get(os.path.join(grafana_url, 'api', 'datasources'))
        datasources = datasources_get.json()
        for datasource in datasources:
            if datasource['name'] == 'InfluxDB':
                return

        # Add new datasource
        datasources_post = session.post(
            os.path.join(grafana_url, 'api', 'datasources'),
            data=json.dumps({
                'access': 'proxy',
                'database': self.database,
                'name': 'InfluxDB',
                'type': 'influxdb',
                'url': 'http://%s:%u' % (influx_host, influx_port)
            }),
            headers={'content-type': 'application/json'})
        logging.info(f"Created Grafana Datasource")

        # Import Dashboard
        # appended id = null to the json
        with open('Model Performance-1612430487867.json') as f:
            dashboard = json.load(f)

        dashboard_post = session.post(
            os.path.join(grafana_url, 'api', 'dashboards', 'db'),
            data=json.dumps({
                'dashboard': dashboard,
                'folderId': 0,
                'overwrite': True
            }),
            headers={'content-type': 'application/json'})
        logging.info(f"Created Grafana Dashboard")

    def push_metrics_2_es(self, accuracy: float, precision: float, recall: float, f1: float):
        record = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'timestamp': datetime.now(),
        }
        self.es.index(index=self.es_indeces['model_metrics'], body=record, doc_type='model_metrics')

    def push_prediction_2_es(self, prediction: int, ground_truth: int):
        record = {
            'prediction': prediction,
            'ground_truth': ground_truth,
            'timestamp': datetime.now(),
        }
        self.es.index(index=self.es_indeces['model_prediction'], body=record, doc_type='model_prediction')
