import tensorflow as tf
import json
import tornado
from typing import Dict
import requests
import logging
import numpy as np
import kfserving
import threading
from influxdb import InfluxDBClient
from datetime import datetime

PREDICTOR_URL_FORMAT = "http://{0}/v1/models/{1}:predict"
PREDICTOR_V2_URL_FORMAT = "http://{0}/v2/models/{1}/infer"
EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)


class AutoencoderTransformer(kfserving.KFModel):
    """ A class object for the data handling activities of Outlier Detection
    Task and returns a KFServing compatible response.

    Args:
        kfserving (class object): The KFModel class from the KFServing
        module is passed here.
    """

    def __init__(self, name: str, predictor_host: str, influx_host: str, influx_port: int, database: str,
                 threshold: float):
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
        self.mask = None
        self.y_true = None
        self.sparse_cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='sum')
        # influxDB Client
        self.client = InfluxDBClient(host=influx_host, port=influx_port)
        self.client.switch_database(database)
        self.threshold = threshold

    async def predict(self, request: Dict) -> Dict:

        if not self.predictor_host:
            raise NotImplementedError
        predict_url = PREDICTOR_URL_FORMAT.format(self.predictor_host, self.name)
        if self.protocol == "v2":
            predict_url = PREDICTOR_V2_URL_FORMAT.format(self.predictor_host, self.name)

        headers = {}

        response = requests.post(predict_url, data=json.dumps(request), headers=headers)

        return json.loads(response.text)

    def preprocess(self, inputs: Dict) -> Dict:
        """Pre-process activity of the Session Input data.

        Args:
            inputs (Dict): KFServing http request

        Returns:
            Dict: Returns the request input after converting it into a tensor
        """
        # logging.info("Preprocessing Inputs %s", inputs)
        np_array = np.array(inputs['instances'])
        self.y_true = np_array[:, :, :1]
        self.mask = np_array[:, :, 0] != 0  # (batch_size, window_length)
        return inputs

    def postprocess(self, request: Dict) -> Dict:
        """Post process function of TFServing on the KFServing side is
        written here.

        Args:
            request (Dict): KFServing http request

        Returns:
            :param request: Dict: Returns the request input after converting it into a tensor
        """
        y_pred = np.array(request['predictions'])  # (batch_size, window_length, n_classes)

        lengths_session = self.mask.sum(axis=1)
        losses = np.empty_like(lengths_session, dtype=np.float64)
        for batch in range(self.y_true.shape[0]):
            total = 0.0
            for i in range(self.y_true.shape[1]):
                if self.y_true[batch, i, 0] > 0:
                    loss = self.sparse_cat_loss(self.y_true[batch, i, 0], y_pred[batch, i, :]).numpy()
                    total = total + loss
                else:
                    break
            losses[batch] = total

        mean_losses = losses / lengths_session
        logging.info("Mean Loss %s", mean_losses)
        logging.info("Total Loss %s", losses)

        response_dict = {'loss': losses.tolist(), 'mean_loss': mean_losses.tolist(), 'is_outlier': []}
        for mean_loss in response_dict['mean_loss']:
            if mean_loss > self.threshold:
                response_dict['is_outlier'].append(1)
            else:
                response_dict['is_outlier'].append(0)

        for i, mean_loss in enumerate(response_dict['mean_loss']):
            thread_push_2_es = threading.Thread(target=self.push_outlier_2_influx,
                                                args=(mean_loss, response_dict['is_outlier'][i],
                                                      self.y_true[i][self.mask[i]]))
            thread_push_2_es.start()
        return response_dict

    def push_outlier_2_influx(self, mean_loss, is_outlier, session: np.ndarray):
        json_body = [{
            "measurement": "outlier",
            "tags": {
            },
            "fields": {
                "mean_loss": mean_loss,
                "is_outlier": is_outlier
            },
            "time": datetime.now()
        }]
        if not self.client.write_points(json_body):
            logging.error("COULD NOT WRITE DATA INTO INFLUXDB!")

    def create_index_es(self, index_name='outlier'):
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

    def push_outlier_2_es(self, mean_loss, is_outlier, np_array: np.ndarray):
        record = {
            'session': np_array.tolist(),
            'mean_loss': mean_loss,
            'is_outlier': is_outlier,
            'timestamp': datetime.now(),
        }
        self.es.index(index=self.es_indeces['outlier'], body=record, doc_type='outlier')
