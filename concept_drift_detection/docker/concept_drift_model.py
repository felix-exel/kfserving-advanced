import tensorflow as tf
import json
import tornado
from typing import Dict
import logging
import numpy as np
import kfserving
import threading
from alibi_detect.utils.saving import load_detector
from alibi_detect.cd import KSDrift, MMDDrift
from influxdb import InfluxDBClient
from datetime import datetime

PREDICTOR_URL_FORMAT = "http://{0}/v1/models/{1}:predict"
PREDICTOR_V2_URL_FORMAT = "http://{0}/v2/models/{1}/infer"
EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)


class ConceptDriftModel(kfserving.KFModel):
    """ A class object for the data handling activities of Concept Drift
    Task and returns a KFServing compatible response.

    Args:
        kfserving (class object): The KFModel class from the KFServing
        modeule is passed here.
    """

    def __init__(self, name: str, influx_host: str, influx_port: int, database: str, model_path: str):
        """Initialize the model name, influx host

        Args:
            name (str): Name of the model.
        """
        super().__init__(name)

        self.timeout = 999999999999
        logging.info("TIMEOUT URL %s", self.timeout)
        self.name = name
        self.batch_size = 1000
        self.batches = None
        self.model = None
        # influxDB Client
        self.client = InfluxDBClient(host=influx_host, port=influx_port)
        self.client.switch_database(database)
        self.model_path = model_path

    def load(self):
        self.model = load_detector(self.model_path)
        self.ready = True

    async def predict(self, request: Dict) -> Dict:

        if self.batches.shape[0] >= self.batch_size:
            preds_ood = self.model.predict(self.batches, return_p_val=True)
            logging.info("Drift %s", preds_ood['data']['is_drift'])
            thread_push_2_es = threading.Thread(target=self.push_conceptdrift_2_influx,
                                                args=[preds_ood['data']['is_drift']])
            thread_push_2_es.start()
            self.batches = None

        else:
            preds_ood = {}
        return preds_ood

    def preprocess(self, inputs: Dict) -> Dict:
        """Pre-process activity of the Session Input data.

        Args:
            inputs (Dict): KFServing http request

        Returns:
            Dict: Returns the request input after converting it into a tensor
        """
        np_array = np.array(inputs['instances'])  # (batch_size, window_length, 52)

        if self.batches is None:
            self.batches = np_array[:, :, :1]
        else:
            self.batches = np.concatenate([self.batches, np_array[:, :, :1]], axis=0)

        return inputs

    def postprocess(self, request: Dict) -> Dict:
        """Post process function of TFServing on the KFServing side is
        written here.

        Args:
            request (Dict): KFServing http request

        Returns:
            :param request: Dict: Returns the request input after converting it into a tensor
        """
        return request

    def push_conceptdrift_2_influx(self, is_concept_drift: int):
        json_body = [{
            "measurement": "concept-drift",
            "tags": {
            },
            "fields": {
                "is_concept_drift": is_concept_drift
            },
            "time": datetime.now()
        }]
        if not self.client.write_points(json_body):
            logging.error("COULD NOT WRITE DATA INTO INFLUXDB!")

    def create_index_es(self, index_name='concept-drift'):
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

    def push_conceptdrift_2_es(self, is_concept_drift: int):
        record = {
            'is_concept_drift': is_concept_drift,
            'timestamp': datetime.now(),
        }
        self.es.index(index=self.es_indeces['concept-drift'], body=record, doc_type='concept-drift')
