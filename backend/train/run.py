"""`fig_config` and code in `server` are copied from Flower Android example."""
import pickle
from logging import getLogger

import requests
from flwr.common import FitRes, Parameters, Scalar
from flwr.server import ServerConfig, start_server
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvgAndroid
from flwr.server.strategy.aggregate import aggregate
from numpy import isnan
from numpy.typing import NDArray

PORT = 8080

logger = getLogger(__name__)


class FedAvgAndroidSave(FedAvgAndroid):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        # This method is initially copied from `server/strategy/fedavg_android.py`
        # in the `flwr` repository.
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = []
        for client, fit_res in results:
            weights = self.parameters_to_ndarrays(fit_res.parameters)
            if any(isnan(weight).any() for weight in weights):
                logger.error(
                    f"aggregate_fit: disgarding weights with NaN from {client}: {weights}."
                )
            else:
                weights_results.append((weights, fit_res.num_examples))
        if weights_results.__len__() == 0:
            raise RuntimeError(
                "aggregate_fit: No valid weights so cannot continue training."
            )
        aggregated = aggregate(weights_results)
        self.signal_save_params(aggregated)
        return self.ndarrays_to_parameters(aggregated), {}

    def signal_save_params(self, params: list[NDArray]):
        # TODO: Port resolution.
        url = "http://localhost:8000/train/params"
        files = {"file": pickle.dumps(params)}
        return requests.post(url, files=files)


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 2,
    }
    return config


def flwr_server(initial_parameters: Parameters | None):
    # TODO: Make configurable.
    strategy = FedAvgAndroidSave(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )

    logger.warning("Starting Flower server.")
    try:
        # Start Flower server for 3 rounds of federated learning
        start_server(
            server_address=f"0.0.0.0:{PORT}",
            config=ServerConfig(num_rounds=3),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        return
    except RuntimeError as err:
        logger.error(err)
