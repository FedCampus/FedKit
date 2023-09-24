from logging import getLogger
from multiprocessing import Process
from threading import Thread

from flwr.common import Parameters
from telemetry.models import TrainingSession
from train.data import ServerData
from train.models import MLModel, ModelParams
from train.run import flwr_server

TF_PORT = 8080
CM_PORT = 8088

logger = getLogger(__name__)


def model_params(model: MLModel):
    try:
        params: ModelParams = model.params.last()  # type: ignore
        if params is None:
            return
        tensors = [param.tobytes() for param in params.decode_params()]
        return Parameters(tensors, tensor_type="numpy.ndarray")
    except RuntimeError as err:
        logger.warning(err)


TEN_MINUTES = 10 * 60


class Server:
    """Spawn a new background Flower server process and monitor it."""

    def __init__(self, model: MLModel, port: int, start_fresh: bool) -> None:
        self.model = model
        self.start_fresh = start_fresh
        params = None if start_fresh else model_params(model)
        self.session = TrainingSession(tflite_model=model)
        self.process = Process(target=flwr_server, args=(params, port, model.coreml))
        self.process.start()
        self.timeout = Thread(target=Process.join, args=(self.process, TEN_MINUTES))
        self.timeout.start()
        self.update_session_end_time()
        logger.warning(f"Started flower server for model {model}")

    def update_session_end_time(self):
        self.session.save()


tf_server: Server | None = None
cm_server: Server | None = None
"""CoreML servers"""


def cleanup_task():
    global tf_server, cm_server
    if tf_server is not None and not tf_server.process.is_alive():
        tf_server = None
    if cm_server is not None and not cm_server.process.is_alive():
        cm_server = None


def server(model: MLModel, start_fresh: bool) -> ServerData:
    """Request a Flower server. Return `(status, port)`.
    `status` is "started" if the server is already running,
    "new" if newly started,
    or "occupied" if the background process is unavailable."""
    global tf_server, cm_server
    server, port = (cm_server, CM_PORT) if model.coreml else (tf_server, TF_PORT)
    cleanup_task()
    if server:
        if server.model == model:
            if start_fresh and not server.start_fresh:
                return ServerData("started_non_fresh", None, None)
            return ServerData("started", server.session.id, port)
        else:
            return ServerData("occupied", None, None)
    else:
        # Start new server.
        server = Server(model, port, start_fresh)
        if model.coreml:
            cm_server = server
        else:
            tf_server = server
        return ServerData("new", server.session.id, port)
