from logging import getLogger
from multiprocessing import Process
from threading import Thread

from flwr.common import Parameters
from telemetry.models import TrainingSession
from train.data import ServerData
from train.models import *
from train.run import PORT, flwr_server

logger = getLogger(__name__)


def model_params(model: TFLiteModel):
    try:
        params: ModelParams = model.params.last()  # type: ignore
        if params is None:
            return
        tensors = [param.tobytes() for param in params.decode_params()]
        return Parameters(tensors, tensor_type="numpy.ndarray")
    except RuntimeError as err:
        logger.warning(err)


TWELVE_HOURS = 12 * 60 * 60


class Server:
    """Spawn a new background Flower server process and monitor it."""

    def __init__(self, model: TFLiteModel, start_fresh: bool) -> None:
        self.model = model
        self.start_fresh = start_fresh
        params = None if start_fresh else model_params(model)
        self.session = TrainingSession(tflite_model=model)
        self.process = Process(target=flwr_server, args=(params,))
        self.process.start()
        self.timeout = Thread(target=Process.join, args=(self.process, TWELVE_HOURS))
        self.timeout.start()
        self.update_session_end_time()
        logger.warning(f"Started flower server for model {model}")

    def update_session_end_time(self):
        self.session.save()


task: Server | None = None


def cleanup_task():
    global task
    if task is not None and not task.process.is_alive():
        task = None


def server(model: TFLiteModel, start_fresh: bool) -> ServerData:
    """Request a Flower server. Return `(status, port)`.
    `status` is "started" if the server is already running,
    "new" if newly started,
    or "occupied" if the background process is unavailable."""
    global task
    cleanup_task()
    if task:
        if task.model == model:
            if start_fresh and not task.start_fresh:
                return ServerData("started_non_fresh", task.session.id, None)
            return ServerData("started", task.session.id, PORT)
        else:
            return ServerData("occupied", None, None)
    else:
        # Start new server.
        task = Server(model, start_fresh)
        return ServerData("new", task.session.id, PORT)
