import copy
import logging
import warnings
from typing import Dict, Optional, Callable

from ray.rllib.utils.from_config import from_config
from ray.util import log_once

from blades.clients.callbacks import ClientCallback
from blades.utils.types import NotProvided, PartialTrainerConfigDict


logger = logging.getLogger(__name__)


class ClientConfig:
    _BUILTIN_CLIENTS = {
        "blades.clients.BaseClient",
        "blades.clients.base_client.BaseClient",
        "blades.clients.Client",
        "blades.clients.client.Client",
        "blades.clients.DPClient",
        "blades.clients.dp_client.DPClient",
    }
    _DATASET_DEFAULTS = {
        "dataset_identifier": "mnist",
        "train_batch_size": 32,
        "eva_batch_size": 64,
    }
    _MODEL_DEFAULTS = {
        "model_identifier": "mlp",
        "model_config": None,
        "loss_identifier": "cross_entropy",
        "loss_kargs": {},
        "loss": None,
    }

    def __init__(self, class_specifier) -> None:
        super().__init__()

        self.uid = None
        self.cls = class_specifier

        # self.lr = 0.1
        self.num_batch_per_round = 1

        # `self.dataset()`
        self.dataset_identifier = "mnist"
        self.train_batch_size = 32
        self.eva_batch_size = 64

        # `self.model()`
        self.model_identifier = "mlp"
        self.model_config = None
        self.loss_identifier = "cross_entropy"
        self.loss_kargs = {}

        # `self.training()`
        self.opt_identifier = "sgd"
        self.opt_kargs = {"lr": 0.1}
        self.lr = 0.1
        self.momentum = 0.0
        self.lr_scheduler = None
        self.lr_scheduler_config = None

        self.is_trainable = True
        self.is_testable = True

        # `self.callbacks()`
        self.callbacks_config = ClientCallback

    def training(
        self,
        num_batch_per_round: int = NotProvided,
        lr: float = NotProvided,
        momentum: float = NotProvided,
    ) -> "ClientConfig":
        """Set the number of batches per round.

        Args:
            num_batch_per_round (int): Number of batches per round.

        Returns:
            ClientConfig: This updated ClientConfig object.
        """
        if num_batch_per_round is not NotProvided:
            self.num_batch_per_round = num_batch_per_round
        if lr is not NotProvided:
            self.lr = lr
            if (
                self._uses_builtin_client()
                and isinstance(self.opt_kargs, dict)
                and "lr" in self.opt_kargs
            ):
                self.opt_kargs["lr"] = lr
        if momentum is not NotProvided:
            self.momentum = momentum
            if (
                self._uses_builtin_client()
                and isinstance(self.opt_kargs, dict)
                and "momentum" in self.opt_kargs
            ):
                self.opt_kargs["momentum"] = momentum
        return self

    def trainable(self, trainable: bool = True) -> "ClientConfig":
        self.is_trainable = trainable
        return self

    def testable(self, testable: bool = True) -> "ClientConfig":
        self.is_testable = testable
        return self

    def client_id(self, uid: str) -> "ClientConfig":
        """Set the client id.

        Args:
            uid (str): Client id.

        Returns:
            ClientConfig: This updated ClientConfig object.
        """
        self.uid = uid
        return self

    def callbacks(self, callbacks_config) -> "ClientConfig":
        """Sets the callbacks configuration.

        Args:
            callbacks_config: Callbacks class, whose methods will be run during
                various phases of training and environment sample collection.
                See the `DefaultCallbacks` class and
                `examples/custom_metrics_and_callbacks.py` for more usage information.

        Returns:
            This updated FedavgClientConfig object.
        """
        if callbacks_config is None:
            callbacks_config = ClientCallback
        # Check, whether given `callbacks` is a callable.
        if not callable(callbacks_config):
            raise ValueError(
                "`config.callbacks_config` must be a callable method that "
                "returns a subclass of DefaultCallbacks, got "
                f"{callbacks_config}!"
            )
        self.callbacks_config = callbacks_config

        return self

    def dataset(
        self,
        dataset_identifier: str = NotProvided,
        *,
        train_batch_size: int = NotProvided,
        eva_batch_size: int = NotProvided,
    ) -> "ClientConfig":
        """Deprecated; configure built-in client data on TrainerConfig instead."""
        self._warn_legacy_method("dataset", "TrainerConfig.data(dataset_config=...)")
        if self._uses_builtin_client() and any(
            value is not NotProvided
            for value in (dataset_identifier, train_batch_size, eva_batch_size)
        ):
            raise ValueError(self._dataset_error())
        if dataset_identifier is not NotProvided:
            self.dataset_identifier = dataset_identifier
        if train_batch_size is not NotProvided:
            self.train_batch_size = train_batch_size
        if eva_batch_size is not NotProvided:
            self.eva_batch_size = eva_batch_size
        return self

    def model(
        self,
        model_identifier: Optional[str] = NotProvided,
        *,
        model_config: Optional[Dict] = NotProvided,
        loss: Optional[str] = NotProvided,
    ) -> "ClientConfig":
        """Deprecated; configure the global model and Task on TrainerConfig."""
        self._warn_legacy_method("model", "TrainerConfig.training(global_model=...)")
        if self._uses_builtin_client() and any(
            value is not NotProvided
            for value in (model_identifier, model_config, loss)
        ):
            raise ValueError(self._model_error())
        if model_identifier is not NotProvided:
            self.model_identifier = model_identifier
        if model_config is not NotProvided:
            self.model_config = model_config
        if loss is not NotProvided:
            self.loss = loss
        return self

    def optimizer(
        self,
        opt_identifier: Optional[str] = NotProvided,
        *,
        opt_kargs: Optional[Dict] = NotProvided,
        lr_scheduler: Optional[str] = NotProvided,
        lr_scheduler_config: Optional[Dict] = NotProvided,
    ) -> "ClientConfig":
        """Deprecated; built-in clients accept SGD lr/momentum via training()."""
        self._warn_legacy_method("optimizer", "ClientConfig.training(lr=..., momentum=...)")
        if self._uses_builtin_client():
            self._validate_optimizer(
                self.opt_identifier if opt_identifier is NotProvided else opt_identifier,
                self.opt_kargs if opt_kargs is NotProvided else opt_kargs,
                self.lr_scheduler if lr_scheduler is NotProvided else lr_scheduler,
                self.lr_scheduler_config
                if lr_scheduler_config is NotProvided
                else lr_scheduler_config,
            )
            if opt_kargs is not NotProvided:
                self.training(**opt_kargs)
        if opt_identifier is not NotProvided:
            self.opt_identifier = opt_identifier
        if opt_kargs is not NotProvided:
            self.opt_kargs = copy.deepcopy(opt_kargs)
        if lr_scheduler is not NotProvided:
            self.lr_scheduler = lr_scheduler
        if lr_scheduler_config is not NotProvided:
            self.lr_scheduler_config = lr_scheduler_config
        return self

    def build(self, use_copy: bool = True) -> Callable:
        """Build the client.

        Args:
            use_copy (bool, optional): Defaults to True.

        Returns:
            Callable: the instantiated client.
        """
        client_cls = self.cls
        # if isinstance(client_cls, str):
        #     raise ValueError("string `client_cls` is not supported in blades yet")
        if client_cls is None:
            raise ValueError("client_cls is not set")
        self.validate()
        if (
            self._uses_builtin_client()
            and self.opt_kargs != {"lr": 0.1}
            and any(value != getattr(self, key) for key, value in self.opt_kargs.items())
        ):
            warnings.warn(
                "ClientConfig.opt_kargs differs from the effective lr/momentum. "
                "Assigning opt_kargs directly does not configure the optimizer; "
                "the client will use lr/momentum. Use ClientConfig.training(), "
                "optimizer(), or update_from_dict() to apply SGD options.",
                FutureWarning,
                stacklevel=2,
            )

        return from_config(
            client_cls,
            config=None,
            **{"client_config": self if not use_copy else copy.deepcopy(self)},
        )

    def to_dict(self) -> dict:
        """Converts all settings into a legacy config dict for backward
        compatibility.

        Returns:
            A complete TrainerConfigDict, usable in backward-compatible Tune/RLlib
            use cases, e.g. w/ `tune.Tuner().fit()`.
        """
        config = copy.deepcopy(vars(self))

        return config

    def _uses_builtin_client(self) -> bool:
        client_cls = self.cls
        if isinstance(client_cls, type):
            client_cls = f"{client_cls.__module__}.{client_cls.__qualname__}"
        return isinstance(client_cls, str) and client_cls in self._BUILTIN_CLIENTS

    @staticmethod
    def _warn_legacy_method(method, replacement):
        warnings.warn(
            f"ClientConfig.{method}() is deprecated. Use {replacement} for built-in "
            "clients; custom clients must consume their own settings.",
            DeprecationWarning,
            stacklevel=3,
        )

    @staticmethod
    def _dataset_error():
        return (
            "Built-in clients do not consume ClientConfig dataset settings. "
            "Use TrainerConfig.data(dataset_config=...) / TrainerConfig.dataset_config "
            "for dataset and batch sizes."
        )

    @staticmethod
    def _model_error():
        return (
            "Built-in clients do not consume ClientConfig model/loss settings. "
            "Use TrainerConfig.training(global_model=...) / TrainerConfig.global_model "
            "for the model, and a custom Task.loss for a custom loss."
        )

    @staticmethod
    def _validate_optimizer(identifier, kwargs, scheduler, scheduler_config):
        guidance = (
            "Built-in clients support only SGD lr/momentum through "
            "ClientConfig.training(lr=..., momentum=...). Override "
            "Task.init_optimizer for another optimizer or a scheduler."
        )
        if not isinstance(identifier, str) or identifier.lower() != "sgd":
            raise ValueError(f"Unsupported client optimizer {identifier!r}. {guidance}")
        if not isinstance(kwargs, dict) or set(kwargs) - {"lr", "momentum"}:
            raise ValueError(f"Unsupported client opt_kargs {kwargs!r}. {guidance}")
        if scheduler is not None or scheduler_config is not None:
            raise ValueError(f"Client learning-rate schedulers are unsupported. {guidance}")

    def validate(self) -> None:
        """Reject legacy settings ignored by built-in clients.

        Legacy defaults remain accepted so existing default dictionaries round-trip.
        Custom clients may consume these fields and arbitrary extra configuration.
        """
        if not self._uses_builtin_client():
            return
        if any(
            getattr(self, key, default) != default
            for key, default in self._DATASET_DEFAULTS.items()
        ):
            raise ValueError(self._dataset_error())
        if any(
            getattr(self, key, default) != default
            for key, default in self._MODEL_DEFAULTS.items()
        ):
            raise ValueError(self._model_error())
        self._validate_optimizer(
            self.opt_identifier,
            self.opt_kargs,
            self.lr_scheduler,
            self.lr_scheduler_config,
        )

    def update_from_dict(
        self,
        config_dict: PartialTrainerConfigDict,
    ) -> "ClientConfig":
        """Update client settings, validating legacy options for built-in clients.

        Custom clients and extra fields remain supported. Legacy SGD options map
        to training settings; explicit lr/momentum take precedence if both appear.

        Args:
            config_dict: Settings to override.

        Returns:
            This updated ClientConfig object.
        """

        # Modify our properties one by one.
        for key, value in config_dict.items():
            # Some keys specify config sub-dicts and therefore should go through the
            # correct methods to properly `.update()` those from given config dict
            # (to not lose any sub-keys).

            if key in ["num_batch_per_round", "lr", "momentum"]:
                self.training(**{key: value})

            # If config key matches a property, just set it, otherwise, warn and set.
            else:
                if not hasattr(self, key) and log_once(
                    "unknown_property_in_algo_config"
                ):
                    logger.warning(
                        f"Cannot create {type(self).__name__} from given "
                        f"`config_dict`! Property {key} not supported."
                    )
                setattr(self, key, copy.deepcopy(value) if key == "opt_kargs" else value)
        self.validate()
        if self._uses_builtin_client() and "opt_kargs" in config_dict:
            # YAML/dict configuration must use the same SGD mapping as optimizer().
            # Explicit training values win when a legacy dictionary contains both.
            training_options = dict(self.opt_kargs)
            training_options.update(
                {key: config_dict[key] for key in ("lr", "momentum") if key in config_dict}
            )
            self.training(**training_options)
            self.opt_kargs = {key: getattr(self, key) for key in self.opt_kargs}
        return self
