# pylint: disable=line-too-long, pointless-string-statement
import os
import warnings
import torch
from typing import Dict, Union, Mapping, Sequence,Optional

from . import Logger

try:
    import wandb

except ImportError:
    wandb = None


class WandBLogger(Logger):

    """

    WandB logger to manage logging of experiments parameters, metrics update, models log, gradient values and other information. The
    logger will log all run into the same experiment. 

    Args:
        name(str): Display name for the run.
        save_dir(str): Path where data is saved (wandb dir by default).
        offline(bool): Run offline (data can be streamed later to wandb servers).
        id(str): Sets the version, mainly used to resume a previous run.
        version(str): Same as id.
        anonymous(bool): Enables or explicitly disables anonymous logging.
        project(str): The name of the project to which this run will belong.
        experiment: Experiment to use instead of creating a new one.
        batch_granularity(bool): Whether to also output the result of each batch in addition to the epochs.
            (Default value = False).
        log_gradient_frequency(int): log gradients and parameters every N batches (Default value = None).
        training_batch_shape(tuples): Shape of a training batch. It will be used for logging architecture of the model on wandb
        

    Example:
        .. code-block:: python

            wandb_logger = WandBLogger(name="First_run",project="Test_project" save_dir="/absolute/path/to/directory", experiment="First experiment")
            wandb_logger.log_config_params(config_params=cfg_dict) # logging the config dictionary

            # our Poutyne experiment
            experiment = Experiment(directory=saving_directory, network=network, device=device, optimizer=optimizer,
                            loss_function=cross_entropy_loss, batch_metrics=[accuracy])

            # Using the WandB logger callback during training
            experiment.train(train_generator=train_loader, valid_generator=valid_loader, epochs=1,
                             seed=42, callbacks=[wandb_logger])

            # You can access the wandb run via the attribute .run if you want to use other wandb features
            image = wandb.Image(an_image, caption="a caption") 
            wandb_logger.run.log({"a exemple": image})

    """

    def __init__(
                    self,
                    name: Optional[str] = None,
                    save_dir: Optional[str] = None,
                    offline: Optional[bool] = False,
                    id: Optional[str] = None,
                    anonymous: Optional[bool] = None,
                    version: Optional[str] = None,
                    project: Optional[str] = None,
                    experiment=None,
                    batch_granularity: Optional[bool] = False,
                    log_gradient_frequency: Optional[bool] = None,
                    training_batch_shape : Optional[tuple] = None
                ) -> None:

        super().__init__(batch_granularity=batch_granularity)

        if wandb is None:
            raise ImportError("WandB needs to be installed to use this callback.")

        anonymous_lut = {True: "allow", False: None}
        self._wandb_init = dict(
            name=name,
            project=project,
            id=version or id,
            dir=save_dir,
            resume="allow",
            anonymous=anonymous_lut.get(anonymous, anonymous),
        )

        if experiment is None:

            if offline:
                os.environ["WANDB_MODE"] = "dryrun"

            if wandb.run is None:
                self.run = wandb.init(**self._wandb_init)
            else:
                warnings.warn(
                "There is already a wandb run experience running. This callback will reuse this run. If you want to start a new one stop this process and call `wandb.finish()` before starting again."
            )
                self.run = wandb.run
        else:
            self.run = experiment

        self.log_gradient_frequency = log_gradient_frequency
        self.training_batch_shape = training_batch_shape

    def _watch_gradient(self) -> None:
        """
            activate gradient watch 
        """
        self.run.watch(self.model.network, log="all", log_freq=self.log_gradient_frequency)


    def _save_architecture(self) -> None:
            """
                Save architecture
            """
            dummies_batch = torch.zeros(self.training_batch_shape)
            save_path = self.run.dir+"/"+self.run.name+"_model.onnx"
            torch.onnx.export(self.model.network,dummies_batch, save_path)
            self.run.watch(self.model.network, log="all", log_freq=self.log_gradient_frequency)
            self.run.save(save_path)


         
    def on_train_begin(self, logs: Dict):
        super().on_train_begin(logs)
        if  self.log_gradient_frequency is not None:
            self._watch_gradient()

        if self.training_batch_shape is not None:
            self._save_architecture()

    def log_config_params(self, config_params: Dict) -> None:
        """
        Args:
            config_params Dict:
                Dictionnary of config parameters of the training to log, such as number of epoch, loss function, optimizer etc.
        """
        self.run.config.update(config_params)

        
    def _on_train_batch_end_write(self, batch_number: int, logs: Dict) -> None:
        """
        Log the batch metric.
        """
        adjusted_logs = self._adjust_logs_granularity(logs, per_batch=True)
        self.run.log(adjusted_logs)

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict) -> None:
        """
        Log the epoch metric.
        """
        adjusted_logs = self._adjust_logs_granularity(logs)
        self.run.log(adjusted_logs)

    def on_train_end(self, logs: Dict):

        self.run.finish()

    def _adjust_logs_granularity(self,logs : Dict, per_batch=False) -> Dict:
        if self.batch_granularity == False:
            adjusted_logs =  logs

        else:
            if per_batch:
                adjusted_logs = {"Batch_"+metric: value for metric,value in logs.items()}
            else:
                adjusted_logs = {"Epoch_"+metric: value for metric,value in logs.items()}

        return adjusted_logs






