import datetime
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from google.cloud import bigquery

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from src.models.components.interfaces import ModelOutput
from src.utils.decorators import RetriesFailedException, retry
from src.utils.tensor_utils import merge_list_of_keyed_tensors_to_single_tensor

log = logging.getLogger(__name__)


class BaseBufferedWriter(BasePredictionWriter):
    def __init__(
        self,
        # Update this based on data size. The goal is to limit the number of writes to BQ without exceeding the memory of your machine.
        flush_frequency: int = 5000,
        write_interval: str = "batch",
        schema: Optional[List[Union[bigquery.SchemaField, pa.Field]]] = None,
        prediction_key_name: Optional[str] = None,
        prediction_name: Optional[str] = None,
    ):
        """
        Args:
            flush_frequency: Number of rows to accumulate before flushing
            write_interval: "batch" or "epoch".
            schema: (Optional) Schema for the output data.
            prediction_key_name: (Optional) The key to key the predictions by e.g. user_id, item_id, etc.
            If not provided, it will be the module's responsibility to either hardcode
            or set it through the module's attributes.
            prediction_name: (Optional) The name for the prediction ie. label, logits, etc.
            If not provided, it will be the module's responsibility to either hardcode
            or set it through the module's attributes.
        """
        super().__init__(write_interval)
        self.flush_frequency = flush_frequency
        # Buffer to accumulate rows before writing
        self.rows_buffer: List[dict] = []
        self.schema = schema
        self.global_rank = None
        self.prediction_key_name = prediction_key_name
        self.prediction_name = prediction_name

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.global_rank = trainer.global_rank if trainer.global_rank else 0
        log.info(f"Rank {self.global_rank} initialized for inference.")
        # If the module does not have the prediction_key_name or prediction_name attributes,
        # we don't do anything as it might be using the previous interface.
        # If the module has the attributes and they are set, we also don't change it as they
        # might be hardcoded in the module.
        # We only set the prediction_key_name or prediction_name if they are not set in the module but are
        # passed in the callback. This allows us to control the prediction_key_name and prediction_name
        # from the callback config.
        # TODO (lneves) Deprecate the way folks do and have all inference pipelines to use this.

        if (
            hasattr(pl_module, "prediction_key_name")
            and pl_module.prediction_key_name is None
            and self.prediction_key_name is not None
        ):
            pl_module.prediction_key_name = self.prediction_key_name
        if (
            hasattr(pl_module, "prediction_name")
            and pl_module.prediction_name is None
            and self.prediction_name is not None
        ):
            pl_module.prediction_name = self.prediction_name

    def flush_buffer(self) -> None:
        """Flush the buffer and then clear it."""
        if self.rows_buffer:
            self._flush_buffer()
            self.rows_buffer.clear()

        else:
            log.info("Buffer is empty, nothing to flush.")

    def _flush_buffer(self) -> None:
        """Override this method to implement the logic for flushing the buffer."""
        raise NotImplementedError(
            "You need to implement the `_flush_buffer` method in your subclass."
        )

    def handle_batch(self, model_output: ModelOutput) -> None:
        """
        Handles a batch of predictions by appending them to the rows buffer and flushing the buffer if it is full.
        Args:
            model_output: The ModelOutput object containing the predictions.
        """
        if model_output is None:
            log.warning(
                f"Rank {self.global_rank} received an empty model output. Skipping this batch. This is expected if the batch is a dummy batch."
            )
            return None
        rows = model_output.list_of_row_format

        if len(rows) + len(self.rows_buffer) < self.flush_frequency:
            self.rows_buffer.extend(rows)
        else:
            rows_left = self.flush_frequency - len(self.rows_buffer)
            self.rows_buffer.extend(rows[:rows_left])
            self.flush_buffer()
            self.rows_buffer.extend(rows[rows_left:])

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: ModelOutput,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Called at the end of each prediction batch.
        We'll accumulate rows in the buffer and only write to BigQuery
        once we reach the flush_frequency.
        """
        self.handle_batch(prediction)

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: List[ModelOutput],
        batch_indices: List[List[int]],
    ) -> None:
        """
        Called at the end of a prediction epoch.
        We'll continue to buffer predictions from all batches in this epoch
        and flush if the buffer exceeds flush_frequency.
        """
        # predictions is typically a list of tensors (one per batch).
        for batch_pred in predictions:
            self.handle_batch(batch_pred)
        # At the very end of the epoch, flush any remaining rows
        self.flush_buffer()

    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """
        Called at the end of the prediction process.
        We'll flush any remaining rows in the buffer.
        """
        self.flush_buffer()
        log.info(f"Rank {self.global_rank} finished writing predictions.")
        # TODO (clark): technically write_on_epoch_end should handle this correctly
        # but if we dont't do this as well here, the number of rows in the final BQ table will
        # always be a multiplier of flush_frequency
        # and we will lose some rows if the last batch is smaller than flush_frequency
        # this is an indicator of something not working fully as expected
        # i'll investigate this issue later

class LocalPickleWriter(BaseBufferedWriter):
    """
    Callback to write predictions to local pickle files during inference.
    """

    def __init__(
        self,
        output_dir: str,
        flush_frequency: int = 1000,
        write_interval: str = "batch",
        should_merge_files_on_main: bool = True,
        should_merge_list_of_keyed_tensors_to_single_tensor: bool = True,
        post_processing_functions: Optional[List[Dict[str, callable]]] = [],
        **kwargs,
    ):
        """
        Args:
            output_dir: Directory to save the pickle files.
            flush_frequency: Number of rows to accumulate
                             before writing to a pickle file.
            write_interval: "batch" or "epoch".
            should_merge_files_on_main: If True, merge all files on the main process after writing.
            should_merge_list_of_keyed_tensors_to_single_tensor: If True, merge list of keyed tensors to a single tensor.
            post_processing_functions: List of ordered post-processing functions to apply to the files.
        """
        super().__init__(
            write_interval=write_interval, flush_frequency=flush_frequency, **kwargs
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.should_merge_files_on_main = should_merge_files_on_main
        self.should_merge_list_of_keyed_tensors_to_single_tensor = (
            should_merge_list_of_keyed_tensors_to_single_tensor
        )
        self.post_processing_functions = post_processing_functions

    def _create_file_path(self) -> str:
        """Create a file path for the pickle file."""
        return f"predictions_{self.global_rank}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')[:-3]}.pkl"

    def _local_file_path(self, file_path: Optional[str] = None) -> str:
        """Create a local file path for the pickle file."""
        return (
            f"{self.output_dir}/{file_path if file_path else self._create_file_path()}"
        )

    @retry()
    def _flush_buffer(self):
        """Flush the buffer to a pickle file."""

        file_path = self._create_file_path()
        with open(self._local_file_path(file_path=file_path), "wb") as f:
            pickle.dump(self.rows_buffer, f)

        log.info(
            f"Global Rank: {self.global_rank} wrote {len(self.rows_buffer)} rows to {self._local_file_path(file_path=file_path)}."
        )

    @retry()
    def on_predict_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        super().on_predict_end(trainer, pl_module)

        if self.should_merge_files_on_main:
            # if we use multiple workers, we need to wait for all of them to finish writing
            # before merging the files
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.barrier()
            if self.global_rank == 0:
                log.info("Merging pickle files on main process.")
                self._merge_files()

            # other processes can continue after merging
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.barrier()

        # conducting post-processing functions on the files
        for process_func in self.post_processing_functions:
            all_files = [f for f in os.listdir(self.output_dir)]
            for file in all_files:
                file_path = os.path.join(self.output_dir, file)
                if process_func.get("main_only", False):
                    if self.global_rank == 0:
                        process_func["function"](file_path)
                else:
                    process_func["function"](file_path)
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.barrier()

    def _merge_files(self):
        """Merge all pickle files in the output directory into a single file."""
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith(".pkl")]
        merged_data = []
        for file in all_files:
            with open(os.path.join(self.output_dir, file), "rb") as f:
                merged_data.extend(pickle.load(f))
            os.remove(os.path.join(self.output_dir, file))
        with open(os.path.join(self.output_dir, "merged_predictions.pkl"), "wb") as f:
            pickle.dump(merged_data, f)
        log.info(f"Merged {len(merged_data)} rows into merged_predictions.pkl.")

        if self.should_merge_list_of_keyed_tensors_to_single_tensor:
            merged_data_tensor = merge_list_of_keyed_tensors_to_single_tensor(
                data=merged_data,
                index_key=self.prediction_key_name,
                value_key=self.prediction_name,
            )
            torch.save(
                merged_data_tensor.cpu(),
                os.path.join(self.output_dir, "merged_predictions_tensor.pt"),
            )
        log.info(
            f"Merged {len(merged_data_tensor)} rows into merged_predictions_tensor.pt. as pytorch tensor"
        )
