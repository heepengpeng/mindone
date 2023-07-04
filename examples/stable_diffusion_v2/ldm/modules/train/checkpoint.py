"""checkpoint manager """
import os
import stat

import mindspore as ms


class CheckpointManager:
    """
    Manage checkpoint files according to ckpt_save_policy of checkpoint.
    Args:
        ckpt_save_dir (str): directory to save the checkpoints
        ckpt_save_policy (str): Checkpoint saving strategy. Option: None, "top_k", or "latest_k".
            None means to save each checkpoint, top_k means to save K checkpoints with the best performance,
            and latest_k means saving the latest K checkpoint. Default: top_k.
        k (int): top k value
        prefer_low_perf (bool): standard for selecting the top k performance. If False, pick top k checkpoints with
            highest performance e.g. accuracy. If True, pick top k checkpoints with the lowest performance, e.g. loss.

    """

    def __init__(self, ckpt_save_dir, ckpt_save_policy="top_k", k=10, prefer_low_perf=False, del_past=True):
        self.ckpt_save_dir = ckpt_save_dir
        self._ckpt_filelist = []
        self.ckpt_save_policy = ckpt_save_policy
        self.k = k

        self.ckpt_queue = []
        self.del_past = del_past
        self.prefer_low_perf = prefer_low_perf

    def get_ckpt_queue(self):
        """Get all the related checkpoint files managed here."""
        return self.ckpt_queue

    @property
    def ckpt_num(self):
        """Get the number of the related checkpoint files managed here."""
        return len(self.ckpt_queue)

    def remove_ckpt_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            if os.path.exists(file_name):
                os.chmod(file_name, stat.S_IWRITE)
                os.remove(file_name)
        except OSError:
            print(f"OSError, failed to remove the older ckpt file {file_name}.")
        except ValueError:
            print(f"ValueError, failed to remove the older ckpt file {file_name}.")

    def save_top_k(self, network, perf, ckpt_name, verbose=True, lora_rank=None):
        """Save and return Top K checkpoint address and accuracy."""
        self.ckpt_queue.append((perf, ckpt_name))
        self.ckpt_queue = sorted(
            self.ckpt_queue, key=lambda x: x[0], reverse=not self.prefer_low_perf
        )  # by default, reverse is True for descending order
        append_dict = {"lora_rank": lora_rank} if lora_rank is not None else None
        if len(self.ckpt_queue) > self.k:
            to_del = self.ckpt_queue.pop(-1)
            # save if the perf is better than the minimum in the heap
            if to_del[1] != ckpt_name:
                ms.save_checkpoint(network, os.path.join(self.ckpt_save_dir, ckpt_name), integrated_save=False,
                                   append_dict=append_dict)
                # del minimum
                self.remove_ckpt_file(os.path.join(self.ckpt_save_dir, to_del[1]))
        else:
            ms.save_checkpoint(network, os.path.join(self.ckpt_save_dir, ckpt_name),
                               append_dict=append_dict)

    def save_latest_k(self, network, ckpt_name, lora_rank):
        append_dict = {"lora_rank": lora_rank} if lora_rank is not None else None
        """Save latest K checkpoint."""
        ms.save_checkpoint(network, os.path.join(self.ckpt_save_dir, ckpt_name), integrated_save=False,
                           append_dict=append_dict)

        print('Checkpoint saved in ', os.path.join(self.ckpt_save_dir, ckpt_name))
        self.ckpt_queue.append(ckpt_name)
        if len(self.ckpt_queue) > self.k:
            to_del = self.ckpt_queue.pop(0)
            if self.del_past:
                self.remove_ckpt_file(os.path.join(self.ckpt_save_dir, to_del))

    def save(self, network, perf=None, ckpt_name=None, lora_rank=None):
        """Save checkpoint according to different save strategy."""
        append_dict = {"lora_rank": lora_rank} if lora_rank is not None else None
        if self.ckpt_save_policy is None:
            ms.save_checkpoint(network, os.path.join(self.ckpt_save_dir, ckpt_name), integrated_save=False,
                               append_dict=append_dict)
        elif self.ckpt_save_policy == "top_k":
            if perf is None:
                raise ValueError(
                    "Evaluation performance is None, but `top_k` ckpt save policy requires evaluation performance"
                )
            self.save_top_k(network, perf, ckpt_name, lora_rank)
            return self.ckpt_queue
        elif self.ckpt_save_policy == "latest_k":
            self.save_latest_k(network, ckpt_name, lora_rank)
            return self.ckpt_queue
        else:
            raise ValueError(
                f"The expected 'ckpt_save_policy' is None, top_k or latest_k, but got: {self.ckpt_save_policy}."
            )
