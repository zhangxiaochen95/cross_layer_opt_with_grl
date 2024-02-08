from abc import abstractmethod


class BaseLearner:
    """Base class of learners to train agents as well as auxiliary modules"""

    def eval(self):
        """Sets auxiliary modules to eval mode."""
        return

    def init_hidden(self):
        """Initializes RNN states for auxiliary modules"""
        return dict()

    def step(self, *args):
        """Gets output of auxiliary modules for env step."""
        return dict()

    def schedule_lr(self):
        """Calls step of learning rate scheduler(s)."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Resets learner before training loop."""
        raise NotImplementedError

    @ abstractmethod
    def update(self, buffer, batch_size: int):
        """Updates parameters of modules from collected data."""
        raise NotImplementedError

    def soft_target_sync(self):
        raise NotImplementedError

    def hard_target_sync(self):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path):
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path):
        raise NotImplementedError
