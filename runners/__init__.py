REGISTRY = {}

from runners.base_runner import BaseRunner
REGISTRY['base'] = BaseRunner

from runners.episode_runner import EpisodeRunner
REGISTRY['episode'] = EpisodeRunner
