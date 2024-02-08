REGISTRY = {}

from modules.encoders.flat_enc import FlatEncoder
REGISTRY['flat'] = FlatEncoder

from modules.encoders.rel_enc import RelationalEncoder
REGISTRY['rel'] = RelationalEncoder
