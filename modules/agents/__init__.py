REGISTRY = {}

from .recurrent_agent import RecurrentAgent
REGISTRY['rnn'] = RecurrentAgent

from .comm_agent import CommunicativeAgent
REGISTRY['comm'] = CommunicativeAgent

from .customized_agents import AdHocRelationalController
REGISTRY['r-adhoc'] = AdHocRelationalController

from .customized_agents import AdHocGraphController
REGISTRY['g-adhoc'] = AdHocGraphController
