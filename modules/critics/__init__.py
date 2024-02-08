REGISTRY = {}

from .customized_q import AdHocRelationalQ
REGISTRY['r-adhoc'] = AdHocRelationalQ

from .customized_q import AdHocGraphQ
REGISTRY['g-adhoc'] = AdHocGraphQ
