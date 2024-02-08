REGISTRY = {}

from envs.multi_ubs_coverage.multi_ubs_coverage import MultiUbsCoverageEnv
REGISTRY['ubs'] = MultiUbsCoverageEnv

from envs.ad_hoc.ad_hoc import AdHocEnv
REGISTRY['ad-hoc'] = AdHocEnv

from envs.mpe.make_mpe import make_mpe
REGISTRY['mpe'] = make_mpe
