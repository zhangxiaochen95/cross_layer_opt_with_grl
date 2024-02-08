REGISTRY = {}

from learners.q_learner import QLearner
REGISTRY['q'] = QLearner

DETERMINISTIC_POLICY_GRADIENT_ALGOS = {'ddpg'}  # Algorithms using deterministic policy gradients
