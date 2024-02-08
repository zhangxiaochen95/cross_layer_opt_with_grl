from run import *

if __name__ == '__main__':

    # Setup common parameters shared across runs.
    base_scene = '1fb'
    exp_tag = base_scene
    common_kwargs = {
        'cuda_idx': 0,
        'env_id': 'ad-hoc',
        'env_kwargs': {'scenario_name': base_scene},

        'record_tests': True,
        'use_wandb': True,

        'total_env_steps': 1000000,
        'steps_per_session': 50000,
        'test_interval': 10000,
        'n_test_episodes': 25,
        'save_interval': 2000000,

        'rollout_len': 20,
        'data_chunk_len': 10,
        'eps_anneal_time': 100000,
        'gamma': 0.98,
        'batch_size': 32,
    }

    mlp_agent = {'hidden_size': 256}
    graph_agent = {'obs': 'graph', 'agent': 'g-adhoc', 'hidden_size': 128}

    # Form a queue of runs.
    queues = {
        # Full power transmission
        'full_rnn': {'algo_name': 'q', **mlp_agent,
                     'env_kwargs': {'scenario_name': f'{base_scene}-full-pow'}},
        'full_1hop-gnn': {'algo_name': 'q', **graph_agent,
                          'env_kwargs': {'scenario_name': f'{base_scene}-full-pow'}},
        'full_2hop-gnn': {'algo_name': 'q', **graph_agent,
                          'env_kwargs': {'scenario_name': f'{base_scene}-full-pow', 'graph_khops': 2}},
        # Random power selection
        'rand_rnn': {'algo_name': 'q', **mlp_agent,
                     'env_kwargs': {'learn_power_control': False}},
        'rand_1hop-gnn': {'algo_name': 'q', **graph_agent,
                          'env_kwargs': {'learn_power_control': False}},
        'rand_2hop-gnn': {'algo_name': 'q', **graph_agent,
                          'env_kwargs': {'learn_power_control': False, 'graph_khops': 2}},
        # Learn cross-layer optimization
        'pc_rnn': {'algo_name': 'q', **mlp_agent},
        'pc_1hop-gnn': {'algo_name': 'q', **graph_agent},
        'pc_2hop-gnn': {'algo_name': 'q', **graph_agent, 'env_kwargs': {'graph_khops': 2}},
    }

    # Assign random seeds.
    seeds = [0, 1, 2]
    # Display run names.
    print(f"Following {len(seeds) * len(queues)} runs are launched in total:")
    for run_name in queues:
        print(run_name)
    # Sequentially start each run.
    for seed in seeds:
        for tag, param_dict in queues.items():
            # Build tag of the run.
            run_tag = exp_tag + '_' + tag if exp_tag is not None else tag

            # Get all kwargs of the run.
            run_kwargs = config_copy(common_kwargs)
            run_kwargs = recursive_dict_update(run_kwargs, param_dict)
            # Extract `env_id`, `env_kwargs` and `algo_name`.
            env_id = run_kwargs['env_id']
            env_kwargs = run_kwargs['env_kwargs']
            algo_name = run_kwargs['algo_name']
            # Drop irrelevant items and `get train_kwargs`.
            train_kwargs = config_copy(run_kwargs)
            del train_kwargs['env_id'], train_kwargs['env_kwargs'], train_kwargs['algo_name']

            # Call run.
            run(env_id, env_kwargs, seed, algo_name, train_kwargs, run_tag, add_suffix=True, suffix=base_scene)
