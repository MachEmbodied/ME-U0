"""RoboDojo evaluation loop for the external ME_U0 WebSocket policy."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        for action_index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_index + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [
                index
                for index, env_idx in enumerate(env_idx_list)
                if env_idx in running
            ]
            actions = [actions[index] for index in active_batch_idx]
            env_idx_list = [env_idx_list[index] for index in active_batch_idx]
            model_client.call(
                func_name="update_obs_batch",
                obs=TASK_ENV.get_obs_batch(env_idx_list),
            )
