import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import seaborn as sns
from env_economic_choice import EconomicChoiceEnv as EconomicChoiceEnv
from env_economic_choice_no_hold import EconomicChoiceEnv as EconomicChoiceEnv_nohold
from env_economic_choice_no_hold_partial import EconomicChoiceEnv as EconomicChoiceEnv_p
from env_partial_cartpole import CartPolePartialObservation
from env_romo import WorkingMemoryEnv
from env_romo_mod import WorkingMemoryEnv as WorkingMemoryEnv_mod
import os
import numpy as np
import tensorflow as tf

# --------------------------
# Helper function: Load model
# --------------------------

def load_model(agent, obs_size, act_size, stage, ckpt_prefix):
    """
    Carga el modelo y las máscaras del stage anterior (stage-1),
    construye las redes para el stage actual, inicializa los optimizadores
    y restaura el checkpoint.

    Args:
        agent: objeto que debe tener atributos
               - actor (con hidden_layers y hidden_size)
               - critic (con hidden_layers)
               - actor_optimizer
               - critic_optimizer
        obs_size: int, tamaño de la dimensión de observación
        act_size: int, tamaño de la dimensión de acción
        stage: int, número de stage actual (p. ej. 6)
        ckpt_prefix: str, prefijo del directorio de checkpoints
                     (sin el sufijo "_<stage>")

    Returns:
        this_ckpt_dir: str, ruta al directorio de checkpoints del stage actual
    """
    prev_ckpt_dir = f"{ckpt_prefix}_{stage-1}"
    this_ckpt_dir = f"{ckpt_prefix}_{stage}"
    os.makedirs(this_ckpt_dir, exist_ok=True)

    # --- Build Networks ---
    actor_input_shape = (None, None, obs_size)

    # Calcular actor_hid_for_critic
    actor_hid_for_critic = obs_size
    actor_hid_for_critic = agent.actor.hidden_size
    critic_input_shape = (None, None, actor_hid_for_critic + act_size)
    agent.actor.build(actor_input_shape)
    agent.critic.build(critic_input_shape)
    print("Actor and Critic networks built.")

    # --- Ensure layers are built (dummy forward) ---
    print("Performing dummy forward to build cells and weights for mask loading...")
    dummy_obs         = tf.zeros((1,1, obs_size), dtype=tf.float32)
    _                 = agent.actor(dummy_obs, training=False)
    dummy_critic_in   = tf.zeros((1,1, agent.critic.input_size), dtype=tf.float32)
    _                 = agent.critic(dummy_critic_in, training=False)

    # --- Load Sparse Masks from prev stage ---
    print(f"Loading masks from stage {stage-1}...")
    # Actor masks
    for i, layer in enumerate(agent.actor.hidden_layers):
        # 1) Kernel mask
        kp = os.path.join(prev_ckpt_dir, f'stage{stage-1}_actor_layer{i}_kernel.npy')
        if os.path.exists(kp) and layer.kernel_constraint is not None:
            mask_k = np.load(kp)
            # Si la capa tiene atributo `cell`, es un GRU (ModifiedGRU o GRU estándar)
            if hasattr(layer, 'cell'):
                dtype_w_in = layer.cell.W_in.dtype
            else:
                # Para Dense usamos el dtype del kernel directamente
                dtype_w_in = layer.kernel.dtype
            layer.kernel_constraint.mask = tf.constant(mask_k, dtype=dtype_w_in)

        # 2) Recurrent mask (sólo si la capa es recurrente)
        rp = os.path.join(prev_ckpt_dir, f'stage{stage-1}_actor_layer{i}_recur.npy')
        if os.path.exists(rp) and getattr(layer, 'recurrent_constraint', None) is not None:
            if hasattr(layer, 'cell'):
                mask_r = np.load(rp)
                layer.recurrent_constraint.mask = tf.constant(mask_r, dtype=layer.cell.W_rec.dtype)
            # Si `layer` es Dense, no existe W_rec y se omite
            
    # Critic masks
    for i, layer in enumerate(agent.critic.hidden_layers):
        # 1) Kernel mask
        kp = os.path.join(prev_ckpt_dir, f'stage{stage-1}_critic_layer{i}_kernel.npy')
        if os.path.exists(kp) and layer.kernel_constraint is not None:
            mask_k = np.load(kp)
            if hasattr(layer, 'cell'):
                dtype_w_in = layer.cell.W_in.dtype
            else:
                dtype_w_in = layer.kernel.dtype
            layer.kernel_constraint.mask = tf.constant(mask_k, dtype=dtype_w_in)

        # 2) Recurrent mask (sólo si la capa es recurrente)
        rp = os.path.join(prev_ckpt_dir, f'stage{stage-1}_critic_layer{i}_recur.npy')
        if os.path.exists(rp) and getattr(layer, 'recurrent_constraint', None) is not None:
            if hasattr(layer, 'cell'):
                mask_r = np.load(rp)
                layer.recurrent_constraint.mask = tf.constant(mask_r, dtype=layer.cell.W_rec.dtype)
            # Si `layer` es Dense, no existe W_rec y se omite
    print("Masks loaded.")

    # --- Dummy Step to Initialize Optimizers ---
    print("Initializing optimizers with dummy step...")

    # 1) Creamos dummy_obs para construir el Actor
    dummy_obs = np.zeros((1, 1, obs_size), dtype=np.float32)

    # 2) Obtenemos una “salida oculta” simulada del Actor con un state de ceros:
    processed_dummy_input = agent.actor.input_fc(dummy_obs, training=False)
    dummy_last_hidden_output = processed_dummy_input
    for hidden_layer in agent.actor.hidden_layers:
        if 'GRU' in agent.actor.layer_type:
            dummy_last_hidden_output, _ = hidden_layer(dummy_last_hidden_output, training=False)
        elif agent.actor.layer_type == 'Dense':
             dummy_last_hidden_output = hidden_layer(dummy_last_hidden_output, training=False)
    hd0 = dummy_last_hidden_output.numpy()[0, 0, :]

    # 3) Concatenamos con “curr_a” one-hot vacío para simular la entrada al Crítico
    dummy_curr_a0 = np.zeros((act_size,), dtype=np.float32)
    dummy_curr_a0[0] = 1.0
    critic_feat0 = np.concatenate([hd0, dummy_curr_a0], axis=0)
    dummy_critic_in = critic_feat0.reshape((1, 1, -1))

    # 4) Hacemos un dummy forward por el Crítico para inicializarlo
    with tf.GradientTape(persistent=True) as tape:
        a_out, _ = agent.actor(dummy_obs, training=True)
        c_out, _ = agent.critic(dummy_critic_in, training=True)
        loss_a   = tf.reduce_mean(tf.square(a_out))
        loss_c   = tf.reduce_mean(tf.square(c_out))
    grads_a = tape.gradient(loss_a, agent.actor.trainable_variables)
    grads_c = tape.gradient(loss_c, agent.critic.trainable_variables)
    agent.actor_optimizer.apply_gradients(zip(grads_a, agent.actor.trainable_variables))
    agent.critic_optimizer.apply_gradients(zip(grads_c, agent.critic.trainable_variables))
    del tape
    print("Optimizers initialized.")

    # --- Restore Checkpoint prev stage ---
    ckpt = tf.train.Checkpoint(
        actor=agent.actor,
        critic=agent.critic,
        actor_optimizer=agent.actor_optimizer,
        critic_optimizer=agent.critic_optimizer
    )
    manager = tf.train.CheckpointManager(ckpt, prev_ckpt_dir, max_to_keep=3)
    print(f"Restoring from checkpoint: {manager.latest_checkpoint}")
    status = ckpt.restore(manager.latest_checkpoint)
    status.assert_existing_objects_matched()
    print("Checkpoint restored successfully.")

    return this_ckpt_dir

# --------------------------
# Helper function: Save model
# --------------------------
def save_model(agent, stage, ckpt_prefix):
    """
    Guarda el checkpoint del modelo y las máscaras sparses para el stage actual.

    Args:
        agent: objeto que debe tener atributos
               - actor (con hidden_layers)
               - critic (con hidden_layers)
               - actor_optimizer
               - critic_optimizer
        stage: int, número de stage actual (p. ej. 6)
        ckpt_prefix: str, prefijo del directorio de checkpoints
                     (sin el sufijo "_<stage>")

    Returns:
        path: str, ruta al checkpoint guardado por tf.train.CheckpointManager.save()
    """
    this_ckpt_dir = f"{ckpt_prefix}_{stage}"
    os.makedirs(this_ckpt_dir, exist_ok=True)

    # --- Save Checkpoint Stage N ---
    ckpt = tf.train.Checkpoint(
        actor=agent.actor,
        critic=agent.critic,
        actor_optimizer=agent.actor_optimizer,
        critic_optimizer=agent.critic_optimizer
    )
    manager = tf.train.CheckpointManager(ckpt, this_ckpt_dir, max_to_keep=3)
    path = manager.save()
    print(f"Checkpoint stage {stage} saved at: {path}")

    # --- Save Sparse Masks Stage N ---
    print(f"Saving masks for stage {stage}...")
    # Actor masks
    for i, layer in enumerate(agent.actor.hidden_layers):
        if layer.kernel_constraint is not None and layer.kernel_constraint.mask is not None:
            np.save(os.path.join(this_ckpt_dir, f'stage{stage}_actor_layer{i}_kernel.npy'),
                    layer.kernel_constraint.mask.numpy())
        if getattr(layer, 'recurrent_constraint', None) is not None and layer.recurrent_constraint.mask is not None:
            np.save(os.path.join(this_ckpt_dir, f'stage{stage}_actor_layer{i}_recur.npy'),
                    layer.recurrent_constraint.mask.numpy())
    # Critic masks
    for i, layer in enumerate(agent.critic.hidden_layers):
        if layer.kernel_constraint is not None and layer.kernel_constraint.mask is not None:
            np.save(os.path.join(this_ckpt_dir, f'stage{stage}_critic_layer{i}_kernel.npy'),
                    layer.kernel_constraint.mask.numpy())
        if getattr(layer, 'recurrent_constraint', None) is not None and layer.recurrent_constraint.mask is not None:
            np.save(os.path.join(this_ckpt_dir, f'stage{stage}_critic_layer{i}_recur.npy'),
                    layer.recurrent_constraint.mask.numpy())
    print(f"Masks saved for stage {stage}.")

    return path

# --------------------------
# Helper function: Load model for validation
# --------------------------
def load_model_for_validation(agent, ckpt_dir):
    """
    Carga pesos del último checkpoint para validación.
    """
    ckpt = tf.train.Checkpoint(actor=agent.actor,
                               critic=agent.critic)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    latest = manager.latest_checkpoint
    if latest is None:
        raise FileNotFoundError(f"No se encontró checkpoint en {ckpt_dir}")
    ckpt.restore(latest).expect_partial()
    print(f"Modelo restaurado desde {latest}")
    return


# --------------------------
# Helper function: Evaluate agent
# --------------------------
def validate_agent(env, agent, num_episodes):
    """
    Validates the loaded agent on an environment without training.
    Collects rewards, measurements, and firing rates.

    Args:
        env: The potentially wrapped environment instance.
        agent: The ActorCriticAgent instance (already loaded with weights).
               Assumes agent.actor and agent.critic have a 'layer_type' attribute
               and potentially 'hidden_size', 'num_layers' attributes if recurrent.
               Assumes necessary methods/call signatures exist on agent.actor/critic.
        num_validation_episodes: Number of episodes to run for validation.

    Returns:
        Tuple containing:
            total_rewards_history (list): Total reward per episode.
            other_measurements_history (list): Measurements collected across episodes.
            actor_firing_rates (np.ndarray or None): Firing rates per layer for the actor,
                                                     shape (units, max_steps, episodes),
                                                     or None if actor uses Dense layers or collection fails.
            critic_firing_rates (np.ndarray or None): Firing rates per layer for the critic,
                                                      shape (units, max_steps, episodes),
                                                      or None if critic uses Dense layers or collection fails.
    """

    total_rewards_history = []
    other_measurements_history = []

    # Determine if networks are recurrent based on the stored layer_type
    actor_is_recurrent = hasattr(agent.actor, 'layer_type') and 'GRU' in agent.actor.layer_type
    critic_is_recurrent = hasattr(agent.critic, 'layer_type') and 'GRU' in agent.critic.layer_type
    act_size = env.action_space.n # Needed for critic input construction

    # Containers to store hidden states across all episodes (raw lists of numpy arrays)
    actor_states_all_raw = [] if actor_is_recurrent else None
    critic_states_all_raw = [] if critic_is_recurrent else None

    max_steps_actor = 0
    max_steps_critic = 0 # Initialize max steps for padding

    # Use a fixed print interval for validation for feedback during long runs
    # Print every 10% of episodes, or at least every 10 episodes.
    print_interval = max(10, num_episodes // 10)


    print(f"Starting validation for {num_episodes} episodes...")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        states_ep = [] # Stores s_t for t=0..T-1
        actions_ep = [] # Stores a_t for t=0..T-1
        rewards_ep = [] # Stores r_t for t=0..T-1
        episode_measurements = [] # Measurements collected *within* the episode
        actor_hidden_states_ep_raw = [] # Stores h_t (state before processing s_t)

        # Initialize hidden states at the start of each episode (h_0)
        # Assume num_layers attribute exists if recurrent, default to 1 if not found but layer_type is GRU
        actor_num_layers = getattr(agent.actor, 'num_layers', 1 if actor_is_recurrent else 0)
        critic_num_layers = getattr(agent.critic, 'num_layers', 1 if critic_is_recurrent else 0)

        current_actor_hidden = [tf.zeros((1, agent.actor.hidden_size), dtype=tf.float32)] * actor_num_layers if actor_is_recurrent else None
        current_critic_hidden = [tf.zeros((1, agent.critic.hidden_size), dtype=tf.float32)] * critic_num_layers if critic_is_recurrent else None

        # Loop for each step within an episode
        while not done:
            # At step t: 'state' is s_t. 'current_actor_hidden' is h_t (state after s_{t-1}).
            # We need to record h_t (state before processing s_t) for step t.
            if actor_is_recurrent:
                 # Store the state from the *beginning* of this time step (h_t)
                 # Assuming current_actor_hidden is a list of states per layer
                 # Flatten and stack states from all layers if multiple layers
                 if actor_num_layers > 1:
                     actor_state_t = np.concatenate([s[0].numpy().flatten() for s in current_actor_hidden])
                 else:
                     actor_state_t = current_actor_hidden[0].numpy().flatten()
                 actor_hidden_states_ep_raw.append(actor_state_t)


            # Agent selects action using the loaded policy (training=False)
            # select_action receives s_t and h_t, returns a_t and h_{t+1}
            action, _, actor_hidden_states_after_select = agent.select_action(
                state,
                actor_hidden_states=current_actor_hidden,
                training=False # Crucial for validation/inference mode
            )

            # Update actor hidden state for the next step (h_{t+1})
            if actor_is_recurrent:
                 current_actor_hidden = actor_hidden_states_after_select

            # Environment steps
            # env.step receives a_t, returns s_{t+1}, r_t, done, info
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store data for this step (corresponds to state s_t, action a_t, reward r_t)
            states_ep.append(state)       # s_t
            actions_ep.append(action)     # a_t
            rewards_ep.append(reward)     # r_t

            # Collect measurements (pass reward for current step r_t)
            # `collect_other_measurements` should handle extracting info from env state/info
            # and typically only returns non-None data when the episode ends.
            measurement = collect_other_measurements(env, done, reward)
            if measurement is not None:
                 episode_measurements.append(measurement)

            # s_{t+1} becomes the state for the next iteration
            state = next_state

        # --- End of Episode ---
        total_episode_reward = sum(rewards_ep)
        total_rewards_history.append(total_episode_reward)

        # Extend overall measurements list with episode-specific measurements
        # Assuming episode_measurements contains only non-None items collected during the episode
        other_measurements_history.extend(episode_measurements)

        # --- Process and Store Firing Rates for this Episode ---
        seq_len = len(states_ep) # Episode length T (number of steps/observations)

        # Actor Firing Rates (if recurrent)
        if actor_is_recurrent and actor_hidden_states_ep_raw:
            # actor_hidden_states_ep_raw contains [h_0, h_1, ..., h_T] where h_t is state before s_t
            # We need states h_t corresponding to states s_t (t from 0 to T-1).
            # This corresponds to [h_0, h_1, ..., h_{T-1}].
            # Transpose from (seq_len, total_units) to (total_units, seq_len)
            actor_states_this_ep = np.array(actor_hidden_states_ep_raw[:seq_len]).T
            actor_states_all_raw.append(actor_states_this_ep)
            max_steps_actor = max(max_steps_actor, seq_len)
        elif actor_is_recurrent and not actor_hidden_states_ep_raw:
             # This case should ideally not happen if num_validation_episodes > 0
             print(f"Warning: No actor hidden states collected for episode {episode}.")
             actor_states_all_raw = None # Indicate collection failure

        # Critic Firing Rates (if recurrent)
        if critic_is_recurrent:
            # Construct critic inputs sequence for the whole episode (length T)
            # Input at index t (0 to T-1) is [feat_t, a_{t-1}] to predict V(s_t).
            # feat_t is actor hidden state h_t (if actor recurrent) or observation s_t (if actor dense).
            critic_inputs_this_ep = []
            for t in range(seq_len):
                # feat_t is actor hidden state h_t (if actor recurrent) or observation s_t (if actor dense)
                if actor_is_recurrent:
                    # h_t is in actor_hidden_states_ep_raw[t] (state before processing s_t)
                    # Ensure actor_hidden_states_ep_raw was successfully populated
                    if actor_states_all_raw is None or t >= len(actor_hidden_states_ep_raw):
                         # This should not happen if actor states were collected correctly
                         print(f"Error: Missing actor state data for critic input at episode {episode}, step {t}.")
                         critic_states_all_raw = None # Indicate collection failure
                         break # Stop processing critic states for this episode

                    # feat = actor_hidden_states_ep_raw[t] # This is already flattened
                    feat = actor_hidden_states_ep_raw[t] # Use the recorded h_t for step t

                else: # Actor is Dense
                    # s_t is in states_ep[t]
                    feat = states_ep[t].flatten()

                # a_{t-1} is actions_ep[t-1] (one-hot), zero vector for t=0
                prev_a = np.zeros(act_size, dtype=np.float32)
                if t > 0:
                    prev_a[ actions_ep[t-1] ] = 1.0

                critic_inputs_this_ep.append(np.concatenate([feat, prev_a]))

            # Only proceed if critic_states_all_raw is not None (no error occurred)
            if critic_states_all_raw is not None:
                # Convert to tensor (batch size 1, sequence length T, feature size)
                critic_inputs_tensor = tf.convert_to_tensor([critic_inputs_this_ep], dtype=tf.float32)

                # Pass the sequence through the critic network in inference mode
                # Assumes agent.critic.call(input_sequence, ...) returns (rnn_output_sequence, final_state)
                try:
                     # This requires agent.critic's call method to handle sequential input
                     # and return the sequence of recurrent outputs as the first element.
                     # The shape should be (1, seq_len, units).
                     critic_rnn_outputs_seq, _ = agent.critic(critic_inputs_tensor, hidden_states=None, training=False) # hidden_states=None for sequence
                     # Take the sequence output (shape (1, seq_len, units)), remove batch dim, transpose
                     critic_states_this_ep = critic_rnn_outputs_seq[0].numpy().T # Shape (units, seq_len)
                     critic_states_all_raw.append(critic_states_this_ep)
                     max_steps_critic = max(max_steps_critic, seq_len)
                except Exception as e:
                     print(f"Warning: Could not get critic RNN outputs for episode {episode}. Error: {e}")
                     print("Ensure agent.critic's call method correctly handles sequential input and returns RNN output sequence as the first element.")
                     # Set critic_states_all_raw to None to indicate failure to collect
                     critic_states_all_raw = None # Stop collecting for subsequent episodes
                     # No need to set critic_is_recurrent = False, loop will just skip collection next time

        if episode % print_interval == 0:
            print(f"Validation Episode {episode}/{num_episodes}\tTotal Reward: {total_episode_reward:.2f}\tEpisode Length: {seq_len}")

    print("Validation episode loop finished.")

    # --- Final Processing of Firing Rates ---
    actor_firing_rates_processed = None
    # Process only if actor is recurrent and collection was successful and list is not empty
    if actor_is_recurrent and actor_states_all_raw and actor_states_all_raw[0] is not None:
        print("Processing actor firing rates...")
        try:
            # Units should be consistent across episodes. Get from the first collected episode.
            units_actor = actor_states_all_raw[0].shape[0]
            # Pad all episodes to max_steps_actor and stack them
            actor_padded = [pad_hidden_states(s, max_steps_actor, units_actor) for s in actor_states_all_raw]
            # Stack along a new dimension (episodes), then transpose to get (units, max_steps, episodes)
            actor_firing_rates_processed = np.array(actor_padded) # Shape (episodes, units, max_steps)
            actor_firing_rates_processed = np.transpose(actor_firing_rates_processed, (1, 2, 0)) # Shape (units, max_steps, episodes)
            print(f"Processed actor firing rates shape: {actor_firing_rates_processed.shape}")
        except Exception as e:
             print(f"Error processing actor firing rates: {e}")
             actor_firing_rates_processed = None


    critic_firing_rates_processed = None
    # Process only if critic is recurrent and collection was successful and list is not empty
    if critic_is_recurrent and critic_states_all_raw and critic_states_all_raw[0] is not None:
        print("Processing critic firing rates...")
        try:
            # Units should be consistent across episodes. Get from the first collected episode.
            units_critic = critic_states_all_raw[0].shape[0]
            # Pad all episodes to max_steps_critic and stack them
            critic_padded = [pad_hidden_states(s, max_steps_critic, units_critic) for s in critic_states_all_raw]
            # Stack along a new dimension (episodes), then transpose
            critic_firing_rates_processed = np.array(critic_padded) # Shape (episodes, units, max_steps)
            critic_firing_rates_processed = np.transpose(critic_firing_rates_processed, (1, 2, 0)) # Shape (units, max_steps, episodes)
            print(f"Processed critic firing rates shape: {critic_firing_rates_processed.shape}")
        except Exception as e:
             print(f"Error processing critic firing rates: {e}")
             critic_firing_rates_processed = None

    return total_rewards_history, other_measurements_history, actor_firing_rates_processed, critic_firing_rates_processed

# --------------------------
# Helper function: Metrics plot
# --------------------------
def plot_metrics(total_rewards, actor_losses, critic_losses, window_1=10, window_2=25):
    """
    Plots the raw metrics and their rolling statistics over windows of window_1 and window_2 episodes.
    Each column corresponds to one metric:
      - Column 1: Total Reward
      - Column 2: Actor Loss
      - Column 3: Critic Loss
    Row 1: Raw metrics.
    Row 2: Rolling mean, median, and std over a window of window_1 episodes.
    Row 3: Rolling mean, median, and std over a window of window_2 episodes.
    """
    def plot_rolling(ax, data, window, label):
        series = pd.Series(data)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_median = series.rolling(window=window, min_periods=1).median()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        ax.plot(episodes, rolling_mean, label="Mean")
        ax.plot(episodes, rolling_median, label="Median")
        ax.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                        alpha=0.2, label="Std")
        ax.set_title(f"Rolling (window={window}) {label}")
        ax.set_xlabel("Episode")
        ax.legend()

    episodes = range(1, len(total_rewards) + 1)
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    axs[0, 0].plot(episodes, total_rewards, label="Raw")
    axs[0, 0].set_title("Raw Total Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].legend()
    
    axs[0, 1].plot(episodes, actor_losses, label="Raw", color="tab:orange")
    axs[0, 1].set_title("Raw Actor Loss")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Actor Loss")
    axs[0, 1].legend()
    
    axs[0, 2].plot(episodes, critic_losses, label="Raw", color="tab:green")
    axs[0, 2].set_title("Raw Critic Loss")
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_ylabel("Critic Loss")
    axs[0, 2].legend()
    
    plot_rolling(axs[1, 0], total_rewards, window_1, "Total Reward")
    plot_rolling(axs[1, 1], actor_losses, window_1, "Actor Loss")
    plot_rolling(axs[1, 2], critic_losses, window_1, "Critic Loss")
    
    plot_rolling(axs[2, 0], total_rewards, window_2, "Total Reward")
    plot_rolling(axs[2, 1], actor_losses, window_2, "Actor Loss")
    plot_rolling(axs[2, 2], critic_losses, window_2, "Critic Loss")
    
    plt.tight_layout()
    plt.show()


def plot_metrics_batch(total_rewards, actor_losses, critic_losses, batch_size=1, window_1=10, window_2=25):
    """
    Plots the raw metrics and their rolling statistics.
    MODIFIED: Handles batch-averaged losses by stretching them to align with the episode axis.
    
    Args:
        total_rewards (list): Rewards collected per episode. Length is num_episodes.
        actor_losses (list): Actor losses. Length can be num_episodes (if batch_size=1) or num_batches.
        critic_losses (list): Critic losses. Length can be num_episodes (if batch_size=1) or num_batches.
        batch_size (int): The batch size used during training. Defaults to 1 for old behavior.
        window_1 (int): First rolling window size.
        window_2 (int): Second rolling window size.
    """
    def plot_rolling(ax, data, window, label):
        series = pd.Series(data)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_median = series.rolling(window=window, min_periods=1).median()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        ax.plot(episodes, rolling_mean, label="Mean")
        ax.plot(episodes, rolling_median, label="Median")
        ax.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                        alpha=0.2, label="Std")
        ax.set_title(f"Rolling (window={window}) {label}")
        ax.set_xlabel("Episode")
        ax.legend()

    num_episodes = len(total_rewards)
    episodes = range(1, num_episodes + 1)

    if batch_size > 1 and len(actor_losses) < num_episodes:
        actor_losses_aligned = np.repeat(actor_losses, batch_size)
        critic_losses_aligned = np.repeat(critic_losses, batch_size)
        actor_losses_aligned = actor_losses_aligned[:num_episodes]
        critic_losses_aligned = critic_losses_aligned[:num_episodes]
    else:
        actor_losses_aligned = actor_losses
        critic_losses_aligned = critic_losses
    
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    # --- Gráficos de Recompensa ---
    axs[0, 0].plot(episodes, total_rewards, label="Raw")
    axs[0, 0].set_title("Raw Total Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].legend()
    
    # --- Gráficos de Pérdidas (usan los datos alineados) ---
    axs[0, 1].plot(episodes, actor_losses_aligned, label="Raw (Batch Avg)", color="tab:orange")
    axs[0, 1].set_title("Raw Actor Loss")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Actor Loss")
    axs[0, 1].legend()
    
    axs[0, 2].plot(episodes, critic_losses_aligned, label="Raw (Batch Avg)", color="tab:green")
    axs[0, 2].set_title("Raw Critic Loss")
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_ylabel("Critic Loss")
    axs[0, 2].legend()
    
    # --- Gráficos de Rolling (usan los datos alineados) ---
    plot_rolling(axs[1, 0], total_rewards, window_1, "Total Reward")
    plot_rolling(axs[1, 1], actor_losses_aligned, window_1, "Actor Loss")
    plot_rolling(axs[1, 2], critic_losses_aligned, window_1, "Critic Loss")
    
    plot_rolling(axs[2, 0], total_rewards, window_2, "Total Reward")
    plot_rolling(axs[2, 1], actor_losses_aligned, window_2, "Actor Loss")
    plot_rolling(axs[2, 2], critic_losses_aligned, window_2, "Critic Loss")
    
    plt.tight_layout()
    plt.show()

# -------------------------
# Helper function: Firing rates plot
# --------------------------
def plot_firing_rates(actor_states_tensor, critic_states_tensor, network_name='Actor'):
    """
    Generates two plots:
    1. Mean firing rates as a function of the number of steps (averaged over units and valid episodes).
       Includes first and last episode firing rates.
    2. Mean firing rates as a function of units and episodes (averaged over valid steps only).

    Args:
        actor_states_tensor (np.ndarray): Tensor of shape (units, steps, episodes) for actor firing rates.
        critic_states_tensor (np.ndarray): Tensor of shape (units, steps, episodes) for critic firing rates.
        network_name (str): Name of the network to be displayed in the plot titles ('Actor' or 'Critic').
    """

    tensors = {'Actor': actor_states_tensor, 'Critic': critic_states_tensor}

    for name, tensor in tensors.items():
        if tensor is None:
            continue

        # Plot 1: Mean firing rates as a function of steps
        mean_over_steps = np.nanmean(tensor, axis=(0, 2))

        plt.figure(figsize=(10, 5))
        plt.plot(mean_over_steps, label='Mean firing rate')

        # Add first and last episodes
        first_episode = np.nanmean(tensor[:, :, 0], axis=0)
        last_episode = np.nanmean(tensor[:, :, -1], axis=0)

        plt.plot(first_episode, label='First episode', alpha=0.6)
        plt.plot(last_episode, label='Last episode', alpha=0.6)

        plt.xlabel('Steps')
        plt.ylabel('Mean firing rate')
        plt.title(f'{name} Mean firing rate over steps')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Mean firing rates as a function of units and episodes (valid steps only)
        mean_over_steps_units_episodes = np.nanmean(tensor, axis=1)

        plt.figure(figsize=(10, 6))
        plt.imshow(mean_over_steps_units_episodes, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Mean firing rate')
        plt.xlabel('Episodes')
        plt.ylabel('Units')
        plt.title(f'{name} Mean firing rate over units and episodes')
        plt.show()

# --------------------------
# Helper function: Discount reward
# --------------------------
def discount_rewards(rewards, gamma):
    """
    Computes discounted rewards.
    Args:
        rewards (list): List of rewards collected in an episode.
        gamma (float): Discount factor.
    Returns:
        np.array: Discounted rewards.
    """
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + gamma * cumulative
        discounted[i] = cumulative
    return discounted

# --------------------------
# Helper function: Padding for the hidden states
# --------------------------

def pad_hidden_states(hidden_states, max_steps, units):
    padded = np.full((units, max_steps), np.nan)
    length = hidden_states.shape[1]
    padded[:, :length] = hidden_states
    return padded

# --------------------------
# Helper function: Other measurements
# --------------------------

def collect_other_measurements(env, done, last_reward):
    """
    Collects environment-specific measurements at the end of an episode.

    Args:
        env: The environment instance.
        done (bool): Flag indicating if the episode ended.
        last_reward (float): The reward received in the final step.

    Returns:
        list or None: A list containing specific measurements for the target environment
                      if the episode ended successfully, otherwise None.
                      For EconomicChoiceEnv successful trials:
                      [juice_pair_LR, offer_pair_BA, chosen_juice_type]
                      e.g., [['A', 'B'], (3, 1), 'A'] means A was Left, B Right;
                            offer was 3 drops B vs 1 drop A; monkey chose A (Left).
    """
    env_instance = env.unwrapped if hasattr(env, 'unwrapped') else env

    # --- Partial CartPole Environment ---
    if isinstance(env_instance, CartPolePartialObservation):
        return None

    # --- Economic Choice Environment ---
    elif isinstance(env_instance, EconomicChoiceEnv) or \
        isinstance(env_instance, EconomicChoiceEnv_nohold) or \
        isinstance(env_instance, EconomicChoiceEnv_p):
        # Check for successful trial completion
        if done and last_reward >= env_instance.R_B:
            juice_pair = env_instance.trial_juice_LR
            offer_pair = env_instance.trial_offer_BA
            chosen_action = env_instance.chosen_action

            if chosen_action == 1:
                chosen_juice_type = juice_pair[0]
            elif chosen_action == 2:
                chosen_juice_type = juice_pair[1]
            else:
                chosen_juice_type = None

            if chosen_juice_type is not None and offer_pair is not None and juice_pair is not None:
                # Return juice assignment, offer amounts (B,A), and chosen type
                return [list(juice_pair), offer_pair, chosen_juice_type]
            else:
                return None
        else:
            # Episode did not end successfully with juice reward
            return None
        
    elif isinstance(env_instance, WorkingMemoryEnv) or isinstance(env_instance, WorkingMemoryEnv_mod):
        if done and env_instance.is_correct_choice is not None:
            params = env_instance.trial_params
            f_low, f_high = params['fpair']
            
            # Reconstruir f1 y f2
            if params['gt_lt'] == '>':
                f1, f2 = float(f_high), float(f_low)
            else:
                f1, f2 = float(f_low), float(f_high)

            return {
                "f1": f1,
                "f2": f2,
                "is_correct": env_instance.is_correct_choice
            }
        else:
            # Episode did not end successfully with reward
            return None

    else:
        # Not the target environment
        return None

# --------------------------
# Helper function: Psycometric curves
# --------------------------

def plot_psychometric_curve(measurements_list, title="Psychometric Curve"):
    """
    Plots the percentage of times Juice B was chosen for each unique B:A offer pair.
    """

    successful_measurements = [m for m in measurements_list if m is not None]

    measurements_df = pd.DataFrame(
            successful_measurements,
            columns=['Juice_Pair_LR', 'Offer_Pair_BA', 'Chosen_Juice']
        )
    
    if measurements_df is None or measurements_df.empty:
        print("Cannot plot psychometric curve: No successful trial data provided.")
        return

    if 'Offer_Pair_BA' not in measurements_df.columns or 'Chosen_Juice' not in measurements_df.columns:
        print("Cannot plot psychometric curve: DataFrame missing required columns ('Offer_Pair_BA', 'Chosen_Juice').")
        return

    # Calculate choice counts per offer type
    choice_counts = measurements_df.groupby('Offer_Pair_BA')['Chosen_Juice'].value_counts().unstack(fill_value=0)

    # Ensure both 'A' and 'B' columns exist, even if one wasn't chosen for some offers
    if 'A' not in choice_counts.columns: choice_counts['A'] = 0
    if 'B' not in choice_counts.columns: choice_counts['B'] = 0

    # Calculate total trials per offer type
    choice_counts['Total'] = choice_counts['A'] + choice_counts['B']

    # Calculate percentage of B choices
    choice_counts['P(Choose B)'] = (choice_counts['B'] / choice_counts['Total']) * 100

    def sort_key(offer_pair_tuple):
        nB, nA = offer_pair_tuple
        # Treat B:0 offers as having a very high relative B value (or infinite)
        if nA == 0:
            return np.inf
        # Sort by the ratio of B drops to A drops
        return nB / nA

    # --- Prepare data for plotting ---
    plot_data = choice_counts.sort_index(key=lambda idx: idx.map(sort_key))

    # Create meaningful labels for the x-axis
    x_labels = [f"{nb}:{na}" for nb, na in plot_data.index]

    # --- Create the plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, plot_data['P(Choose B)'], marker='o', linestyle='-')

    plt.xlabel("Offer Type (# Drops B : # Drops A)")
    plt.ylabel("Percentage Choice (%) - Chose B")
    plt.title(title)
    plt.ylim([-5, 105])
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

# --------------------------
# Helper function: Activity per neuron
# --------------------------
def activity_per_neuron(offers_list, firing_rates, first_steps=150, last_steps=150, n_rows=5, n_cols=10):
    """
    Plots mean firing rates per model unit for specified time windows across fixed-order offer types,
    using subplots for individual units, with individual Y-limits per subplot (min to max of that neuron's values).

    Parameters
    ----------
    offers_list : list
        List of length N_trials, each element like [[left_label, right_label], (left_drops, right_drops), choice],
        or None for invalid trials.
    firing_rates : np.ndarray
        Array of shape (n_units, max_timesteps, n_trials) with firing rates (NaN-padded for shorter episodes).
    first_steps : int, optional
        Number of time steps from episode start to average over (default=150).
    last_steps : int, optional
        Number of time steps from episode end to average over (default=150).
    n_rows : int, optional
        Number of subplot rows (default=5).
    n_cols : int, optional
        Number of subplot columns (default=10).

    Returns
    -------
    None
    """
    # Define the fixed offer order
    fixed_order = [(0, 1), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (10, 1), (2, 0)]
    cat_labels = [f"{a}:{b}" for a, b in fixed_order]
    n_cats = len(fixed_order)

    n_units, max_ts, n_trials = firing_rates.shape

    # Filter valid trials and map to A,B counts
    valid_trials = []
    trial_ratios = []
    for i, entry in enumerate(offers_list):
        if entry is None:
            continue
        labels, counts, _ = entry
        left_label, right_label = labels
        left_cnt, right_cnt = counts
        # Determine A_cnt, B_cnt regardless of side
        if left_label == 'A':
            A_cnt, B_cnt = left_cnt, right_cnt
        else:
            A_cnt, B_cnt = right_cnt, left_cnt
        valid_trials.append(i)
        trial_ratios.append((A_cnt, B_cnt))

    # Preallocate result arrays
    fr_start = np.zeros((n_units, n_cats))
    fr_end = np.zeros((n_units, n_cats))

    # Compute means per fixed category
    for ci, ratio in enumerate(fixed_order):
        # find trials matching this ratio exactly
        trial_idxs = [ti for ti, r in zip(valid_trials, trial_ratios) if r == ratio]
        if not trial_idxs:
            continue
        # Mean for first window
        fr1 = np.nanmean(firing_rates[:, :first_steps, trial_idxs], axis=(1, 2))
        # Mean for last window
        fr2_trials = []
        for ti in trial_idxs:
            valid_ts = ~np.all(np.isnan(firing_rates[:, :, ti]), axis=0)
            if not np.any(valid_ts):
                continue
            last_idx = np.where(valid_ts)[0].max()
            start_idx = max(0, last_idx - last_steps + 1)
            fr2_trials.append(np.nanmean(firing_rates[:, start_idx:last_idx+1, ti], axis=1))
        fr2 = np.nanmean(np.stack(fr2_trials, axis=1), axis=1) if fr2_trials else np.zeros(n_units)
        # Store
        fr_start[:, ci] = fr1
        fr_end[:, ci] = fr2

    # Create subplots for all units
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*1.8), sharex=True, sharey=False)
    axes = axes.flatten()
    x = np.arange(n_cats)

    for u in range(n_units):
        ax = axes[u]
        y1 = fr_start[u]
        y2 = fr_end[u]
        ax.plot(x, y2, marker='o', linestyle='-', alpha=0.7, linewidth=0.8)
        ax.plot(x, y1, marker='o', linestyle='-', color='grey', alpha=0.7, linewidth=0.8)
        ax.set_title(f'Neuron {u+1}', fontsize=6)
        # Determine individual y-limits
        combined = np.concatenate([y1, y2])
        finite = combined[np.isfinite(combined)]
        if finite.size > 0:
            ymin, ymax = finite.min(), finite.max()
            if np.isclose(ymin, ymax):
                delta = ymin * 0.1 if ymin != 0 else 1
                ymin -= delta
                ymax += delta
            ax.set_ylim(ymin, ymax)
        # X axis ticks
        if u // n_cols == n_rows - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(cat_labels, rotation=45, fontsize=6)
        else:
            ax.set_xticks([])
        # Y-axis label on first column
        if u % n_cols == 0:
            ax.set_ylabel('FR')

    # Remove unused axes
    for idx in range(n_units, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.suptitle('Firing Rates: primeros vs últimos time steps por neurona', y=1.02)
    plt.show()

# --------------------------
# Helper function: Romo plot
# --------------------------
def plot_performance_matrix(all_measurements: list):
    """
    Procesa una lista de resultados de ensayos y genera un gráfico de rendimiento.

    Args:
        all_measurements (list): Una lista de diccionarios, donde cada diccionario
                                 representa un ensayo y contiene las claves 'f1', 'f2',
                                 y 'is_correct'.
    """
    # Filtramos los None (episodios que han sido abortados)
    all_measurements = [m for m in all_measurements if m is not None]

    # Verificación de datos
    if not all_measurements:
        print("La lista de mediciones está vacía. No se puede generar el gráfico.")
        return

    # Procesamiento de datos
    df = pd.DataFrame(all_measurements)

    # Agrupar por cada par (f1, f2) y calcular la media de 'is_correct'.
    performance = df.groupby(['f1', 'f2'])['is_correct'].mean().reset_index()
    performance['percentage'] = (performance['is_correct'] * 100).round(1).astype(int)
    print("\nPorcentaje de acierto calculado por condición:")
    print(performance)

    # Visualización
    fig, ax = plt.subplots(figsize=(8, 6.5))

    vmin = 25
    vmax = 100
    cmap = plt.get_cmap('jet')
    norm = Normalize(vmin=vmin, vmax=vmax)

    scatter = ax.scatter(
        x=performance['f1'],
        y=performance['f2'],
        s=400,
        marker='s',
        c=performance['percentage'],
        cmap=cmap,
        norm=norm,
        edgecolor='black'
    )

    ax.set_xlabel('$f_1$ (Hz)', fontsize=14)
    ax.set_ylabel('$f_2$ (Hz)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 45)
    ax.set_aspect('equal', adjustable='box')
    # Línea diagonal f1 = f2
    line_range = [0, 45]
    ax.plot(line_range, line_range, 'k--', linewidth=1.5, label='$f_1 = f_2$')
    ax.text(35, 37, '$f_1 = f_2$', rotation=45, ha='center', va='center', fontsize=12)
    # Añadir la barra de colores
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('performance', fontsize=12, rotation=270, labelpad=20)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# --------------------------
# Helper function: Mean firing rate in Romo
# --------------------------
def plot_mean_firing_rate_trace(firing_rates, dt=10, title="Mean Firing Rate Trace (All Neurons)"):
    """
    Grafica la tasa de disparo media de todas las neuronas a lo largo del tiempo.

    Muestra una línea para la media a través de los episodios y una zona sombreada
    para la desviación estándar.

    Args:
        firing_rates (np.ndarray): Tensor de tasas de disparo con forma
                                   (units, steps, episodes).
        dt (int): Duración de un time step en ms, para el eje x.
        title (str): Título del gráfico.
    """
    if firing_rates is None or firing_rates.size == 0:
        print("No firing rate data to plot.")
        return

    # 1. Calcular la media a través de las neuronas para cada episodio
    mean_rate_per_episode = np.nanmean(firing_rates, axis=0)

    # 2. Calcular la media y la desviación estándar a través de los episodios
    mean_trace = np.nanmean(mean_rate_per_episode, axis=1)
    std_trace = np.nanstd(mean_rate_per_episode, axis=1)

    # Crear el eje de tiempo en segundos
    time_steps = np.arange(mean_trace.shape[0])
    time_sec = time_steps * (dt / 1000.0)

    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(time_sec, mean_trace, label="Mean Firing Rate")
    plt.fill_between(
        time_sec,
        mean_trace - std_trace,
        mean_trace + std_trace,
        alpha=0.2,
        label="Std. Dev."
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Mean Firing Rate (a.u.)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ------------------------------------
# Helper function: Sesgo de contracción
#-------------------------------------
def sesgo_contraccion(all_measurements: list):
    """
    Procesa los resultados de los ensayos y genera un gráfico de rendimiento en forma de V.
    La búsqueda de clases ahora es agnóstica al orden de f1 y f2.
    """
    CLASS_NUMBER_MAP = {
        (26, 34): 1, (22, 30): 2, (18, 26): 3, (14, 22): 4, (10, 18): 5,
        (34, 26): 6, (30, 22): 7, (26, 18): 8, (22, 14): 9, (18, 10): 10    
    }

    valid_measurements = [m for m in all_measurements if m is not None]
    if not valid_measurements:
        print("No hay mediciones válidas para generar el gráfico.")
        return

    df = pd.DataFrame(valid_measurements)

    def find_class_number(row):
        # Convertir a enteros para una comparación segura
        f1_int = int(row['f1'])
        f2_int = int(row['f2'])
        
        # Crear la tupla en el orden original y en el orden inverso
        pair_forward = (f1_int, f2_int)
        pair_reversed = (f2_int, f1_int)
        
        # Intentar buscar la tupla original. Si no, la inversa.
        return CLASS_NUMBER_MAP.get(pair_forward, CLASS_NUMBER_MAP.get(pair_reversed))

    df['class_number'] = df.apply(find_class_number, axis=1)

    df.dropna(subset=['class_number'], inplace=True)
    df['class_number'] = df['class_number'].astype(int)
    
    performance = df.groupby('class_number')['is_correct'].agg(['mean', 'sem']).reset_index()
    performance['mean_perc'] = performance['mean'] * 100
    performance['sem_perc'] = performance['sem'] * 100
    performance = performance.sort_values('class_number')

    print("\nRendimiento por Número de Clase:")
    print(performance)
    
    # Visualización
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x=performance['class_number'], y=performance['mean_perc'],
                yerr=performance['sem_perc'], marker='o', linestyle='-',
                capsize=4, label='Overall Performance')
    ax.set_xlabel("Class number", fontsize=14, color='coral')
    ax.set_ylabel("Performance (%)", fontsize=14)
    ax.set_title("Performance by Stimulus Class", fontsize=16)
    ax.set_xticks(np.arange(1, 13, 2))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(20, 101)
    ax.set_yticks(np.arange(20, 101, 10))
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --------------------
# Actividad por clase para f1 y f2
# --------------------
def plot_activity_by_stimulus_class(all_measurements, firing_rates, f1_timesteps, f2_timesteps, n_rows=10, n_cols=15):
    """
    Genera dos gráficos que muestran la tasa de disparo media de cada neurona en función
    de la clase de estímulo. Un gráfico corresponde al período de estímulo f1 y el
    otro al período f2.

    Args:
        all_measurements (list): Lista de mediciones de los ensayos, donde cada elemento
                                 es un diccionario que contiene 'f1', 'f2'. Los elementos
                                 pueden ser None para ensayos abortados.
        firing_rates (np.ndarray): Tensor de tasas de disparo con forma (unidades, pasos, episodios).
        f1_timesteps (list or range): Índices de tiempo correspondientes a la época f1.
        f2_timesteps (list or range): Índices de tiempo correspondientes a la época f2.
        n_rows (int): Número de filas de subtramas en la figura.
        n_cols (int): Número de columnas de subtramas en la figura.
    """
    # Mapa para clasificar cada par (f1, f2) en una clase numérica.
    CLASS_NUMBER_MAP = {
        (26, 34): 1, (22, 30): 2, (18, 26): 3, (14, 22): 4, (10, 18): 5,
        (34, 26): 6, (30, 22): 7, (26, 18): 8, (22, 14): 9, (18, 10): 10
    }
    
    # 1. Procesar mediciones para mapear cada ensayo a una clase de estímulo
    valid_measurements = [(i, m) for i, m in enumerate(all_measurements) if m is not None]
    if not valid_measurements:
        print("No hay mediciones válidas para generar los gráficos.")
        return

    trial_indices_map = {}
    for class_num in range(1, 11):
        trial_indices_map[class_num] = []

    for trial_idx, measurement in valid_measurements:
        f1 = int(measurement['f1'])
        f2 = int(measurement['f2'])
        pair = (f1, f2)
        
        class_num = CLASS_NUMBER_MAP.get(pair)
        if class_num is not None:
            trial_indices_map[class_num].append(trial_idx)

    # 2. Calcular la actividad media para cada neurona y clase de estímulo
    n_units, _, _ = firing_rates.shape
    n_classes = len(CLASS_NUMBER_MAP) // 2 * 2 # 10 clases
    
    mean_activity_f1 = np.full((n_units, n_classes + 1), np.nan)
    mean_activity_f2 = np.full((n_units, n_classes + 1), np.nan)

    for class_num, indices in trial_indices_map.items():
        if not indices:
            continue
        
        # Actividad media durante el período F1 para los ensayos de esta clase
        activity_slice_f1 = firing_rates[:, f1_timesteps, :][:, :, indices]
        mean_activity_f1[:, class_num] = np.nanmean(activity_slice_f1, axis=(1, 2))
        
        # Actividad media durante el período F2 para los ensayos de esta clase
        activity_slice_f2 = firing_rates[:, f2_timesteps, :][:, :, indices]
        mean_activity_f2[:, class_num] = np.nanmean(activity_slice_f2, axis=(1, 2))

    # 3. Generar los gráficos
    x_axis = np.arange(1, n_classes + 1)
    
    # Gráfico para el período F1
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.2), sharex=True)
    fig1.suptitle('Actividad Neuronal Media durante F1 vs. Clase de Estímulo', fontsize=16, y=1.02)
    axes1 = axes1.flatten()
    for u in range(n_units):
        ax = axes1[u]
        ax.plot(x_axis, mean_activity_f1[u, 1:], marker='o', linestyle='-', markersize=4)
        ax.set_title(f'Neurona {u+1}', fontsize=8)
        if u % n_cols == 0:
            ax.set_ylabel('FR (a.u.)', fontsize=8)
        if u >= (n_units - n_cols):
            ax.set_xlabel('Clase', fontsize=8)
            ax.set_xticks(x_axis)
    # Ocultar ejes no utilizados
    for i in range(n_units, len(axes1)): fig1.delaxes(axes1[i])
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

    # Gráfico para el período F2
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.2), sharex=True)
    fig2.suptitle('Actividad Neuronal Media durante F2 vs. Clase de Estímulo', fontsize=16, y=1.02)
    axes2 = axes2.flatten()
    for u in range(n_units):
        ax = axes2[u]
        ax.plot(x_axis, mean_activity_f2[u, 1:], marker='o', linestyle='-', markersize=4, color='coral')
        ax.set_title(f'Neurona {u+1}', fontsize=8)
        if u % n_cols == 0:
            ax.set_ylabel('FR (a.u.)', fontsize=8)
        if u >= (n_units - n_cols):
            ax.set_xlabel('Clase', fontsize=8)
            ax.set_xticks(x_axis)

    for i in range(n_units, len(axes2)): fig2.delaxes(axes2[i])
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()