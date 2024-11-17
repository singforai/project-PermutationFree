import torch
import itertools
from algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space,  num_agents, device=torch.device("cpu")):
        self.device = device
        self.lr = args["lr"]
        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        
        self.hidden_size: int = args["hidden_size"]
        self.check_permutation_free = args["check_permutation_free"]    

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        
        self.threadshold = 1e-7
        self.num_agents = num_agents
        self.action_dim = act_space.n

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """

        if self.check_permutation_free:
            self.is_permutation_free(obs, cent_obs, rnn_states_actor, rnn_states_critic, masks, available_actions)
            
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic
        )

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, obs, rnn_states, rnn_states_critic, masks, available_actions=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


    def is_permutation_free(self, obs, cent_obs, rnn_states_actor, rnn_states_critic, masks, available_actions):
        
        obs_input_dim = obs.shape[1]
        cent_obs_input_dim = cent_obs.shape[1]
        obs = obs.reshape(-1, self.num_agents, obs_input_dim)
        cent_obs = cent_obs.reshape(-1, self.num_agents, cent_obs_input_dim)
        rnn_states_actor = rnn_states_actor.reshape(-1, self.num_agents, 1, self.hidden_size)
        rnn_states_critic = rnn_states_critic.reshape(-1, self.num_agents, 1, self.hidden_size) 
        masks = masks.reshape(-1, self.num_agents, 1)
        available_actions = available_actions.reshape(-1, self.num_agents, 1, self.action_dim)
        
        possible_permutations = list(itertools.permutations(range(self.num_agents)))
        
        actions_set = []
        rnn_states_actor_output_set = []
        rnn_states_critic_output_set = []
        
        values_set = []
        
        for possible_permutation in possible_permutations:
            permuted_obs = obs[:, possible_permutation, :].reshape(-1, obs_input_dim)
            permuted_cent_obs = cent_obs[:, possible_permutation, :].reshape(-1, cent_obs_input_dim) 
            permuted_rnn_states_actor = rnn_states_actor[:, possible_permutation, :, :].reshape(-1, 1, self.hidden_size)
            permuted_rnn_states_critic = rnn_states_critic[:, possible_permutation, :, :].reshape(-1, 1, self.hidden_size)  
            permuted_masks = masks[:, possible_permutation, :].reshape(-1, 1)
            permuted_available_actions = available_actions[:, possible_permutation, :, :].reshape(-1, self.action_dim)
            
            actions, _, rnn_states_actor_output = self.actor(
                permuted_obs,
                permuted_rnn_states_actor,
                permuted_masks,
                permuted_available_actions,
                deterministic = True
            )

            values, rnn_states_critic_output = self.critic(permuted_cent_obs, permuted_rnn_states_critic, permuted_masks)
            
            reverse_permutation = [0] * len(possible_permutation)
            for i, p in enumerate(possible_permutation):
                reverse_permutation[p] = i
                
            actions = actions.reshape(
                -1, self.num_agents, 1
            )[:, reverse_permutation, :].reshape(-1)
            
            rnn_states_actor_output = rnn_states_actor_output.reshape(
                -1, self.num_agents, 1, self.hidden_size
            )[:, reverse_permutation, :, :].reshape(-1)
            
            rnn_states_critic_output = rnn_states_critic_output.reshape(
                -1, self.num_agents, 1, self.hidden_size
            )[:, reverse_permutation, :, :].reshape(-1)
                    
            actions_set.append(actions.reshape(-1))
            rnn_states_actor_output_set.append(rnn_states_actor_output.reshape(-1))
            rnn_states_critic_output_set.append(rnn_states_critic_output_set.reshape(-1))
            
            values_set.append(values.reshape(-1))
            
        
        actions_set = torch.stack(actions_set)
        values_set = torch.stack(values_set)
        rnn_states_actor_output_set = torch.stack(rnn_states_actor_output_set)
        rnn_states_critic_output_set = torch.stack(rnn_states_critic_output_set)
        
        if torch.allclose(actions_set[0], actions_set, atol=self.threadshold) and \
            torch.allclose(rnn_states_actor_output_set[0], rnn_states_actor_output_set, atol=self.threadshold) and \
            torch.allclose(rnn_states_actor_output_set[0], rnn_states_actor_output_set, atol=self.threadshold):
            print("Model is Permutation equivariant!")
        else:
            print("Model is not Permutation equivariant!")

        if torch.allclose(values_set[0], values_set, atol=self.threadshold):
            print("Model is Permutation invariant!")
        else:
            print("Model is not Permutation invariant!")