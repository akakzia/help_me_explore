import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.replay_buffer import ReplayBuffer
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from updates import update_gnns, update_gnns_vds
from utils import hard_update


"""
SAC with HER (MPI-version)
"""

class RLAgent:
    def __init__(self, args, compute_rew, goal_sampler):

        self.args = args
        self.alpha = args.alpha
        self.env_params = args.env_params

        self.goal_sampler = goal_sampler

        self.total_iter = 0

        self.freq_target_update = args.freq_target_update

        # create the network
        self.architecture = self.args.architecture

        if self.args.agent == 'VDSAgent':
            from rl_modules.rn_vds_modules import RnSemantic
            self.model = RnSemantic(self.env_params, args)
        elif self.architecture == 'flat':
            from rl_modules.flat_models import FlatSemantic
            self.model = FlatSemantic(self.env_params)
        elif self.architecture == 'interaction_network':
            from rl_modules.interaction_models import InSemantic
            self.model = InSemantic(self.env_params, args)
        elif self.architecture == 'full_gn':
            from rl_modules.gn_models import GnSemantic
            self.model = GnSemantic(self.env_params, args)
        elif self.architecture == 'relation_network':
            from rl_modules.rn_models import RnSemantic
            self.model = RnSemantic(self.env_params, args)
        elif self.architecture == 'deep_sets':
            from rl_modules.deepsets_models import DsSemantic
            self.model = DsSemantic(self.env_params, args)
        else:
            raise NotImplementedError

        # if use GPU
        if self.args.cuda:
            self.model.actor.cuda()
            self.model.critic.cuda()
            self.model.critic_target.cuda()
            self.model.value_network.cuda()
            if self.args.agent == 'VDSAgent':
                self.model.vds_q_values.cuda()
                self.model.vds_target_q_values.cuda()
        # sync the networks across the CPUs
        sync_networks(self.model.critic)
        sync_networks(self.model.actor)
        hard_update(self.model.critic_target, self.model.critic)
        sync_networks(self.model.critic_target)
        sync_networks(self.model.value_network)
        if self.args.agent == 'VDSAgent':
            sync_networks(self.model.vds_q_values)
            hard_update(self.model.vds_target_q_values, self.model.vds_q_values)

        # create the optimizer
        self.policy_optim = torch.optim.Adam(list(self.model.actor.parameters()), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(list(self.model.critic.parameters()), lr=self.args.lr_critic)
        self.value_optim = torch.optim.Adam(list(self.model.value_network.parameters()), lr=self.args.lr_critic)
        if self.args.agent == 'VDSAgent':
            self.vds_optim = torch.optim.Adam(list(self.model.vds_q_values.parameters()),
                                             lr=self.args.lr_critic)
        
        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)

        # Target Entropy
        if self.args.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.env_params['action'])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.lr_entropy)

        # her sampler
        self.her_module = her_sampler(self.args, compute_rew)

        # create the replay buffer
        self.buffer = ReplayBuffer(env_params=self.env_params,
                                  buffer_size=self.args.buffer_size,
                                  sample_func=self.her_module.sample_her_transitions,
                                  goal_sampler=self.goal_sampler
        )

        # Create social replay buffer for social episodes
        # create the replay buffer
        self.social_buffer = ReplayBuffer(env_params=self.env_params,
                                        buffer_size=self.args.buffer_size,
                                        sample_func=self.her_module.sample_her_transitions,
                                        goal_sampler=self.goal_sampler
        )

    def act(self, obs, ag, g, no_noise):
        """ Use the policy model to perform a forward pass and sample action """
        with torch.no_grad():
            # normalize policy inputs
            obs_norm = self.o_norm.normalize(obs)
            ag_tensor = torch.tensor(ag, dtype=torch.float32).unsqueeze(0)
            g_tensor = torch.tensor(g, dtype=torch.float32).unsqueeze(0)

            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                obs_tensor = obs_tensor.cuda()
                g_tensor = g_tensor.cuda()
                ag_tensor = ag_tensor.cuda()
            self.model.policy_forward_pass(obs_tensor, ag_tensor, g_tensor, no_noise=no_noise)
            if self.args.cuda:
                action = self.model.pi_tensor.cpu().numpy()[0]
            else:
                action = self.model.pi_tensor.numpy()[0]
                
        return action.copy()
    
    def store(self, episodes, episodes_type):
        """ Store episodes in the corresponding buffer according to type (either social or individual) """
        if episodes_type == 'autotelic':
            self.buffer.store_episode(episode_batch=episodes)
        else:
            self.social_buffer.store_episode(episode_batch=episodes)

    def train(self):
        """ Train the network """
        self.total_iter += 1
        self._update_network()

        # soft update
        if self.total_iter % self.freq_target_update == 0:
            self._soft_update_target_network(self.model.critic_target, self.model.critic)

    def _update_normalizer(self, episode):
        """ Update the normalizer
        Input: Most recent episode """
        mb_obs = episode['obs']
        mb_ag = episode['ag']
        mb_g = episode['g']
        mb_actions = episode['act']
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0]
        # create the new buffer to store them
        buffer_temp = {'obs': np.expand_dims(mb_obs, 0),
                       'ag': np.expand_dims(mb_ag, 0),
                       'g': np.expand_dims(mb_g, 0),
                       'actions': np.expand_dims(mb_actions, 0),
                       'obs_next': np.expand_dims(mb_obs_next, 0),
                       'ag_next': np.expand_dims(mb_ag_next, 0),
                       }

        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        # recompute the stats
        self.o_norm.recompute_stats()

    def _preproc_og(self, o, g):
        """ Preprocesses observations and goals to avoid collapse """
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def _soft_update_target_network(self, target, source):
        """ Perform soft update using target networks and polyak hyparameter """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _update_network(self):
        """ Update the considered networks """
        # sample from buffer, this is done with LP is multi-head is true
        if np.random.uniform() < self.goal_sampler.query_proba and self.social_buffer.current_size > 0:
            transitions = self.social_buffer.sample(self.args.batch_size)
        else:
            transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g, ag, ag_next, actions, rewards = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag'], \
                                                      transitions['ag_next'], transitions['actions'], transitions['r']
        anchor_g = transitions['anchor_g']

        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['ag'] = self._preproc_og(o, ag)
        _, transitions['ag_next'] = self._preproc_og(o, ag_next)
        _, transitions['anchor_g'] = self._preproc_og(o, anchor_g)

        anchor_rewards = transitions['anchor_r']
        final_rewards = transitions['final_reward']

        # apply normalization
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = transitions['g']
        ag_norm = transitions['ag']
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = transitions['ag_next']

        anchor_g_norm = transitions['anchor_g']

        if self.args.agent == 'VDSAgent':
            update_gnns_vds(self.model, self.policy_optim, self.critic_optim, self.value_optim, self.alpha, self.log_alpha, self.target_entropy, 
                        self.alpha_optim, self.vds_optim, obs_norm, ag_norm, g_norm, anchor_g_norm, obs_next_norm, ag_next_norm, actions, rewards, anchor_rewards,
                        final_rewards, self.args)
        else:
            update_gnns(self.model, self.policy_optim, self.critic_optim, self.value_optim, self.alpha, self.log_alpha, self.target_entropy, 
                        self.alpha_optim, obs_norm, ag_norm, g_norm, anchor_g_norm, obs_next_norm, ag_next_norm, actions, rewards, anchor_rewards,
                        final_rewards, self.args)

    def save(self, model_path, epoch):
        """ Save model """
        torch.save([self.o_norm.mean, self.o_norm.std,
                    self.model.actor.state_dict(), self.model.critic.state_dict(), self.model.value_network.state_dict()],
                    model_path + '/model_{}.pt'.format(epoch))

    def load(self, model_path, args):
        """ Load model """
        o_mean, o_std, actor, critic, value_network = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.actor.load_state_dict(actor)
        self.model.critic.load_state_dict(critic)
        self.model.value_network.load_state_dict(value_network)
        self.model.actor.eval()
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
