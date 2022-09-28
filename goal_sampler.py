from dis import dis
import torch
import numpy as np
from utils import generate_stacks_to_class, get_eval_goals
from utils import INSTRUCTIONS
from mpi4py import MPI
from goal_evaluator import GoalEvaluator
import pickle


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.agent = args.agent

        self.n_blocks = args.n_blocks
        self.goal_dim = args.env_params['goal']

        self.args = args
        # Keep track of the number of discovered goals
        self.nb_discovered_goals = 0

        # Define lists to store discovered goals as arrays and as strings
        self.discovered_goals = []
        self.discovered_goals_str = []
        self.discovered_goals_oracle_ids = []

        # Define mapping dict between goals and indexes of discovery
        self.goal_to_id = {}
        # self.id_to_goal = {}

        # Define list of counts of goal visits. Indexes of the list are indexes of goals. Values are the visits
        self.visits = []

        # Initialize value estimations list
        self.values_goals = []

        # Query arguments
        self.fixed_queries = args.fixed_queries
        self.query_proba = args.fixed_query_proba if self.fixed_queries else 0.
        self.min_queue_length = args.min_queue_length 
        self.max_queue_length = args.max_queue_length
        self.beta = args.beta
        self.progress_function = args.progress_function

        # Initialize goal_evaluator
        self.goal_evaluator = GoalEvaluator(args)

        # Cycle counter
        self.n_cycles = 0

        self.stacks_to_class = generate_stacks_to_class()
        self.discovered_goals_per_stacks = {e:0 for e in set(self.stacks_to_class.values())}
        self.discovered_goals_per_stacks['others'] = 0 # for goals that are not in the stack classes

        self.use_stability_condition = args.use_stability_condition

        self.ss_b_pairs = []
        self.beyond = []

        self.init_stats()
    
    def setup_policy(self, policy):
        """ Sets up the policy """
        self.goal_evaluator.setup_policy(policy)

    def sample_goals(self, n_goals=1, evaluation=False):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation:
            goals = []
            for instruction in INSTRUCTIONS:
                goal = get_eval_goals(instruction, n=self.n_blocks)
                goals.append(goal.squeeze(0))
            goals = np.array(goals)
        else:
            if len(self.discovered_goals) == 0:
                goals = - np.ones((n_goals, self.goal_dim))
            else:
                # sample uniformly from discovered goals
                goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                goals = np.array(self.discovered_goals)[goal_ids]
        return goals
    
    def sample_rare_goal(self, n=50):
        """ Samples goals in sparsely explored areas of the goal space (goals that are the least visited) """
        visits_buffer = np.array(self.visits)
        if len(visits_buffer) == 0:
            return tuple(- np.ones(self.goal_dim))
        goal_id = np.random.choice(np.argsort(visits_buffer)[:n])
        goal = self.discovered_goals[goal_id]

        return tuple(goal)
    
    def sample_intermediate_complexity_goal(self, n=50):
        """ Samples goals based on their goal achievement value """
        if len(self.values_goals) == 0:
            return tuple(- np.ones(self.goal_dim))
        
        last_values = self.values_goals[-1].squeeze()
        goal_id = np.random.choice(np.argsort(last_values)[:n])
        goal = self.discovered_goals[goal_id]

        return tuple(goal)

    def sample_lp_goals(self, n_goals=1, n=50):
        """ Samples goals based on their goal achievement value """
        if len(self.values_goals) == 0:
            return - np.ones((n_goals, self.goal_dim))
        
        last_values = self.values_goals[-1].squeeze()
        goal_ids = np.random.choice(np.argsort(last_values)[:n], size=n_goals)
        goals = np.array(self.discovered_goals)[goal_ids]

        return goals
    
    def sample_vds_goals(self, initial_obs, n_goals=1):
        """ Sample goals based on value disagreement 
        We need initial observation to perform forward pass in Q networks """
        n = min(1000, len(self.discovered_goals))
        goals = self.sample_goals(n_goals=n)
        observations = np.repeat(np.expand_dims(initial_obs['observation'], axis=0), n, axis=0)
        ag = np.repeat(np.expand_dims(initial_obs['achieved_goal'], axis=0), n, axis=0)

        obs_norm = self.goal_evaluator.policy.o_norm.normalize(observations)

        obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32)
        g_tensor = torch.tensor(goals, dtype=torch.float32)
        ag_tensor = torch.tensor(ag, dtype=torch.float32)
        if self.args.cuda:
            obs_norm_tensor = obs_norm_tensor.cuda()
            g_tensor = g_tensor.cuda()
            ag_tensor = ag_tensor.cuda()
        
        with torch.no_grad():
            self.goal_evaluator.policy.model.forward_pass(obs_norm_tensor, ag_tensor, g_tensor)
            qf1_vd, qf2_vd, qf3_vd = self.goal_evaluator.policy.model.target_q1_vd_tensor.cpu().numpy(), self.goal_evaluator.policy.model.target_q2_vd_tensor.cpu().numpy(),\
                                        self.goal_evaluator.policy.model.target_q3_vd_tensor.cpu().numpy()
            qs = np.concatenate([qf1_vd, qf2_vd, qf3_vd], axis=1)
            scores = np.std(qs, axis=1)**2
            normalized_scores = scores / np.sum(scores)
            selected_goals = goals[np.random.choice(np.arange(n), p=normalized_scores, size=n_goals)]
        
        return selected_goals

    def sample_uniform_goal(self):
        """ Samples goals based on their goal achievement value """
        goal_id = np.random.choice(range(len(self.discovered_goals)))
        goal = self.discovered_goals[goal_id]

        return tuple(goal)
        
    def update(self, episodes):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        # Update the goal memory
        episodes = self.update_goal_memory(episodes)

        # update goal estimations
        if self.rank == 0.:
            # Compute goal values
            norm_values = self.goal_evaluator.estimate_goal_value(goals=np.array(self.discovered_goals)).reshape(len(self.discovered_goals), -1)
            self.values_goals.append(norm_values)
        
        self.sync_queries()
        return episodes

    def update_goal_memory(self, episodes):
        """ Given a batch of episodes, gathered from all workers, updates:
        1. the list of discovered goals (arrays and strings)
        2. the list of discovered goals' ids
        3. the number of discovered goals
        4. the bidict oracle id <-> goal str """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                # Retrive last achieved goal
                last_ag = e['ag'][-1]
                # Retrieve goal
                goal = e['g'][-1]
                if self.use_stability_condition:
                    # Compute boolean conditions to determine the discovered goal stability 
                    # the goal is stable for the last 10 steps
                    condition_stability = np.sum([str(last_ag) == str(el) for el in e['ag'][-10:]]) == 10.
                else:
                    # Always true
                    condition_stability = True
                # Add last achieved goal to memory if first time encountered
                if str(last_ag) not in self.discovered_goals_str and condition_stability:
                    self.discovered_goals.append(last_ag.copy())
                    self.discovered_goals_str.append(str(last_ag))
                    self.discovered_goals_oracle_ids.append(self.nb_discovered_goals)
                    self.goal_to_id[str(last_ag)] = self.nb_discovered_goals
                    # self.id_to_goal[self.nb_discovered_goals] = str(last_ag)
                    self.visits.append(1)

                    # Check to which stack class corresponds the discovered goal
                    above_predicates = last_ag[10:30]
                    try:
                        c = self.stacks_to_class[str(above_predicates)]
                        self.discovered_goals_per_stacks[c] += 1
                    except KeyError:
                        self.discovered_goals_per_stacks['others'] += 1

                    # Increment number of discovered goals (to increment the id !)
                    self.nb_discovered_goals += 1
                
                # if goal already encountered before, we are sure its index exists in the visits list
                # Update number of visits
                elif str(last_ag) in self.discovered_goals_str and condition_stability and self.agent not in  ['HME', 'Beyond']:
                    self.visits[self.goal_to_id[str(last_ag)]] += 1
                # Add goal if not already encountered (to include internalized pairs in discovere buffer)

                if self.agent == 'HME' or self.agent == 'Beyond':
                    if str(goal) not in self.discovered_goals_str: 
                        self.discovered_goals.append(goal.copy())
                        self.discovered_goals_str.append(str(goal))
                        self.discovered_goals_oracle_ids.append(self.nb_discovered_goals)

                        # Increment number of discovered goals (to increment the id !)
                        self.nb_discovered_goals += 1

        for e in episodes:
            # Set final reward
            e['final_reward'] = np.zeros_like(e['rewards']) + e['rewards'][-1]
        
        self.sync()
        return episodes

    def sync(self):
        """ Synchronize the goal sampler's attributes between all workers """
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
        self.discovered_goals_oracle_ids = MPI.COMM_WORLD.bcast(self.discovered_goals_oracle_ids, root=0)
        self.nb_discovered_goals = MPI.COMM_WORLD.bcast(self.nb_discovered_goals, root=0)
        self.visits = MPI.COMM_WORLD.bcast(self.visits, root=0)
        self.goal_to_id = MPI.COMM_WORLD.bcast(self.goal_to_id, root=0)
        # self.id_to_goal = MPI.COMM_WORLD.bcast(self.id_to_goal, root=0)
    
    def update_query_proba(self):
        # Compute Query Probabilities
        if not self.fixed_queries:
            if len(self.values_goals) > self.min_queue_length:
                delta_value_goals = abs(self.values_goals[0] - self.values_goals[-1][:len(self.values_goals[0])])
                if self.progress_function == 'mean':
                    progress = np.mean(delta_value_goals) 
                elif self.progress_function == 'max':
                    progress = np.max(delta_value_goals)
                
                self.query_proba = np.exp(- self.beta * progress)

    def sync_queries(self):
        """ Synchronize the query's attributes between all workers """
        self.values_goals = MPI.COMM_WORLD.bcast(self.values_goals, root=0)
        self.values_goals = self.values_goals[-self.max_queue_length:]
        # self.query_proba = MPI.COMM_WORLD.bcast(self.query_proba, root=0)

    def update_internalization(self, ss_b_pairs, b):
        """ Update lists of stepping stone / beyond and beyond goals """
        self.ss_b_pairs = ss_b_pairs
        self.beyond = b

    def generate_intermediate_goal(self, goal):
        """ Given a goal, uses goal evaluator to generate intermediate goal that maximize the value """
        # res = []
        # for eval_goal in goals:
        #     repeat_goal = np.repeat(np.expand_dims(eval_goal, axis=0), repeats=len(self.discovered_goals), axis=0)
        #     norm_goals = self.goal_evaluator.estimate_goal_value(goals=repeat_goal, ag=self.discovered_goals)
        #     ind = np.argsort(norm_goals)[-2:]
        #     adjacent_goal = self.discovered_goals[ind[0]] if str(self.discovered_goals[ind[0]]) != str(eval_goal) else self.discovered_goals[ind[1]]
        #     res.append(adjacent_goal)
        
        # res = np.array(res)
        repeat_goal = np.repeat(np.expand_dims(goal, axis=0), repeats=len(self.discovered_goals), axis=0)
        norm_goals = self.goal_evaluator.estimate_goal_value(goals=repeat_goal, ag=self.discovered_goals)
        ind = np.argsort(norm_goals)[-2:]
        adjacent_goal = self.discovered_goals[ind[0]] if str(self.discovered_goals[ind[0]]) != str(goal) else self.discovered_goals[ind[1]]

        return adjacent_goal

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        n = len(INSTRUCTIONS)
        for i in np.arange(1, n+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []
            self.stats['# class_teacher {}'.format(i)] = []
            self.stats['# class_agent {}'.format(i)] = []
        
        # Init for each stack class
        stack_classes = set(self.stacks_to_class.values())
        for c in stack_classes:
            self.stats[f'discovered_{c}'] = []
        # Add class that contains all goals that do not correspond to the stack_classes
        self.stats['discovered_others'] = []

        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        self.stats['nb_internalized_pairs'] = []
        self.stats['proposed_ss'] = []
        self.stats['proposed_beyond'] = []
        self.stats['query_proba'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update', 'update_graph', 
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr,  agent_stats, goals_per_class, proposed_ss, proposed_beyond, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals))
        for g_id in np.arange(1, len(av_res) + 1):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])

        for k, v in self.discovered_goals_per_stacks.items():
            self.stats[f'discovered_{k}'].append(v)
        self.stats['proposed_ss'].append(proposed_ss)
        self.stats['proposed_beyond'].append(proposed_beyond)
        for k in goals_per_class.keys():
            self.stats['# class_teacher {}'.format(k)].append(goals_per_class[k])
            self.stats['# class_agent {}'.format(k)].append(agent_stats[k])
