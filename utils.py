import numpy as np
from datetime import datetime
import os
import json
import subprocess
import os.path
import sys
from itertools import permutations, combinations


def get_env_params(env):
    obs = env.reset()

    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

def hard_update(target, source):
    """ Perform hard update, used to copy critic networks in target network during the initialization of the latter """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def merge_mini_episodes_and_relabel(generated_episodes):
        """ Given a list of mini episodes, merge them into a single one a relabel according to final goal """
        # Concatenate mini-episodes and perform data augmentation
        updated_episodes = []
        for episode in generated_episodes:
            merged_mini_episodes = {k: np.concatenate([v[:100], episode[1][k]]) for k, v in episode[0].items()}
            # Relabel mini episodes according to final goal
            merged_mini_episodes['g'][:] = merged_mini_episodes['g'][-1]
            updated_episodes.append(merged_mini_episodes)
        
        return updated_episodes
        
def init_storage(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # path to save the model
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logdir = os.path.join(args.save_dir, f'{date_time}_{args.agent}_beta={args.beta}_{args.oracle_name}')
    # path to save evaluations
    model_path = os.path.join(logdir, 'models')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return logdir, model_path


def get_stat_func(line='mean', err='std'):

    if line == 'mean':
        def line_f(a):
            return np.nanmean(a, axis=0)
    elif line == 'median':
        def line_f(a):
            return np.nanmedian(a, axis=0)
    else:
        raise NotImplementedError

    if err == 'std':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0)
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0)
    elif err == 'sem':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
    elif err == 'range':
        def err_plus(a):
            return np.nanmax(a, axis=0)
        def err_minus(a):
            return np.nanmin(a, axis=0)
    elif err == 'interquartile':
        def err_plus(a):
            return np.nanpercentile(a, q=75, axis=0)
        def err_minus(a):
            return np.nanpercentile(a, q=25, axis=0)
    else:
        raise NotImplementedError

    return line_f, err_minus, err_plus


class CompressPDF:
    """
    author: Pure Python
    url: https://www.purepython.org
    copyright: CC BY-NC 4.0
    Forked date: 2018-01-07 / First version MIT license -- free to use as you want, cheers.
    Original Author: Sylvain Carlioz, 6/03/2017
    Simple python wrapper script to use ghoscript function to compress PDF files.
    With this class you can compress and or fix a folder with (corrupt) PDF files.
    You can also use this class within your own scripts just do a
    import CompressPDF
    Compression levels:
        0: default
        1: prepress
        2: printer
        3: ebook
        4: screen
    Dependency: Ghostscript.
    On MacOSX install via command line `brew install ghostscript`.
    """
    def __init__(self, compress_level=0, show_info=False):
        self.compress_level = compress_level

        self.quality = {
            0: '/default',
            1: '/prepress',
            2: '/printer',
            3: '/ebook',
            4: '/screen'
        }

        self.show_compress_info = show_info

    def compress(self, file=None, new_file=None):
        """
        Function to compress PDF via Ghostscript command line interface
        :param file: old file that needs to be compressed
        :param new_file: new file that is commpressed
        :return: True or False, to do a cleanup when needed
        """
        try:
            if not os.path.isfile(file):
                print("Error: invalid path for input PDF file")
                sys.exit(1)

            # Check if file is a PDF by extension
            filename, file_extension = os.path.splitext(file)
            if file_extension != '.pdf':
                raise Exception("Error: input file is not a PDF")
                return False

            if self.show_compress_info:
                initial_size = os.path.getsize(file)

            subprocess.call(['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
                            '-dPDFSETTINGS={}'.format(self.quality[self.compress_level]),
                            '-dNOPAUSE', '-dQUIET', '-dBATCH',
                            '-sOutputFile={}'.format(new_file),
                             file]
            )


            if self.show_compress_info:
                final_size = os.path.getsize(new_file)
                ratio = 1 - (final_size / initial_size)
                print("Compression by {0:.0%}.".format(ratio))
                print("Final file size is {0:.1f}MB".format(final_size / 1000000))

            return True
        except Exception as error:
            print('Caught this error: ' + repr(error))
        except subprocess.CalledProcessError as e:
            print("Unexpected error:".format(e.output))
            return False


def invert_dict(d):
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = key
            else:
                pass
    return inverse


def get_graph_structure(n):
    """ Given the number of blocks (nodes), returns :
    edges: in the form [to, from]
    incoming_edges: for each node, the indexes of the incoming edges
    predicate_ids: the ids of the predicates takes for each edge """
    map_list = list(combinations(np.arange(n), 2)) + list(permutations(np.arange(n), 2)) + [(i, i) for i in range(n)]
    edges = list(permutations(np.arange(n), 2))
    obj_ids = np.arange(n)
    n_comb = n * (n-1) // 2

    incoming_edges = []
    for obj_id in obj_ids:
        temp = []
        for i, pair in enumerate(permutations(np.arange(n), 2)):
            if obj_id == pair[0]:
                temp.append(i)
        incoming_edges.append(temp)

    predicate_ids = []
    for pair in permutations(np.arange(n), 2):
        ids_g = [i for i in range(len(map_list))
                 if (set(map_list[i]) == set(pair) and i < n_comb)
                 or (map_list[i] == pair and 30 > i >= n_comb) or (map_list[i] == (pair[0], pair[0]))
                 or (map_list[i] == (pair[1], pair[1]))]
        predicate_ids.append(ids_g)

    return edges, incoming_edges, predicate_ids


def get_idxs_per_relation(n):
    """ For each possible relation between any pair of objects, outputs the corresponding predicate indexes in the goal vector"""
    map_list = list(combinations(np.arange(n), 2)) + list(permutations(np.arange(n), 2))
    all_relations = list(combinations(np.arange(n), 2))
    return np.array([np.array([i for i in range(len(map_list)) if set(map_list[i]) == set(r)]) for r in all_relations])


def get_idxs_per_object(n):
    """ For each objects, outputs the predicates indexes that include the corresponding object"""
    map_list = list(combinations(np.arange(n), 2)) + list(permutations(np.arange(n), 2)) + [(i, i) for i in range(n)]
    obj_ids = np.arange(n)
    return np.array([np.array([i for i in range(len(map_list)) if obj_id in map_list[i]]) for obj_id in obj_ids])


def get_eval_goals(instruction, n, nb_goals=1):
    """ Given an instruction and the total number of objects on the table, outputs a corresponding semantic goal"""
    res = []
    n_blocks = n
    n_comb = n_blocks * (n_blocks - 1) // 2
    n_perm = n_blocks * (n_blocks - 1)
    goal_dim = n_comb + n_perm + n_blocks
    try:
        predicate, pairs = instruction.split('_')
    except ValueError:
        predicate, pairs_1, pairs_2 = instruction.split('_')
    # tower + pyramid
    if predicate == 'mixed':
        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=n_blocks, replace=False)
            tower_objects = objects[:2]
            pyramid_objects = objects[2:]
            # Create on_table predicates

            on_table_config = -np.ones(n_blocks)
            # The base of the tower
            on_table_config[tower_objects[-1]] = 1.
            # The base of the pyramid
            on_table_config[pyramid_objects[0]] = 1.
            on_table_config[pyramid_objects[1]] = 1.
            for j in range(1):
                obj_ids = (tower_objects[j], tower_objects[j+1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k+10)

            for j in range(1):
                obj_ids = (pyramid_objects[j], pyramid_objects[j + 1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                    if set((pyramid_objects[j], pyramid_objects[-1])) == set(c):
                        id.append(k)
                    if j == 2 - 2 and set((pyramid_objects[j + 1], pyramid_objects[-1])) == set(c):
                        id.append(k)
            for j in range(2):
                obj_ids = (pyramid_objects[-1], pyramid_objects[j])
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k + n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            g[-n_blocks:] = on_table_config
            res.append(g)
        return np.array(res)

    # two towers
    if predicate == '2stacks':
        stack_size_1 = int(pairs_1)
        stack_size_2 = int(pairs_2)

        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=stack_size_1 + stack_size_2, replace=False)
            # on_table config creation
            on_table_config = np.ones(n_blocks)
            for j in range(stack_size_1 - 1):
                on_table_config[objects[j]] = -1.
                obj_ids = (objects[j], objects[j + 1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k + n_comb)
            for j in range(stack_size_1, stack_size_1+stack_size_2-1):
                on_table_config[objects[j]] = -1.
                obj_ids = (objects[j], objects[j + 1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k + n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            g[-n_blocks:] = on_table_config
            res.append(g)
        return np.array(res)

    # pyramid
    if predicate == 'pyramid':
        n_base = int(pairs)-1
        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=n_base+1, replace=False)
            # Create on_table_predicates
            on_table_config = np.ones(n_blocks)
            on_table_config[objects[-1]] = -1.
            for j in range(n_base-1):
                obj_ids = (objects[j], objects[j + 1])
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                    if set((objects[j], objects[-1])) == set(c):
                        id.append(k)
                    if j == n_base - 2 and set((objects[j+1], objects[-1])) == set(c):
                        id.append(k)
            for j in range(n_base):
                obj_ids = (objects[-1], objects[j])
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k+n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            g[-n_blocks:] = on_table_config
            res.append(g)
        return np.array(res)

    if predicate == 'stack':
        stack_size = int(pairs)
        close_pairs = 0
    else:
        stack_size = 1
        close_pairs = int(pairs)
    # no stacks whatsoever
    if stack_size == 1:
        ids = []
        for _ in range(nb_goals):
            id = np.random.choice(np.arange(n_comb), size=close_pairs, replace=False)
            ids.append(id)
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            # all on table
            g[-n_blocks:] = np.ones(n_blocks)
            res.append(g)
        return np.array(res)
    # one tower
    else:
        ids = []
        for _ in range(nb_goals):
            id = []
            objects = np.random.choice(np.arange(n_blocks), size=stack_size, replace=False)
            # Create on_table predicates
            on_table_config = np.ones(n_blocks)
            for j in range(stack_size-1):
                obj_ids = (objects[j], objects[j+1])
                on_table_config[objects[j]] = -1.
                for k, c in enumerate(combinations(np.arange(n_blocks), 2)):
                    if set(obj_ids) == set(c):
                        id.append(k)
                for k, c in enumerate(permutations(np.arange(n_blocks), 2)):
                    if obj_ids == c:
                        id.append(k+n_comb)
            ids.append(np.array(id))
        for id in ids:
            g = -np.ones(goal_dim)
            g[id] = 1.
            g[-n_blocks:] = on_table_config
            res.append(g)
        return np.array(res)


def generate_stacks_dict(list_classes, n_blocks=5, n_trials=100):
    """ Given a list of classes from SP, outputs a dictionary id_class -> partial goals that it contains """
    n_combinations = int(n_blocks * (n_blocks - 1 ) / 2)
    class_id_to_goals = {}
    stacks_to_class_id = {}
    for i, c in enumerate(list_classes):
        eval_goals = []
        for _ in range(n_trials):
            eval_goal = get_eval_goals(c, n=n_blocks)
            eval_goals.append(eval_goal.squeeze(0))
        eval_goals = np.array(eval_goals)

        unique_goals = np.unique(eval_goals, axis=0)
        class_id_to_goals[i] = [str(e[n_combinations:30]) for e in unique_goals]
        for g in unique_goals:
            stacks_to_class_id[str(g[n_combinations:30])] = list_classes[i]

    return stacks_to_class_id

def generate_stacks_to_class():
        """ Get evaluation map that correspond each goal to a class involving stacks, based only on the above predicates
        This map ignores the close predicates
        i.e. if there is only a stack of 2 blocks, we ignore the close predicates
        This is used to check what types of goals the agent is discovering """
        stacks_classes = ['stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3', 'mixed_2_3', 'stack_4', 'stack_5']
        stacks_to_class = generate_stacks_dict(list_classes=stacks_classes, n_blocks=5, n_trials=2000)

        return stacks_to_class

def apply_on_table_config(g):
    """ Appends the goal with unary on_table predicates """
    g = np.array(g)
    map_list = list(permutations(np.arange(5), 2)) 
    on_table_config = np.ones(5)
    for i in range(10, 30):
        if g[i] == 1.:
            on_table_config[map_list[i-10][0]] = -1.
    
    res = np.concatenate([g, on_table_config])
    return tuple(res)

INSTRUCTIONS = ['close_1', 'close_2', 'close_3', 'stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3',
                'mixed_2_3', 'stack_4', 'stack_5']

