import argparse
import os
import numpy as np
import scipy.stats as st

from utils.data_utils import check_extension, save_dataset

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()

def generate_nesting_data(dataset_size, nesting_size, case=1):
    if case == 1:
        loc1 = np.random.uniform(size=(dataset_size, nesting_size, 2))
        loc2 = np.random.uniform(size=(dataset_size, nesting_size, 2))
    elif case == 2:
        loc1 = np.random.uniform(size=(dataset_size, nesting_size, 2))
        loc2 = loc1 + np.random.uniform(size=(dataset_size, nesting_size, 2), low=-0.1, high=0.1)
    elif case in (3, 4):
        # Plate spec parameters
        mean_estimated = [11616.92823418, 2569.503305]
        cov_estimated = [
            [19010400.78123422, 1497149.40004837],
            [1497149.40004837, 502740.1651592]
        ]
        mvn = st.multivariate_normal(mean=mean_estimated, cov=cov_estimated)

        def sample_positive_from_mvn(mvn, size):
            samples = []
            while len(samples) < size:
                sample = mvn.rvs()
                if np.all(sample > 0):
                    samples.append(sample)
            return np.array(samples)

        data = sample_positive_from_mvn(mvn, dataset_size)
        norm = np.max(data, axis=-1)

        raw1 = np.random.uniform(
            size=(dataset_size, nesting_size, 2)
        ) * data[:, None, :] / norm[:, None, None]
        raw2 = np.random.uniform(
            size=(dataset_size, nesting_size, 2)
        ) * data[:, None, :] / norm[:, None, None]

        if case == 3:
            loc1, loc2 = raw1, raw2
        else:
            # Scale to [0,1]^2 based on min/max of combined
            combined = np.concatenate([raw1, raw2], axis=1)  # (dataset_size, 2*nesting_size, 2)
            mins = combined.min(axis=1)
            maxs = combined.max(axis=1)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            loc1 = (raw1 - mins[:, None, :]) / ranges[:, None, :]
            loc2 = (raw2 - mins[:, None, :]) / ranges[:, None, :]
    else:
        raise ValueError(f"Unsupported case: {case}")

    start = np.zeros((dataset_size, 2))
    loc = np.stack([loc1, loc2], axis=2).reshape((dataset_size, nesting_size * 2, 2))
    loc_paired = np.stack([loc2, loc1], axis=2).reshape((dataset_size, nesting_size * 2, 2))

    return list(zip(
        start.tolist(),
        loc.tolist(),
        loc_paired.tolist()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all',
                        help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")
    parser.add_argument('--case', type=int, default=3, help='Dataset distribution')

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    # assert opts.filename is None or (len(opts.problem) == 1 and len(opts.graph_sizes) == 1), \
    #     "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'nesting': [None],
        'vrp': [None],
        'pctsp': [None],
        'op': ['const', 'unif', 'dist']
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size)
                elif problem == 'nesting':
                    dataset = generate_nesting_data(opts.dataset_size, graph_size, case=opts.case)
                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(dataset[0])

                save_dataset(dataset, filename)