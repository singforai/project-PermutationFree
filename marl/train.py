import sys

from config import get_config
from runner import RUNNER_REGISTRY
from utils.configs_tools import get_defaults_yaml_args

def main(args):
    exp_args = vars(get_config(args))
    algo_args, env_args = get_defaults_yaml_args(
        algo = exp_args["algorithm_name"], 
        env = exp_args["env_name"]
    )

    runner = RUNNER_REGISTRY[exp_args["env_name"]](
        exp_args = exp_args, 
        algo_args = algo_args, 
        env_args = env_args,
    )
    runner.run()
    runner.close()
    
if __name__ == "__main__":
    main(args=sys.argv[1:])