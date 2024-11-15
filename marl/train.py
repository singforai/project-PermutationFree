import sys
import argparse

from utils.configs_tools import get_defaults_yaml_args

sys.dont_write_bytecode = True

def get_config(args):
    
    parser = argparse.ArgumentParser(
        description='MARL', 
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--env_name", type=str, default='smacv2', choices=['smacv2' , 'grf'], help="Name of environment")
    parser.add_argument("--algorithm_name", type=str, default='mappo', choices=['mast',"mappo"], help="Name of algorithm")
    parser.add_argument("--use_gpu", action='store_false', default=True, help="Whether to use GPU or CPU for training model.")
    parser.add_argument("--num_gpu", type=int, default=0, help="GPU number to use for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for numpy/torch")
    parser.add_argument("--experiment_name", type=str, default="test", help="an identifier to distinguish different experiment.")
    
    parser.add_argument("--group_name", type=str, default="test", help="group name for wandb")
    parser.add_argument("--user_name", type=str, default='singfor7012', help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="[for wandb usage], by default True, will log date to wandb server")
    
    parser.add_argument("--cuda", action='store_false', default=True, help="Whether to use CUDA.")
    parser.add_argument("--torch_threads", type=int, default=4, help="arg to torch.set_num_threads")
    
    parser.add_argument("--use_render", action='store_true', default=False, help="whether to use render")
    parser.add_argument("--render_episodes", type=int, default=10, help="number of episodes to render")
    
    parser.add_argument("--log_dir", type=str, default='./examples/results', help="logging directory")
    parser.add_argument("--model_dir", default=None, help="if set, load models from this directory; otherwise, randomly initialise the models")
    
    parser.add_argument("--log_interval", type=int, default=5, help="logging interval")
    parser.add_argument("--eval_interval", type=int, default=5, help="evaluation interval")
    
    parser.add_argument("--save_model", action="store_true", default=False, help="whether to save model")
    parser.add_argument("--save_interval", type=int, default=1000, help="model saving interval")
    
    parser.add_argument("--check_permutation_free", action = "store_true", default = False, help = "whether to check permutation free")
    
    parser.add_argument("--map_name", type=str, default='protoss_5_vs_5', choices = [
        "protoss_5_vs_5",
        "terran_5_vs_5",
        "zerg_5_vs_5",
        "protoss_10_vs_10",
        "terran_10_vs_10",
        "zerg_10_vs_10",
        "protoss_10_vs_11",
        "terran_10_vs_11",
        "zerg_10_vs_11",
        "protoss_20_vs_20",
        "terran_20_vs_20",
        "zerg_20_vs_20",
        "protoss_20_vs_23",
        "terran_20_vs_23",
        "zerg_20_vs_23"
    ])
    args = parser.parse_known_args(args)[0]
    return args

def main(args):
    exp_args = vars(get_config(args))
    algo_args, env_args = get_defaults_yaml_args(
        algo = exp_args["algorithm_name"], 
        env = exp_args["env_name"]
    )

    from runner.smac_runner import SMACRunner as Runner
    runner = Runner(
        exp_args = exp_args, 
        algo_args = algo_args, 
        env_args = env_args,
    )
    runner.run()
    runner.close()
    
if __name__ == "__main__":
    main(args=sys.argv[1:])