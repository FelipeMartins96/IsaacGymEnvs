import isaacgym

import hydra
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from omegaconf import DictConfig, OmegaConf
from tasks import isaacgym_task_map

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
    env = isaacgym_task_map[cfg.task_name](
        omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless
    )

    for i in range(1000):
        obs, rws, dones, info = env.step(env.zero_actions())
    

if __name__ == "__main__":
    launch_rlg_hydra()
