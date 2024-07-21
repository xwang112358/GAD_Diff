import hydra
from omegaconf import DictConfig, OmegaConf
# load yaml file by hydra

@hydra.main(config_path='./configs', config_name='config')
def my_app(cfg):
    # Access the YAML file contents through the cfg object
    print(type(cfg))
    print(cfg.general.name)
if __name__ == "__main__":
    my_app()