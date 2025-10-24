import hydra
from omegaconf import OmegaConf

# Define the addn resolver function
def addn(*args):
    """Adds multiple numbers."""
    result = 0
    for x in args:
        result += float(x) # 숫자로 변환하여 덧셈
    return int(result)

# Register the addn resolver
OmegaConf.register_new_resolver("addn", addn)

@hydra.main(config_path=None)
def register_resolvers(cfg):
    pass

# Define the resolver function
def replace_slash(value: str) -> str:
    return value.replace('/', '_')

# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)

if __name__ == "__main__":
    register_resolvers()

