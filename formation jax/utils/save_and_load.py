import os
import pickle
from unicodedata import east_asian_width
import yaml

def save_jax_model(model_path, params, state=None, opt_state=None,
                   config_model=None, config_opt=None):
    os.makedirs(model_path + '/model', exist_ok=True)
    if opt_state or config_opt:
        os.makedirs(model_path + '/optimizer', exist_ok=True)

    params_path = os.path.join(model_path, 'model', 'params.pickle')
    state_path = os.path.join(model_path, 'model', 'state.pickle')
    opt_state_path = os.path.join(model_path, 'optimizer', 'state.pickle')
    config_model_path = os.path.join(model_path, 'model', 'config.yaml')
    config_opt_path = os.path.join(model_path, 'optimizer', 'config.yaml')

    with open(params_path, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if state:
        with open(state_path, 'wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if opt_state:
        with open(opt_state_path, 'wb') as handle:
            pickle.dump(opt_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if config_model:
        with open(config_model_path, 'w') as yaml_file:
            yaml.dump(config_model, yaml_file, default_flow_style=False)
        print(f"Model saved at '{model_path}'.\n")
    if config_opt:
        with open(config_opt_path, 'w') as yaml_file:
            yaml.dump(config_opt, yaml_file, default_flow_style=False)


def load_jax_model(model_path):
    params_path = os.path.join(model_path, 'model', 'params.pickle')
    state_path = os.path.join(model_path, 'model', 'state.pickle')
    opt_state_path = os.path.join(model_path, 'optimizer', 'state.pickle')
    config_model_path = os.path.join(model_path, 'model', 'config.yaml')
    config_opt_path = os.path.join(model_path, 'optimizer', 'config.yaml')
    configs = {}
    with open(params_path, 'rb') as handle:
        configs['params'] = pickle.load(handle)
    if os.path.isfile(state_path):
        with open(state_path, 'rb') as handle:
            configs['state'] = pickle.load(handle)
    if os.path.isfile(opt_state_path):
        with open(opt_state_path, 'rb') as handle:
            configs['opt_state'] = pickle.load(handle)
    if os.path.isfile(config_model_path):
        with open(config_model_path, 'r') as yaml_file:
            configs['model_config'] = yaml.load(yaml_file,
                                                Loader=yaml.FullLoader)
    if os.path.isfile(config_opt_path):
        with open(config_opt_path, 'r') as yaml_file:
            configs['opt_config'] = yaml.load(yaml_file,
                                              Loader=yaml.FullLoader)
    return configs
