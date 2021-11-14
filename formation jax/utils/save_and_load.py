import os
import pickle

def save_jax_model(params, state, model_path):
    os.makedirs(model_path+'/params', exist_ok=True)
    os.makedirs(model_path+'/state', exist_ok=True)
    param_path = os.path.join(model_path, 'params', 'params.pickle')
    state_path = os.path.join(model_path, 'state', 'state.pickle')
    with open(param_path, 'wb') as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(state_path, 'wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved at '{model_path}'.\n")


def load_jax_model(model_path):
    param_path = os.path.join(model_path, 'params', 'params.pickle')
    state_path = os.path.join(model_path, 'state', 'state.pickle')
    with open(param_path, 'rb') as handle:
        params = pickle.load(handle)
    with open(state_path, 'rb') as handle:
        state = pickle.load(handle)
    return params, state