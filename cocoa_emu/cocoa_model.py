import numpy as np

from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.model import Model
#from cobaya.conventions import kinds, _timing, _params, _prior, _packages_path, packages_path_input
from cobaya.conventions import packages_path_input

def get_model(yaml_file):
    info  = yaml_load_file(yaml_file)
    updated_info = update_info(info)
    # model =  Model(updated_info[_params], updated_info[kinds.likelihood],
    #            updated_info.get(_prior), updated_info.get(kinds.theory),
    #            packages_path=info.get(_packages_path), timing=updated_info.get(_timing),
    #            allow_renames=False, stop_at_error=info.get("stop_at_error", False))
    model =  Model(updated_info["params"], updated_info["likelihood"], #KZ: updated cobaya function
                 updated_info.get("prior"), updated_info.get("theory"),
                 packages_path=info.get(packages_path_input),
                 timing=updated_info.get("timing"),
                 stop_at_error=info.get("stop_at_error", False))
    return model

class CocoaModel:
    def __init__(self, configfile, likelihood):
        self.model      = get_model(configfile)
        self.likelihood = likelihood
        
    def calculate_data_vector(self, params_values, baryon_scenario=None):        
        likelihood   = self.model.likelihood[self.likelihood]
        input_params = self.model.parameterization.to_input(params_values)
        self.model.provider.set_current_input_params(input_params)
        #print("1")
        #KZ start
        
        #for (component, like_index), param_dep in zip(self.model._component_order.items(),
        #                                              self.model._params_of_dependencies):
        #    depend_list = [input_params[p] for p in param_dep]
        #    params = {p: input_params[p] for p in component.input_params}
        #    print("component \n")
        #    print(component)

        #    print(depend_list)
        #    print(params)
        #    compute_success = component.check_cache_and_compute(
        #        params, want_derived=False,
        #        dependency_params=depend_list, cached=False)
        #    print('compute success')
        #    print(compute_success)
        #for (component, index), param_dep in zip(self.model._component_order.items(), 
        #                                         self.model._params_of_dependencies):
        #    depend_list = [input_params[p] for p in param_dep]
        #    params = {p: input_params[p] for p in component.input_params}
        #    compute_success = component.check_cache_and_compute(want_derived=False,
        #                                 dependency_params=depend_list, cached=False, **params)
        
        #KZ end
        #print("2")
        if baryon_scenario is None:
            print("2")
            data_vector = likelihood.get_datavector(**input_params)
        else:
            data_vector = likelihood.compute_baryon_datavector_masked_reduced_dim(baryon_scenario, **input_params)

        print("4")
        return np.array(data_vector)