# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class GenMWPConfig():
    def __init__(self, args):
        self.seed = args.seed
        self.model_name = args.model_name
        self.model_max_length = args.model_max_length
        self.root_dataset_name = args.root_dataset_name
        self.group_name = args.group_name
        self.setting_name = args.setting_name
        self.max_epoch = args.max_epoch
    
    def setup(self, perturb):
        # build perturb-specific params
        self.perturb = perturb
        
        dataset_name = f'{self.root_dataset_name}-{perturb}-{self.setting_name}' if perturb != 'original' else self.root_dataset_name
        self.dataset_name = dataset_name
        
        exp_id = f'{self.model_name}_{dataset_name}'
        self.exp_id = exp_id

        save_dir = f'{self.group_name}/{self.model_name}_{dataset_name}-{self.seed}'
        self.save_dir = save_dir

    def setup_test(self, perturb, train_perturb):
        self.setup(perturb)
        
        dataset_name = f'{self.root_dataset_name}-{train_perturb}-{self.setting_name}' if train_perturb != 'original' else self.root_dataset_name
        load_dir = f'{self.group_name}/{self.model_name}_{dataset_name}-{self.seed}'
        self.load_dir = load_dir

