# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class GenDROPConfig():
    def __init__(self, args):
        self.seed = args.seed
        self.model_name = args.model_name
        self.model_max_length = args.model_max_length
        self.root_dataset_name = args.root_dataset_name
        self.batch_size = args.batch_size
        self.group_name = args.group_name
        self.setting_name = args.setting_name
        self.max_epochs = args.max_epochs
    
    # TODO: refactor this setup function
    def setup(self, perturb):
        # build perturb-specific params
        self.perturb = perturb
        
        dataset_name = f'{self.root_dataset_name}-{perturb}-{self.setting_name}' if perturb != 'original' else self.root_dataset_name
        self.dataset_name = dataset_name
        
        exp_id = f'{self.model_name}_{dataset_name}'
        self.exp_id = exp_id

        save_dir = f'{self.group_name}/{self.model_name}_{dataset_name}-{self.seed}'
        self.save_dir = save_dir

    def setup_train(self, perturb):
        self.setup(perturb)
        self.gpu_num = 4

    def setup_test(self, perturb, trained_perturb):
        self.setup(perturb)

        # set up load_dir, which identifies which directory our trained model is at
        dataset_name = f'{self.root_dataset_name}-{trained_perturb}-{self.setting_name}' if trained_perturb != 'original' else self.root_dataset_name
        load_dir = f'{self.group_name}/{self.model_name}_{dataset_name}-{self.seed}'
        self.load_dir = load_dir
        self.gpu_num = 1

