import torch

class Distillation():

    def __init__(self,models_path,opt):
        self.models = 