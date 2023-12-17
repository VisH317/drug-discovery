from torch import nn, Tensor
from .moltransformer import MolTransformer

class MolTransformerClassifier(MolTransformer):

    def __init__(self, num_features: int = 8):
        super().__init__()

        self.num_features = num_features
        self.classifiers = nn.ModuleList([ nn.Sequential(nn.Linear(self.d_model, 2), nn.Softmax()) for _ in range(num_features) ])

    def forward(self, input: Tensor, top: Tensor, feature: int):
        if feature >= self.num_features: raise IndexError(f"Feature {feature} does not exist")
        out = super().forward(input, top)
        return self.classifiers[feature](out)