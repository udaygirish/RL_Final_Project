from IMG_POS_CNN import CombinedFeatureNetwork
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import A2C

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=9):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = CombinedFeatureNetwork()

    def forward(self, observations):
        return self.cnn(observations)

class CustomCNNPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(CustomCNNPolicy, self).__init__(*args, **kwargs,
                                              features_extractor_class=CustomFeatureExtractor,
                                              features_extractor_kwargs=dict(features_dim=9))

model = A2C(policy=CustomCNNPolicy, env='LunarLander-v2', verbose=1).learn(total_timesteps=100000)