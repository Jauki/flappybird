import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


## Old own implementation when i started wiht the Deep-Learning QNetwork
# import numpy as np
# class QNetwork:
#     def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001):
#         self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
#         self.b1 = np.zeros((1, hidden_dim))
#         self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
#         self.b2 = np.zeros((1, output_dim))
#         self.lr = lr

#     def forward_prop(self, x):
#         self.z1 = np.dot(x, self.W1) + self.b1
#         self.a1 = np.maximum(0, self.z1)  # ReLU
#         self.z2 = np.dot(self.a1, self.W2) + self.b2
#         return self.z2

#     def backward(self, x, target_q, action):
#         """
#         One-step gradient update for Q-learning with MSE-like objective.
#         """
#         output = self.forward_prop(x)

#         # Copy so we don't affect the original forward pass
#         target = output.copy()
#         target[0, action] = target_q

#         d_output = target - output
#         d_W2 = np.dot(self.a1.T, d_output)
#         d_b2 = np.sum(d_output, axis=0, keepdims=True)
#         d_a1 = np.dot(d_output, self.W2.T)
#         d_z1 = d_a1 * (self.z1 > 0)  # derivative of ReLU

#         d_W1 = np.dot(x.T, d_z1)
#         d_b1 = np.sum(d_z1, axis=0, keepdims=True)

#         # Gradient ascent on the loss means we add
#         self.W2 += self.lr * d_W2
#         self.b2 += self.lr * d_b2
#         self.W1 += self.lr * d_W1
#         self.b1 += self.lr * d_b1

#     def get_params(self):
#         """
#         Returns a dict of the network's parameters
#         for cloning or saving.
#         """
#         return {
#             "W1": self.W1.copy(),
#             "b1": self.b1.copy(),
#             "W2": self.W2.copy(),
#             "b2": self.b2.copy()
#         }

#     def set_params(self, params):
#         """
#         Loads the network's parameters from a dict.
#         """
#         self.W1 = params["W1"].copy()
#         self.b1 = params["b1"].copy()
#         self.W2 = params["W2"].copy()
#         self.b2 = params["b2"].copy()

# def mutate_network(network, mutation_prob, mutation_std):
#     # For each parameter array, we add noise with probability = mutation_prob
#     for param_name in ["W1", "b1", "W2", "b2"]:
#         param = getattr(network, param_name)
#         mask = (np.random.rand(*param.shape) < mutation_prob)
#         noise = np.random.normal(0, mutation_std, size=param.shape)
#         param += mask * noise 
        



## this code was mostly sticked together by using chat gpt and reading pytorch documentation and stackoverflow really just a mess
class GA_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GA_Network, self).__init__()
        # Simple feed-forward net
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    ## feeds the input to the nn
    def forward_prop(self, x):
        """
        x: numpy array shape (1, input_dim).
        Returns numpy array shape (1, output_dim).
        """
        x_torch = torch.FloatTensor(x)
        with torch.no_grad():
            out = self.net(x_torch)
        return out.cpu().numpy()

    def get_params(self):
        """
        Return a dict of parameter_name -> Tensor for external saving/mutation.
        """
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_params(self, params):
        """
        Load a dict of parameter_name -> Tensor into this model.
        """
        current = self.state_dict()
        for k, v in params.items():
            current[k] = v
        self.load_state_dict(current)


def mutate_network(network, mutation_prob, mutation_std):
    """
    Mutate network parameters in-place with random Gaussian noise
    (applied with probability `mutation_prob` per weight).
    """
    with torch.no_grad():
        for name, param in network.named_parameters():
            mask = (torch.rand_like(param) < mutation_prob)
            noise = torch.randn_like(param) * mutation_std
            param += mask * noise


def crossover_networks(parent1, parent2):
    """
    Combine parameters from two parents. 
    Example: half from parent1, half from parent2.
    Returns a new param dict for the child.
    """
    p1 = parent1.get_params()
    p2 = parent2.get_params()

    child_params = {}
    for key in p1.keys():
        # 50/50 split of each parameter's entries
        tensor1 = p1[key]
        tensor2 = p2[key]
        mask = torch.rand_like(tensor1) < 0.5
        child_tensor = torch.where(mask, tensor1, tensor2)
        child_params[key] = child_tensor
    return child_params
