import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

# NOTE: The original code uses an additional transformation from policy, reward and value logits (i.e. value_suppot_vector ect.) to the actual polcy, value etc.
# in the current implementation, we skip this additional transform. and map directly onto the values.


class MuZeroNet(nn.Module):
    """Class containing all MuZero nets"""

    def __init__(
        self,
        rpr_input_s,
        action_s,
        lr,
        device,
        reward_s=1,
        h1_s=256,  # 64 seems to work worse! 256 was quite good for the easy case
        reprs_output_size=64,
        weight_decay=1e-4,
        TD_return=False,
    ):
        super().__init__()
        self.dev = device

        self.num_actions = action_s
        self.TD_return = TD_return
        self.reprs_output_size = reprs_output_size

        # NOTE: currently use support for both value and rwd prediction
        if TD_return:
            self.support_size = 33
        else:
            self.support_size = 1

        self.representation_net = nn.Sequential(
            nn.Linear(rpr_input_s, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, reprs_output_size),
        )

        self.dynamic_net = nn.Sequential(
            nn.Linear(reprs_output_size + action_s, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, reprs_output_size),
        )

        self.rwd_net = nn.Sequential(
            nn.Linear(reprs_output_size, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, self.support_size),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(reprs_output_size, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, action_s),
        )

        self.value_net = nn.Sequential(
            nn.Linear(reprs_output_size, h1_s),
            nn.ReLU(),
            nn.Linear(h1_s, self.support_size),
        )

        self.optimiser = opt.Adam(self.parameters(), lr)  # , weight_decay=weight_decay)

    @torch.no_grad()
    def initial_inference(self, x):
        """During self-play, given environment observation, use representation function to predict initial hidden state.
        Then use prediction function to predict policy probabilities and state value (on the hidden state).
        """

        # Representation
        h_state = self.represent(x)

        # Prediction
        pi_logits, value = self.prediction(h_state)

        pi_probs = F.softmax(pi_logits, dim=-1)  # NOTE: dim ?

        rwd = torch.zeros_like(
            value
        )  # NOTE: Not sure why it doesn't predict rwd for initial inference

        pi_probs = pi_probs.squeeze(0).cpu().numpy()
        value = value.squeeze(0).cpu().item()
        rwd = rwd.squeeze(0).cpu().item()
        h_state = h_state.squeeze(0).cpu().numpy()

        return h_state, rwd, pi_probs, value

    @torch.no_grad()
    def recurrent_inference(self, h_state, action):
        """During self-play, given hidden state at timestep `t-1` and action at t,
        use dynamics function to predict the reward and next hidden state,
        and use prediction function to predict policy probabilities and state value (on new hidden state).
        """

        # Dynamic
        h_state, rwd = self.dynamics(h_state, action)

        # Prediction
        pi_logits, value = self.prediction(h_state)

        pi_probs = F.softmax(pi_logits, dim=-1)  # NOTE: dim ?

        pi_probs = pi_probs.squeeze(0).cpu().numpy()
        value = value.squeeze(0).cpu().item()
        rwd = rwd.squeeze(0).cpu().item()
        h_state = h_state.squeeze(0).cpu().numpy()

        return h_state, rwd, pi_probs, value

    def update(self, loss):

        self.optimiser.zero_grad()
        loss.backward()

        self.optimiser.step()

    def represent(self, x):
        h_state = self.representation_net(x)
        norm_h_state = self.normalize_h_state(h_state)
        return norm_h_state

    def dynamics(self, h_state, action):
        x = torch.cat([h_state, action], dim=-1)
        new_h_state = self.dynamic_net(x)
        rwd_prediction = self.rwd_net(new_h_state)

        if self.TD_return:
            rwd_prediction = self.logits_to_transformed_expected_value(rwd_prediction)

        norm_h_state = self.normalize_h_state(new_h_state)
        return norm_h_state, rwd_prediction
    
    def dynamics_logits(self, h_state, action):
        x = torch.cat([h_state, action], dim=-1)
        new_h_state = self.dynamic_net(x)
        rwd_prediction_logits = self.rwd_net(new_h_state)

        return new_h_state, rwd_prediction_logits

    def prediction(self, h):
        pi_logits = self.policy_net(h)

        value_logits = self.value_net(h)

        # Use transformation with support vector only for TD-returns
        if self.TD_return:
            value_logits = self.logits_to_transformed_expected_value(value_logits)

        return pi_logits, value_logits
    
    def prediction_logits(self, h):
        pi_logits = self.policy_net(h)

        value_logits = self.value_net(h)
        return pi_logits, value_logits

    def real_to_logits(self, scalar):
        """Convert a real value to a categorical distribution (logits) over a fixed support.

        Given a real value (the target returns), do the following operations:
            - apply hyperbolic transform function
            - clamp the scalar to be in the range of [min_value, max_value]
            - convert to categorical distribution (logits)
        Args:
            scalar: a 2D tensor of real values, shape [B, 1].
            supports: a 1D tensor of support for the computation, shape [N,].

        Returns:
            a 2D tensor which represent the categorical distribution (logits), shape [B, N].
        """
        max_value = (self.support_size - 1) // 2
        min_value = -max_value

        # Apply inverse transform function
        scalar = self._signed_hyperbolic(scalar, eps=1e-3)
        
        # Clamp the scalar to be in the range of [min_value, max_value]
        scalar = torch.clamp(scalar, min_value, max_value)

        # Convert to categorical distribution (logits)
        logits = self._transform_to_2hot(scalar,min_value,max_value)
        return logits

    def logits_to_transformed_expected_value(self, logits):
        """
        Given raw logits (could be either reward or state value), do the following operations:
            - apply softmax
            - compute the expected scalar value
            - apply `signed_parabolic` transform function

        Args:
            logits: 2D tensor raw logits of the network output, shape [B, N].
            supports: vector of support for the computeation, shape [N,].

        Returns:
            a 2D tensor which represent the transformed expected value, shape [B, 1].
        """
        max_value = (self.support_size - 1) // 2
        min_value = -max_value

        # Compute expected scalar
        probs = torch.softmax(logits, dim=-1)
        x = self._transform_from_2hot(probs, min_value, max_value)

        # Apply transform function
        x = self._signed_parabolic(x)
        return x

    def _transform_from_2hot(self, probs, min_value, max_value):
        """Transforms from a categorical distribution to a scalar."""
        support_space = torch.linspace(
            min_value, max_value, self.support_size, device=self.dev
        )
        support_space = support_space.expand_as(probs)
        scalar = torch.sum(probs * support_space, dim=-1, keepdim=True)
        return scalar

    def _transform_to_2hot(self,scalar,min_value,max_value):
        """Transforms from a scalar to a categorical distribution (logits)."""
        if len(scalar.shape) == 1:
            scalar = scalar.unsqueeze(1)
        support_space = torch.linspace(
            min_value, max_value, self.support_size, device=self.dev
        )
        support_space = support_space.expand(scalar.size(0),-1)

        rel_distance = -torch.abs(support_space - scalar)

        below = torch.where((rel_distance >= -1) & (rel_distance <= 0),rel_distance+1,torch.tensor(0., device=self.dev))
        above = torch.where((rel_distance >= 0) & (rel_distance <= 1),rel_distance,torch.tensor(0., device=self.dev))

        logits = below + above

        return logits
    

    def _signed_hyperbolic(self, x, eps=1e-3):
        """Inverse of signed_parabolic."""
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps*x

    def _signed_parabolic(self, x, eps=1e-3):
        """Signed parabolic transform, inverse of signed_hyperbolic."""
        z = torch.sqrt(1 + 4 * eps * (eps + 1 + torch.abs(x))) / 2 / eps - 1 / 2 / eps
        return torch.sign(x) * (torch.square(z) - 1)

    def normalize_h_state(self, h_state):
        _min = h_state.min(dim=-1, keepdim=True)[0]
        _max = h_state.max(dim=-1, keepdim=True)[0]
        return (h_state - _min) / (
            _max - _min + 1e-8
        )  ## Add small constant to avoid division by 0

    def set_pol_pertubation(self, pertub_magnitude):
        self.perturb_p_magnitude = pertub_magnitude

    def reset_param(self, l):
        k = np.sqrt(1 / self.reprs_output_size)
        if isinstance(l, nn.Linear):
            nn.init.uniform_(l.weight, a=-k, b=k)
            nn.init.uniform_(l.bias, a=-k, b=k)
