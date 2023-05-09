import unittest
import numpy as np
from unittest.mock import Mock
from DQN.train_dqn import train_dqn

class TestTrainDQN(unittest.TestCase):

    def setUp(self):
        self.env = Mock()
        self.env.reset.return_value = [1, 2, 3]
        self.env.step.return_value = ([4, 5, 6], 1.0, False)

    def test_train_dqn(self):
        Q, losses, rewards = train_dqn(self.env)

        self.assertIsNotNone(Q)
        self.assertGreater(len(losses), 0)
        self.assertGreater(len(rewards), 0)
    
    def test_q_network_init():
        q_network = train_dqn.Q_Network(input_size=5, hidden_size=10, output_size=3)
        assert len(q_network.children()) == 3
        assert q_network.fc1.W.shape == (10, 5)
        assert q_network.fc2.W.shape == (10, 10)
        assert q_network.fc3.W.shape == (3, 10)

    def test_q_network_call():
        q_network = train_dqn.Q_Network(input_size=5, hidden_size=10, output_size=3)
        x = np.random.rand(1, 5).astype(np.float32)
        output = q_network(x)
        assert output.shape == (1, 3)
