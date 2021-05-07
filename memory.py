from collections import namedtuple
import random
import torch
import numpy as np
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SumTreeNode:
    """Sum Tree Node."""
            
    def __init__(self, left, right, is_leaf = False, idx = None):
        """
        Params
        ======
            left (SumTreeNode): left node connection.
            right (SumTreeNode): right node connection.
            is_leaf (boolean): Random seed.
            idx (boolean): Node idx.
        """
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            if self.right is not None:
                self.prob = self.left.prob + self.right.prob
            else:
                self.prob = self.left.prob
        else:
            self.idx = idx
        self.parent = None
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self
    @classmethod
    def create_leaf(cls, prob, exp, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.prob = prob
        leaf.experience = exp
        return leaf
    
class SumTree:
    """SumTree Implementation.
    
    
    "the value of a parent node is the sum of its children. Leaf nodes store the 
    transition priorities and the internal nodes are intermediate sums, with the
    parent node containing the sum over all priorities, ptotal. This provides a 
    efficient way of calculating the cumulative sum of priorities, allowing O(log N)
    updates and sampling" - Schaul et al., 2016
    """
    
    write = 0
    
    def __init__(self, n_leafs):
        """
        Params
        ======
            n_leafs (boolean): Number of leafs in the SumTree.
        """
        self.n_leafs = n_leafs
        self.root, self.tree = self._create_tree(n_leafs)
        self.n = 0
    
    
    def _create_tree(self, n_leafs): 
        """Populate de SumTree with SumTreeNode's."""
        nodes = [SumTreeNode.create_leaf(p, p, i) for i, p in enumerate([0]*n_leafs)]
        leaf_nodes = nodes
        while len(nodes) > 1:
            _nodes = []
            for i in range(0,len(nodes),2):
                if i + 1 == len(nodes):
                    _nodes.append(SumTreeNode(nodes[i], None))
                else:
                    _nodes.append(SumTreeNode(nodes[i], nodes[i+1]))
            nodes = _nodes
        return nodes[0], leaf_nodes
    
    def _retrieve(self, prob, node):
        """Retrive next node based on 'prob' value."""     
        if node.is_leaf:
            return node
        if node.left.prob >= prob:
            return self._retrieve(prob, node.left)
        else:
            return self._retrieve(prob - node.left.prob, node.right)
        
    def _propagate_changes(self, change, node):
        """Propagate probability changes."""
        node.prob += change
        if node.parent is not None:
            self._propagate_changes(change, node.parent)
        
    def total(self):
        """Total sum of leafs probabilities."""
        return self.root.prob
        
    def sample(self, prob):
        """Sample leaf node based on 'prob' value."""
        return self._retrieve(prob, self.root)
    
    def update(self, new_prob, node = None, idx = None):
        """Update leaf node probability."""
        if node is None and idx is None:
            print('Missing node or idx argument')
        if idx is not None:
            node = self.tree[idx]
        change = new_prob - node.prob
        node.prob = new_prob
        self._propagate_changes(change, node.parent)
  
    def add(self, exp, prob):
        """Add experience 'exp' with probability 'prob' to the SumTree."""
        node = self.tree[self.write]
        self.update(prob, node=node)
        node.experience = exp
        
        if self.write == self.n_leafs - 1:
            self.write = 0
        else:
            self.write += 1
            
        if self.n < self.n_leafs:
            self.n += 1
            
    def printTree(self, with_exp=False):
        """Print SumTree nodes. 
        Params
        ======
            with_exp (boolean): If True print node experience and probability, else print only probability.
        """
        size = len(self.tree)
        nodes = [self.root]
        while len(nodes) <= size:
            nodes_ = []
            for n in nodes:
                if n is not None:
                    if with_exp and len(nodes) == size:
                        print(f'{n.experience}({n.prob})', end=" ")
                    else:
                        print(f'{n.prob}', end=" ")
                    nodes_.append(n.left)
                    nodes_.append(n.right)
            print('\r')
            nodes = nodes_
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
            
            


class PrioritizedMemory:
    """Prioritized memory implementation based on proportional prioritization method of Schaul et al., 2016.
    
    REFERENCE
    ========
    Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.
    https://arxiv.org/pdf/1511.05952.pdf
    """
    
    prob_e = .2 # 'e' small positive constant that prevents the edge-case of transitions not being revisited once their _
                # error is zero.
    alpha = 0.6 # alpha of Schaul et al., 2016 paper (0.6 is the paper best parameter).

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a PrioritizedReplayMemory object.

        Params
        ======
            action_size (int): dimension of each action.
            buffer_size (int): maximum size of buffer.
            batch_size (int): size of each training batch.
            seed (int): random seed.
        """
        self.action_size = action_size
        self.memory = SumTree(buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self._max_prob = 1 - self.prob_e

    def _priority(self, TD_delta):
        """Compute the priority 'pi'."""
        return (abs(TD_delta) + self.prob_e)**self.alpha
    
#     def add(self, state, action, reward, next_state, done, TD_delta):
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        exp = self.experience(state, action, reward, next_state, done)
        prob = self._priority(self._max_prob)
        self.memory.add(exp, prob)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory.
        
        Implemented accordingly to the original paper:
        
        'To sample a minibatch of size k, the range [0, ptotal] is divided equally into k ranges.
        Next, a value is uniformly sampled from each range. Finally the transitions that correspond to each
        of these sampled values are retrieved from the tree.' - Schaul et al., 2016
        
        """
        prob_sum = self.memory.total()
        prob_step = prob_sum/self.batch_size
        batch_nodes = []
        
        for i in range(self.batch_size):
            lower_boundary = i*prob_step
            upper_boundary = (i + 1)*prob_step
            
            prob = np.random.uniform(lower_boundary,upper_boundary)
            batch_nodes.append(self.memory.sample(prob))

        states = torch.from_numpy(np.vstack([node.experience.state for node in batch_nodes if node is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([node.experience.action for node in batch_nodes if node is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([node.experience.reward for node in batch_nodes if node is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([node.experience.next_state for node in batch_nodes if node is not None]))\
        .float().to(device)
        dones = torch.from_numpy(np.vstack([node.experience.done for node in batch_nodes if node is not None]).astype(np.uint8))\
        .float().to(device)
        node_idxs = [node.idx for node in batch_nodes if node is not None]
        
        return (states, actions, rewards, next_states, dones, node_idxs)

    def update(self, node_idx, TD_delta):
        """Update node probabilities."""
        new_prob = self._priority(TD_delta)
        self.memory.update(new_prob, idx=node_idx)
        self._max_prob = max(self._max_prob, TD_delta)
    
    def get_prob(self, node_idx):
        """Get probability of node by idx."""
        return self.memory.tree[node_idx].prob
        
    def __len__(self):
        """Return the current size of internal memory."""
        return self.memory.n      