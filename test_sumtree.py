import unittest
import random
import numpy as np
from memory import SumTree, PrioritizedMemory


class TestSumTree(unittest.TestCase):
    """Test SumTree implementation"""
    
    
    def test_SumTree(self):
        print('')
        print('---- TEST SUM TREE --- ')
        size = 10
        mem = SumTree(size)
        probs = [p for p in range(size)]
        for p in probs:
            mem.add((p), p)
         
        print('Sum Tree')
        mem.printTree(True)
        
        # Test initialization
        self.assertIs(mem.total(), sum(probs))
        self.assertIs(mem.n, size)
        
        # Test sampling
        self.assertIs(mem.sample(mem.total()).prob, probs[9])
        self.assertIs(mem.sample(mem.total()).experience, probs[9])
        self.assertIs(mem.sample(0).prob, probs[0])
        self.assertIs(mem.sample(0).experience, probs[0])
        
        # Test probability update
        mem.update(mem.tree[4].prob + 10, idx = 4)
        mem.update(mem.tree[9].prob + 10, idx = 9)
        print('')
        print('Change probability of leaf idx [4]')
        mem.printTree(True)
        self.assertIs(mem.total(), sum(probs) + 20)
        self.assertIs(mem.tree[4].parent.prob, 19)
        
        # Test add when full
        mem.add((50), 5)
        mem.add((80), 8)
        print('')
        print('Add two new leafs with memory full "(50),5" and "(80),8"')
        mem.printTree(True)
        self.assertIs(mem.total(), sum(probs) + 20 - 1 + 13)

class TestPrioritizedMem(unittest.TestCase):
    """Test Prioritized Experience Replay implementation"""
    def test_PriorityMemory(self):
        action_size = 4
        BUFFER_SIZE = 32*4
        BATCH_SIZE = 32
        seed = 420

        mem = PrioritizedMemory(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        for i in range(BATCH_SIZE):
            for j in range(int(BUFFER_SIZE/BATCH_SIZE)):
                mem.add(i, i, i, i, i)
         
        # Test initial probabilities
        self.assertEqual([n.prob for n in mem.memory.tree], [1.0]*BUFFER_SIZE)
        
        # Test Update probability
        probs_ = []
        for i in range(BUFFER_SIZE):
            new_prob = random.uniform(0.01, 3)
            p = (new_prob + mem.prob_e)**mem.alpha
            probs_.append(p)
            mem.update(i,new_prob)
            self.assertAlmostEqual(mem.memory.tree[i].prob, p)   

        self.assertAlmostEqual(mem.memory.total(), sum(probs_))
        
        # Test sampling
        step = mem.memory.total() / BATCH_SIZE
        probs = np.array([n.prob for n in mem.memory.tree])
        sorted_prob_idx = np.argsort([n.prob for n in mem.memory.tree])
        sorted_prob = probs[sorted_prob_idx]
        end_first_bin = mem.memory.sample(step).prob
        end_first_bin_idx = np.where(sorted_prob == end_first_bin)[0][0]
        exps = np.array([n.experience[0] for n in mem.memory.tree])
        sorted_exps = exps[sorted_prob_idx]
        
        start_last_bin = mem.memory.sample(mem.memory.total() - step).prob
        start_last_bin_idx = np.where(sorted_prob == start_last_bin)[0][0]
        
        samp = mem.sample()[0].numpy()[0][0]
        first_exps =sorted_exps[:end_first_bin_idx]
        self.assertTrue(samp in first_exps)
        samp = mem.sample()[0].numpy()[0][-1]
        last_exps = sorted_exps[start_last_bin_idx:]
        self.assertTrue(samp in last_exps)
        
        #Test get Prob
        self.assertAlmostEqual(mem.get_prob(42), probs_[42])
   
        #Test add sample when memory is full
        total_ = mem.memory.total()
        new_exps = []
        for i in range(1,BATCH_SIZE+1):
            mem.add(i*100, i*100, i*100, i*100, i*100)
            new_exps.append(i*100)
            self.assertAlmostEqual(mem.get_prob(i-1), sorted_prob[-1])
            
        samp = mem.sample()[0].numpy()[0][-1]
        self.assertTrue(samp in ([sorted_exps[-1]] + new_exps))

        
def run_tests():
    unittest.main(TestSumTree(),argv=[''],verbosity=2, exit=False)
    unittest.main(TestPrioritizedMem(),argv=[''],verbosity=2, exit=False)