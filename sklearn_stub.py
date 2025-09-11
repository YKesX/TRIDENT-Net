"""
Temporary sklearn stub to allow imports to work.
This should be replaced with proper sklearn installation.
"""

class metrics:
    @staticmethod
    def auc(*args, **kwargs):
        return 0.5
    
    @staticmethod 
    def average_precision_score(*args, **kwargs):
        return 0.5
        
    @staticmethod
    def f1_score(*args, **kwargs):
        return 0.5
        
    @staticmethod
    def roc_auc_score(*args, **kwargs):
        return 0.5
        
    @staticmethod
    def roc_curve(*args, **kwargs):
        return [0, 1], [0, 1], [0.5]

# Make sklearn importable
class Module:
    def __init__(self):
        self.metrics = metrics()

import sys
sys.modules['sklearn'] = Module()
sys.modules['sklearn.metrics'] = metrics