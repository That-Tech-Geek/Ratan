"""Dependency-free benchmark metrics for Attune.

These functions are deliberately simple and auditable. They do not claim clinical validity.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class BinaryMetrics:
    tp:int; tn:int; fp:int; fn:int
    @property
    def precision(self): return self.tp/(self.tp+self.fp) if self.tp+self.fp else 0.0
    @property
    def recall(self): return self.tp/(self.tp+self.fn) if self.tp+self.fn else 0.0
    @property
    def false_negative_rate(self): return self.fn/(self.tp+self.fn) if self.tp+self.fn else 0.0
    @property
    def false_positive_rate(self): return self.fp/(self.fp+self.tn) if self.fp+self.tn else 0.0

def binary_metrics(expected, predicted, positive="crisis"):
    tp=tn=fp=fn=0
    for e,p in zip(expected,predicted):
        if e==positive and p==positive: tp+=1
        elif e!=positive and p!=positive: tn+=1
        elif e!=positive: fp+=1
        else: fn+=1
    return BinaryMetrics(tp,tn,fp,fn)
