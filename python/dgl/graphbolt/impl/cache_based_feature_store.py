"""Torch-based feature store for GraphBolt."""
from typing import List

import numpy as np

import torch

from ..feature_store import FeatureStore
from .ondisk_metadata import OnDiskFeatureData

__all__ = ["CacheBasedFeatureStore"]

class CacheBase:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        
    def put(self, key: int, val: torch.Tensor):
        raise NotImplementedError
    
    def get(self, key: torch.Tensor):
        raise NotImplementedError

class LRUCache(CacheBase):

    class ListNode:
        def __init__(self, key: int, val: torch.Tensor):
            self.key = key
            self.val = val
            self.pre = None
            self.nxt = None

        def delete(self):
            if self.pre != None:
                self.pre.nxt = self.nxt
            if self.nxt != None:
                self.nxt.pre = self.pre
            self.pre = self.nxt = None

    def __init__(self, fallback_store: FeatureStore, cache_size: int):
        super().__init__(cache_size)
        self.fallback = fallback_store
        self.sample = fallback_store.read(torch.LongTensor([0]))
        self.val_shape = list(self.sample.shape)
        self.list_head = None
        self.list_tail = None
        self.cache = {}
        self.counter = 0
        self.hit_counter = 0
        self.get_counter = 0

    def _add_to_list_tail(self, node: ListNode):
        if self.list_tail == None:
            self.list_tail = node
        else:
            self.list_tail.nxt = node
            node.pre = self.list_tail
            node.nxt = None
            self.list_tail = node
        if self.list_head == None:
            self.list_head = node
    
    def _move_to_list_tail(self, node: ListNode):
        if self.list_tail == None:
            self.list_head = node
            self.list_tail = node
            return
        if self.list_tail == node:
            return
        if self.list_head == node:
            self.list_head = node.nxt
        node.delete()
        self.list_tail.nxt = node
        node.pre = self.list_tail
        self.list_tail = node
    
    def put(self, key: int, val: torch.Tensor):
        if self.counter < self.cache_size:
            newnode = self.ListNode(key, val)
            self.cache[key] = newnode
            self._add_to_list_tail(newnode)
            self.counter += 1
        else:
            replaced_node = self.list_head.key
            self.list_head = self.list_head.nxt
            del self.cache[replaced_node]
            newnode = self.ListNode(key, val)
            newnode.pre = self.list_tail
            newnode.nxt = None
            self.cache[key] = newnode
            self.list_tail.nxt = newnode
            self.list_tail = newnode
    
    def get(self, keys: torch.Tensor):
        if keys is None:
            return self.fallback.read()
        if keys.shape[0] == 0:
            return torch.Tensor([])
        cache_tensor = torch.tensor(list(self.cache.keys()))
        hit = torch.isin(keys, cache_tensor)

        missing_keys = keys[hit == False]
        missing_vals = self.fallback.read(missing_keys)
        hit_count = keys.shape[0] - missing_keys.shape[0]
        rest_count = self.cache_size - hit_count
        self.get_counter += keys.shape[0]
        self.hit_counter += hit_count
        print(keys.shape[0], hit_count, self.cache_size - len(self.cache))

        vcount = 0

        res_shape = self.val_shape
        res_shape[0] = keys.shape[0]
        res = self.sample.expand(res_shape).clone()
        
        for i in range(keys.shape[0]):
            nid = int(keys[i])
            if hit[i]:
                vcount += 1
                res[i] = self.cache[nid].val
                self._move_to_list_tail(self.cache[nid])

        if missing_keys.shape[0] <= rest_count:
            for i in range(missing_keys.shape[0]):
                if not int(missing_keys[i]) in self.cache:
                    self.put(int(missing_keys[i]), missing_vals[i])
            res[hit == False] = missing_vals
        else:
            for i in range(rest_count):
                if not int(missing_keys[i]) in self.cache:
                    vcount += 1
                    self.put(int(missing_keys[i]), missing_vals[i])
            res[hit == False] = missing_vals
        return res

class CacheBasedFeatureStore(FeatureStore):

    def __init__(self, fallback_store: FeatureStore, cache_size: int = 1024, replacement_policy: str = None):
        super().__init__()
        assert isinstance(fallback_store, FeatureStore), "Error"
        self.cache_size = cache_size
        self.cache = LRUCache(fallback_store, cache_size)

    def read(self, ids: torch.Tensor = None):
        return self.cache.get(ids)
    
    def hit_rate(self):
        return self.cache.hit_counter / self.cache.get_counter if self.cache.get_counter > 0 else 0
