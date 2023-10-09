import json
import os
import sys
import re
import operator

from utils.util import warning

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "=": operator.eq,
    "==": operator.eq,
    "in": lambda v, l: v in l
}

def return_filter_rule(s):
    match = re.match(r"(?P<op>(>=|<=|>|<|=|==|default|others|in))(?P<value>.*)", s)
    assert match is not None, s
    rst = match.groupdict()
    op = rst["op"]
    value = rst["value"]
    if len(value) > 0:
        value = eval(value)
    # import code
    # code.interact(local=locals())
    def _rule_func(inp):
        if op in ["default", "others"]:
            return True
        return OPS[op](inp, value)
    return _rule_func


class ClusterRouter:
    ''' Decide which entry a query is routed to according to a specific dimension
    '''
    def __init__(self, filter_dim):
        self.filter_dim = filter_dim
        self.filter_entries = []

    @staticmethod
    def recur_load(hub_info, finalize_entry):
        _learner_hub = ClusterRouter(hub_info["filter_dim"])
        for _entry_dict in hub_info["filter_entries"]:
            entry_rule_str, _entry = _entry_dict["rule"], _entry_dict["entry"]
            entry_rule = return_filter_rule(entry_rule_str)
            if isinstance(_entry, dict):
                _inner_learner_hub = ClusterRouter.recur_load(_entry, finalize_entry)
                _learner_hub.filter_entries.append((entry_rule_str, entry_rule, _inner_learner_hub))
            else:
                _learner_hub.filter_entries.append((entry_rule_str, entry_rule, finalize_entry(_entry)))
        return _learner_hub

    @staticmethod
    def load_init(path, finalize_entry):
        with open(path, 'r') as fp:
            hub_info = json.load(fp)
        return ClusterRouter.recur_load(hub_info, finalize_entry)

    def switch(self, query_dict):
        if self.filter_dim is None:
            return self.filter_entries[0][-1]
        assert self.filter_dim in query_dict, (self.filter_dim, query_dict)
        for _str, _entry_rule, _entry in self.filter_entries:
            if _entry_rule(query_dict[self.filter_dim]):
                ### Hit entry
                if isinstance(_entry, ClusterRouter):
                    return _entry.switch(query_dict)
                else:
                    return _entry
            else:
                ### Missing, next rule
                pass
        warning(f"[{self.__class__.__name__}] Fail to find an entry for {query_dict}")
        return None


class BaseClusterHub:
    ''' A wrap of clustering entries, used to router a query to the corresponding entry
    There are two kinds of entries: router entry for routing (filtering) and target entry
    '''
    def __init__(self):
        self.cluster_entry = None
        
    def load_from_json(self, path):
        self.cluster_entry = ClusterRouter.load_init(path, self.finalize_entry)
    
    def add_entry_fron_config(self, target_entry, _config):
        ''' target_entry is the action we take if the filters return True
        '''
        assert self.cluster_entry is None
        filters_to_handle = []

        fix_seq_len = _config.get("FIX_SEQ_LEN", None)
        if fix_seq_len is not None:
            filter_rule_str = f"in{fix_seq_len}"
            filters_to_handle.append((filter_rule_str, "leaf_node_no", return_filter_rule(filter_rule_str)))

        flop_bound = _config.get("FLOP_BOUND", None)
        if flop_bound is not None:
            lower, upper = flop_bound
            if lower is not None:
                filter_rule_str = f">={lower}"
                filters_to_handle.append((filter_rule_str, "FLOPs", return_filter_rule(filter_rule_str)))
            if upper is not None:
                filter_rule_str = f"<{lower}"
                filters_to_handle.append((filter_rule_str, "FLOPs", return_filter_rule(filter_rule_str)))
        
        self.cluster_entry = ClusterRouter(None)
        _cur_entry = self.cluster_entry
        for _id, (filter_rule_str, filter_dim, filter_rule) in enumerate(filters_to_handle):
            _cur_entry.filter_dim = filter_dim
            _child_entry = ClusterRouter(None)
            if _id == len(filters_to_handle) - 1:
                ### Last one
                _cur_entry.filter_entries.append((filter_rule_str, filter_rule, self.finalize_entry(target_entry)))
            else:
                _cur_entry.filter_entries.append((filter_rule_str, filter_rule, _child_entry))
                _cur_entry = _child_entry

    def finalize_entry(self, target_entry):
        ''' The function to further process the entry, e.g., parse a learner from the entry
        In this BaseClusterHub, only return the origin entry
        '''
        return target_entry

    def switch(self, query_dict):
        '''
        Parameters
        ----------
        query_dict: dict
            A dict contains the query for entry
        '''
        return self.cluster_entry.switch(query_dict)

    def to_json(self):
        if self.cluster_entry is None:
            return None
        return self.to_json_per_entry(self.cluster_entry)

    def to_json_per_entry(self, _entry: ClusterRouter):
        rst = {
            "filter_dim": _entry.filter_dim,
            "filter_entries": []
        }

        for filter_rule_str, filter_rule, filter_entry in _entry.filter_entries:
            if isinstance(filter_entry, ClusterRouter):
                rst["filter_entries"].append({
                    "rule": filter_rule_str,
                    "entry": self.to_json_per_entry(filter_entry)
                })
            else:
                rst["filter_entries"].append({
                    "rule": filter_rule_str,
                    "entry": filter_entry
                })
        return rst
            
if __name__ == "__main__":
    learner_hub = BaseClusterHub()
    learner_hub.load_from_json(sys.argv[1])
    target_entry = learner_hub.switch({"leaf_node_no": 5, "flops": 1e5})
    print(target_entry)