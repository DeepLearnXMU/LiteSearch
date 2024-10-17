DEBUG = False

class Node:
    def __init__(self, content, value, parent, timestep, tree, leaf=False, budget=0):
        self.content = content
        self.parent = parent
        self.children = []
        self.value = value
        self.timestep = timestep
        self.leaf = leaf
        self.tree = tree
        self.budget = budget

    def get_depth(self):
        return len(self.return_path()) + 1

    def return_path(self):
        if self.content is None:
            return []
        if self.parent is None:
            return [self.content]
        return self.parent.return_path() + [self.content]

    def print_path(self):
        return "\n".join(self.return_path())

    def append_child(self, child_content, child_value, child_timestep, child_leaf=False, child_budget=0):
        child = Node(child_content, child_value, self, child_timestep, self.tree, child_leaf, child_budget)
        self.children.append(child)
        self.tree.all_nodes.append(child)
        return child

    def left_budget_size(self):
        return self.budget - len(self.children) if not self.leaf else 0

class Tree:
    def __init__(self, question, answer, additional_info=None):
        self.question = question
        self.answer = answer # provided, but will not used when searching
        self.all_nodes = []
        self.root = None # wait init
        self.additional_info = additional_info

    def init_root_node(self, value):
        root_budget = self.calculate_budget(value, 1)
        self.root = Node(None, value, None, 0, self, False, root_budget)
        self.all_nodes.append(self.root)
        if DEBUG:
            print("root budget:", root_budget)
    
    def select_best_node(self):
        available_nodes = [node for node in self.all_nodes if node.left_budget_size() > 0 or node.leaf]
        if DEBUG:
            print("available nodes:", len(available_nodes))
        if len(available_nodes) == 0:
            return None
        best_node = max(available_nodes, key=lambda x: x.value)
        if best_node.leaf:
            if DEBUG:
                print("best node is leaf:", best_node.value)
            return None
        elif best_node.parent is not None and best_node.value < 0.01: # bad node
            return None
        else:
            if DEBUG:
                print("best node is not leaf:", best_node.value)
            return best_node

    def expand_node(self, node, child_content, child_value, timestep, leaf=False):
        child_budget = self.calculate_budget(child_value, node.get_depth() + 1)
        if DEBUG:
            print("child value:", child_value, "child budget:", child_budget, "is leaf:", leaf, "text:", child_content)
        return node.append_child(child_content, child_value, timestep, leaf, child_budget)

    def calculate_budget(self, value, depth):
        if value <= 0:
            return 0
        C = 1
        if "greedy_value" in self.additional_info:
            greedy_value = self.additional_info["greedy_value"]
            value = (value + greedy_value * 1 / depth) / (1 + 1 / depth)
        MAX_BUDGET = self.additional_info["max_budget"]
        EXPECTED = self.additional_info["expected"]
        while C < MAX_BUDGET:
            if 1 - (1 - value) ** C > EXPECTED:
                break
            C += 1
        return min(max(round(C * 1 / depth), 2), MAX_BUDGET)

    def return_best_path(self, use_greedy=True):
        has_greedy = "greedy" in self.additional_info and "greedy_value" in self.additional_info and use_greedy
        leaf_nodes = [node for node in self.all_nodes if node.leaf]
        if leaf_nodes:
            state = max(leaf_nodes, key=lambda x: x.value)
            if has_greedy and self.additional_info["greedy_value"] > state.value:
                return self.additional_info["greedy"], self.additional_info["greedy_value"]
            else:
                return state.print_path(), state.value
        if has_greedy:
            return self.additional_info["greedy"], self.additional_info["greedy_value"]
        else:
            return None, None

    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes])

    def is_finished(self, min_value):
        best_value = self.return_best_path(use_greedy=False)[1]
        if best_value is None:
            return False
        if DEBUG:
            print("best value:", best_value, "greedy value:", self.additional_info["greedy_value"])
        return best_value > max(min_value, self.additional_info["greedy_value"] - 0.01 if "greedy_value" in self.additional_info else 0)
    
    def get_leaf_num(self):
        return len([node for node in self.all_nodes if node.leaf])
