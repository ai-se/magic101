import sys
import xml.etree.ElementTree
import re
import validators
import urllib

__author__ = "Jianfeng Chen"
__copyright__ = "Copyright (C) 2016 Jianfeng Chen"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "jchen37@ncsu.edu"

"""
Supporting methods.
See docs in feature_model.py
"""


class Node(object):
    def __init__(self, identification, parent=None, node_type='o'):
        self.id = identification
        self.parent = parent
        self.node_type = node_type
        self.children = []
        if node_type == 'g':
            self.g_u = 1
            self.g_d = 0

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

    def __repr__(self):
        # return '\n id: %s\n type:%s \n' % (
        #     self.id,
        #     self.node_type)
        return '%s|%s' % (self.node_type, self.id)


class Constraint(object):
    def __init__(self, identification, literals, literals_pos):
        self.id = identification
        self.literals = literals
        self.li_pos = literals_pos

    def __repr__(self):
        return self.id + '\n' + str(self.literals) + '\n' + str(self.li_pos)

    def is_correct(self, ft, filled_form):
        """
        not supported for filled_form containing -1
        try to apply is_violate if needed.
        """
        for li, pos in zip(self.literals, self.li_pos):
            i = ft.find_fea_index(li)
            if int(pos) == filled_form[i]:
                return True
        return False

    def is_violated(self, ft, filled_form):
        for li, pos in zip(self.literals, self.li_pos):
            i = ft.find_fea_index(li)
            filled = filled_form[i]
            if filled == -1 or int(pos) == filled_form[i]:
                return False
        return True


class FeatureTree(object):
    def __init__(self):
        self.root = None
        self.features = []
        self.groups = []
        self.leaves = []
        self.con = []
        self.featureNum = 0
        self.subtree_index_dict = dict()

    def set_root(self, root):
        self.root = root

    def add_constraint(self, con):
        self.con.append(con)

    def find_fea_index(self, id_or_nodeObj):
        if type(id_or_nodeObj) is not str:
            identification = id_or_nodeObj.id
        else:
            identification = id_or_nodeObj

        if not hasattr(self, 'fea_index_dict'):
            self.fea_index_dict = dict()
            for f_i, f in enumerate(self.features):
                self.fea_index_dict[f.id] = f_i

        return self.fea_index_dict[identification]

    # fetch all the features in the tree basing on the children structure
    def set_features_list(self):
        def setting_feature_list(node):
            if node.node_type == 'g':
                node.g_u = int(node.g_u) if node.g_u != sys.maxint else len(node.children)
                node.g_d = int(node.g_d) if node.g_d != sys.maxint else len(node.children)
                self.features.append(node)
                self.groups.append(node)
            if node.node_type != 'g':
                self.features.append(node)
            if len(node.children) == 0:
                self.leaves.append(node)
            for i in node.children:
                setting_feature_list(i)

        setting_feature_list(self.root)
        self.featureNum = len(self.features)

    def post_order(self, node, func, extra_args=[]):
        """children, then the root"""
        if node.children:
            for c in node.children:
                self.post_order(c, func, extra_args)
        func(node, *extra_args)

    def pre_order(self, node, func, extra_args=[]):
        """root, then children"""
        func(node, *extra_args)
        if node.children:
            for c in node.children:
                self.pre_order(c, func, extra_args)

    def check_fulfill_valid(self, fill):
        """
        checking a given fulfill lst whether consistent with the feature model tree structure
        :param fill:
        :return:
        """

        def find(x):
            return fill[self.find_fea_index(x)]

        def check_node(node):
            if not node.children:
                return True

            if find(node) == 0:
                return True

            child_sum = sum([find(c) for c in node.children])

            for m_child in filter(lambda x: x.node_type in ['m', 'r'], node.children):
                if find(m_child) == 0:
                    # print m_child
                    # pdb.set_trace()
                    return False

            if node.node_type is 'g':
                if not (node.g_d <= child_sum <= node.g_u):
                    # print node
                    # pdb.set_trace()
                    return False

            for child in node.children:
                if find(child) == 1:
                    t = check_node(child)
                    if not t:
                        # print child
                        # pdb.set_trace()
                        return False
            return True

        if fill[0] == 0:
            return False

        return check_node(self.root)

    def fill_form4all_fea(self, form):
        # setting the form by the structure of feature tree
        # leaves should be filled in the form in advanced
        # all not filled feature should be -1 in the form
        def filling(node):
            index = self.features.index(node)
            if form[index] != -1:
                return
            # handling the group features
            if node.node_type == 'g':
                sum = 0
                for c in node.children:
                    i_index = self.features.index(c)
                    sum += form[i_index]
                form[index] = 1 if node.g_d <= sum <= node.g_u else 0
                return

            """
            # the child is a group
            if node.children[0].node_type == 'g':
                form[index] = form[index+1]
                return
            """

            # handling the other type of node
            m_child = [x for x in node.children if x.node_type in ['m', 'r', 'g']]
            o_child = [x for x in node.children if x.node_type == 'o']
            if len(m_child) == 0:  # all children are optional
                s = 0
                for o in o_child:
                    i_index = self.features.index(o)
                    s += form[i_index]
                form[index] = 1 if s > 0 else 0
                return
            for m in m_child:
                i_index = self.features.index(m)
                if form[i_index] == 0:
                    form[index] = 0
                    return
            form[index] = 1
            return

        self.post_order(self.root, filling)

    def fill_subtree_0(self, subtree_root, fulfill):
        """
        Setting the subtree rooted by node zeros.
        Fulfill vector will be modified
        NOTHING WILL BE RETURNED
        """

        def fill_zero(node, fill_vec):
            node_index = self.features.index(node)
            fill_vec[node_index] = 0

        self.post_order(subtree_root, fill_zero, [fulfill])

    def get_subtree_index(self, subtree_root):
        def fetch_indices(node, lst):
            lst.append(self.find_fea_index(node))

        lst = []
        self.post_order(subtree_root, fetch_indices, [lst])
        return lst

    def get_subtree_index_dict(self):
        for f_i, f in enumerate(self.features):
            self.subtree_index_dict[f_i] = sorted(self.get_subtree_index(f))

    def get_feature_num(self):
        return len(self.features) - len(self.groups)

    def get_cons_num(self):
        return len(self.con)

    def get_tree_height(self):
        height = 1
        for fea in self.features:
            h = 0
            cursor = fea
            while cursor.parent:
                h += 1
                cursor = cursor.parent
            height = max(height, h)

        return height

    def load_ft_from_url(ft, url):
        if validators.url(url):
            url = urllib.urlopen(url)
        # load the feature tree and constraints
        tree = xml.etree.ElementTree.parse(url)
        root = tree.getroot()

        for child in root:
            if child.tag == 'feature_tree':
                feature_tree = child.text
            if child.tag == 'constraints':
                constraints = child.text
        # parse the feature tree text
        feas = feature_tree.split("\n")
        feas = filter(bool, feas)
        common_feature_pattern = re.compile('(\t*):([romg]?).*\W(\w+)\W.*')
        group_pattern = re.compile('\t*:g \W(\w+)\W \W(\d),([\d\*])\W.*')
        layer_dict = dict()
        for f in feas:
            m = common_feature_pattern.match(f)
            """
            m.group(1) layer
            m.group(2) type
            m.group(3) id
            """
            layer = len(m.group(1))
            t = m.group(2)
            if t == 'r':
                tree_root = Node(identification=m.group(3), node_type='r')
                layer_dict[layer] = tree_root
                ft.set_root(tree_root)
            elif t == 'g':
                mg = group_pattern.match(f)
                """
                mg.group(1) id
                mg.group(2) down_count
                mg.group(3) up_count
                """
                gNode = Node(identification=mg.group(1), parent=layer_dict[layer - 1], node_type='g')
                layer_dict[layer] = gNode
                if mg.group(3) == '*':
                    gNode.g_u = sys.maxint
                else:
                    gNode.g_u = mg.group(3)
                gNode.g_d = mg.group(2)
                layer_dict[layer] = gNode
                gNode.parent.add_child(gNode)
            else:
                treeNode = Node(identification=m.group(3), parent=layer_dict[layer - 1], node_type=t)
                layer_dict[layer] = treeNode
                treeNode.parent.add_child(treeNode)

        # parse the constraints
        cons = constraints.split('\n')
        cons = filter(bool, cons)
        common_con_pattern = re.compile('(\w+):(~?)(\w+)(.*)\s*')
        common_more_con_pattern = re.compile('\s+(or) (~?)(\w+)(.*)\s*')

        for cc in cons:
            literal = []
            li_pos = []
            m = common_con_pattern.match(cc)
            con_id = m.group(1)
            li_pos.append(not bool(m.group(2)))
            literal.append(m.group(3))
            while m.group(4):
                cc = m.group(4)
                m = common_more_con_pattern.match(cc)
                li_pos.append(not bool(m.group(2)))
                literal.append(m.group(3))
            """
             con_id: constraint identifier
             literal: literals
             li_pos: whether is positive or each literals
            """
            con_stmt = Constraint(identification=con_id, literals=literal, literals_pos=li_pos)
            ft.add_constraint(con_stmt)

        ft.set_features_list()
