#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import codecs
import pprint
import time
import math

import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import modularity
from pyvis.network import Network 

# import numpy as np
# from matplotlib.pyplot import cm

from document import Document
 
# sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
# sys.stdin = codecs.getreader('utf_8')(sys.stdin)

class Util:
    @staticmethod
    # Pretty-print a Python object
    def pp(obj):
        pp = pprint.PrettyPrinter(indent=4, width=160)
        s = pp.pformat(obj)
        return s    
  
    @staticmethod
    # Read file name from the console
    def get_file_name():
        if (len(sys.argv) != 2):
            print("Usage: #python %s file-name" % sys.argv[0])
            sys.exit()
        return sys.argv[1]
 
class KeyGraph:
    def __init__(self, document, M=30, K=12):
        self.document = document
        self.base = self.compute_base(M)
        self.G_C = self.compute_hubs(K)

#   Compute base of frequently co-occurring words
    def compute_base(self, M):
        mtime0 = time.time()

        print("Compute base: top %d words and edges" % M)
        # Sort words by their frequency (in ascending order)
        freq_count = self.document.freq_count()
        words_freq = sorted(freq_count.items(), key=lambda x: x[1])
        
        # Compute unique words        
        self.words = [w for w, f in words_freq]

        print("Unique words:", len(self.words))
        
        # Calculate word frequency in sentences
        self.wfs = self.calculate_wfs()
        
        # Determine high frequency words
        # Include all words with the higher or same frequency than the Mth word
        wf_min = words_freq[-M][1] if len(words_freq) > M else 0
        hf = [w for w, f in words_freq if f >= wf_min]

        # Store high frequency words
        self.top_m_words = hf

        # Adjust M to the number of high frequency words
        M = len(hf)

        print("Adjust M to", M)
        print("High frequency words:", len(hf))

        # Calculate co-occurrence degree of high-frequency words
        co = self.calculate_co_occurrence(hf)

        # Keep only the tightest links
        # Include all links with the higher of same co-occurrence degree as the Mth link
        c_min = co[-M][2] if len(co) > M else 0
        co = [[i, j] for i, j, c in co if c >= c_min]

        print(Util.pp(co))

        mtime1 = time.time()
        print("Execution time of compute base before find clusters: %.4f seconds" % (mtime1 - mtime0))

        # Compute the clusters (which are the basis for islands)
        self.find_clusters(co)

        mtime2 = time.time()
        print("Execution time for find clusters: %4f" % (mtime2 - mtime1))

        return co
    
#   Calculate word frequency in sentences
    def calculate_wfs(self):
        wfs = {}
        for w in self.words:
            for s_idx, s in enumerate(self.document.sentences):
                if w not in wfs:
                    wfs[w] = {}
                wfs[w][s_idx] = s.count(w)
        return wfs
    
#   Calculate co-occurrence degree of high-frequency words
    def calculate_co_occurrence(self, hf):
        co = {}
        for hf1 in hf:
            co[hf1] = {}
            for hf2 in hf[hf.index(hf1)+1:]:
                co[hf1][hf2] = 0
                for s in self.document.sentences:
                    # Why sum products, not min, as in Ohsawa (1998)?
                    # co[hf1][hf2] += s.count(hf1) * s.count(hf2)
                    co[hf1][hf2] += min(s.count(hf1), s.count(hf2))
        co_list = []
        for x in co.keys():
            for y in co[x].keys():
                co_list.append([x, y, co[x][y]])
        co_list.sort(key=lambda a: a[2])
        return co_list

#   Detect communities in the base
#   The base is a list of pairs of words that are co-occurring in the document
#   Clusters will be used to define islands of connected words, however, the edges
#   between the clusters do not be removed to do that
    def find_clusters(self, base):
        G = nx.Graph()
        for i, j in base:
            G.add_edge(i, j)
        
        communities = girvan_newman(G)
        communities_by_quality = [(c, modularity(G, c)) for c in communities]
        c_best = sorted([(c, m) for c, m in communities_by_quality], key=lambda x: x[1], reverse=True)
        c_best = c_best[0][0]
        # print(Util.pp(communities_by_quality))
        print("Clusters:", modularity(G, c_best), c_best)
        
        # only include clusters of more than one node (for now)
        # self.clusters = [c for c in c_best if len(c) > 1]

        # Include all clusters (do not remove edges between clusters)
        self.clusters = c_best

        # for cluster in c_best:
        #     print(G.subgraph(cluster).edges())

        # The following code is for the visualization of the clusters:
        # We only want to show the connections of black nodes within clusters, not between clusters
        # This makes it easier to see the clusters (existing concepts) and the new connections
        # between them created by the red nodes (chances)
        self.new_base = [edge for cluster in c_best for edge in G.subgraph(cluster).edges()]
 
#   Compute hubs that connect words in the base
    def compute_hubs(self, K):
        print("Compute hubs: top %d key terms and bridges" % K)
        # Extract nodes in the base
        G_base = set([x for pair in self.base for x in pair])

        # Remove high frequency words from G_base, leaving non-high frequency words
        self.words = [w for w in self.words if w not in G_base]

        print("Non-high frequency words:", len(self.words))

        # Compute key terms that connect clusters
        key = self.key(self.words)

        # print("Key terms:", Util.pp(key))

        # Sort terms in D by keys
        # Include all words with the higher or same frequency than the Kth word
        high_key = sorted(key.items(), key=lambda x: x[1])
        k_min = high_key[-K][1] if len(high_key) > K else 0
        high_key = [w for w, k in high_key if k >= k_min]

        # Adjust K to the number of high key words
        K = len(high_key)
        
        print("Adjusted K:", K)
        print(Util.pp(high_key))
 
        # Calculate columns
        C = self.columns(high_key, G_base)
        
        print("Columns:", Util.pp(C))

        # Compute the top links between key terms (red nodes) and columns
        # Include all links with the higher of same co-occurrence degree as the Kth link
        c_min = C[-K][2] if len(C) > K else 0
        G_C = [[i, j] for i, j, c in C if c >= c_min]
        
        return G_C
        
    # Compute key terms that connect clusters
    def key(self, words):
        # optimization: compute the neighbors of all clusters ahead of time
        neighbors = {}
        for g_idx, g in enumerate(self.clusters):
            neighbors[g_idx] = self.neighbors(g)
        # key is a dictionary of the form　key = {w: key value}
        key = {}
        for w in words:
            # print("keyword: {}".format(w))
            product = 1.0
            for g_idx, g in enumerate(self.clusters):
                # print("g", g_)
                # print("neighbors", neighbors)
                based = self.based(w, g)
                # print("based", based)
                product *= 1 - based/neighbors[g_idx]
            key[w] = 1.0 - product
        return key

    # Count of words in sentences including words in cluster g 
    def neighbors(self, g):
        neighbors = 0
        for s, sentence in enumerate(self.document.sentences):
            g_s = 0
            for t in g:
                g_s += self.wfs[t][s]
            # print("g_s", g_s)
            for w in sentence:
                # print(s, w)
                w_s = self.wfs[w][s]
                if w in g:
                    # print("w in g")
                    neighbors += + w_s * (g_s - w_s)
                else:
                    # print("w not in g")
                    neighbors += w_s * g_s
        return neighbors
        
    # Count how many times w appeared in D based on concept represented by cluster g
    def based(self, w, g):
        based = 0
        for s, sentence in enumerate(self.document.sentences):
            # print(s, w)
            g_s = 0
            for t in g:
                g_s += self.wfs[t][s]
            # print("g_s", g_s)
            w_s = self.wfs[w][s]
            if w in g:
                # print("w in g")
                based += w_s * (g_s - w_s)
            else:
                # print("w not in g")
                based += w_s * g_s
        return based
    
    # Calculate columns c(wi,wj)
    def columns(self, hk, base):
        c = {}
        for k in hk:
            c[k] = {}
            for b in base:
                c[k][b] = 0
                for s in self.document.sentences:
                    c[k][b] += min(s.count(k), s.count(b))
        n_clusters = self.clusters_touching(c)
        print("Clusters touching:", Util.pp(n_clusters))
        c_list = [] 
        for k in c.keys():
            for b in c[k].keys():
                if n_clusters[k] > 1 and c[k][b] > 0:
                    c_list.append([k, b, c[k][b]])
        c_list.sort(key=lambda a: a[2])
        return c_list 

    # How many clusters does each column touch
    def clusters_touching(self, c):
        n_clusters = {}
        for k in c.keys():
            # print("k", k)
            n_clusters[k] = 0
            for g in self.clusters:
                # print("g", g)
                in_cluster = 0
                for b in c[k].keys():
                    # print("b", b)
                    if c[k][b] > 0 and b in g:
                        # print("b in g")
                        in_cluster = 1
                n_clusters[k] += in_cluster
        return n_clusters
    
    def draw(self):
        G = nx.Graph()
        
        # color_range = cm.rainbow(np.linspace(0, 1, len(self.clusters)))

        # Add all nodes in clusters
        for k, cluster in enumerate(self.clusters):
            G.add_nodes_from(cluster, color='black')

        # Add edges for nodes in clusters
        for i, j in self.base:
            if (i, j) in self.new_base:
                G.add_edge(i, j, smooth=True)
        
        # Add edges for nodes in key terms
        for i, j in self.G_C:
            G.add_node(i, color='red')
            G.add_edge(i, j, color='red', smooth=True)

        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))

        return G

    def write_to_file(self, G, fname, M, K):
        network = Network("600px", "100%")
        network.from_nx(G)
        network.save_graph(f"./graphs/{fname}_{M}_{K}.html")
        # In pyvis 0.3.2, the following code looks for a non-existent template
        # network.show(f"./graphs/{fname}_{M}_{K}.html")
             
#-----------Main----------------
if __name__ == "__main__":
    stime = time.time() 
    
#   Create a document
    fname = Util.get_file_name()
    doc = Document(file_name = 'txt_files/' + fname + '.txt')
        
#   Create a keygraph
    kg = KeyGraph(doc, M=5, K=5) # default: M=30, K=12
    print("clusters", kg.clusters)

    mtime = time.time()
    G = kg.draw()
    kg.write_to_file(G, fname, M, K)
    print("Time to draw keygraph: %.4f", (mtime - stime))
    
    etime = time.time()
    print("Execution time: %.4f seconds" % (etime - stime))
