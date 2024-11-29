import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import random
import functools
from igraph import *
from utils import MAP, MRR

SEED = 0
DATA = ['../data/facebook.txt', '../data/arxiv.txt', '../data/bio-celegans-dir/bio-celegans-dir.edges', '../data/cora/cora.cites']
ALGO = ['AA', 'PA', 'CN', 'JC', 'Katz']
DATA_ID = 0
ALGO_ID = 2
TRAIN_SPLIT = 0.6
MAX_DEPTH = 2
BETA = 0.1
K = 5


def parse_file(file):
	'''returns edge list, stored in the file'''
	with open(file, 'r') as f:
		edges = np.array(list(map(lambda x: list(map(int, x.split())), f.read().splitlines())))
		print(edges)
	return edges

def split_graph(edge_list):
	'''returns train-test split graphs'''
	n = len(edge_list)
	indices = set(range(n))
	train_idx = set(random.sample(range(n), int(TRAIN_SPLIT*n)))
	test_idx = indices - train_idx
	train_edges = [edge_list[i] for i in train_idx]
	test_edges = [edge_list[i] for i in test_idx]
	return train_edges, test_edges

def vertices(edges):
	'''returns list of vertices from the edges'''
	vert = set()
	for u, v in edges:
		vert.add(u)
		vert.add(v)
	return vert

def rank_list_top_K(graph, sim, i, k):
	non_ngbr = filter(lambda x: i!=x and graph[i, x]!=1, range(len(sim[i])))
	return sorted(non_ngbr, key=lambda x: sim[i][x], reverse=True)[:k]

# def evaluate(sim, train_graph, test_graph, train_vertices, test_vertices):
# 	precision, recall = 0, 0
# 	c = 0
# 	for i in test_vertices:
# 		if i in train_vertices:
# 			true_ngbr = set(test_graph.neighbors(i))
# 			K2 = min(K, len(true_ngbr))
			
# 			top_k = set(rank_list_top_K(train_graph, sim, i, K2))
# 			precision += len(top_k & true_ngbr)/float(K2)
# 			recall += len(top_k & true_ngbr)/float(len(true_ngbr))
# 			c += 1
# 	print("precision: {:.2f}".format(precision/c))
# 	print("recall: {:.2f}".format(recall/c))

def evaluate(sim, train_graph, test_graph, train_vertices, test_vertices):
	# mAP = 0.0
	# mRR = 0.0
	# num = 0.0
	# for u in test_vertices:
	# 	if u in train_vertices:
	# 		precision, rr = 0.0, 0.0
	# 		cnt = 0.0
	# 		true_ngbr = set(test_graph.neighbors(u))
	# 		K2 = min(K, len(true_ngbr))
	# 		top_k = set(rank_list_top_K(train_graph, sim, u, K2))

	# 		found_first = False
	# 		for i, v in enumerate(top_k):
	# 			if v in true_ngbr:
	# 				cnt += 1.0
	# 				precision += cnt*1.0/(i+1.0)
	# 				if not(found_first):
	# 					rr += 1.0/(i+1.0)
	# 					found_first = True
	# 		avg_pre = precision/len(true_ngbr)
	# 		# print(u, "avg_pre:", avg_pre)
	# 		# print("top_k:", top_k)
	# 		# print("true_ngbr:", true_ngbr)
	# 		# print()
	# 		mAP += avg_pre
	# 		mRR += rr
	# 		num += 1.0
	# mAP /= num
	# mRR /= num

	# K2 = min(K, len(true_ngbr))
	n = sim.shape[0]
	indexlist = []
	for i in range(n):
		for j in range(i+1, n):
			indexlist.append([i, j, sim[i,j]])

	mAP = MAP(indexlist, test_graph, test_vertices, sim.shape[0])
	mRR = MRR(indexlist, test_graph, test_vertices, sim.shape[0])

	print("MAP: {:.2f}".format(mAP))
	print("MRR: {:.2f}".format(mRR))
	


class LinkPred:
	def __init__(self, file):
		edges = parse_file(file)
		self.train_edges, self.test_edges = split_graph(edges)
		self.train_graph = Graph(self.train_edges)
		self.test_graph = Graph(self.test_edges)
		self.train_n = self.train_graph.vcount()
		self.train_vertices = vertices(self.train_edges)
		self.test_vertices = vertices(self.test_edges)
		self.algo = ALGO[ALGO_ID]

	def simple_sim(self, u, v):
		if (u%1000==0 and v%1000==0):
			print(u, v)
		if (u==v) or (u not in self.train_vertices) or (v not in self.train_vertices):
			return 0.0

		if self.algo == 'PA':
			# print(u, v)
			return self.train_graph.degree(u) * self.train_graph.degree(v)
		else:
			u_ngbrs = set(self.train_graph.neighbors(u))
			v_ngbrs = set(self.train_graph.neighbors(v))
			if self.algo == 'AA':
				# degrees = list(map(lambda x: self.train_graph.degree(x), (u_ngbrs & v_ngbrs) ))
				# return np.sum(1.0/np.log(degrees))
				return sum([1.0/math.log(self.train_graph.degree(x)) for x in (u_ngbrs & v_ngbrs)])
			elif self.algo == 'CN':
				return len(u_ngbrs & v_ngbrs)
			elif self.algo == 'JC':
				return len(u_ngbrs & v_ngbrs)/float(len(u_ngbrs | v_ngbrs))

	def katz_sim(self, u, v):
		if (u%100==0 and v%1000==0):
			print(u, v)
		score = 0.0
		ngbr = self.train_adjlist[u]

		i = 1
		while i <= MAX_DEPTH:
			path_count = ngbr.count(v)
			if path_count > 0:
				score += (BETA**i) * path_count

			# new_ngbr = list(map(lambda x: self.train_adjlist[x], ngbr))
			# new_ngbr = functools.reduce(operator.iconcat, new_ngbr, [])
			new_ngbr = []
			for x in ngbr:
				new_ngbr.extend(self.train_adjlist[x])
			ngbr = new_ngbr
			i += 1

		return score

	def run_algo(self, algo):
		self.algo = algo
		if self.algo != 'Katz':
			vec_sim = np.vectorize(self.simple_sim)
		else:
			tmp_dict = self.train_graph.get_adjlist()
			self.train_adjlist = {}
			for i, l in enumerate(tmp_dict):
				self.train_adjlist[i] = l
			vec_sim = np.vectorize(self.katz_sim)

		arr = np.arange(self.train_n)
		sim = vec_sim(arr.reshape(-1, 1), arr.reshape(1, -1))
		print(sim[1][:100])
		evaluate(sim, self.train_graph, self.test_graph, self.train_vertices, self.test_vertices)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Benchmark Heuristic-based Link prediction methods')
	parser.add_argument('-data', metavar='D', type=int, default=DATA_ID, help='Choose Data [facebook, arxiv]')
	parser.add_argument('-algo', metavar='A', type=str, default=ALGO[ALGO_ID], help='Choose Algorithm [AA, PA, CN, JC, Katz]')
	parser.add_argument('-seed', metavar='S', type=int, default=SEED, help='random seed for experiments')

	args = parser.parse_args()
	SEED = args.seed
	DATA_ID = args.data
	# ALGO_ID = args.algo 

	# random.seed(SEED)
	# np.random.seed(SEED)

	G = LinkPred(DATA[DATA_ID])
	G.run_algo(args.algo)
