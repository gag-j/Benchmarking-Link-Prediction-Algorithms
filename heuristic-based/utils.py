import networkx as nx
import numpy as np


def MAP(index_list, test_graph, test_vertices, tot_vetices, topk = None):
	ans = np.zeros((tot_vetices,))
	count = np.zeros((tot_vetices,))
	corr = np.zeros((tot_vetices,))

	index_list.sort(reverse = True, key=lambda x:x[2])
	for i,(a,b,_) in enumerate(index_list):
		if not topk is None:
			if i > topk:
				break
		if a > b:
			temp = a
			a = b
			b = temp
		count[a] += 1
		count[b] += 1
		if a in test_vertices and b in set(test_graph.neighbors(a)):
			corr[a] += 1
			corr[b] += 1

			ans[a] += corr[a]/count[a]
			ans[b] += corr[b]/count[b]

	corr_indices = corr > 0
	out = np.divide(ans[corr_indices],corr[corr_indices])
	return out.mean()

def MRR(index_list, test_graph, test_vertices, tot_vetices, topk = None):
	ans = np.zeros((tot_vetices,))-1
	count = np.zeros((tot_vetices,))
	index_list.sort(reverse = True, key=lambda x:x[2])
	for i,(a,b,_) in enumerate(index_list):
		if not topk is None:
			if i > topk:
				break
		if a > b:
			temp = a
			a = b
			b = temp
		count[a] += 1
		count[b] += 1
		if a in test_vertices and b in set(test_graph.neighbors(a)):
			if ans[a] == -1:
				ans[a] = 1/count[a]
			if ans[b] == -1:
				ans[b] = 1/count[b]
	ans_indices = np.invert(ans == -1)
	count_indices = np.invert(count == 0)
	return ans[ans_indices].sum()/ans_indices.sum()

