#!/usr/bin/python2.7
# -*- coding: latin-1 -*-
"""
Python Library implementing basic graph manipulation and traversal.
Use 'from Graph import *' in terminal for direct use.
"""


########################################################################[[IMPORTS]]#####################################################################################################~
from Queue import Queue
import pprint
import re

########################################################################[[CLASS]]#######################################################################################################~
class Node(object):
	"""Node class to store attributes and neighbors"""
	# constructor
	def __init__(self, id, attributes = None):
		self.id = id
		if attributes is None: attributes = {}
		self.attributes = attributes
		self.neighbors = []  # list of neighbors id (ajacency list implementation)

	def __str__(self):
		"""Simply prints node content"""
		return "  id: %s, attributes: %s, neighbors: %s" % (self.id, self.attributes, ','.join(self.neighbors))



########################################################################[[CLASS]]#######################################################################################################~
class GOTerm(Node):
	"""	GeneOntology term node (subclasses Node) class to store attributes and neighbors. """
	# constructor

	def __init__(self, id, name = "", namespace = "", Def = "", attributes = None):
		super(GOTerm, self).__init__(id, attributes)
		self.name = name
		self.namespace = namespace
		self.Def = Def # 'def' is a python term. Therefore we use "Def" to avoid obfuscation.

	def __str__(self):
		"""Simply prints GOTerm content"""
		return "  id: %s, attributes: %s, neighbors: %s \n name: %s, namespace: %s" % (self.id, self.attributes, ','.join(self.neighbors), self.name, self.namespace )



########################################################################[[CLASS]]#######################################################################################################~
class GeneProduct(Node):
	""" GeneOntology annotations"""
	# constructor
	def __init__(self, id, name = "", aliases = [], attributes = None):
		super(GeneProduct, self).__init__(id, attributes)
		self.name = name
		if aliases is None: aliases = []
		self.aliases = aliases

	def __str__(self):
		"""Simply prints GeneProduct content"""
		return "  id: %s, attributes: %s, neighbors: %s \n name: %s, aliases:[ %s ]\n " % (self.id, self.attributes, ','.join(self.neighbors), self.name, ','.join(self.aliases))



########################################################################[[CLASS]]#######################################################################################################~
class Edge(object):
	"""Edge class to store edge attributes"""
	def __init__(self, source, destination, attributes = None):
		self.source      = source
		self.destination = destination
		if attributes is None: attributes = {}
		self.attributes  = attributes

	def __str__(self):
		"""Simply prints edge"""
		return "  %s --- %s , attributes: %s" % (self.source.id, self.destination.id, self.attributes)



########################################################################[[CLASS]]#######################################################################################################~
class Graph(object):
	"""
	Abstract class representing a graph. It stores attributes for the graph, a dictionnary of nodes by id and the edges as a list.
	"""
	# constructor
	def __init__(self):
		self.attributes = {}
		self.nodesById  = {}
		self.edges      = []

	def __str__(self):
		"""
		Dumps all nodes and edges of the graph.
		pprint is used to make matrices more readable. 
		"""
		pp = pprint.PrettyPrinter(indent=5)
		s = 'Attributes:\n'
		for a in sorted(self.attributes.keys()):
			s += a+': '
			if type(self.attributes[a]) == dict: s += '\n' #print dictionaries one line down for better readability.
			s += pp.pformat(self.attributes[a]) + '\n\n'
		s += '\n\nNodes (%r)\n' % len(self.nodesById.keys())
		for n in self.nodes(): s+= str(n)+'\n'
		s+= '\nEdges (%s)\n' % (len(self.edges))
		for e in sorted(self.edges, key=lambda edge: edge.source.id): s+= str(e)+'\n'
		s+= '\n'
		return s



	# modifiers
	###########
	def addNode(self, node, attributes = None):
		"""
		Add a node to the graph.
		It can be called with an already existing node:
			n = Node('youpi')
			g.addNode(n)
		or it can be called with an id:
			g.addNode('youpi')
		Attributes can be passed through a dictionnary:
			g.addNode('yopla', { 'color': 'yellow', 'flavour': 'lemon' })
		"""
		if isinstance(node, Node): #  node variable is a Node object
			if node.id in self.nodesById:
				return self.nodesById[node.id] # return it if already in this graph
			if attributes is None: # initialize if null
				node.attributes = {}
			else: # copy attributes 
				assert type(attributes) is dict, "Must pass dictionnary as attributes parameter."
				#Having something else than a dictionnary would break every graph traversal method.
				for a in attributes:
					node.attributes[a] = attributes[a]
			# add it to our dict
			self.nodesById[node.id] = node
			return node # return new node object
		else: # not a node instance, assume it is an id, need to create one
			if node in self.nodesById: # return it if already in this graph
				return self.nodesById[node]
			# initialize attributes if needed
			if attributes is None: # initialize if null
				attributes = {}
			else: assert type(attributes) is dict, "Must pass dictionnary as attributes parameter." 
			#Having something else than a dictionnary would break every graph traversal method.
			# create node
			n = Node(id = node, attributes=attributes)
			self.nodesById[ n.id ] = n
			return n # return new node object



	# accessors
	###########
	def node(self, obj): # convert node id or node to node (to ensure we have a node)
		"""
		Utility function to ensure we have a node:
			n = g.node(n)
		if n is a Node instance then it does nothing, but if n is the node id then it is replaced by the Node instance.
		"""
		if isinstance(obj, Node):
			return obj
		else:
			for n in self.nodesById.values():
				if n.id == obj: return n
			return None


	def nodes(self): # get all nodes as a vector
		"""
		Return the set of nodes of the graph as a list sorted lexicographically by their id.
		"""
		return sorted(self.nodesById.values(), key=lambda node: node.id)


	def neighbors(self, node):
		"""
		Returns the set of Nodes (as a list) that are accessible from node.
		In the case of a directed graph, this represents the successors.
		"""
		node = self.node(node)
		assert node is not None, 'Node error: Node given as parameter not registered as part of this graph.' 
		neighbors = []
		for nei in node.neighbors:
			neighbors.append(self.nodesById[nei])
		return neighbors


	def isAcyclic(self):
		"""
		Tests if the graph is devoid of any cycling paths.
		Calls dfs() and derives decision from its outcome.
		Nodes gain attributes given by dfs()
		"""
		self.dfs()
		for edge in self.edges:
			if edge.attributes["dfs_type"] == "back":
				return False
		return True

	def subgraph(self, nodes):
		"""
		Returns the subgraph containing the nodes given as argument.
		 - Only given nodes are kept.
		 - All edges between pairs of kept nodes are kept.
		"""
		subgraph = self.__class__() #Subgraph will be of the same type as graph.
		for node in nodes.values():
			n = node.__class__(node.id)
			for att in node.__dict__:
				setattr(n, att, node.__getattribute__(att))
			n.neighbors = []
			subgraph.addNode(n) 
		for edge in self.edges:
			if edge.source in nodes.values() and edge.destination in nodes.values():
				s = edge.source.__class__(edge.source.id)
				for att in edge.source.__dict__:
					setattr(s, att, edge.source.__getattribute__(att))
				s.neighbors = []
				d = edge.destination.__class__(edge.destination.id)
				for att in edge.destination.__dict__:
					setattr(d, att, edge.destination.__getattribute__(att))
				d.neighbors = []
				subgraph.addEdge(s, d, edge.attributes)
		return subgraph


	def adjacencyMatrix(self, weight = None):
		"""
		Constructs an adjacency matrix as a two-level dictionary from the graph with weighting values.
		weight attributes must correspond to an attribute present on all edges.
		Only works properly if weight is a quantitative (numerical) value.
		After execution, graph have additional attribute:
		  - AdjMatrix: Distance adjacency matrix as a two-level dictionnary. 
		"""
		for e in self.edges:
			assert weight in e.attributes.keys(), "Attribute error: %r parameter doesn't match any attribute in at least one of the graph nodes." %weight

		matrix = {}
		for i in self.nodesById:
			matrix[i] = {}
			for j in self.nodesById:
				matrix[i][j] = float('inf')
				matrix[i][i] = 0.0
				if self.edge(i,j) != None:
					if weight != None: matrix[i][j] = float(self.edge(i,j).attributes[weight])
					else: matrix[i][j] = 1.0
		self.attributes["AdjMatrix"] = matrix
		return None



	# traversal
	###########

	## Depth First Search travserval:
	def dfs(self):
		"""
		Depth First Search travserval of the graph.
		Makes call to __dfs_Visit(node).
		After execution, nodes have additional attributes:
		  - dfs_color: should be black (node has been processed)
		  - dfs_predecessor: node id of predecessor in the traversal
		  - dfs_in: time when the node started to be processed
		  - dfs_out: time when the node processing finished
		edges have an additional attribute:
		  - dfs_type: dfs classification of edges:
		     ¤ tree: the edge is part of a dfs tree
		     ¤ back: the edge returns on a known node and creates a cycle
		     ¤ forward: from a node to a reachable node in the dfs tree
		     ¤ cross: from a node to a node from an other dfs tree
		"""
		#Set color of all nodes to WHITE.
		for node in self.nodes():
			node.attributes["dfs_color"] = "WHITE"
			node.attributes["dfs_predecessor"] = None
		# Start timer attribute. Stored in graph for convenience.
		self.attributes["dfs_time"] = 0
		for node in self.nodes():
			# Start or restart __dfs_Visit
			if node.attributes["dfs_color"] == "WHITE":
				self.dfs_Visit(node)
		del self.attributes["dfs_time"] #Not needed anymore once method has finished.
		return None


	def dfs_Visit(self, node): 
		"""
		Not meant to be used outside of class or inherited classes.
		Recursion of dfs() method.
		"""
		# Set node to GREY to avoid checking the same node twice.
		node.attributes["dfs_color"] = "GREY"
		self.attributes["dfs_time"] += 1
		node.attributes["dfs_in"] = self.attributes["dfs_time"]
		for nei in self.neighbors(node):
			# Only look unchecked neighbors. Call recursively to follow paths. Keep track of the path.
			if nei.attributes["dfs_color"] == "WHITE":
				nei.attributes["dfs_predecessor"] = node.id 
				self.dfs_Visit(nei) 
				self.edge(node, nei).attributes["dfs_type"] = "tree"
			# Conditions to class edges:
			elif nei.attributes["dfs_color"] == "GREY":
				self.edge(node, nei).attributes["dfs_type"] = "back"
			elif node.attributes["dfs_in"] > nei.attributes["dfs_in"]:
				self.edge(node, nei).attributes["dfs_type"] = "cross"
			else: self.edge(node, nei).attributes["dfs_type"] = "forward"
		node.attributes["dfs_color"] = "BLACK"
		self.attributes["dfs_time"] += 1
		node.attributes["dfs_out"] = self.attributes["dfs_time"]
		return None
	


	## Breadth First Search traverval:
	def bfs(self,source):
		"""
		Breadth First Search travserval of the graph.
		Uses Queue object for efficiency (there is no 'leftpop' in python)
		After execution, nodes have additional attributes:
		  - bfs_color: At the end: black is successor of node, white if not (predecessor of node or its successors)
		  - bfs_predecessor: node id of predecessor in the traversal.
		  - bfs_in: time when the node started to be processed
		Returns a dictionary of all traversed nodes.
		"""
		# Assert that needed attributes exist in the current graph:
		source = self.node(source)
		assert source is not None, 'Node error: source given as parameter not registered as part of this graph.' 

		# Initialization:
		for n in self.nodesById.values():
			n.attributes["bfs_color"] = "WHITE"
			n.attributes["bfs_in"] = float("inf")
			n.attributes["bfs_predecessor"] = None
		source.attributes["bfs_color"] = "GREY"
		source.attributes["bfs_in"] = 0
		traversedNodes = {source.id:source}
		Q = Queue(maxsize=0)
		Q.put(source)

		#Main execution:
		while Q.empty() == False:
			n = Q.get()
			for nei in self.neighbors(n):
				if nei.attributes["bfs_color"] == "WHITE":
					nei.attributes["bfs_color"] = "GREY"
					nei.attributes["bfs_in"] = n.attributes["bfs_in"] +1
					nei.attributes["bfs_predecessor"] = n.id
					traversedNodes[nei.id] = nei
					Q.put(nei)
			n.attributes["bfs_color"] = "BLACK"
		return traversedNodes



	## Shortest Path Bellman-Ford travserval:

	def bellman_ford(self, source, weight):
		"""
		Bellman-Ford single source shortest path traversal of the graph.
		After execution, nodes have additional attributes:
		  - BF_d<-src: Distance(or cost) of the shortest path from source to the node
		  - bfs_predecessor: node id of predecessor in the shortest path from source to the node
		"""
		# Assert that needed attributes exist in the current graph:
		for e in self.edges:
			assert weight in e.attributes.keys(), "Attribute error: %r parameter doesn't match any attribute in at least one of the graph nodes." %weight
		source = self.node(source)
		assert source is not None, 'Node error: source given as parameter not registered as part of this graph.' 

		#Main execution:
		self.BF_initializeSingleSource(source)
		for i in xrange(1, len(self.edges)-1): 
			for e in self.edges:
				self.BF_relax(e.source, e.destination, float(e.attributes[weight]))

	def BF_initializeSingleSource(self, source): 
		"""
		Not meant to be used outside of class or inherited classes.
		Part of bellman_ford method.
		"""
		for n in self.nodes():
			n.attributes["BF_d<-src"] = float("inf")
			n.attributes["BF_path_pred"] = None
		source.attributes["BF_d<-src"] = 0
		return None

	def BF_relax(self, source, destination, weight): 
		"""
		Not meant to be used outside of class or inherited classes.
		Part of bellman_ford method.
		"""
		if destination.attributes["BF_d<-src"] > source.attributes["BF_d<-src"] + weight:
			destination.attributes["BF_d<-src"] = source.attributes["BF_d<-src"] + weight
			destination.attributes["BF_path_pred"] = source.id



	## Shortest Paths Floyd-Warshall traversal:
	def floyd_warshall(self, weight):
		"""
		Floyd Warshall shortest paths traversal of the graph.
		After execution, nodes have additional attributes:
		  - FW_Dmatrix: shortest distance adjacency matrix as a two-level dictionary. Not to be confused with distance adjaceccy matrix.
		  - FW_Nmatrix: Next node adjacency matrix as a two-level dictionary. Used for path reconstruction with FW_shortest path.
		"""
		# Assert that needed attributes exist in the current graph:
		for e in self.edges:
			assert weight in e.attributes.keys(), "Attribute error: %r parameter doesn't match any attribute in at least one of the graph nodes." %weight

		# Matrices initialization:
		self.adjacencyMatrix(weight)
		D = self.attributes["AdjMatrix"]
		N = {}
		for i in D.keys():
			N[i] = {}
			for j in D.keys():
				N[i][j] = None
		for e in self.edges:
			N[e.source.id][e.destination.id] = e.destination.id

		# Main execution: compares lengths with additional node added to see if it the new path is cheaper until all triplets have been done.
		for k in D.keys():
			for i in D.keys():
				for j in D.keys():
					if D[i][k] + D[k][j] < D[i][j]:
						D[i][j] = D[i][k] + D[k][j]
						N[i][j] = N[i][k]
		self.attributes["FW_Dmatrix"] = D
		self.attributes["FW_Nmatrix"] = N
		return None


	def FW_shortest_path(self, i, j):
		"""
		Path reconstruction method for the floyd-warshall graph traversal method. Use to get the nodes contained in the shortest calculated path.
		Call this function only after performing the floyd_warshall() method. 
		Returns a list of node ids.
		"""
		# Assert that needed attributes exist in the current graph:
		assert "FW_Dmatrix" in self.attributes.keys(), "Attribute Error: FW_Dmatrix missing from graph attributes.\n Be sure to run floyd_warshall method on this graph before calling FW_shortest_path method."
		assert "FW_Nmatrix" in self.attributes.keys(), "Attribute Error: FW_Nmatrix missing from graph attributes.\n Be sure to run floyd_warshall method on this graph before calling FW_shortest_path method."
		# AdjMatrix is keyed with ids. Allows this function to work even with Node objects:
		if isinstance(i, Node): i = i.id 
		if isinstance(j, Node): j = j.id
		assert i in self.nodesById.keys(), 'Node error: %r not registered as part of this graph.' %i
		assert j in self.nodesById.keys(), 'Node error: %r not registered as part of this graph.' %j

		D = self.attributes["FW_Dmatrix"]
		N = self.attributes["FW_Nmatrix"]
		if D[i][j] == float('inf'):	
			return ('No path from '+ i + ' to ' + j)
		path = [i]
		k = N[i][j]
		while k:
			path.append(k)
			k = N[k][j]
		return path


	def diameter(self):
		"""
		Finds the diameter (= longest of the shortest paths for all pairs of verteces).
		After execution, graph had additional attributes:
		  - diameter: float distance of the diameter of the graph.
		  - dia_path: path corresponding to the diameter distance.
		"""
		assert "FW_Dmatrix" in self.attributes.keys(), "No shortest path distances found. Be sure to run floyd_warshall() on this graph before calling this method."

		self.attributes["diameter"] = 0.0
		for s in self.attributes["FW_Dmatrix"]:
			for d in self.attributes["FW_Dmatrix"][s]:
				if self.attributes["FW_Dmatrix"][s][d] > self.attributes["diameter"]:
					self.attributes["diameter"] = self.attributes["FW_Dmatrix"][s][d]
					source = s
					destination = d
		self.attributes["dia_path"] = self.FW_shortest_path(source, destination)
		return None



	# File Loaders:
	###############
	def parse_pbar(self, currentLine, numLines):
		"""
		Simple progress bar printout for file loaders to keep track of parsing functions for big files.
		To be used in read loops of parsing functions.
		Each printout overwrites the previous one. The progress bar is overwritten by the next print output after the end of the parser function.
		Example printout:
		>>> Loading...[#########-----------]Lines read:24486/50000 | 48.97% 
		"""
		barLength = 20
		barFilled = (currentLine/float(numLines))*barLength
		percent = (currentLine/float(numLines))*100
		print 'Loading...' + '[' + '#' * int(barFilled) + '-' * (barLength - int(barFilled)) + ']' + 'Lines read:' + str(currentLine) + '/' + str(numLines) + ' | ' + str(round(percent, 2)) + '%' + ' ' * 10 + ' \r',


	def loadSIF(self, filename):
		"""
		Loads a graph in a (simplified) SIF (Simple Interactipon Format, cf. Cytoscape doc).
		Assumed input:
		   node1	relation	node2
		   chaussettes	avant	chaussures
		   pantalon	avant	chaussures
		   ...
		"""
		with open(filename) as f:
			# SKIP COLUMNS NAMES
			tmp = f.readline()
			# PROCESS THE REMAINING LINES
			row = f.readline().rstrip()
			while row:
				vals = row.split('\t')
				self.addEdge(vals[0], vals[2])
				row = f.readline().rstrip()
		return None


	def loadTAB(self, filename):
		"""
		Loads a graph in Cytoscape tab format
		Assumed input:
		   id1	id2	weight	color	...
		   A	B	6	blue	...
		"""
		with open(filename) as f: 
			# GET COLUMNS NAMES
			tmp = f.readline().rstrip()
			attNames= tmp.split('\t')
			# REMOVES FIRST TWO COLUMNS WHICH CORRESPONDS TO THE LABELS OF THE CONNECTED VERTICES
			attNames.pop(0)
			attNames.pop(0)
			# PROCESS THE REMAINING LINES
			row = f.readline().rstrip()
			while row:
				vals = row.split('\t')
				v1 = vals.pop(0)
				v2 = vals.pop(0)
				att = {}
				for i in xrange(len(attNames)):
					att[ attNames[i] ] = vals[i]
				self.addEdge(v1, v2, att)
				row = f.readline().rstrip() # NEXT LINE



########################################################################[[CLASS]]#######################################################################################################~
class UnirectedGraph(Graph):
	"""	Class implementing a directed graph (subclasses Graph) as adjacency lists. """
	# constructor
	def __init__(self):
		super(UnirectedGraph, self).__init__()

	# modifiers
	###########
	def addEdge(self, source, destination, attributes = None):
		"""
		Add an edge to the graph.
		The source and destination can be either nodes or node ids.
		If the nodes do not exist, they will be created and added to the graph:
		   g=DirectedGraph()
		   g.addEdge('chaussettes','chaussures')
		is equivalent to:
		   src=Node('chaussettes')
		   dst=Node('chaussures')
		   g.addEdge(src,dst)
		Attributes can be set on edges:
		   g.addEdge('Toulouse', 'Bordeaux', { 'dist' : 250 } )
		"""
		if attributes is not None:
			assert type(attributes) is dict, "Must pass dictionnary as attributes parameter." 
			#Having something else than a dictionnary would break every graph traversal method.
		s = self.addNode(source) # Adds node if necessary as defined in addNode.
		d = self.addNode(destination)
		#Test if edge is not already in the graph.
		for e in self.edges:
			if e.source.id == s.id and e.destination.id == d.id:
				return e #Edge exists. Return it.
		#Else add a new edge to the graph.
		s.neighbors.append(d.id)
		d.neighbors.append(s.id)
		edge = Edge(s,d,attributes)
		self.edges.append(edge)
		return edge


	# accessors
	###########
	def isUndirected(self):
		"""
		returns True!
		"""
		return True

	def edge(self, source, destination):
		"""
		Used to ensure we have an edge from source to destination, and retrieve attributes:
		  e = g.edge('Toulouse', 'Bordeaux')
		  print e.attributes['dist']
		"""
		source = self.node(source)
		destination = self.node(destination)

		for e in self.edges:
			if e.source == source and e.destination == destination: return e
			elif e.destination == source and e.source == destination: return e
		return None



########################################################################[[CLASS]]#######################################################################################################~
class DirectedGraph(Graph):
	"""	Class implementing a directed graph (subclasses Graph) as adjacency lists. """
	# constructor
	def __init__(self):
		super(DirectedGraph, self).__init__()

	# modifiers
	###########
	def addEdge(self, source, destination, attributes = None):
		"""
		Add an edge to the graph.
		The source and destination can be either nodes or node ids.
		If the nodes do not exist, they will be created and added to the graph:
		   g=DirectedGraph()
		   g.addEdge('chaussettes','chaussures')
		is equivalent to:
		   src=Node('chaussettes')
		   dst=Node('chaussures')
		   g.addEdge(src,dst)
		Attributes can be set on edges:
		   g.addEdge('Toulouse', 'Bordeaux', { 'dist' : 250 } )
		"""
		if attributes is not None:
			assert type(attributes) is dict, "Must pass dictionnary as attributes parameter." 
			#Having something else than a dictionnary would break every graph traversal method.
		s = self.addNode(source) # Adds node if necessary as defined in addNode.
		d = self.addNode(destination)
		#Test if edge is not already in the graph.
		for e in self.edges:
			if e.source.id == s.id and e.destination.id == d.id:
				return e #Edge exists. Return it.
		#Else add a new edge to the graph.
		s.neighbors.append(d.id)
		edge = Edge(s,d,attributes)
		self.edges.append(edge)
		return edge


	# accessors
	###########
	def isDirected(self):
		"""
		returns True!
		"""
		return True



	def edge(self, source, destination):
		"""
		Used to ensure we have an edge from source to destination, and retrieve attributes:
		  e = g.edge('Toulouse', 'Bordeaux')
		  print e.attributes['dist']
		"""
		source = self.node(source)
		destination = self.node(destination)

		for e in self.edges:
			if e.source == source and e.destination == destination: return e
		return None


	def getTransposed(self):
		"""
		returns the transposition of this graph. 
		 - The transposed graph contains the same nodes as this graph.
		 - The orientation of all edges is inverted.
		"""
		tGraph = self.__class__() #Subgraph will be the same type of graph.
		for node in self.nodesById.values(): 
			n = node.__class__(node.id) #Nodes will be of the same type as given nodes.
			for att in node.__dict__:
				setattr(n, att, node.__getattribute__(att))
			n.neighbors = []
			#Doesn't carry the neighbors adjacency list, It has to be recreated.
			tGraph.addNode(n)			
		for edge in self.edges:
			s = edge.source.__class__(edge.source.id)
			for att in edge.source.__dict__:
				setattr(s, att, edge.source.__getattribute__(att))
			s.neighbors = []
			d = edge.destination.__class__(edge.destination.id)
			for att in edge.destination.__dict__:
				setattr(d, att, edge.destination.__getattribute__(att))
			d.neighbors = []
			tGraph.addEdge(d, s, edge.attributes)
		return tGraph


	def predecessors(self, node):
		"""
		Returns node's predecessors sorted lexicographically by id.
		"""
		node = self.node(node)
		assert node is not None, 'Node error: Node given as parameter not registered as part of this graph.'
		predecessors = []
		for pred in self.nodesById.values():
			if node in self.neighbors(pred): predecessors.append(pred)
		return sorted(predecessors, key = lambda node: node.id)


	# traversal
	###########
	def topologicalSort(self):
		"""
		Performs a topological sort of the nodes of the graph. Nodes are return as a list.
		Absence of cycle is tested.
		"""
		assert self.isAcyclic() == True, "Graph Error: Topological sort cannot be performed on a graph that contains cycles."
		nodes = self.nodes()
		return sorted(nodes, key = lambda node: node.attributes['dfs_out'], reverse = True)



########################################################################[[CLASS]]#######################################################################################################~
class GeneOntology(DirectedGraph):
	""" 
	Class implementing an ontology and its annotations (subclasses DirectedGraph) as adjacency lists.
	Contains two subtypes of nodes: GOTerm (Ontology terms) and GeneProduct (annotations).
	"""
	# constructor
	def __init__(self):
		super(GeneOntology, self).__init__()


	# accessors
	###########

	def isGO(self):
		"""
		returns True!
		"""
		return True


	def isComplete(self):
		"""
		Verifies that the GeneOntology is complete.
		All terms should at the very least have a name and namespace attribute:
		  Terms without either of these would have been added through relationships but would not be referenced as such in the .obo file.
		All terms must be part of at least one edge:
		  Isolated terms with 0 neighbors shouldn't exist in an Ontology. 
		"""
		for term in self.GOTerms().values():
			if term.name == "" : return False
			if term.namespace == "" : return False
			#if len(self.neighbors(term)) + len(self.predecessors(term)) == 0 : return False
		return True


	def GOTerms(self):
		"""
		returns all GOTerm nodes from graph in a dictionary with id as keys.
		"""
		GOTerms = {}
		for node in self.nodesById.values():
			if isinstance(node, GOTerm): GOTerms[node.id] = node
		return GOTerms


	def GeneProducts(self):
		"""
		returns all GeneProduct nodes from graph in a dictionary with id as keys.
		"""
		geneProducts = {}
		for node in self.nodesById.values():
			if isinstance(node, GeneProduct): geneProducts[node.id] = node
		return geneProducts


	def biologicalProcesses(self):
		"""
		returns all GOTerm nodes with namespace 'biological_process' from graph in a dictionary with id as keys.
		"""
		biologicalProcesses = {}
		for term in self.GOTerms().values():
			if term.namespace == 'biological_process': biologicalProcesses[term.id] = term
		return biologicalProcesses


	def moldecularFunctions(self):
		"""
		returns all GOTerm nodes with namespace 'molecular_function' from graph in a dictionary with id as keys.
		"""
		moldecularFunctions = {}
		for term in self.GOTerms().values():
			if term.namespace == 'molecular_function': moldecularFunctions[term.id] = term
		return moldecularFunctions


	def cellularComponents(self):
		"""
		returns all GOTerm nodes with namespace 'cellular_component' from graph in a dictionary with id as keys.
		"""
		cellularComponents = {}
		for term in self.GOTerms().values():
			if term.namespace == 'cellular_component': cellularComponents[term.id] = term
		return cellularComponents



	# traversal
	###########

	def getDirectGOTerms(self, gene):
		"""
		returns all GOTerm type nodes directly associated to gene GeneProduct type node in a list sorted lexicographically by id.
		"""
		gene = self.node(gene)
		assert 'GO_loaded' in self.attributes.keys(), 'Graph error: No gene ontology has been yet loaded.'
		assert 'annotations_loaded' in self.attributes.keys(), 'Graph error: No annotations have been yet loaded.'
		assert gene is not None, 'Node error: GeneProduct given as parameter not registered as part of this graph.' 
		assert isinstance(gene, GeneProduct), 'Node error: node given as parameter not a GeneProduct type node.' 
		directGOTerms = {}
		for nei in self.neighbors(gene):
			#GOTerm neighbors are the destination of product_of type edges from a GeneProduct source.
			directGOTerms[nei.id] = nei
		return directGOTerms


	def getDirectGeneProducts(self, term):
		"""
		returns all GeneProduct type nodes directly associated to gene GOTerm type node in a list sorted lexicographically by id.
		"""
		term = self.node(term)
		assert 'GO_loaded' in self.attributes.keys(), 'Graph error: No gene ontology has been yet loaded.'
		assert 'annotations_loaded' in self.attributes.keys(), 'Graph error: No annotations have been yet loaded.'
		assert term is not None, 'Node error: GOTerm given as parameter not registered as part of this graph.' 
		assert isinstance(term, GOTerm), 'Node error: node given as parameter not a GOTerm type node.' 
		directGeneProducts = {}
		for pred in tGraph.predecessors(term):
			#GeneProduct neighbors are the source of product_of type edges to a GeneProduct destination.
			if isinstance(pred, GeneProduct): directGeneProducts[pred.id] = pred
		return directGeneProducts


	def getAllGOTerms(self, gene):
		"""
		Uses bfs() to return all terms (GOTerm type nodes) directly or indirectly associated to gene (GeneProduct type node) in a list sorted lexicographically by id.
		"""
		# Assert that needed attributes exist in the current graph:
		gene = self.node(gene)
		assert 'GO_loaded' in self.attributes.keys(), 'Graph error: No gene ontology has been yet loaded.'
		assert 'annotations_loaded' in self.attributes.keys(), 'Graph error: No annotations have been yet loaded.'
		assert gene is not None, 'Node error: geneProduct node given as parameter not registered as part of this graph.' 
		assert isinstance(gene, GeneProduct), 'Node error: node given as parameter not a GeneProduct type node.'

		allGOTerms = self.bfs(gene)
		del allGOTerms[gene.id]
		return allGOTerms


	def getAllGeneProducts(self, term):
		"""
		Uses bfs() on transposed GO to return all genes (GeneProduct type nodes) directly or indirectly associated to term (GOTerm type node) in a list sorted lexicographically by id.
		"""
		# Assert that needed attributes exist in the current graph:
		term = self.node(term)
		assert 'GO_loaded' in self.attributes.keys(), 'Graph error: No gene ontology has been yet loaded.'
		assert 'annotations_loaded' in self.attributes.keys(), 'Graph error: No annotations have been yet loaded.'
		assert term is not None, 'Node error: GOTerm node given as parameter not registered as part of this graph.' 		
		assert isinstance(term, GOTerm), 'Node error: node given as parameter not a GOTerm type node.'

		allGeneProducts = {}
		tGO = self.getTransposed()
		for node in tGO.bfs(term).values():
			if isinstance(node, GeneProduct): allGeneProducts[node.id] = node
		return allGeneProducts


	def ontologyDepth(self, ontology):
		"""
		Computes depth (longest path) of one of the three ontologies: 
		 - Biological processes: ontology = 'biological_process'
		 - Molecular functions: ontology = 'molecular_function'
		 - Cellular components: ontology = 'cellular_component'
		Creates a new graph instance to store relevant terms and edges.
		Performs a depth search first algorithm on transposed graph to sort terms in topological order.
		Topological ordering allows to calculate longest path in linear time since ontologies are directed acyclic graphs.
		"""
		assert 'GO_loaded' in self.attributes.keys(), 'Graph error: No gene ontology has been yet loaded.'
		assert ontology == 'biological_process' or ontology == 'molecular_function' or ontology == 'cellular_component', " %r is not a valid ontology category (namespace)" % (ontology)
		#Ontology extraction:
		if ontology == 'biological_process': subgraph = self.subgraph(self.biologicalProcesses())
		elif ontology == 'molecular_function': subgraph = self.subgraph(self.moldecularFunctions())
		elif ontology == 'cellular_component': subgraph = self.subgraph(self.cellularComponents())
		#Transposition and topological sorting:
		tGraph = subgraph.getTransposed()
		topo = tGraph.topologicalSort()
		#Distances initialization:
		dist = {}
		for term in topo: dist[term] = float('-inf')
		dist[topo[0]] = 0 #Root
		#Main execution. Computes longest distances between root and all other terms:
		for term in topo:
			if dist[term] != float('-inf'):
				for pred in tGraph.neighbors(term):
					if dist[pred] < dist[term] + 1: 
						dist[pred] = dist[term] + 1
		return max(dist.values())



	# File Loaders:
	###############

	def loadOBO(self, filename):
		""" 
		Loads a Gene Ontology .obo file into a GeneOntology Graph.
		Warning: .obo files are generaly big so this function may take several minutes to complete. Progress is displayed to keep track of the function.
		Nodes are GOTerm objects. A new GOTerm object is added to the graph for each [Term].
		 - Obsolete terms are not kept.
		 - GOTerm objects' id attributes correspond to the 'id' tag.
		 - GOTerm objects' name attributes correspond to the 'name' tag.
		 - GOTerm objects' namespace attributes correspond to the 'namespace' tag.
		 - GOTerm objects' Def attributes correspond to the 'def' tag.
		A new Edge object is added to the graph for each 'is_a' tag and for each 'relationship' tag.
		 - Edge objects' source attribute correspond to the GOTerm object which id is the current term's 'id' tag.
		 - Edge objects' destination attribute correspond to the GOTerm object which id is the 'is_a'/'relationship' tag.
		 - Edge objects gain a new attributes key: "type" which value is 'is_a' or else (relationships) depending on the tag.
		Other tags are not accounted for.
		"""
		with open(filename) as f:
			numLines = open(filename).read().count('\n') + 1
			currentLine = 1
			line = f.readline()#FIRST LINE 
			while line != "":
			#This loop only searches for [Term] tags to start the regex loop.
				if line == "[Term]\n":
				# The [Term] tag marks a new information block to treat and turn into a GOTerm node.
				# rstrip() is used in this loop to stop when an empty newline is found since it marks the end of a term.
					line = f.readline().rstrip() #First Line for this block.
					while line:
						currentLine += 1
						self.parse_pbar(currentLine, numLines)
						tmp = re.match('id: (GO:\d+)', line)
						if tmp: 
							id  = tmp.group(1)
							term = self.addNode(GOTerm(id))
							line = f.readline().rstrip() #Next Line
							continue
						tmp = re.match('name: (.+)', line)
						if tmp:
							if "obsolete" not in tmp.group(1):
								term.name = tmp.group(1)
								line = f.readline().rstrip() #Next Line
								continue
							else:
								del self.nodesById[term.id]
								line = f.readline().rstrip() #Next Line
								break
						tmp = re.match('namespace: (.+)', line)
						if tmp: 
							term.namespace = tmp.group(1)
							line = f.readline().rstrip() #Next Line
							continue
						tmp = re.match('def: (.+)', line)
						if tmp:
							term.Def = tmp.group(1)
							line = f.readline().rstrip() #Next Line
							continue
						tmp = re.match('is_a: (GO:\d+)', line)
						if tmp:
							dest = GOTerm(tmp.group(1))
							edge = self.addEdge(term, dest)
							edge.attributes["type"] = "is_a"
							line = f.readline().rstrip() #Next Line
							continue
						tmp = re.match('relationship: (.+) (GO:\d+) ', line)
						if tmp:
							if tmp.group(1) == "part_of" or tmp.group(1) == "has_part":
								dest = GOTerm(tmp.group(2))
								edge = self.addEdge(term, dest)
								edge.attributes["type"] = tmp.group(1)
							line = f.readline().rstrip() #Next Line
							continue
						line = f.readline().rstrip() #Next Line for this block.
				currentLine += 1
				self.parse_pbar(currentLine, numLines)
				line = f.readline() #NEXT LINE
				#Move to next block. Skips other types of blocks.	
			assert self.isComplete(), "Graph error: This ontology appears to be incomplete. Use help(GeneOntology.isComplete) for more details"
			self.attributes['GO_loaded'] = len(self.GOTerms())
			print 'Ontology:' + str(self.attributes['GO_loaded']) + ' terms loaded ' + ' '* 100


	def loadAnnotations(self, filename):
		"""
		Loads a proteome annotation file for gene ontology onto a previously loaded geneOntology graph.
		Warning: annotation files are generaly big so this function may take several minutes to complete. Progress is displayed to keep track of the function.
		Annotations corresponding to GOTerms not registered in the graph are not added to avoid creating these possibly obsolete terms 
		(because of differences between the times these files were implemented).
		Added nodes are GeneProduct objects:
		 - GeneProduct objects' id attrubute corresponds to the 'DB_Object_ID tag' (column 2).
		 - GeneProduct objects' name attrubute corresponds to the 'DB_Object_Symbol' tag (column 3).
		 - GeneProduct objects gain a new attributes key: "evidence-code" which value corresponds to the 'Evidence Code' tag (column 7).
		 - GeneProduct objects' aliases attribute is a list of names given by the 'DB_Object_Synonym' tag (column 11). 
		A new Edge object is added to the graph for each added GeneProduct object.
		 - Edge objects' source attribute correspond to the added GeneProduct object.
		 - Edge objects' destination attribute corespond to the corresponding GOTerm in the gene ontology given by the 'GO ID' tag (column 5).
		"""
		assert 'GO_loaded' in self.attributes.keys(), 'Graph error: Annotations must be loaded after the Gene Ontology. Use loadOBO(filename) to load the desired GO.obo file.'
		with open(filename) as f:
			numLines = open(filename).read().count('\n') + 1
			terms = self.GOTerms()
			currentLine = 0
			line = f.readline().rstrip() #first line
			while line:
				self.parse_pbar(currentLine, numLines)
				currentLine += 1
				if line[0] == '!':
					#ignore this line.
					line = f.readline().rstrip()
					continue
				cols = line.split('\t', 10)
				if cols[4] in terms.keys():
					gene = GeneProduct(cols[1])
					gene.name = cols[2]
					edge = self.addEdge(gene, cols[4])
					edge.attributes['type'] = 'product_of'
					gene.attributes['evidence-code'] = cols[6]
					gene.aliases = cols[10].split('|')
				line = f.readline().rstrip() #next line
			self.attributes['annotations_loaded'] = len(self.GeneProducts())
			print 'Annotations:' + str(self.attributes['annotations_loaded']) + ' products loaded ' + ' ' * 100




########################################################################[[MAIN]]#######################################################################################################~

if __name__ == "__main__":
# TESTS (Uncomment to run):
	print "=== Graph.py library tests: ==="
	## DirectedGraph implementation:
	##################################

	#~ print "Graph.py library tests:"

	#~ g = DirectedGraph()
	#~ g = UnirectedGraph()

	#~ print "Testing Graph.loadSIF:"
	#~ g.loadSIF('dressing.sif')
	#~ g.loadSIF('dia29.sif')
	#~ print g

	#~ print "Testing Graph.loadTAB:"
	#~ g.loadTAB('Bellman-Ford.tab')
	#~ g.loadTAB('Floyd-Warshall.tab')
	#~ print g

	#~ print "Testing accessors:"
	#~ print g.node('ceinture')
	#~ print g.node('veste')
	#~ print g.edge('ceinture', 'veste')
	#~ print g.edge('veste', 'ceinture')
	#~ print g.neighbors('ceinture')
	#~ print g.subgraph({node: g.nodesById[node] for node in ('ceinture', 'veste', 'pantalon', 'cravate', 'chemise')})
	#~ print g.getTransposed()
	#~ g.adjacencyMatrix('weight')

	#~ print "Testing DFS"
	#~ g.dfs()
	#~ print g

	#~ print "Testing BFS"
	#~ g.bfs('sous-vetements')
	#~ print g

	## Use this with .tab (weighted graph) files:
	#~ print "Testing Bellman-Ford"
	#~ g.bellman_ford('A', 'weight')
	#~ print g

	## Use this with .tab (weighted graph) files:
	#~ print "Testing Floyd-Warshall"
	#~ g.floyd_warshall('weight')
	#~ print g
	#~ print "Distance from A to C: "+ str(g.attributes["FW_Dmatrix"]['A']['C']) + " | Path: " + str(g.FW_shortest_path('A', 'C'))
	#~ print "Distance from A to B: "+ str(g.attributes["FW_Dmatrix"]['A']['B']) + " | Path: " + str(g.FW_shortest_path('A', 'B'))
	#~ print "Distance from A to B: "+ str(g.attributes["FW_Dmatrix"]['E']['A']) + " | Path: " + str(g.FW_shortest_path('E', 'A'))
	
	#~ print "Testing diameter:"
	#~ g.diameter()
	#~ print g
	#~ print "The graph is acyclic? " + str(g.isAcyclic())
	#~ print "Topological sort:"
	#~ nodes = g.topologicalSort()
	#~ for n in nodes: print "%s (%i) " % (n.id , n.attributes['dfs_out'] )


	#~ Gene Ontology implementation:
	################################

	#~ go = GeneOntology()
	#~ print 'testing GeneOntology.loadOBO'
	#~ go.loadOBO("go-basic.obo")
	#~ go.loadOBO("goslim_generic.obo")
	#~ print 'testing  GeneOntology.loadAnnotations'
	#~ go.loadAnnotations("gene_association.goa_arabidopsis")
	#~ go.loadAnnotations("gene_sample.goa_arabidopsis")

	#~ print "Testing accessors"
	#~ print go.isComplete()
	#~ go.GOTerms()
	#~ go.GeneProducts()
	#~ go.biologicalProcesses()
	#~ go.moldecularFunctions()
	#~ go.cellularComponents()

	#~ print "Testing direct traversal:"
	#~ dgp = go.getDirectGeneProducts('GO:0003723')
	#~ for g in dgp: print g + " ",
	#~ print "\n"
	#~ dgt = go.getDirectGOTerms('A9LNK9')
	#~ for t in dgt: print t + " ",

	#~ print "Testing breadth first traversal:"
	#~ agp = go.getAllGeneProducts('GO:0003723')
	#~ for g in agp: print g + " ",
	#~ print "\n"
	#~ agt = go.getAllGOTerms('A9LNK9')
	#~ for t in agt: print t + " ",

	#~ print "Testing ontologyDepth():"
	#~ print go.ontologyDepth('biological_process')
	#~ print go.ontologyDepth('molecular_function')
	#~ print go.ontologyDepth('cellular_component')
