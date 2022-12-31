import networkx as nx
import csv
G=nx.karate_club_graph()
edge=G.edges
# with open("../data/karate_edges.csv",'w',newline='') as karatefile:
#     writer=csv.writer(karatefile)
#     for item in edge:
#         writer.writerow(item)
