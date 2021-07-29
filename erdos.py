import time
import random
import itertools
# from collections import defaultdict
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis import network as net
from stvis import pv_static
# import pickle
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")
# import pandas as pd

st.set_page_config(layout="wide", page_title="NBA Erdos", page_icon=":basketball: :basketball: :basketball:")
start1 = time.time()
# loadMessage = st.empty()
# with loadMessage.beta_container():
#     st.title('BEEP BOOP BOOP BOP :robot_face:')
#     st.title("DASHBOARD IS LOADING....")

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def loadData():
    out = []
    out.append(nx.read_gpickle("nbaErdos/data/completeGraph.gpickle"))
    for file in ["completeEdges", "completeNodes", "imageDict", "nameDict"]:
        out.append(pd.read_pickle("nbaErdos/data/" + file + ".pkl"))
        # temp = []
        # with (open("nbaErdos/data/"+ i + ".pkl", "rb")) as openfile:
        #     while True:
        #         try:
        #             temp.append(pickle.load(openfile))
        #         except EOFError:
        #             out.append(temp[0])
        #             break

    out.append(dict((v,k) for k,v in sorted(out[4].items(), key=lambda x: x[1])))

    return out


# @st.cache()
# def readData(path1, path2):
#     playerDf = pd.read_csv(path1)
#     rostersDf = playerDf.pivot_table(index="Team", columns="Season", values="Player Key", aggfunc=list)
#     imageDf = pd.read_csv(path2)
#
#     return playerDf, rostersDf, imageDf
#
# playerDf, rosterDf, imageDf = readData("playerDf.csv", "imageDf.csv")
#
# @st.cache()
# def playerLookups(df1, df2):
#     """
#     Returns dictionaries used to lookup player names and images (from BBREF)
#     using player primary keys.
#     """
#     nameDict = df1[["Player", "Player Key"]].set_index("Player Key").to_dict()["Player"]
#     imageDict = df2[["Player Key", "Image"]].set_index("Player Key").to_dict()["Image"]
#     selectBoxNames = dict((v,k) for k,v in sorted(nameDict.items(), key=lambda x: x[1]))
#
#     return nameDict, imageDict, selectBoxNames
#
#
# nameDict, imageDict, selectboxNames = playerLookups(playerDf, imageDf)
#
# @st.cache()
# def genGraphComponents(df):
#     """
#     Returns edges and nodes to generate a complete graph of player collaborations. Also tracks
#     relevant edge metadata used in displayed graph including seasons as teammates and mutual teams.
#     """
#     nodes, edges = set(), {}
#     teams, seasons = df.index, df.columns
#     for i, team in enumerate(teams):
#         for j, season in enumerate(seasons):
#             roster = df.iloc[i, j]
#             try:
#                 rosterEdges = list(itertools.permutations(roster, 2))
#                 for k in rosterEdges:
#                     nodes.add(k[0])
#                     nodes.add(k[1])
#                     if k not in edges.keys():  # non defined teammate edges
#                         edges[k] = [set(), defaultdict(list)]
#                         edges[k][0].add(season)
#                         edges[k][1][team].append(season)
#                     elif team not in edges[k][1].keys():
#                         edges[k][0].add(season)
#                         edges[k][1][team].append(season)
#                     else:
#                          if season not in edges[k][1][team]:   # prevents doublecounting of team + season
#                             edges[k][0].add(season)
#                             edges[k][1][team].append(season)
#
#             except:    # teams that didn't exist in a given season are omitted eg. Supersonics in 2014
#                 pass
#
#     return [edges, nodes]
#
# completeEdges, completeNodes = genGraphComponents(rosterDf)
#
# @st.cache()
# def genCompleteGraph(edges, nodes):
#     """
#     Returns the complete Networkx collaboration graph that is used for finding shortest paths
#     between nodes.
#     """
#     completeGraph = nx.Graph()
#     for key, value in completeEdges.items():
#         completeGraph.add_node(key[0])
#         completeGraph.add_node(key[1])
#         completeGraph.add_edge(key[0], key[1])
#
#     return completeGraph
#
# completeGraph = genCompleteGraph(completeEdges, completeNodes)

# def randomPlayers():
    # return random.random_choice(nameDict.keys(), 2)

def nodeNeighbors(nodes, adjacencyList):
    """
    Helper function to format a node's neighbors with HTML as part of a node's tooltips.
    """
    for node in nodes:
        teammates = [nameDict[i] for i in adjacencyList[node["id"]]]
        node["title"] += "<br> <br> Teammates: <br> &emsp; &emsp;" + teammates[0] + "<br> &emsp; &emsp;"
        node['title'] += "<br> &emsp; &emsp;".join(teammates[1:])


def edgeTitle(edge, metadata):
    """
    Helper function to format an edge's metadata with HTML as part of an edge's tooltip.
    """
    names = ", ".join([nameDict[i] for i in edge])
    seasonsAsTeammates = "Seasons as teammates: {}".format(len(metadata[0]))
    mutualTeams = ""
    for team, years in metadata[1].items():
        mutualTeams += team + ": " + ", ".join(years) + "<br>"
    mutualTeams = mutualTeams[:-4]    # remove trailing line break tag

    return names + "<br>" + seasonsAsTeammates + "<br>" + mutualTeams


def addNode(graph, value, source, target):
    """
    Helper function that adds nodes to displayed graph. Source and target nodes are larger and
    marked by a thick green or red border.
    """
    if value == source:
        graph.add_node(value, title=nameDict[value], label=" ",
                       shape="circularImage", image=imageDict[value],
                       size=120, borderWidth=12, borderWidthSelected=16, color="green", mass=9)
    elif value == target:
        graph.add_node(value, title=nameDict[value], label=" ",
                       shape="circularImage", image=imageDict[value],
                       size=120, borderWidth=12, borderWidthSelected=16, color="red", mass=9)
    else:
        graph.add_node(value, title=nameDict[value], label=" ",
                       shape="circularImage", image=imageDict[value],
                       size=85, borderWidth=3, borderWidthSelected=10, color="white", mass=9)


def invalidInputCheck(source, target):
    if source not in completeNodes:
        return "{} was not on an NBA roster from 1980-2021 or did not qualify by playing more than 2500 career minutes".format(source)

    elif target not in completeNodes:
        return "{} was not on an NBA roster from 1980-2021 or did not qualify by playing more than 2500 career minutes".format(target)

    elif target not in completeNodes and source not in completeNodes:
        return "{} and {} were not on an NBA roster from 1980-2021 or did not qualify by playing more than 2500 career minutes".format(source, target)



def shortestPathsGraph(source, target):
    """
    Returns a pyvis graph that displays all shortest paths between the source and target nodes and
    text that contains the source player's "erdos" number, and the number of shortest paths between
    source and target nodes.
    """
    text = []

    # If a source or target doesn't exist in graph
    # out.append(invalidInputCheck(source, target))

    paths = [path for path in nx.all_shortest_paths(completeGraph, source, target)]
    if len(paths) == 0:
        return None, "There were no paths found between {} and {}".format(source, target)

    else:
        # pathPlayers = [[nameDict for i in paths for j in i]
        G = net.Network(width="100%", height="850px", bgcolor="#222222")
        # G = net.Network(width="1300px", height="700px", bgcolor="#222222")
        G.set_options('''
        var options = {
            "edges": {
              "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 2
              }
            },
            "hoverWidth": 10,
            "color": {
              "inherit": true
            },
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "horizontal"
            }
        },
          "interaction": {
            "tooltipDelay": 0,
            "hover": true
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -750,
                "centralGravity": 0.14,
                "springLength": 150,
                "springConstant": 0.0005,
                "damping": 0.07,
                "avoidOverlap": 0.7
            },
            "maxVelocity": 40,
            "minVelocity": 3.5,
            "timestep": 0.5,
            "stabilization": {
                "enabled": true,
                "iterations": 5000,
                "updateInterval": 100
                }
            }
        }
        ''')

        if len(paths) >= 10 and len(paths) < 30:
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -3000
            G.options["physics"]["barnesHut"]["springLength"] = 300
        elif len(paths) >= 30 and len(paths) < 70:
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -5000
            G.options["physics"]["barnesHut"]["springLength"] = 500
        elif len(paths) >= 70 and len(paths) < 200:
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -22500
            G.options["physics"]["barnesHut"]["springLength"] = 800
            G.options["physics"]["timestep"] = 1
        elif len(paths) >= 200:
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -65000
            G.options["physics"]["barnesHut"]["springLength"] = 1600
            G.options["physics"]["timestep"] = 1


        for i in paths:
            for j in list(itertools.permutations(i, 2)):
                if j not in completeEdges.keys():     # edges that are non-existent
                    pass
                else:
                    addNode(G, j[0], source, target)
                    addNode(G, j[1], source, target)
                    G.add_edge(j[0], j[1], title=edgeTitle(j, completeEdges[j]), width=3,
                               selectionWidth=16, color="#ffdc17", arrows="to", arrowStrikethrough=False)

        nodeNeighbors(G.nodes, G.get_adj_list())

        text.append("{} has a {} number of {}".format(nameDict[source],
                                                      " ".join(nameDict[target].split(" ")[1:]),
                                                      len(paths[0])-1))
        if len(paths) == 1:     # Players that are directly connected as teammates
            text.append("There is {} shortest path of length {} between {} and {} consisting of {} players".format(len(paths), len(paths[0])-1,
                                                                                                                       nameDict[source], nameDict[target],
                                                                                                                       len(G.nodes)-2))
        else:
            text.append("There are {} shortest paths of length {} between {} and {} consisting of {} players".format(len(paths), len(paths[0])-1,
                                                                                                                     nameDict[source], nameDict[target],
                                                                                                                     len(G.nodes)-2))

        # for i in paths:
        #     for j in
    return G, text, pd.DataFrame(paths)

# playerDf, rosterDf, imageDf = readData("playerDf.csv", "imageDf.csv")
# nameDict, imageDict = playerLookups(playerDf, imageDf)
# selectboxNames = dict((v,k) for k,v in sorted(nameDict.items(), key=lambda x: x[1]))

# completeEdges, completeNodes = genGraphComponents(rosterDf)
# completeGraph = genCompleteGraph(completeEdges, completeNodes)

dataObjects = loadData()
completeGraph = dataObjects[0]
completeEdges, completeNodes = dataObjects[1], dataObjects[2]
imageDict, nameDict, selectboxNames = dataObjects[3], dataObjects[4], dataObjects[5]
# loadMessage.empty()

# page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# background-size: cover;
# }
# </style>
# '''
#
# st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("NBA Erdos :basketball: :basketball: :basketball:")
st.subheader("Find shortest paths of mutual teammates between two players!")


with st.form("Find paths between two players"):
    empty_1, selectBox_1, empty_2, selectBox_2, empty_3 = st.beta_columns([0.2, 0.5, 0.5, 0.45, 0.3])
    empty_4, image_1, empty_5, image_2 = st.beta_columns([0.5, 1, 0.5, 1])
    button_1, button_2, emptyRight = st.beta_columns(([0.15, 0.2, 1]))

    with selectBox_1:
        player_1 = st.selectbox(label="Player 1", options=list(selectboxNames.values()),
                                format_func=lambda x: nameDict[x], index=696, on_change=None)
    with selectBox_2:
        player_2 = st.selectbox(label="Player 2", options=list(selectboxNames.values()),
                                format_func=lambda x: nameDict[x], index=1087, on_change=None)

    with button_1:
        enterButton = st.form_submit_button("Find paths!")
    with button_2:
        randomButton = st.form_submit_button("I'm feeling lucky!")


graphCol = st.beta_container()

with graphCol:
    if enterButton:
        graph, text, pathsDf = shortestPathsGraph(player_1, player_2)
        # st.title(message)
        # graph.show("out.html")
        # # st.markdown(graph.html, unsafe_allow_html=True)
        # components.html(graph.html, height=850)
        # # pv_static(graph)
        # st.success("Query took: {} seconds".format(np.round(time.time() - start1, 4)))
    elif randomButton:
        player_1, player_2 = random.sample(list(nameDict.keys()), 2)
        # graph, message = shortestPathsGraph(player_1, player_2)
        # st.title(message)
        # graph.show("out.html")
        # # st.markdown(graph.html, unsafe_allow_html=True)
        # components.html(graph.html, height=850)
        # # pv_static(graph)
        # st.success("Query took: {} seconds".format(np.round(time.time() - start1, 4)))
    # else:
        # st.success("App loaded in {} seconds".format(np.round(time.time() - start1, 4)))

    graph, text, pathsDf = shortestPathsGraph(player_1, player_2)
    if graph == None:
        st.title(text)
    else:
        st.title(text[0])
        st.subheader(text[1])

        graph.write_html("graph.html")
        # st.markdown(graph.html, unsafe_allow_html=True)
        components.html(graph.html, height=850)
        # st.write(type(pathsDf))
        # pd.DataFrame(pathsDf.apply(lambda x: nameDict[x], axis=0))
        st.success("Query took: {} seconds".format(np.round(time.time() - start1, 4)))
    # pv_static(graph)


with image_1:
    st.image(imageDict[player_1], width=150)
with image_2:
    st.image(imageDict[player_2], width=150)
#
#
#
#
#
#
# # with st.beta_expander("Details about this project"):
# #     st.markdown("* What tools are used in this project sdf")
#     # st.markdown("This dashboard is essentially a collaboration graph between NBA players where a \
#     # collaboration is denoted by players being mutual teammates. For a given set of players, the resulting
#     # graph represents the shortest paths of mutual teammates that connect one player to another.
#     # The data used for this project was
#     # scraped from Basketball Reference using roster data for every team from 1980-2021 and players that
#     # logged less than 2500 career minutes are omitted.
#     # This is not at all a novel idea and
#     # has been famously implemented in mathematics and film with the Erdos and Bacon numbers.")
