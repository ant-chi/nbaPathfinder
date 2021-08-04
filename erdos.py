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
import base64
import numpy as np

st.set_page_config(layout="wide", page_title="NBA Erdos", page_icon=":basketball: :basketball: :basketball:")
start = time.time()


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def loadGraph():
    out = []
    completeGraph = nx.read_gpickle("nbaErdos/data/completeGraph.gpickle")
    completeEdges = pd.read_pickle("nbaErdos/data/completeEdges.pkl")
    completeNodes = pd.read_pickle("nbaErdos/data/completeNodes.pkl")
    # for file in ["completeEdges", "completeNodes"]:#, "imageDict", "nameDict"]:
        # out.append(pd.read_pickle("nbaErdos/data/" + file + ".pkl"))
        # temp = []
        # with (open("nbaErdos/data/"+ i + ".pkl", "rb")) as openfile:
        #     while True:
        #         try:
        #             temp.append(pickle.load(openfile))
        #         except EOFError:
        #             out.append(temp[0])
        #             break

    # out.append(dict((v,k) for k,v in sorted(out[4].items(), key=lambda x: x[1])))

    return completeGraph, completeEdges, completeNodes

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def loadData():
    playerDf = pd.read_csv("nbaErdos/data/playerDf.csv")
    imageDf = pd.read_csv("nbaErdos/data/imageDf.csv")

    temp = playerDf.pivot_table(index="Player Key", values=["Team","Season"], aggfunc=set).to_dict()
    teams, seasons = temp["Team"], temp["Season"]
    names = playerDf[["Player", "Player Key"]].set_index("Player Key").to_dict()["Player"]
    selectboxNames = dict((v,k) for k,v in sorted(names.items(), key=lambda x: x[1]))
    images = imageDf[["Player Key", "Image"]].set_index("Player Key").to_dict()["Image"]

    return teams, seasons, names, selectboxNames, images




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
        teammates = [names[i] for i in adjacencyList[node["id"]]]
        node["title"] += "<br> <br> Teammates: <br> &emsp; &emsp;" + teammates[0] + "<br> &emsp; &emsp;"
        node['title'] += "<br> &emsp; &emsp;".join(teammates[1:])


def edgeTitle(edge, metadata):
    """
    Helper function to format an edge's metadata with HTML as part of an edge's tooltip.
    """
    players = "<b>" + ", ".join([names[i] for i in edge]) + "</b>"
    seasonsAsTeammates = "Seasons as teammates: {}".format(len(metadata[0]))
    mutualTeams = ""
    for team, years in metadata[1].items():
        mutualTeams += team + ": " + ", ".join(years) + "<br>"
    mutualTeams = mutualTeams[:-4]    # remove trailing line break tag

    return players + "<br>" + seasonsAsTeammates + "<br>" + mutualTeams

def nodeTitle(nodeValue):
    # return str(names[nodeValue])
    title = "<b>{}</b>".format(names[nodeValue])
    title += "<br> Years of Experience: {}".format(len(seasons[nodeValue]))
    title += "<br> <br> Teams: <br> &emsp; &emsp;" + "<br> &emsp; &emsp;".join(teams[nodeValue])
    return title


def addNode(graph, value, source, target):
    """
    Helper function that adds nodes to displayed graph. Source and target nodes are larger and
    marked by a thick green or red border.
    """
    if value == source:
        graph.add_node(value, title=nodeTitle(value), label=" ",
                       shape="circularImage", image=images[value],
                       size=120, borderWidth=12, borderWidthSelected=16, color="green", mass=9)
    elif value == target:
        graph.add_node(value, title=nodeTitle(value), label=" ",
                       shape="circularImage", image=images[value],
                       size=120, borderWidth=12, borderWidthSelected=16, color="red", mass=9)
    else:
        graph.add_node(value, title=nodeTitle(value), label=" ",
                       shape="circularImage", image=images[value],
                       size=85, borderWidth=3, borderWidthSelected=10, color="white", mass=9)



def pathsDf(paths):
    data = []
    for path in paths:
        row = []
        for player in path:
            row.append(names[player])
        data.append(row)

    df = pd.DataFrame(data)
    df.columns = ["Source"] + ["Node {}".format(i+1) for i in range(len(paths[0])-2)] + ["Target"]

    return df



def shortestPathsGraph(source, target):
    """
    Returns a pyvis graph that displays all shortest paths between the source and target nodes and
    text that contains the source player's "erdos" number, and the number of shortest paths between
    source and target nodes.
    """

    if nx.has_path(completeGraph, source, target):
        text = []
        paths = list(nx.all_shortest_paths(completeGraph, source, target))

        G = net.Network(width="100%", height="750px", bgcolor="#222222")
        # G = net.Network(width="1300px", height="700px", bgcolor="#222222")
        G.set_options('''
        var options = {
            "edges": {"arrows": {"to": {"enabled": true,
                                        "scaleFactor": 2}},
                      "hoverWidth": 10,
                      "color": {"inherit": true},
                      "smooth": {"type": "cubicBezier",
                                 "forceDirection": "horizontal"}
            },
            "interaction": {"tooltipDelay": 0,
                            "hover": true
            },
            "physics": {"barnesHut": {"gravitationalConstant": -750,
                                      "centralGravity": 0.14,
                                      "springLength": 150,
                                      "springConstant": 0.0005,
                                      "damping": 0.07,
                                      "avoidOverlap": 0.7},
                        "maxVelocity": 40,
                        "minVelocity": 3.5,
                        "timestep": 0.5,
                        "stabilization": {"enabled": true,
                                          "iterations": 5000,
                                          "updateInterval": 100}
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
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -25000
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

        text.append("{} has a {} number of {}".format(names[source],
                                                      " ".join(names[target].split(" ")[1:]),
                                                      len(paths[0])-1))

        if len(paths) == 1:     # Players that are directly connected as teammates
            text.append("There is {} shortest path of length {} between {} and {}".format(len(paths), len(paths[0])-1,
                                                                                          names[source], names[target]))
        else:
            text.append("There are {} shortest paths consisting of {} players between {} and {}".format(len(paths), len(G.nodes)-2,
                                                                                                        names[source], names[target]))
        return G, text, pathsDf(paths)

    else:    # when there is no path between source and target
        return None, "There were no paths found between {} and {}".format(source, target), None



completeGraph, completeEdges, completeNodes = loadGraph()
teams, seasons, names, selectboxNames, images = loadData()


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

    player_1 = selectBox_1.selectbox(label="Player 1", options=list(selectboxNames.values()),
                                     format_func=lambda x: names[x], index=696, on_change=None)

    player_2 = selectBox_2.selectbox(label="Player 2", options=list(selectboxNames.values()),
                                     format_func=lambda x: names[x], index=1087, on_change=None)

    enterButton =  button_1.form_submit_button("Find paths!")
    randomButton = button_2.form_submit_button("I'm feeling lucky!")


graphCol = st.beta_container()
pathsExpander = st.beta_expander("View Paths")

with graphCol:
    if randomButton:
        player_1, player_2 = random.sample(list(names.keys()), 2)

    graph, text, pathsDf = shortestPathsGraph(player_1, player_2)

    if graph == None:   # When no paths are found
        st.title(text)
    else:
        st.title(text[0])
        st.subheader(text[1])

        graph.write_html("graph.html")
        components.html(graph.html, height=750)

        # pv_static(graph)
        with pathsExpander:
            st.dataframe(pathsDf)
            st.markdown("Each row represents a path of mutual teammates that connects the source and \
            target players. <br> A row can be read in either direction since this is an\
            undirected collaboration graph where order does not matter.", unsafe_allow_html=True)

        st.success("Query took: {} seconds".format(np.round(time.time() - start, 4)))

        # f = open("graph.html", "r")
        # data = f.read()
        # f.close()
        # b64 = base64.b64encode(data.encode()).decode()
        b64 = base64.b64encode(graph.html.encode()).decode()
        htmlDownload = f'<a href="data:text/html;base64,{b64}">Save visualization as HTML</a> (Right-Click + \"Save Link As\")'
        st.markdown(htmlDownload, unsafe_allow_html=True)


image_1.image(images[player_1], width=150)
image_2.image(images[player_2], width=150)
