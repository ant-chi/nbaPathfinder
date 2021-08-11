import time
import itertools
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis import network as net
# from stvis import pv_static
import base64
import numpy as np

st.set_page_config(layout="wide", page_title="NBA Erdos", page_icon=":basketball:")
start = time.time()


# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
# def loadGraph():
#     # completeGraph = nx.read_gpickle("data/completeGraph.gpickle")
#     completeEdges = pd.read_pickle("data/completeEdges.pkl")
#     completeNodes = pd.read_pickle("data/completeNodes.pkl")
#
#     return completeGraph, completeEdges, completeNodes



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def loadData():
    playerDf = pd.read_csv("data/playerDf.csv")
    imageDf = pd.read_csv("data/imageDf.csv")

    temp = playerDf.pivot_table(index="Player Key", values=["Team","Season"], aggfunc=set).to_dict()
    teams, seasons = temp["Team"], temp["Season"]
    names = playerDf[["Player", "Player Key"]].set_index("Player Key").to_dict()["Player"]
    selectboxNames = dict((v,k) for k,v in sorted(names.items(), key=lambda x: x[1]))
    images = imageDf[["Player Key", "Image"]].set_index("Player Key").to_dict()["Image"]

    return teams, seasons, names, selectboxNames, images



def nodeNeighbors(nodes, adjacencyList):
    """
    Format a node's neighbors with HTML as part of a node's tooltips.
    """
    for node in nodes:
        teammates = [names[i] for i in adjacencyList[node["id"]]]
        node["title"] += "<br> <br> Teammates: <br> &emsp; &emsp;" + teammates[0] + "<br> &emsp; &emsp;"
        node['title'] += "<br> &emsp; &emsp;".join(teammates[1:])



def edgeTitle(edge, metadata):
    """
    Format an edge's metadata with HTML as part of an edge's tooltip.
    """
    text = []
    text.append("<b>" + ", ".join([names[i] for i in edge]) + "</b>")
    text.append("Seasons as teammates: {}".format(len(metadata[0])))

    mutualTeams = ""
    for team, years in metadata[1].items():
        years = [i[:4] for i in years]
        mutualTeams += team + ": " + ", ".join(years) + "<br>"
    text.append(mutualTeams[:-4])       # remove trailing line break tag

    return "<br>".join(text)



def nodeTitle(nodeValue):
    """
    Format a node's metadata with HTML as part of a node's tooltip
    """
    text = []
    text.append("<b>{}</b>".format(names[nodeValue]))
    text.append("<img src={} alt='Player Image' width='100'>".format(images[nodeValue]))
    text.append("Years of Experience: {}".format(len(seasons[nodeValue])))
    text.append("<br> Teams: <br> &emsp; &emsp;" + "<br> &emsp; &emsp;".join(teams[nodeValue]))
    title = "<br>".join(text)

    return title



def addNode(graph, value, source, target):
    """
    Adds nodes to displayed graph. Source and target nodes are larger and marked by a thick green
    or red border.
    """
    if value == source:
        graph.add_node(value, title=nodeTitle(value), label=" ",
                       shape="circularImage", image=images[value],
                       size=120, borderWidth=12, borderWidthSelected=18, color="green", mass=9)
    elif value == target:
        graph.add_node(value, title=nodeTitle(value), label=" ",
                       shape="circularImage", image=images[value],
                       size=120, borderWidth=12, borderWidthSelected=18, color="red", mass=9)
    else:
        graph.add_node(value, title=nodeTitle(value), label=" ",
                       shape="circularImage", image=images[value],
                       size=85, borderWidth=3, borderWidthSelected=10, color="white", mass=9)



def pathsDf(paths):
    """
    Converts shortest paths into a dataframe
    """
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
    information about a player's erdos number.
    """

    if nx.has_path(completeGraph, source, target):
        text = []
        paths = list(nx.all_shortest_paths(completeGraph, source, target))

        G = net.Network(width="100%", height="700px", bgcolor="#222222")
        # G = net.Network(width="1300px", height="700px", bgcolor="#222222")
        G.set_options("""
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
            "physics": {"barnesHut": {"gravitationalConstant": -1000,
                                      "centralGravity": 0.14,
                                      "springLength": 150,
                                      "springConstant": 0.0005,
                                      "damping": 0.07,
                                      "avoidOverlap": 0.7},
                        "maxVelocity": 35,
                        "minVelocity": 4,
                        "timestep": 0.5,
                        "stabilization": {"enabled": true,
                                          "iterations": 5000,
                                          "updateInterval": 100}
            }
        }
        """)

        if len(paths) >= 10 and len(paths) < 30:
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -3000
            G.options["physics"]["barnesHut"]["springLength"] = 300
        elif len(paths) >= 30 and len(paths) < 70:
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -5500
            G.options["physics"]["barnesHut"]["springLength"] = 600
        elif len(paths) >= 70 and len(paths) < 200:
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -32500
            G.options["physics"]["barnesHut"]["springLength"] = 900
            G.options["physics"]["timestep"] = 1
        elif len(paths) >= 200:
            G.options["physics"]["barnesHut"]["gravitationalConstant"] = -80000
            G.options["physics"]["barnesHut"]["springLength"] = 2500
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



# completeGraph, completeEdges, completeNodes = loadGraph()
teams, seasons, names, selectboxNames, images = loadData()

st.title("NBA Pathfinder :basketball:")
st.subheader("Find shortest paths of mutual teammates between two players!")

with st.form(" "):
    empty_1, selectBox_1, empty_2, selectBox_2, empty_3 = st.beta_columns([0.25, 0.45, 0.5, 0.45, 0.25])
    empty_4, image_1, empty_5, image_2 = st.beta_columns([0.6, 1, 0.6, 1])
    button_1, button_2, empty_6 = st.beta_columns(([0.2, 0.25, 1]))

    player_1 = selectBox_1.selectbox(label="Player 1", options=list(selectboxNames.values()),
                                     format_func=lambda x: names[x], index=696, on_change=None)

    player_2 = selectBox_2.selectbox(label="Player 2", options=list(selectboxNames.values()),
                                     format_func=lambda x: names[x], index=1087, on_change=None)

    enterButton =  button_1.form_submit_button("Find paths!")
    randomButton = button_2.form_submit_button("I'm feeling lucky! 🍀")


# graphCol = st.beta_container()
# pathsExpander = st.beta_expander("View paths")
#
# with graphCol:
#     if randomButton:
#         player_1, player_2 = np.random.choice(list(names.keys()), 2, replace=False)
#
#     graph, text, pathsDf = shortestPathsGraph(player_1, player_2)
#
#     if graph == None:   # When no paths are found
#         st.title(text)
#     else:
#         st.title(text[0])
#         st.subheader(text[1])
#
#         graph.write_html("graph.html")
#         components.html(graph.html, height=700)
#         st.write("This graph is fully interactive and has a physics engine! Try dragging, clicking \
#                   on, and hovering over nodes and edges!")
#         # pv_static(graph)
#         with pathsExpander:
#             st.dataframe(pathsDf)
#             st.markdown("Each row represents a path of mutual teammates that connects the source and \
#             target players. <br> A row can be read in either direction since this is an \
#             undirected collaboration graph where order does not matter.", unsafe_allow_html=True)
#
#         st.success("Query took: {} seconds".format(np.round(time.time() - start, 4)))
#
#         # f = open("graph.html", "r")
#         # data = f.read()
#         # f.close()
#         # b64 = base64.b64encode(data.encode()).decode()
#         b64 = base64.b64encode(graph.html.encode()).decode()
#         htmlDownload = f'<a href="data:text/html;base64,{b64}">Save visualization as HTML</a> (Right-Click + \"Save Link As\")'
#         st.markdown(htmlDownload, unsafe_allow_html=True)
#
#
# image_1.image(images[player_1], width=150)
# image_2.image(images[player_2], width=150)


with st.beta_expander("Details about this app"):
    st.markdown(
    """
    -   How does this app work?
        -   The goal of this app is to find and visualize all the shortest paths between two NBA
        players by connecting them through mutual teammates. The resulting output will be a
        collaboration network of players similar to the famous
        [Paul Erdos](https://en.wikipedia.org/wiki/Erd%C5%91s_number) and
        [Kevin Bacon](https://en.wikipedia.org/wiki/Six_Degrees_of_Kevin_Bacon) numbers in
        mathematics and film.
    -   How do I interpret a graph?
        -   Each node represents a player and each edge represents a teammate relationship. The
        source and target players are represented as large nodes with a green and red border. The
        arrows on edges provide the direction of travel from the source to the target player,
        although order doesn’t matter since this is an undirected graph.
    -   What tools did you use?
        -   Player and roster data are scraped from [Basketball Reference](https://www.basketball-reference.com/)
        using requests and BeautifulSoup
        -   Basic tabular operations are done with Pandas
        -   Networkx and Pyvis are used to find shortest paths and visualize graphs
        -   Streamlit is used to build and deploy this app  :sunglasses:
    -   The graph doesn't seem correct!
        -   A majority of issues exist because data prior to the 1980-81 seasons was not scraped, so
        players that were active in prior seasons may not be accurately represented. For example,
        searching for paths between the legendary George Gervin and Larry Kenon will not show that
        they were teammates for 5 seasons on the San Antonio Spurs from 1976-1980. Some issues also
        pop up for players that have had their contracts bought out or waived since they still appear
        on a team's payroll. For example, searching for paths between Luol Deng and Lebron James show
        that they are teammates on the Los Angeles Lakers even though this is not true.
    -   Why isn't Player XYZ showing up in the search bar?
        -   Players that have logged less then 2500 career minutes or were not on an NBA roster
        during the 1980-2021 seasons will not appear in the search bar.
    -   The tooltip popups are blocking the rest of the graph.
        -   Clicking on an edge or node instead of hovering over them will hide the tooltips, while
        still highlighting connected edges and nodes.
    """)
