import datetime
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import os, shutil
from vega import Vega
import json

DATA_FOLDER = "../../data/"
DUMP_FOLDER = "./data/"
DISTANCE_FOLDER = "../../distances/"


def add_distances(pairing, distances, distance_cols=None):
    if distance_cols is None:
        distance_cols = [c for c in distances.columns if
                         c not in ["templateID", "candidateID", "templateType", "candidateType"]]
    df = pairing.copy()
    for c in distance_cols:
        df[c] = np.NaN
    for index, row in df.iterrows():
        scores = distances[
            (distances["templateID"] == row["templateID"]) & (distances["candidateID"] == row["candidateID"])]
        if len(scores.index) > 0:
            for c in distance_cols:
                df.loc[index, c] = scores.iloc[0][c]
        else:
            for c in distance_cols:
                df.loc[index, c] = np.NaN
    return df


def edge_status(edge, pairing, otherEdges, input_is_template):
    # Edge status is 'common' if an edge of same type exists on both sides
    # Not comparing location or time!
    e_col = "templateID" if input_is_template else "candidateID"
    match_col = "candidateID" if input_is_template else "templateID"

    matching_src = pairing[pairing[e_col] == edge["Source"]][match_col].values
    matching_tgt = pairing[pairing[e_col] == edge["Target"]][match_col].values

    if len(matching_src) == 0 or len(matching_tgt) == 0:
        return "out of pairing"

    matching_edges = otherEdges[(otherEdges["Source"] == matching_src[0]) & \
                                (otherEdges["Target"] == matching_tgt[0]) & \
                                (otherEdges["eType"] == edge["eType"])]
    if len(matching_edges.index) > 0:
        return "common"
    else:
        return "missing" if input_is_template else "supplementary"


def flatten_communication_edges(edges):
    edges = edges[edges["eType"].isin([0, 1])]
    edges = edges.groupby(by=["Source", "Target", "eType"], as_index=False) \
        .aggregate(lambda c: c.unique().tolist())
    return edges


def annotate_edges(pairing, c_edges, t_edges, t_status_col, c_status_col):
    c_edges[c_status_col] = c_edges.apply(
        lambda row: edge_status(row, pairing, t_edges, input_is_template=False), axis=1)
    t_edges[t_status_col] = t_edges.apply(
        lambda row: edge_status(row, pairing, c_edges, input_is_template=True), axis=1)
    return c_edges, t_edges


def get_spending_profile(nodes, edges, categories):
    edges = edges.merge(nodes, how="left", left_on="Source", right_on="NodeID").rename(
        columns={"NodeType": "SourceType"}).drop("NodeID", axis=1)
    edges = edges.merge(nodes, how="left", left_on="Target", right_on="NodeID").rename(
        columns={"NodeType": "TargetType"}).drop("NodeID", axis=1)

    spent = edges.loc[(edges['SourceType'] == 1) & (edges['TargetType'] == 4), ['Source', 'Target', 'Weight']]
    spent = spent.rename(columns={"Source": "Person", "Target": "NodeID", "Weight": "Amount"})
    spent['Amount'] = -1 * spent['Amount']

    received = edges.loc[(edges['SourceType'] == 4) & (edges['TargetType'] == 1), ['Source', 'Target', 'Weight']]
    received = received.rename(columns={"Target": "Person", "Source": "NodeID", "Weight": "Amount"})

    financial = pd.concat([spent, received], sort=False)
    financial = financial.groupby(['Person', 'NodeID']).agg({'Amount': 'sum' }).reset_index()

    cartesian = (financial[['Person']].drop_duplicates().assign(key=1)
                 .merge(categories.assign(key=1), how='outer', on="key").drop("key", axis=1))

    financial = cartesian.merge(financial, how='left', on=['Person', 'NodeID'])
    return pd.pivot_table(financial, values='Amount', index=['Person'], columns=['NodeID'])


def squarify_edge_list(edges):
    edge_matrix = edges.groupby(by=["Source", "Target"]).agg(['count'])
    edge_matrix.columns = edge_matrix.columns.get_level_values(1)
    edge_matrix = edge_matrix.unstack(level=-1).fillna(0).astype(int)
    edge_matrix.columns = edge_matrix.columns.get_level_values(1)

    # Add missing columns
    missing_columns = set(edge_matrix.index) - set(edge_matrix.columns)
    for c in missing_columns:
        edge_matrix[c] = 0
    # Add missing rows
    missing_rows = set(edge_matrix.columns) - set(edge_matrix.index)
    for r in missing_rows:
        edge_matrix.loc[r] = [0] * len(edge_matrix.columns)
    return edge_matrix


def get_country(nodes, edges):
    person_ids = nodes[nodes["NodeType"] == 1]["NodeID"].unique()
    person_country = edges[edges["eType"] == 1][["Source", "SourceLocation"]] \
        .groupby(by="Source") \
        .aggregate(lambda c: str(int(c.unique().tolist()[0])))
    person_country = person_country.rename(columns={"SourceLocation": "Country"})
    return person_country["Country"].fillna("")


def annotate_nodes(nodes, edges, categories):
    nodes_ = nodes[nodes["NodeType"] == 1]
    edges_ = edges[edges["eType"].isin([0, 1])][["Source", "Target", "Weight"]]

    # Add spending profile
    category_dict = categories.set_index("NodeID")["Category"].to_dict()
    spending = get_spending_profile(nodes, edges, categories)
    spending = spending.rename(columns=category_dict)
    spending.columns = spending.columns.get_level_values(0)
    if len(spending.index) > 0:
        nodes_ = nodes_.join(spending, on="NodeID", how="left")
        nodes_["NodeID"] = nodes_["NodeID"].astype(int)
    else:
        for c in spending.columns:
            nodes_[c] = np.NaN

    # Add order minimizing crossings
    edge_matrix = squarify_edge_list(edges_)
    index_order = edge_matrix.index
    edge_matrix = csr_matrix(edge_matrix.values)
    node_order = reverse_cuthill_mckee(edge_matrix)
    node_order = pd.Series(node_order, index=index_order, name="edgeOrder")

    # Add node without edges
    missing_rows = set(nodes_["NodeID"].unique()) - set(node_order.index)
    for r in missing_rows:
        node_order.loc[r] = len(node_order.index)
    nodes_ = nodes_.join(node_order.to_frame(), on="NodeID")

    # Profile: add person country and demographic flags
    node_countries = get_country(nodes, edges)
    nodes_ = nodes_.join(node_countries.to_frame(), on="NodeID")

    if len(spending.index) > 0:
        nodes_["F_Smoker"] = nodes_["Tobacco"] > 0
        nodes_["F_Education"] = nodes_["Education"] > 0
        nodes_["F_Gas"] = nodes_["Natural gas"] > 0
        nodes_["F_Property"] = nodes_["Property taxes"] > 0
        nodes_["F_Mortgage"] = nodes_["Mortgage payments"] > 0
        nodes_["F_Renting"] = nodes_["Rented dwellings"] > 0
        nodes_["F_Rich"] = nodes_["Money income before taxes"] > 50000
    else:
        for c in ["F_Smoker", "F_Education", "F_Gas", "F_Property", "F_Mortgage", "F_Renting", "F_Rich"]:
            nodes_[c] = np.NaN

    # nodes_ = nodes_.fillna(0)
    return nodes_


def add_neighbor_coherence(pairing, template_edges, candidate_edges, template_status_col):
    # Only using communication (phone & email) for now (aggregated)
    for index, row in pairing.iterrows():
        tid = row["templateID"]
        cid = row["candidateID"]
        tedges = template_edges[(template_edges["Source"] == tid) | (template_edges["Target"] == tid)]
        cedges = candidate_edges[(candidate_edges["Source"] == cid) | (candidate_edges["Target"] == cid)]
        tneighbors = set(tedges["Source"].unique()) | set(tedges["Target"].unique()) - {tid}
        cneighbors = set(cedges["Source"].unique()) | set(cedges["Target"].unique()) - {cid}
        tcommon = len(tedges[tedges[template_status_col] == "common"].index)
        pairing.loc[index, "templateNeighbors"] = len(tneighbors)
        pairing.loc[index, "candidateNeighbors"] = len(cneighbors)
        pairing.loc[index, "nbCommonCommNeighbors"] = tcommon
    pairing["templateNeighbors"] = pairing["templateNeighbors"].astype(int)
    pairing["candidateNeighbors"] = pairing["candidateNeighbors"].astype(int)
    return pairing


def add_demographic_coherence(pairing, template_nodes, candidate_nodes):
    property_columns = ["Country", "F_Smoker", "F_Education", "F_Property", "F_Mortgage", "F_Gas", "F_Rich",
                        "F_Renting"]

    template_node_props = template_nodes[property_columns]
    template_node_props = template_node_props.rename(columns={c: "template" + c for c in property_columns})
    candidate_node_props = candidate_nodes[property_columns]
    candidate_node_props = candidate_node_props.rename(columns={c: "candidate" + c for c in property_columns})
    pairing_with_props = pairing.join(template_node_props, on="templateID")
    pairing_with_props = pairing_with_props.join(candidate_node_props, on="candidateID")

    # Derive comparison flag columns
    for c in property_columns:
        pairing_with_props[c] = pairing_with_props["template" + c] == pairing_with_props["candidate" + c]
    # Dump prop columns
    for c in ["template" + c for c in property_columns] + ["candidate" + c for c in property_columns]:
        del pairing_with_props[c]
    return pairing_with_props


def normalize_timestamps(edges, is_template=False):
    # Normalize to large graph time, apply before grouping!
    origin = datetime.datetime(2025, 1, 1, 0, 0)
    if is_template:
        # Exactly 14 days shift
        edges["Time"] = edges["Time"].astype(int) + datetime.timedelta(days=14).total_seconds()
    # Processed format to unix timestamps
    if np.issubdtype(edges["Time"].dtype, np.number):
        # Time is in seconds from 12:00 AM Jan. 1, 2025 (like Unix timestamps, except that time=0 is the year 2025, not 1970)
        edges["Time"] = pd.to_datetime(edges["Time"], unit='s', origin=pd.to_datetime(origin)).values.astype(
            np.int64) // 10 ** 9
    else:
        # Time is formatted as string
        edges["Time"] = pd.to_datetime(edges["Time"], infer_datetime_format=True).values.astype(np.int64) // 10 ** 9
    return edges


def normalize_metric_values(df):
    distance_cols = [c for c in df.columns if
                     c not in ["templateID", "candidateID", "graphName", "templateType", "candidateType"]]
    for c in distance_cols:
        if "Cosine" in c:  # Cosine similarity
            df.loc[:, c] = (1 - df[c]) / 2
        if "Jaccard" in c:  # Jaccard index
            df.loc[:, c] = 1 - df[c]
        if "dtw" in c:  # Time distance
            df.loc[:, c] = df[c] / df[c].max()
        if "gl" in c:  # Graphlet distance
            df.loc[:, c] = df[c] / df[c].max()
    return df


def extract_subgraph_nodes(all_nodes, candidate_edges):
    candidate_node_ids = set(candidate_edges["Source"].unique()) | set(candidate_edges["Target"].unique())
    return all_nodes[all_nodes["NodeID"].isin(candidate_node_ids)]


def process_dataframes_candidate_pairing(template_edges, template_nodes, candidate_graphs, distances, pairings,
                                         categories, dump_folder=None):
    # Create output file if provided and necessary
    if dump_folder and not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

    # Add spending profile, demographic properties and flags
    template_nodes = annotate_nodes(template_nodes, template_edges, categories)
    # Template edges will be further annotated for every candidate graph
    template_edges = normalize_timestamps(template_edges, is_template=True)

    for candidate_name, candidate_graph in candidate_graphs.items():
        candidate_nodes = candidate_graph["nodes"]
        candidate_edges = candidate_graph["edges"]
        candidate_nodes = annotate_nodes(candidate_nodes, candidate_edges, categories)
        candidate_edges = normalize_timestamps(candidate_edges, is_template=False)

        for dist_name in pairings[candidate_name].keys():
            distance_cols = [c for c in distances[candidate_name].columns if
                             c not in ["templateID", "candidateID", "graphName", "templateType", "candidateType"]]
            template_status_col = "Status_" + candidate_name + "_" + dist_name
            candidate_status_col = "Status_" + dist_name
            pairing = pairings[candidate_name][dist_name]
            pairing = add_distances(pairing, distances[candidate_name], distance_cols)
            pairing = add_demographic_coherence(pairing, template_nodes, candidate_nodes)
            candidate_edges, template_edges = annotate_edges(pairing, candidate_edges, template_edges,
                                                             template_status_col, candidate_status_col)
            pairing = add_neighbor_coherence(pairing, template_edges, candidate_edges, template_status_col)
            if dump_folder:
                pairing.to_csv(dump_folder + '/pairing_' + dist_name + '_' + candidate_name + '.csv', index=False)
        if dump_folder:
            candidate_nodes.to_csv(dump_folder + '/candidate_nodes_' + candidate_name + '.csv', index=False)
            candidate_edges.to_csv(dump_folder + '/candidate_edges_' + candidate_name + '.csv', index=False)
    if dump_folder:
        template_edges.to_csv(dump_folder + '/template_edges.csv', index=False)
        template_nodes.to_csv(dump_folder + '/template_nodes.csv', index=False)


def process_dataframes_single_pairing(template_edges, template_nodes,
                                      candidate_edges, candidate_nodes,
                                      pairing, distances, categories, pairingMetric=None):
    pairingMetric = "" if pairingMetric is None else pairingMetric
    template_edges = normalize_timestamps(template_edges, is_template=True)
    template_nodes = annotate_nodes(template_nodes, template_edges, categories)
    candidate_nodes = annotate_nodes(candidate_nodes, candidate_edges, categories)
    if len(distances.index) > 0:
        pairing = add_distances(pairing, distances)
    pairing = add_demographic_coherence(pairing, template_nodes, candidate_nodes)
    candidate_edges = normalize_timestamps(candidate_edges)
    candidate_edges, template_edges = annotate_edges(pairing, candidate_edges, template_edges, "Status" + pairingMetric,
                                                     "Status")
    pairing = add_neighbor_coherence(pairing, template_edges, candidate_edges, "Status" + pairingMetric)
    return template_edges, template_nodes, candidate_edges, candidate_nodes, pairing


def view(template_edges, template_nodes, candidate_edges, candidate_nodes, pairing, distances, categories):
    # Checking format of template_nodes and candidate_nodes
    for c in ["NodeID", "NodeType"]:
        if c not in template_nodes.columns:
            raise Exception("Missing '" + c + "' column in template_nodes")
        if c not in candidate_nodes.columns:
            raise Exception("Missing '" + c + "' column in candidate_nodes")
    # Checking format of template_edges and candidate_edges
    for c in ["Source", "Target", "eType"]:
        if c not in template_edges.columns:
            raise Exception("Missing '" + c + "' column in template_edges")
        if c not in candidate_edges.columns:
            raise Exception("Missing '" + c + "' column in candidate_edges")
            # Node set is defined by edges
    template_node_set = set(template_edges["Source"].unique()) | set(template_edges["Target"].unique())
    candidate_node_set = set(candidate_edges["Source"].unique()) | set(candidate_edges["Target"].unique())
    # Node set for type 'Person'
    template_node_set = set(
        template_nodes[(template_nodes["NodeID"].isin(template_node_set)) & template_nodes["NodeType"] == 0][
            "NodeID"].unique())
    candidate_node_set = set(
        candidate_nodes[(candidate_nodes["NodeID"].isin(candidate_node_set)) & candidate_nodes["NodeType"] == 0][
            "NodeID"].unique())

    # Checking format of distances
    metric_cols = [c for c in distances.columns if
                   c not in ["templateID", "candidateID", "templateType", "candidateType", "graphName"]]
    if "templateID" not in distances.columns or "candidateID" not in distances.columns:
        raise Exception('Missing templateID or candidateID in distance dataframe')
    if len(distances.index) > 0:
        if set(distances["templateID"].unique()) <= template_node_set:
            raise Exception("Some template Person are missing amongst given pairwise distances")
        if set(distances["candidateID"].unique()) <= candidate_node_set:
            raise Exception("Some candidate Person are missing amongst given pairwise distances")

    # Annotating dataframes // TODO: Do not change input dataframe
    template_edges, template_nodes, candidate_edges, candidate_nodes, pairing = process_dataframes_single_pairing(
        template_edges,
        template_nodes,
        candidate_edges,
        candidate_nodes,
        pairing,
        distances,
        categories)

    with open('./spec_minimal.vg.json') as json_obj:
        spec_no_data = json.load(json_obj)

        obj_from_name = lambda obj, name: next(obj_item for obj_item in obj if obj_item["name"] == name)
        index_from_name = lambda obj, name: [i for i, obj_item in enumerate(obj) if obj_item["name"] == name][0]

        # Wire datasets
        datasets = ["template-edges", "candidate-edges", "pairing", "template-nodes", "candidate-nodes"]
        for dataset_label in datasets:
            dataset = obj_from_name(spec_no_data["data"], dataset_label)
            del dataset["url"]
            del dataset["format"]
        spec_with_data = spec_no_data.copy()
        for i, data in enumerate([template_edges, candidate_edges, pairing, template_nodes, candidate_nodes]):
            dataset = obj_from_name(spec_with_data["data"], datasets[i])
            dataset["values"] = data.to_dict('records')

        # Wire metric list
        metrics = obj_from_name(spec_with_data["data"], "sortingMetrics")
        metrics["values"] = [{"value": c} for c in metric_cols]
        obj_from_name(spec_with_data["marks"], "arcview")["data"][0]["transform"][0]["fields"] = metric_cols
        # Adapt height to number of metrics
        templateY = obj_from_name(spec_with_data["signals"], "templateAxisOffset")["value"]
        obj_from_name(spec_with_data["signals"], "candidateAxisOffset")["value"] = templateY + 10*(len(metric_cols)+1)+2
        return Vega(spec_with_data)


def add_time_travel(distances, subgraph_name):
    travel_channel = pd.read_csv(
        DISTANCE_FOLDER + "/time_distances_travel_channel/" + "dtw_" + subgraph_name + ".csv")  # TemplateId,Id,Distance
    travel_channel = travel_channel.rename(
        columns={'TemplateId': 'templateID', 'Id': 'candidateID', 'Distance': 'dtwTravel'})
    travel_channel["graphName"] = subgraph_name
    distances = distances.merge(travel_channel, how="outer", on=["templateID", "candidateID"])
    return distances

def add_graphlet(distances, node_ids):
    gl0 = pd.read_csv(DISTANCE_FOLDER + "gl_0_undir5_similarity.csv")  # TemplateNode,PersonNode,graphletSimilarity
    gl0 = gl0.rename(columns={'TemplateNode': 'templateID', 'PersonNode': 'candidateID', 'graphletSimilarity': 'glEmail-undir5'})
    gl0 = gl0[["templateID", "candidateID", "glEmail-undir5"]][gl0["candidateID"].isin(node_ids)]
    gl0.loc[:,"glEmail-undir5"] = 1 - gl0["glEmail-undir5"]
    distances = distances.merge(gl0, how="outer", on=["templateID", "candidateID"])
    gl1 = pd.read_csv(DISTANCE_FOLDER + "gl_0_undir5_similarity.csv")  # TemplateNode,PersonNode,graphletSimilarity
    gl1 = gl1.rename(columns={'TemplateNode': 'templateID', 'PersonNode': 'candidateID', 'graphletSimilarity': 'glPhone-undir5'})
    gl1 = gl1[["templateID", "candidateID", "glPhone-undir5"]][gl1["candidateID"].isin(node_ids)]
    gl1.loc[:,"glPhone-undir5"] = 1 - gl1["glPhone-undir5"]
    distances = distances.merge(gl1, how="outer", on=["templateID", "candidateID"])
    return distances

def cleanup_distance_cols(df):
    # Drop column that are not used
    col_to_drop = ["graphName", "templateType", "candidateType", "profileJaccard"]
    df = df.drop(columns=[c for c in col_to_drop if c in df.columns])
    return df

def process_distances(distances, subgraph_name, node_ids):
    df = cleanup_distance_cols(distances[distances["graphName"] == "Q1-" + subgraph_name])
    df = add_time_travel(df,"Q1-" + subgraph_name)
    # Make everything a distance [0,1], normalize from distances of all graphs
    return normalize_metric_values(df)

def process_candidates():
    # Load files
    candidate_graph_names = ["Graph" + str(i) for i in range(1, 6)]
    template_edges = pd.read_csv(DATA_FOLDER + "CGCS-Template.csv")
    template_nodes = pd.read_csv(DATA_FOLDER + "CGCS-Template-NodeTypes.csv")
    all_nodes = pd.read_csv(DATA_FOLDER + "CGCS-GraphData-NodeTypes.csv")
    categories = pd.read_csv(DATA_FOLDER + "DemographicCategories.csv")

    edges_per_graph = {name: pd.read_csv(DATA_FOLDER + "Q1-" + name + ".csv") for name in candidate_graph_names}
    nodes_per_graph = {name: extract_subgraph_nodes(all_nodes, edges_per_graph[name]) for name in candidate_graph_names}
    graphs = {name: {"nodes": nodes_per_graph[name], "edges": edges_per_graph[name]} for name in candidate_graph_names}

    # Process distances
    distances = pd.read_csv(DISTANCE_FOLDER + "/distances_candidates.csv")
    distances_per_graph = {name: process_distances(distances, name, graphs[name]["nodes"]["NodeID"].unique())
                                for name in  candidate_graph_names}

    # Process pairings
    distance_to_pairing_file = {
        "demographicsCosine": "demographics_distance",
        "travelJaccard": "travel_similarity",
        "dtwAllEdges": "ts_distances",
        "dtwCommunication": "ts_only_communication_channel",
        "glPhone-undir5": "graphlets/0-undir5",
        "glEmail-undir5": "graphlets/1-undir5"

    }
    distance_cols = distance_to_pairing_file.keys()
    pairing_file_folder = "../../python/matchings/"
    pairings_per_graph = { \
        name: { \
            dist_name: pd.read_csv(pairing_file_folder + distance_to_pairing_file[dist_name] + "_Q1-" + name + ".csv") \
            for dist_name in distance_cols} \
        for name in candidate_graph_names}

    process_dataframes_candidate_pairing(template_edges, template_nodes, graphs, distances_per_graph,
                                         pairings_per_graph,
                                         categories, DUMP_FOLDER + "/candidates/")


def cleanup_pairing(pairing, candidate_nodes):
    candidate_person_ids = (candidate_nodes[candidate_nodes["NodeType"] == 1]["NodeID"]).unique()
    # Only person pairs
    pairing = pairing[pairing["candidateID"].isin(candidate_person_ids)]
    # Force unique pair, taking the first
    pairing = pairing.groupby(by=["templateID"], as_index=False).aggregate(lambda c: c.unique()[0])
    return pairing

def get_distance_list(subgraph_name, node_ids):
    # Travel, demographic and graphlet distances for the complete graph
    travel = pd.read_csv(
        DISTANCE_FOLDER + "travelJaccard.csv")  # TemplateNode,PersonNode,nTripTemplate,nTripPerson,travelJaccard
    travel = travel.rename(columns={'TemplateNode': 'templateID', 'PersonNode': 'candidateID'})
    travel = travel[["templateID", "candidateID", "travelJaccard"]][travel["candidateID"].isin(node_ids)]
    demographics = pd.read_csv(
        DISTANCE_FOLDER + "demographicsCosine.csv")  # TemplateNode,PersonNode,NumMatch,AvgDiff,Cosine,Pearson
    demographics = demographics.rename(
        columns={'TemplateNode': 'templateID', 'PersonNode': 'candidateID', 'Cosine': 'demographicsCosine'})
    demographics = demographics[["templateID", "candidateID", "demographicsCosine"]][
        demographics["candidateID"].isin(node_ids)]
    distance_merged = demographics.merge(travel, how="outer", on=["templateID", "candidateID"])
    distance_merged = add_graphlet(distance_merged, node_ids)

    # Time distances only for the graph
    try:
        all_channels = pd.read_csv(
            DISTANCE_FOLDER + "/time_distances_all_channels/" + "dtw_" + subgraph_name + ".csv")  # TemplateId,Id,Distance
        all_channels = all_channels.rename(
            columns={'TemplateId': 'templateID', 'Id': 'candidateID', 'Distance': 'dtwAllEdges'})
        com_channels = pd.read_csv(
            DISTANCE_FOLDER + "/time_distances_communication_channels/" + "dtw_" + subgraph_name + ".csv")  # TemplateId,Id,Distance
        com_channels = com_channels.rename(
            columns={'TemplateId': 'templateID', 'Id': 'candidateID', 'Distance': 'dtwCommunication'})
        travel_channels = pd.read_csv(
            DISTANCE_FOLDER + "/time_distances_travel_channel/" + "dtw_" + subgraph_name + ".csv")  # TemplateId,Id,Distance
        travel_channels = travel_channels.rename(
            columns={'TemplateId': 'templateID', 'Id': 'candidateID', 'Distance': 'dtwTravel'})
        distance_merged = distance_merged.merge(all_channels, how="outer", on=["templateID", "candidateID"])
        distance_merged = distance_merged.merge(com_channels, how="outer", on=["templateID", "candidateID"])
        distance_merged = distance_merged.merge(travel_channels, how="outer", on=["templateID", "candidateID"])
    except FileNotFoundError:
        pass
    return distance_merged

def process_single():
    MATCH_FOLDER = "../../python/extracted_matchs_seed/"
    OPTIONS = [
        ("best", DATA_FOLDER + "/final_matching/matched_nodes_with_demographics_filtered_with_node66"),
        ("seed1-travels-demo-time-greedy2",
         MATCH_FOLDER + "/seed1/match_travels_demographics_timeFilter_greedy2"),
        ("seed3-travels-demo-time-greedy2",
         MATCH_FOLDER + "/seed3/match_travels_demographics_timeFilter_greedy2")

    ]
    CATEGORIES = pd.read_csv(DATA_FOLDER + "DemographicCategories.csv")
    print([o[0] for o in OPTIONS])

    for candidate_name, candidate_file in OPTIONS:
        print(candidate_file)

        if "best" in candidate_name:
            pairing = pd.read_csv(DATA_FOLDER + "/final_matching/matched_so_far.csv")
            pairing.columns = ['Unamed', 'templateID', 'candidateID', 'Match']
            pairing = pairing.drop(['Unamed', 'Match'], axis=1)
        else:
            pairing = pd.read_json(candidate_file + ".json", orient='index')
            pairing.columns = ["candidateID"]
            pairing["templateID"] = pairing.index
        template_edges = pd.read_csv(DATA_FOLDER + "CGCS-Template.csv")
        template_nodes = pd.read_csv(DATA_FOLDER + "CGCS-Template-NodeTypes.csv")
        candidate_edges = pd.read_csv(candidate_file + ".csv")
        candidate_nodes = pd.read_csv(DATA_FOLDER + "CGCS-GraphData-NodeTypes.csv")
        candidate_nodes = extract_subgraph_nodes(candidate_nodes, candidate_edges)
        pairing = cleanup_pairing(pairing, candidate_nodes)

        graph_folder = candidate_file.split('/')[-2]
        file_name = candidate_file.split('/')[-1]
        graph_name = file_name + "_" + graph_folder if "seed" in graph_folder else file_name
        distances = get_distance_list(graph_name, pairing["candidateID"].unique())
        distances = normalize_metric_values(distances)

        template_edges, template_nodes, candidate_edges, candidate_nodes, pairing = process_dataframes_single_pairing(
            template_edges, template_nodes,
            candidate_edges, candidate_nodes,
            pairing, distances, CATEGORIES)

        dump_folder = DUMP_FOLDER + candidate_name + "/"
        if not os.path.exists(dump_folder):
            os.makedirs(dump_folder)
        template_edges.to_csv(dump_folder + '/template_edges.csv', index=False)
        candidate_edges.to_csv(dump_folder + '/candidate_edges.csv', index=False)
        template_nodes.to_csv(dump_folder + '/template_nodes.csv', index=False)
        candidate_nodes.to_csv(dump_folder + '/candidate_nodes.csv', index=False)
        pairing.to_csv(dump_folder + '/pairing.csv', index=False)


def empty_data_folder():
    ok = input("About to wipe out content of folder '" + DUMP_FOLDER + "', press Enter to continue...")
    for root, dirs, files in os.walk(DUMP_FOLDER):
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))


if __name__ == "__main__":
    empty_data_folder()
    process_candidates()
    process_single()
