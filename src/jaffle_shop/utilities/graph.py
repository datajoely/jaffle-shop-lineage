import contextlib
import random
from html import escape
import tempfile

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import networkx as nx
from ibis.common.graph import Graph
from pyvis.network import Network

from jaffle_shop.utilities.context import all_logging_disabled


def _get_type(node):
    with contextlib.suppress(AttributeError, NotImplementedError):
        return escape(str(node.dtype))

    try:
        schema = node.schema
    except (AttributeError, NotImplementedError):
        # TODO(kszucs): this branch should be removed
        try:
            # As a last resort try get the name of the output_type class
            return node.output_type.__name__
        except (AttributeError, NotImplementedError):
            return "\u2205"  # empty set character
    except com.IbisError:
        assert isinstance(node, ops.Join)
        left_table_name = getattr(node.left, "name", None) or ops.genname()
        left_schema = node.left.schema
        right_table_name = getattr(node.right, "name", None) or ops.genname()
        right_schema = node.right.schema
        pairs = [
            (f"{left_table_name}.{left_column}", type)
            for left_column, type in left_schema.items()
        ] + [
            (f"{right_table_name}.{right_column}", type)
            for right_column, type in right_schema.items()
        ]
        schema = ibis.schema(pairs)

    return dict(zip(schema.names, schema.types))


def _get_data(node):
    typename = _get_type(node)  # Already an escaped string
    name = type(node).__name__
    nodename = (
        node.name
        if isinstance(
            node,
            (
                ops.Literal,
                ops.TableColumn,
                ops.Alias,
                ops.PhysicalTable,
                ops.window.RangeWindowFrame,
            ),
        )
        else None
    )
    data = dict(
        color_category=name,
        description=nodename,
        label=name,
        types=typename,
        unique_key=nodename,
        kind="ibis",
    )
    return data


def _ibis_expr_to_graph(expr):
    graph = Graph.from_bfs(expr.op())
    g = nx.DiGraph()

    seen_nodes = set()
    edges = set()

    for v, us in graph.items():
        v_data = _get_data(v)
        vhash = str(hash(v))
        if v not in seen_nodes:
            g.add_node(vhash, **v_data)
            seen_nodes.add(v)

        for u in us:
            u_data = _get_data(u)
            uhash = str(hash(u))
            if u not in seen_nodes:
                g.add_node(uhash, **u_data)
                seen_nodes.add(u)
            if (edge := (u, v)) not in edges:
                for name, arg in zip(v.argnames, v.args):
                    if isinstance(arg, tuple) and u in arg:
                        index = arg.index(u)
                        name = f"{name}[{index}]"
                        break
                    elif arg == u:
                        break
                else:
                    name = None

                g.add_edge(uhash, vhash, label=name)
                edges.add(edge)
    return g


def kedro_pipeline_to_graph(pipeline, catalog):
    kedro_graph = nx.DiGraph()

    for dataset_name in pipeline.datasets():
        if hasattr(catalog.datasets, dataset_name.replace(":", "__")):
            dataset_type = getattr(
                catalog.datasets, dataset_name.replace(":", "__")
            ).__class__.__name__
        else:
            dataset_type = "MemoryDataset"

        kedro_graph.add_node(
            dataset_name,
            color_category="dataset",
            description=dataset_type,
            label=dataset_name,
            unique_key=dataset_name,
            kind="kedro",
        )

    for task in pipeline.nodes:
        kedro_graph.add_node(
            task.name,
            color_category="task",
            description=task.name,
            label=task.name.split("(")[0],
            unique_key=task.name.split("(")[0],
            kind="kedro",
        )
        for input_data in task.inputs:
            kedro_graph.add_edge(input_data, task.name)
        for output_data in task.outputs:
            kedro_graph.add_edge(task.name, output_data)

    return kedro_graph


def _generate_random_color(seed):
    random.seed(seed)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color_code = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return color_code


def nx_graph_to_pyvis(nx_graph, notebook=True) -> Network:
    nt = Network(
        notebook=notebook,
        cdn_resources="in_line",
    )

    # Add nodes to Pyvis Network and color them based on degree
    for node in nx_graph.nodes:
        attr_color = _generate_random_color(
            nx_graph.nodes[node].get("color_category", "unknown")
        )
        label_value = nx_graph.nodes[node].get("label")
        title_value = nx_graph.nodes[node].get("description")
        shape = "square" if nx_graph.nodes[node].get('kind') == "kedro" else "dot"
        nt.add_node(node, label=label_value, color=attr_color, title=title_value, shape=shape)

    # Add edges to Pyvis Network
    for edge in nx_graph.edges():
        nt.add_edge(edge[0], edge[1])

    return nt

def kedro_node_ibis_plan(node, catalog):
    with all_logging_disabled():
        node_outputs = node.run(
            {
                x: catalog.load(x).head(0) if "params:" not in x else catalog.load(x)
                for x in node.inputs
            }
        )

        ibis_output_graphs = {
            output: _ibis_expr_to_graph(g) for output, g in node_outputs.items()
        }

        # todo we do a head(0) in every execution so need to 
        # remove this from the graph artificially

        last_toposort = {
            output: list(nx.topological_generations(g))[-1]
            for output, g in ibis_output_graphs.items()
        }

        nx_ibis_graphs = nx.compose_all(ibis_output_graphs.values())

        for output_name, node_id in last_toposort.items():
            working_node = nx_ibis_graphs.nodes[node_id[0]]
            working_node["description"] = output_name
            working_node["unique_key"] = output_name
            working_node["kind"] = "ibis"

    return nx_ibis_graphs


def get_lineage_for_kedro_node(pipe, node, data_catalog):
    def _retrieve_node_by_desc(g, key, value, kind):
        for i, n in g.nodes(data=True):
            if n.get(key) == value and n.get("kind") == kind:
                return i

    pipeline_graph = kedro_pipeline_to_graph(pipe, data_catalog)
    node_graph = kedro_node_ibis_plan(node, data_catalog)
    combined_graphs = nx.disjoint_union(node_graph, pipeline_graph)

    for i in node.inputs:
        
        kedro_edge = _retrieve_node_by_desc(
            g=combined_graphs, key="unique_key", value=i, kind="kedro"
        )
        ibis_edge = _retrieve_node_by_desc(
            g=combined_graphs, key="unique_key", value=i, kind="ibis"
        )
        combined_graphs.add_edge(kedro_edge, ibis_edge, width=10)

    for o in node.outputs:
        kedro_edge = _retrieve_node_by_desc(
            g=combined_graphs, key="unique_key", value=o, kind="kedro"
        )
        ibis_edge = _retrieve_node_by_desc(
            g=combined_graphs, key="unique_key", value=o, kind="ibis"
        )
        combined_graphs.add_edge(ibis_edge, kedro_edge, width=10)

    return combined_graphs
