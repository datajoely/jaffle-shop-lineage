import contextlib
import random
from html import escape

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import networkx as nx
from ibis.common.graph import Graph
from pyvis.network import Network

from jaffle_shop.utilities.context import all_logging_disabled


def get_type(node):
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


def get_data(node):
    typename = get_type(node)  # Already an escaped string
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
    data = {"data_name": nodename, "operation_name": name, "types": typename}
    return data


def ibis_expr_to_graph(expr):
    graph = Graph.from_bfs(expr.op())
    g = nx.DiGraph()

    seen_nodes = set()
    edges = set()

    for v, us in graph.items():
        v_data = get_data(v)
        vhash = str(hash(v))
        if v not in seen_nodes:
            g.add_node(vhash, **v_data)
            seen_nodes.add(v)

        for u in us:
            u_data = get_data(u)
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
            dataset_name, name=dataset_name, obj=dataset_type, attr_type="dataset"
        )

    for task in pipeline.nodes:
        kedro_graph.add_node(task.name, attr_type="task")  # , kedro_node=task
        for input_data in task.inputs:
            kedro_graph.add_edge(input_data, task.name)
        for output_data in task.outputs:
            kedro_graph.add_edge(task.name, output_data)

    return kedro_graph


def generate_random_color(seed):
    # Set a seed for reproducibility
    random.seed(seed)

    # Generate random RGB values
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    # Format the RGB values into a hex color code
    color_code = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return color_code


def render_nx_graph(
    nx_graph, notebook=True, color_field=None, label_attr=None, title_attr=None
):
    nt = Network(
        notebook=notebook,
        cdn_resources="in_line",
    )

    # Add nodes to Pyvis Network and color them based on degree
    for node in nx_graph.nodes:
        if color_field:
            attr_color = generate_random_color(
                nx_graph.nodes[node].get(color_field, "unknown")
            )
        else:
            attr_color = "#97c2fc"

        if label_attr:
            label_value = nx_graph.nodes[node].get(label_attr)
        else:
            label_value = node

        if title_attr:
            title_value = nx_graph.nodes[node].get(title_attr)
        else:
            title_value = "data"

        nt.add_node(node, label=label_value, color=attr_color, title=title_value)

    # Add edges to Pyvis Network
    for edge in nx_graph.edges():
        nt.add_edge(edge[0], edge[1])

    nt.show("graph.html")
    return nt


def render_kedro_node(node, catalog):
    with all_logging_disabled():
        node_outputs = node.run(
            {
                x: catalog.load(x).head(0) if "params:" not in x else catalog.load(x)
                for x in node.inputs
            }
        )
        nx_ibis_graphs = nx.compose_all(
            [ibis_expr_to_graph(v) for v in node_outputs.values()]
        )

        return render_nx_graph(
            nx_ibis_graphs,
            color_field="operation_name",
            label_attr="operation_name",
            title_attr="data_name",
        )
