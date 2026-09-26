from collections import defaultdict


def build_reply_graph(episode_df, predictions):
    """
    Build an undirected graph from reply relationships.

    If:
        M4 -> M2

    then M4 and M2 belong to the same discussion.
    """

    graph = defaultdict(set)

    message_ids = episode_df["message_id"].tolist()

    for message_id in message_ids:
        graph[message_id]

    for prediction in predictions:

        child = prediction["message_id"]
        parent = prediction["reply_to_message_id"]

        if parent == "NONE":
            continue

        if parent not in graph:
            continue

        graph[child].add(parent)
        graph[parent].add(child)

    return graph


def connected_components(graph, message_order):
    """
    Find discussion components.

    Components are ordered according to the first message
    appearing in the original timestamp order.
    """

    visited = set()
    components = []

    order = {
        message_id: index
        for index, message_id
        in enumerate(message_order)
    }

    for start in message_order:

        if start in visited:
            continue

        stack = [start]
        component = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            for neighbor in graph[current]:

                if neighbor not in visited:
                    stack.append(neighbor)

        component.sort(
            key=lambda x: order[x]
        )

        components.append(component)

    components.sort(
        key=lambda component:
        order[component[0]]
    )

    return components


def assign_thread_ids(
    episode_df,
    predictions,
):
    """
    Assign T1, T2, T3... within an episode.

    The challenge only compares thread labels within
    an episode, so local numbering is sufficient.
    """

    message_order = (
        episode_df["message_id"].tolist()
    )

    graph = build_reply_graph(
        episode_df,
        predictions,
    )

    components = connected_components(
        graph,
        message_order,
    )

    message_to_thread = {}

    for thread_number, component in enumerate(
        components,
        start=1,
    ):

        thread_id = (
            f"{episode_df.iloc[0]['episode_id']}"
            f"-T{thread_number}"
        )

        for message_id in component:
            message_to_thread[
                message_id
            ] = thread_id

    result = episode_df[
        [
            "episode_id",
            "message_id",
        ]
    ].copy()

    result[
        "reply_to_message_id"
    ] = [
        prediction["reply_to_message_id"]
        for prediction in predictions
    ]

    result["thread_id"] = (
        result["message_id"]
        .map(message_to_thread)
    )

    return result