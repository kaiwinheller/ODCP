from collections import defaultdict, deque

def find_order(n, conditions):
    # Build the graph with reversed edges based on the new requirement
    graph = defaultdict(list)
    indegree = {i: 0 for i in range(n)}  # In-degree for all nodes
    for num, successors in conditions:
        for succ in successors:
            graph[num].append(succ)
            indegree[succ] += 1

    # Perform topological sort
    queue = deque([node for node in indegree if indegree[node] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) == n:
        return order
    else:
        return "Invalid ordering"  # Cycle detected or not all nodes covered

# Example
n = 5
conditions = [
    (0, [1]),  # 0 should precede 1
    (1, [3]),  # 1 should precede 3
    (2, [3]),  # 2 should precede 3
    (3, [2, 4])   # 3 should precede 4
]

print(find_order(n, conditions))