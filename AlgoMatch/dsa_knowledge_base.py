# A knowledge base containing advanced DSA patterns, their complexities, and usage scenarios.
dsa_patterns = [
    {
        "name": "Segment Trees",
        "description": "A tree data structure used for storing information about intervals, or segments. It allows querying which of the stored segments contain a given point.",
        "use_cases": "Range sum/min/max queries, dynamic updates on array elements, handling intervals.",
        "time_complexity": "O(log N) for updates and range queries",
        "space_complexity": "O(N)"
    },
    {
        "name": "Trie (Prefix Tree)",
        "description": "A tree-like data structure that proves to be quite efficient for solving problems related to strings. It provides fast retrieval and stores characters efficiently.",
        "use_cases": "Autocomplete systems, spell checking, prefix matching, searching words in a dictionary.",
        "time_complexity": "O(L) where L is the string length for insert and search operations",
        "space_complexity": "O(N * M) where N is number of strings and M is max length"
    },
    {
        "name": "Topological Sort",
        "description": "Linear ordering of its vertices such that for every directed edge uv from vertex u to vertex v, u comes before v in the ordering. Only possible in Directed Acyclic Graphs (DAG).",
        "use_cases": "Course schedule dependencies, package/build dependency systems, task scheduling.",
        "time_complexity": "O(V + E) where V is vertices and E is edges",
        "space_complexity": "O(V)"
    },
    {
        "name": "Two Pointers",
        "description": "Using two pointers to iterate through an array or list, often moving towards each other or in the same direction to find pairs or process elements efficiently.",
        "use_cases": "Finding pairs in a sorted array that sum to a target, reversing an array, palindrome checking.",
        "time_complexity": "O(N)",
        "space_complexity": "O(1)"
    },
    {
        "name": "Sliding Window",
        "description": "A dynamic subset of two pointers where the distance between pointers defines a 'window' that slides over the contiguous array or data structure.",
        "use_cases": "Maximum sum subarray of size K, longest substring with K distinct characters, network throttling.",
        "time_complexity": "O(N)",
        "space_complexity": "O(1) or O(K)"
    },
    {
        "name": "Union-Find (Disjoint Set)",
        "description": "A data structure that tracks a set of elements partitioned into a number of disjoint (non-overlapping) subsets. Supports efficient union and find operations.",
        "use_cases": "Kruskal's Minimum Spanning Tree (MST), cycle detection in undirected graphs, social network connectedness.",
        "time_complexity": "O(α(N)) essentially constant time due to path compression and union by rank",
        "space_complexity": "O(N)"
    },
    {
        "name": "Monotonic Stack",
        "description": "A stack whose elements dynamically adjust to remain strictly increasing or strictly decreasing.",
        "use_cases": "Finding the Next Greater Element, Next Smaller Element, Largest Rectangle in a Histogram, daily temperatures.",
        "time_complexity": "O(N)",
        "space_complexity": "O(N)"
    },
    {
        "name": "Dynamic Programming (Knapsack 0/1 Pattern)",
        "description": "Breaking down a problem into nested, overlapping subproblems and storing the history of results (memoization or tabulation) to avoid recalculations.",
        "use_cases": "Choosing items with weight and value to maximize total value without exceeding capacity, resource allocation.",
        "time_complexity": "O(N * W) where W is the capacity",
        "space_complexity": "O(W)"
    },
    {
        "name": "Floyd-Warshall",
        "description": "A dynamic programming algorithm for finding shortest paths in a directed weighted graph with positive or negative edge weights (no negative cycles).",
        "use_cases": "Finding all-pairs shortest path in a graph, network routing protocols.",
        "time_complexity": "O(V^3)",
        "space_complexity": "O(V^2)"
    },
    {
        "name": "Knuth-Morris-Pratt (KMP)",
        "description": "A string-searching algorithm that searches for occurrences of a word within a main text string utilizing a Longest Prefix Suffix (LPS) array to skip redundant comparisons.",
        "use_cases": "Efficient pattern matching in massive documents, searching for a substring, DNA sequence matching.",
        "time_complexity": "O(N + M)",
        "space_complexity": "O(M)"
    },
    {
        "name": "Two Heaps Pattern",
        "description": "Utilizes two Priority Queues (a Max Heap and a Min Heap) to divide a set of numbers into two halves. It ensures that the elements in the first half are always smaller than or equal to the elements in the second half.",
        "use_cases": "Finding the Running Median of a data stream, Schedule optimization problems, maximizing capital in investments.",
        "time_complexity": "O(log N) for insertions, O(1) for finding the median",
        "space_complexity": "O(N)"
    }
]
