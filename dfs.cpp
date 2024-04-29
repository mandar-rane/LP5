#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>

using namespace std;

// Represents an undirected graph using adjacency lists
class Graph {
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency lists

public:
    Graph(int V) {
        this->V = V; // Set the number of vertices to the provided value V
        this->adj = vector<vector<int>>(V); // Initialize the adjacency list vector with V empty vectors
    }

    // Add an edge between v and w
    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v);
    }


    // Depth First Search
    void DFS(int start) {
        vector<bool> visited(V, false);
        stack<int> s;

        visited[start] = true;
        s.push(start);

        while (!s.empty()) {
            int v;
            #pragma omp critical
            {
                v = s.top();
                s.pop();
            }

            cout << v << " ";

            // Visit all adjacent vertices of v
            #pragma omp parallel for
            for (int w : adj[v]) {
                if (!visited[w]) {
                    #pragma omp critical
                    {
                        visited[w] = true;
                        s.push(w);
                    }
                }
            }
        }
    }
};

int main() {
    // Create a graph with 5 vertices
    Graph g(5);

    // Add edges
    g.addEdge(2, 1);
    g.addEdge(4, 2);
    g.addEdge(1, 3);
    g.addEdge(3, 4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);


    cout << "DFS starting from vertex 0: ";

    double start_time = omp_get_wtime();
    g.DFS(0);
    double end_time = omp_get_wtime();
    cout << endl;

    cout << "Time taken: " << end_time - start_time << " seconds" <<endl;

    return 0;
}
