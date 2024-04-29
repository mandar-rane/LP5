#include <iostream>
#include <vector>
#include <queue>
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

    // Breadth First Search
    void BFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int v;
            #pragma omp critical
            {
                v = q.front();
                q.pop();
            }

            cout << v << " ";

            // Visit all adjacent vertices of v
            #pragma omp parallel for
            for (int w : adj[v]) {
                if (!visited[w]) {
                    #pragma omp critical
                    {
                        visited[w] = true;
                        q.push(w);
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

    cout << "BFS starting from vertex 0: ";
    double start_time = omp_get_wtime();
    g.BFS(0);
    double end_time = omp_get_wtime();
    cout << endl;
    cout << "Time taken: " << end_time - start_time << " seconds" <<endl;

    return 0;
}
