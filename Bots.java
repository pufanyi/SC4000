/* perform Kosaraju's algorithm
count SCCs with more than one bot -> no. of botnets
SCC that only have one bot is a solo bot
*/

import java.io.*;
import java.util.*;

class Kattio extends PrintWriter {
    public Kattio(InputStream i) {
        super(new BufferedOutputStream(System.out));
        r = new BufferedReader(new InputStreamReader(i));
    }

    public Kattio(InputStream i, OutputStream o) {
        super(new BufferedOutputStream(o));
        r = new BufferedReader(new InputStreamReader(i));
    }

    public boolean hasMoreTokens() {
        return peekToken() != null;
    }

    public int getInt() {
        return Integer.parseInt(nextToken());
    }

    public double getDouble() {
        return Double.parseDouble(nextToken());
    }

    public long getLong() {
        return Long.parseLong(nextToken());
    }

    public String getWord() {
        return nextToken();
    }

    private BufferedReader r;
    private String line;
    private StringTokenizer st;
    private String token;

    private String peekToken() {
        if (token == null)
            try {
                while (st == null || !st.hasMoreTokens()) {
                    line = r.readLine();
                    if (line == null)
                        return null;
                    st = new StringTokenizer(line);
                }
                token = st.nextToken();
            } catch (IOException e) {
            }
        return token;
    }

    private String nextToken() {
        String ans = peekToken();
        token = null;
        return ans;
    }
}

public class Bots {
    static boolean[] visited;
    static Stack<Integer> K;
    static List<Integer>[] adjList;
    static int[] parent;
    static List<List<Integer>> SCCs;

    static void DFS(int u) {
        visited[u] = true;
        for (int v : adjList[u]) {
            if (!visited[v]) {
                parent[v] = u;
                DFS(v);
            }
        }
    }

    static void DFS_SCC(int u, List<Integer> scc) {
        visited[u] = true;
        scc.add(u);
        for (int v : adjList[u]) {
            if (!visited[v]) {
                parent[v] = u;
                DFS_SCC(v, scc);
            }
        }
    }

    static Stack<Integer> DFStopo(List<Integer>[] adjList) {
        K = new Stack<>();
        visited = new boolean[adjList.length];
        parent = new int[adjList.length];
        for (int i = 0; i < adjList.length; i++) {
            visited[i] = false;
            parent[i] = -1;
        }

        for (int i = 0; i < adjList.length; i++) {
            if (!visited[i]) {
                DFS(i);
                K.push(i);
            }
        }

        return K;
    }

    public static void main(String[] args) {
        Kattio io = new Kattio(System.in, System.out);
        int N = io.getInt();
        int M = io.getInt();
        adjList = new ArrayList[N];
        visited = new boolean[N];
        for (int i = 0; i < N; i++) {
            adjList[i] = new ArrayList<>();
        }
        for (int i = 0; i < M; i++) {
            int P = io.getInt();
            int Q = io.getInt();
            adjList[P].add(Q);
        }

        K = DFStopo(adjList);
        SCCs = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            visited[i] = false;
        }
        while (!K.isEmpty()) {
            int u = K.pop();
            if (!visited[u]) {
                List<Integer> scc = new ArrayList<>();
                DFS_SCC(u, scc);
                SCCs.add(scc);
            }
        }

        int solobots = 0;
        int botnets = 0;
        for (List<Integer> scc : SCCs) {
            if (scc.size() == 1) {
                solobots++;
            } else {
                botnets++;
            }
        }

        io.println(solobots + " " + botnets);
        io.close();

    }
}