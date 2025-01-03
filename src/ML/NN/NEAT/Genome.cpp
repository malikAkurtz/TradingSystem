#include "Genome.h"
#include "Entity.h"

Genome::Genome() {};

Genome::Genome(std::vector<ConnectionGene> connection_genes, std::vector<NodeGene> node_genes) : connection_genes(connection_genes), node_genes(node_genes) {};


std::string nodeToString(NodeType type)
{
    switch (type)
    {
        case INPUT: return "INPUT";
        case HIDDEN: return "HIDDEN";
        case OUTPUT: return "OUTPUT";
        case BIAS: return "BIAS";
        default:
            return "UNKNOWN";
    }
}

void Genome::mutateAddConnection()
{
    int initial_attempts = 10 * this->node_genes.size();
    int attempts = initial_attempts;
    // pick random in node
    NodeGene &node_gene_in = node_genes[rand() % this->node_genes.size()];
    // pick random out node
    NodeGene &node_gene_out = node_genes[rand() % this->node_genes.size()];

    std::map<int, int> id_to_depth = this->mapIDtoDepth();
    bool connection_is_valid = false;

    while (!connection_is_valid && attempts > 0)
    {
        node_gene_in = node_genes[rand() % this->node_genes.size()];
        node_gene_out = node_genes[rand() % this->node_genes.size()];

        debugMessage("mutateAddConnection", "Attempting Connection: In Node: " + std::to_string(node_gene_in.node_id) + " Out Node: " + std::to_string(node_gene_out.node_id));

        // if the source connection is the output, skip
        if (node_gene_in.node_type == OUTPUT) 
        {
            debugMessage("mutateAddConnection", "Rejected: In node is OUTPUT");
            continue;
        }
        // if the destination connection is an input node, skip
        if (node_gene_out.node_type == INPUT) 
        {
            debugMessage("mutateAddConnection", "Rejected: Out node is INPUT");
            continue;
        }
        // if the connection is to the same node, skip
        if (node_gene_in.node_id == node_gene_out.node_id) 
        {
            debugMessage("mutateAddConnection", "Rejected: Same node");
            continue;
        }
        if (id_to_depth[node_gene_in.node_id] >= id_to_depth[node_gene_out.node_id]) 
        {
            debugMessage("mutateAddConnection", "Depth Condition not met");
            continue;
        }

        bool exists = false;
        for (ConnectionGene connection_gene : connection_genes)
        {
            if (connection_gene.node_in == node_gene_in.node_id && connection_gene.node_out == node_gene_out.node_id)
            {
                exists = true;
                break;
            }
        }


        if (exists)
        {
            debugMessage("mutateAddConnection", "Connection Already Exists, Finding Another");
            attempts--;
            continue;
        }

        connection_is_valid = true;
        attempts--;
    }

    if (!connection_is_valid) 
    {   
        debugMessage("mutateAddConnection", "Warning: No valid connection found after " + std::to_string(attempts) + " attempts");
        return;
    }
    debugMessage("mutateAddConnection", "Added a connection from: " + std::to_string(node_gene_in.node_id) + " To: " + std::to_string(node_gene_out.node_id));

    global_innovation_number++;

    // std::cout << "We are trying: in->id=" << node_gene_in->node_id
    //       << ", in->type=" << nodeToString(node_gene_in->node_type)
    //       << " out->id=" << node_gene_out->node_id
    //       << ", out->type=" << nodeToString(node_gene_out->node_type)
    //       << std::endl;
    this->connection_genes.emplace_back(
        ConnectionGene(
            node_gene_in.node_id, node_gene_out.node_id, 
            static_cast<double>(rand()) / RAND_MAX - 0.5, 
            true, 
            global_innovation_number)
            );
}

void Genome::mutateAddNode()
{   
    // 1.) pick a random existing connection
    ConnectionGene &random_connection_gene = this->connection_genes[rand() % this->connection_genes.size()];


    int initial_attempts = 10 * this->connection_genes.size();
    int attempts = initial_attempts;

    while (random_connection_gene.enabled == false)
    {
        random_connection_gene = this->connection_genes[rand() % this->connection_genes.size()];
        attempts--;
        if (attempts <= 0)
        {
            return;
        }
    }

    debugMessage("mutateAddNode", "Connection Gene Selected for Node Placement: " + random_connection_gene.toString());

    double prev_weight = random_connection_gene.weight;
    // disable the existing connection
    random_connection_gene.enabled = false;

    // get in node
    int base_node_in = random_connection_gene.node_in;
    // get out node
    int final_node_out = random_connection_gene.node_out;

    // 2.) generate the new node_id
    int new_node_id = node_genes.size();
    // create the node gene and add it
    this->node_genes.emplace_back(NodeGene(new_node_id, HIDDEN));

    // 3.) create connection gene to connect base_node_in to new_node_id
    this->connection_genes.emplace_back(ConnectionGene(base_node_in, new_node_id, 1, true, global_innovation_number++));

    // create the second connection gene
    this->connection_genes.emplace_back(ConnectionGene(new_node_id, final_node_out, prev_weight, true, global_innovation_number++));

    std::sort(this->connection_genes.begin(), this->connection_genes.end(), [](const ConnectionGene &a, const ConnectionGene &b)
              { return a.innovation_number < b.innovation_number;});

    std::cout
        << "Added Node: " << new_node_id << " With node_in: " << base_node_in << " And node_out: " << final_node_out << std::endl;
}

void Genome::mutateChangeWeight()
{
    ConnectionGene &random_connection_gene = this->connection_genes[rand() % this->connection_genes.size()];
    // debugMessage("mutateChangeWeight", "Connection Gene Selected for Weight Replacement: " + random_connection_gene.toString());

    double random_weight = static_cast<double>(rand()) / RAND_MAX - 0.5;

    random_connection_gene.weight = random_weight;

}

std::map<int, Node> Genome::mapIDtoNode()
{
    std::map<int, Node> id_to_node;
    // for every node in the NodeGene sequence of the genome
    for (const NodeGene& node_gene : this->node_genes)
    {
        // add it to the map which maps ids to nodes
        id_to_node.emplace(node_gene.node_id, Node(node_gene));
    }

    return id_to_node;
}

std::map<int, int> Genome::mapIDtoDepth()
{
    std::map<int, int> id_to_depth;

    for (const NodeGene& node_gene : this->node_genes)
    {
        id_to_depth[node_gene.node_id] = 0;
    }
    
    bool change_occurred = true;

    while (change_occurred)
    {
        change_occurred = false;
        for (const ConnectionGene &connection_gene : connection_genes)
        {
            // std::cout << "Connection gene when mapping looks like: " << std::endl;
            // std::cout << connection_gene.toString() << std::endl;
            int conn_node_in = connection_gene.node_in;
            int conn_node_out = connection_gene.node_out;
            
            int prev_conn_node_out_depth = id_to_depth[conn_node_out];

            id_to_depth[conn_node_out] = std::max(id_to_depth[conn_node_out], id_to_depth[conn_node_in] + 1);

            if (prev_conn_node_out_depth < id_to_depth[conn_node_out])
            {
                change_occurred = true;
            }
        }
    }
    
    return id_to_depth;
}

void Genome::assignConnectionsToNodes(std::map<int, Node>& id_to_node)
{
    for (const auto& cg : this->connection_genes)
    {
        int node_out = cg.node_out;

        auto it = id_to_node.find(node_out);
        if (it != id_to_node.end()) {
            it->second.connections_in.emplace_back(Connection(cg));
        } else {
            std::cerr << "Error: Node " << node_out << " not found in id_to_node." << std::endl;
        }
    }
}


std::string Genome::toString() const
{
    std::string result = "Genome:\n";
    
    result += "NodeGenes:\n";
    for (const auto& node : this->node_genes)
    {
        result += "  " + node.toString() + "\n";
    }

    result += "ConnectionGenes:\n";
    for (const auto& connection : this->connection_genes)
    {
        result += "  " + connection.toString() + "\n";
    }

    return result;
}



std::string Genome::toGraphviz() const
{
    std::string result = "digraph Genome {\n";
    result += "  rankdir=LR;\n"; // Left-to-right layout

    // Subgraph for input layer
    result += "  subgraph cluster_0 {\n";
    result += "    label=\"Input Layer\";\n";
    for (const auto& node : this->node_genes)
    {
        if (node.node_type == INPUT)
        {
            result += "    " + std::to_string(node.node_id) + ";\n";
        }
    }
    result += "  }\n";

    // Subgraph for hidden layers
    result += "  subgraph cluster_1 {\n";
    result += "    label=\"Hidden Layer\";\n";
    for (const auto& node : this->node_genes)
    {
        if (node.node_type == HIDDEN)
        {
            result += "    " + std::to_string(node.node_id) + ";\n";
        }
    }
    result += "  }\n";

    // Subgraph for output layer
    result += "  subgraph cluster_2 {\n";
    result += "    label=\"Output Layer\";\n";
    for (const auto& node : this->node_genes)
    {
        if (node.node_type == OUTPUT)
        {
            result += "    " + std::to_string(node.node_id) + ";\n";
        }
    }
    result += "  }\n";

    // Add connections
    for (const auto& connection : this->connection_genes)
    {
        if (connection.enabled)
        {
            result += "  " + std::to_string(connection.node_in) + " -> " + std::to_string(connection.node_out) +
                      " [label=\"w: " + std::to_string(connection.weight) + "\"];\n";
        }
    }

    result += "}\n";
    return result;
}





void Genome::saveToDotFile(const std::string& filename) const
{
    std::ofstream file(filename); // Open file for writing
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Generate the Graphviz-compatible dot content
    std::string dotContent = this->toGraphviz();
    
    // Write the content to the file
    file << dotContent;
    file.close(); // Close the file
    std::cout << "Graphviz content saved to " << filename << std::endl;
}

