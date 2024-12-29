#include "Genome.h"


Genome::Genome(std::vector<ConnectionGene> connection_genes, std::vector<NodeGene> node_genes) : connection_genes(connection_genes), node_genes(node_genes) {};

void Genome::mutateAddConnection()
{
    int attempts = 100;
    // pick random in node
    NodeGene *node_gene_in;
    // pick random out node
    NodeGene* node_gene_out;
    bool connection_exists = false;
    do
    {
        connection_exists = false;
        node_gene_in = &node_genes[rand() % this->node_genes.size()];
        std::cout << "Node Gene In Selected: " << node_gene_in->node_id << std::endl;
        node_gene_out = &node_genes[rand() % this->node_genes.size()];
        std::cout << "Node Gene out Selected: " << node_gene_out->node_id << std::endl;
        for (ConnectionGene connection_gene : connection_genes)
        {
            if (connection_gene.node_in == node_gene_in->node_id && connection_gene.node_out == node_gene_out->node_id)
            {
                connection_exists = true;
                break;
                std::cout << "Connection Already Exists, Finding Another" << std::endl;
            }
        }
        attempts--;
        if (attempts == 0) 
        {
            return; // coudlnt find a valid place to put a connection
        }
    } while ((node_gene_in->node_type == OUTPUT) || (node_gene_in->node_id == node_gene_out->node_id) || (node_gene_out->node_type == INPUT) || connection_exists);

    global_innovation_number++;
    int innovation_number = global_innovation_number;

    this->connection_genes.emplace_back(ConnectionGene(node_gene_in->node_id, node_gene_out->node_id, static_cast<double>(rand()) / RAND_MAX - 0.5, true, innovation_number));
    std::cout << "Added Connection" << std::endl;
}

void Genome::mutateAddNode()
{   
    // 1.) pick a random existing connection
    ConnectionGene *random_connection_gene = &connection_genes[rand() % this->connection_genes.size()];
    // disable the existing connection
    random_connection_gene->enabled = false;

    // get in node
    int base_node_in = random_connection_gene->node_in;
    // get out node
    int final_node_out = random_connection_gene->node_out;

    // 2.) generate the new node_id
    int new_node_id = node_genes.size() + 1;
    // create the node gene and add it
    this->node_genes.emplace_back(NodeGene(new_node_id, HIDDEN));

    global_innovation_number++;
    // 3.) create connection gene to connect base_node_in to new_node_id
    this->connection_genes.emplace_back(ConnectionGene(base_node_in, new_node_id, 1, true, global_innovation_number));

    global_innovation_number++;
    // create the second connection gene
    this->connection_genes.emplace_back(ConnectionGene(new_node_id, final_node_out, random_connection_gene->weight, true, global_innovation_number));

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
