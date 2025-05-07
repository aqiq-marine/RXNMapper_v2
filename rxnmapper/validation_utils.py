import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import categorical_node_match, categorical_edge_match
from rxn.chemutils.conversion import smiles_to_mol
from rxnmapper.smiles_utils import process_reaction_with_product_maps_atoms, generate_atom_mapped_reaction_atoms, create_rxn_smiles_from_rxn_equation


def sort_canonicalize_rxn_and_maps(rxn_smi:str) -> str:
    """
    Canonicalizes the smiles reaction and standardizes the atom-maps order
    """
    try:
        rxn_smi_can_dict = process_reaction_with_product_maps_atoms(rxn_smi)

        if rxn_smi_can_dict is None:
            raise ValueError("process_reaction_with_product_maps_atom() returned None")
        
        if not isinstance(rxn_smi_can_dict, tuple):
            print(f" Invalid structure: {rxn_smi_can_dict}")
            raise ValueError("Expected tuple output")
        
        rxn_can, _ = generate_atom_mapped_reaction_atoms(*rxn_smi_can_dict, canonical=True)

        rxn_smi_can  = create_rxn_smiles_from_rxn_equation(rxn_can)
        return rxn_smi_can
    
    except Exception as e:
        print(f"{e}")
        return ''
    

def build_atom_graph(mol:Chem.Mol) -> nx.graph:
    """ 
    Construct a graph from an RDKit molecule. 
    Stores the atom index, atomic number, formal charge, degree, and mapping number as node attributes.
    The bond types are also stored as edge attributes.
    
    Args:
        mol: RDKit mol object
    
    Returns:
        nx.Graph: graph of the mol object, with node attributes {atomic_number, formal_charge, degree, atom_map number}
         and node attribute {bond_type}
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():   
        map_num = int(atom.GetProp("molAtomMapNumber") if atom.HasProp("molAtomMapNumber") else 0)
        G.add_node(atom.GetIdx(), 
                   atomic_num=atom.GetAtomicNum(), 
                   formal_charge=atom.GetFormalCharge(), 
                   degree=atom.GetDegree(), 
                   atom_map=map_num, 
        )
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 
                   bond_type=str(bond.GetBondType()))
    return G


def get_atom_map_attr_dict(G:nx.Graph) -> Dict:
    '''
    Returns a dict {node:atom_map} where node is the name of each node, and atom_map is the corresponding "atom_map"
    attribute.

    Args:
        G: NetworkX graph of a molecule, with defined "atom_map" node attribute
    
    Returns:
        Dict: {node index:atom_map}, where atom_map is 0 if not defined
    '''
    assert isinstance(G, nx.Graph), f"Expected a NetworkX graph, got {type(G)}"
    try:
        atom_map_attr_dict = nx.get_node_attributes(G, "atom_map")
        assert len(atom_map_attr_dict)==G.number_of_nodes(), "Every node must have an 'atom_map' attribute"
        return atom_map_attr_dict
    except:
        print(f"Node attribute 'atom_map' needs to be defined")


def has_g_maps(G : nx.Graph) -> bool:
    '''
    Returns True if at least one of G nodes represents an atom-mapped atom, i.e. defined "atom_map" attribute (>0).
    Returns False otherwise

    Args:
        G: NetworkX graph of a molecule, with defined "atom_map" node attribute
    
    Returns:
        bool: True if at least one node of G has a defined "atom_map" attribute >0, False otherwise.
    '''

    assert isinstance(G, nx.Graph), f"Expected a NetworkX graph, got {type(G)}"
    atom_map_attr_dict = get_atom_map_attr_dict(G)
    
    return max(list(atom_map_attr_dict.values())) > 0


def union_of_reac_prod_graphs_lists(reacG_list:List[nx.Graph], prodG_list:List[nx.Graph]) -> nx.Graph:
    """
    Creates a union graph from multiple reactant and product graphs.
    Each node is renamed with a prefix ('R-' or 'P-') and an index to ensure uniqueness.

    Args:
        reacG_list (List[nx.Graph]): List of NetworkX graphs for reactants.
        prodG_list (List[nx.Graph]): List of NetworkX graphs for products.

    Returns:
        nx.Graph: A unified graph with unique node names and preserved node/edge attributes.
    """
    # Test inputs
    for name, graphs in [("reactant_graphs", reacG_list), ("product_graphs", prodG_list)]:
        if not isinstance(graphs, list):
            raise TypeError(f"{name} must be a list, got {type(graphs)}")
        if not graphs:
            raise ValueError(f"{name} must not be empty")
        if not all(isinstance(g, nx.Graph) for g in graphs):
            raise TypeError(f"All elements in {name} must be instances of networkx.Graph")

    # create a union graph containing all reactants and products with atom-maps,
    #  labeling their nodes with distinct prefixes        
    union_G = nx.Graph()
    
    def add_graphs(graphs, prefix):
        for i, G in enumerate(graphs):
            mapping = {n: f"{prefix}{i}-{n}" for n in G.nodes}
            renamed_G = nx.relabel_nodes(G, mapping)
            union_G.update(renamed_G)
    
    add_graphs(reacG_list, prefix='R-')
    add_graphs(prodG_list, prefix='P-')
    
    return union_G


def create_rxn_g(unionG:nx.Graph) -> nx.Graph:
    '''
    Takes a union graph containing reactant and product graphs, with their respective node names starting with "R-"
    or "P-". Outputs a graph of the reaction where the nodes of reactants and products sharing the same atom_map 
    number are connected by a special edge with the 'mapping' label.

    Args:
        unionG (nx.Graph):  NetworkX graph containing several non-connected graphs of reactant(s) and product(s) 
        from a reaction with node names starting with at least a 'R-' or 'P-' prefix to ensure that each node name 
        is unique.

    Returns:
        nx.Graph: Reaction graph with special edges connecting the reactants' and products' nodes that share the 
        same "atom_map" attribute. The special edges have a 'mapping' label. 
    '''

    assert isinstance(unionG, nx.Graph), f"Expected a NetworkX graph, got {type(unionG)}"
    
    atom_map_attr_dict = get_atom_map_attr_dict(unionG)
    
    atom_map_to_nodes = defaultdict(list)
    for node, amap in atom_map_attr_dict.items():
        if amap>0:
            atom_map_to_nodes[amap].append(node)
    
    for amap_val, nodes in atom_map_to_nodes.items():
        reactant_nodes = [n for n in nodes if n.startswith("R-")]
        product_nodes = [n for n in nodes if n.startswith("P-")]

        for p in product_nodes:
            for r in reactant_nodes:
                unionG.add_edge(p, r, label='mapping')

    return unionG


def remove_node_attr_from_g(G:nx.Graph, attr_name="atom_map"):
    '''
    Removes the 'attr_name' node attribute from all nodes in graph G

    Args:
        G: NetworkX graph
    
    Returns:
        G without the node attribute 'attr_name'
    '''
    assert isinstance(G, nx.Graph), f"Expected a NetworkX graph, got {type(G)}"

    G2 = G.copy()
    for node in G2.nodes:
        G2.nodes[node].pop(attr_name, None)
    return G2


def rxn_g_from_rxn_smi(rxn_smi:str) -> nx.Graph:
    '''
    Converts a reaction SMILES string into a reaction graph.

    The function parses a reaction SMILES (with atom-mapping), canonicalizes it,
    constructs molecular graphs for each reactant and product, filters out molecules
    without any mapped atoms, and builds a unified reaction graph. Reactant and
    product atoms are tagged with 'R-' and 'P-' prefixes respectively, and atom-mapped
    pairs are connected with special edges.

    Args:
        rxn_smi (str): A reaction SMILES string with atom-mapping.

    Returns:
        nx.Graph: A reaction graph where nodes represent atoms (from reactants and
        products) and special edges connect nodes (atoms) with shared atom-mapping indices.   
    '''

    assert isinstance(rxn_smi, str), f"Expected a reaction SMILES string, got {type(rxn_smi)}"

    # canonicalize smiles and atom-mapping
    can_rxn_smi = sort_canonicalize_rxn_and_maps(rxn_smi)

    # convert to RDKit reaction object
    rxn = AllChem.ReactionFromSmarts(can_rxn_smi, useSmiles=True)

    # build graphs for reactants and products
    graphs_reacs = [(build_atom_graph(mol), mol) for mol in rxn.GetReactants()]
    graphs_prods = [(build_atom_graph(mol), mol) for mol in rxn.GetProducts()]

    # keep only graphs of molecules containing at least one mapped atom
    graphs_reacs_molmap = [g for g, mol in graphs_reacs if has_g_maps(g)]
    graphs_prods_molmap = [g for g, mol in graphs_prods if has_g_maps(g)]

    # create a graph with all mapped molecules in it, with a 'R-'/'P-' prefix on each node
    unionG = union_of_reac_prod_graphs_lists(reacG_list=graphs_reacs_molmap, prodG_list=graphs_prods_molmap)
    
    # build the reaction graph
    rxn_G = create_rxn_g(unionG=unionG)

    return rxn_G


def define_node_edge_matches():
    '''
    Defines the node and edge matches as comparison functions for categorical attributes 
    for evaluation of isomorphism as colored graphs within the NetworkX package.

    Args:
        None

    Returns:
            Node match (function): NetworkX comparison function for 'atomic_num', 'formal_charge', and 'degree' attributes
            Edge match (function): NetworkX comparison function for 'bond_type', and 'label' attributes 

    '''
    node_match = categorical_node_match(["atomic_num", "formal_charge", "degree"], [None, None, None])
    edge_match = categorical_edge_match(["bond_type", "label"], [None, None])
    return node_match, edge_match


def are_two_rxn_AM_equ(gt_smi:str, pred_smi:str) -> bool:
    '''
    Compares two atom-mapped reaction SMILES of the same reaction and checks if both atom-mappings 
    are equivalent (i.e. if they are chemically equivalent). Atom-mappings of tautomeric structures 
    are not considered equivalent, as template extraction of both reaction SMILES would yield different 
    reaction templates. 
    Overall process:
        - canonicalization of SMILES and atom-maps
        - for each reaction, build NetworkX graphs of all reactants and products
        - for each reaction, build a union graph of all reactants and products that have at least one atom-mapped atom.
        - build reaction graphs, with edges linking nodes (i.e. atoms) sharing the same "atom_map" attribute
        - check for isomorphism as colored graphs for both reaction graphs
    
    Args:
        - gt_smi (str):     Atom-mapped reaction SMILES of the ground truth mapping
        - pred_smi (str):   Atom-mapped reaction SMILES of the predicted mapping

    Returns:
        bool:   True if both atom-mappings are equivalent, False otherwise
    '''

    assert isinstance(gt_smi, str), f"Expected an atom-mapped reaction SMILES string, got {type(gt_smi)}"
    assert isinstance(pred_smi, str), f"Expected an atom-mapped reaction SMILES string, got {type(pred_smi)}"

    rxn_gt_G = rxn_g_from_rxn_smi(rxn_smi=gt_smi)
    rxn_pred_G = rxn_g_from_rxn_smi(rxn_smi=pred_smi)

    node_match, edge_match = define_node_edge_matches()

    return nx.is_isomorphic(rxn_gt_G, rxn_pred_G, node_match=node_match, edge_match=edge_match)


