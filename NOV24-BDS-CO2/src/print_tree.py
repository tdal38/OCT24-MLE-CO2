import os
from treelib import Tree

def build_tree(root):
    tree = Tree()
    tree.create_node(os.path.basename(root), root) 

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        filenames = [f for f in filenames if not f.startswith('.')]
        for d in dirnames:
            full_path = os.path.join(dirpath, d)
            tree.create_node(d, full_path, parent=dirpath)
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            tree.create_node(f, full_path, parent=dirpath)
    return tree

if __name__ == "__main__":
    root_path = "/Users/bouchaibchelaoui/Desktop/DATASCIENTEST/PROJET_CO2_DST/nov24_bds_co2"
    tree = build_tree(root_path)
    # Affichage de l'arborescence sous forme de chaîne lisible
    print(tree)
