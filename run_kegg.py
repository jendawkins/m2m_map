from Bio.KEGG import REST
import argparse
import pickle as pkl
import urllib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-info", "--info", help="module, pathway, compound, etc to be read by KEGG API", type=str)
    parser.add_argument("-path", "--path", help="pathway for saving pickle file", type=str)
    args = parser.parse_args()


    # try:
    # print(args.info)
    doc = REST.kegg_get(args.info).read()

    with open(args.path + '/' + args.info + '.pkl', 'wb') as f:
        pkl.dump(doc, f)