# use environment M2M_CodeBase!!!
import argparse
import pubchempy as pcp
import pickle as pkl
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-key", "--key", help="key", type=str)
    parser.add_argument("-val", "--val", help="val", type=str)
    parser.add_argument("-path", "--path", help="pathway for saving pickle file", type=str)
    args = parser.parse_args()

    # try:
    # print(args.info)
    temp = pcp.get_compounds(args.val, args.key, listkey_count = 5)
    doc = []
    for t in temp:
        doc.append(t.isomeric_smiles)

    if not os.path.isdir(args.path):
        os.mkdir(args.path)
    if not os.path.isdir(args.path + '/' + args.key):
        os.mkdir(args.path + '/' + args.key)

    with open(args.path + '/' + args.key + '/' + args.val + '.pkl', 'wb') as f:
        pkl.dump(doc, f)