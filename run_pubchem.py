import argparse
import pubchempy as pcp
import pickle as pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", "--name", help="compound name", type=str)
    parser.add_argument("-path", "--path", help="pathway for saving pickle file", type=str)
    args = parser.parse_args()
    print(args.name)

    # try:
    # print(args.info)
    doc = pcp.get_compounds(args.name, 'name')

    with open(args.path + '/' + args.name + '.pkl', 'wb') as f:
        pkl.dump(doc, f)