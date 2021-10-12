from argparse import ArgumentParser
from astropy.table import Table, vstack

from astropy.utils.metadata import enable_merge_strategies
from astropy.utils.metadata import MergeStrategy

class MergeNumbersAsList(MergeStrategy):
    types = ((int, float, str),  # left side types
             (int, float, str))  # right side types

    @classmethod
    def merge(cls, left, right):
        if left == right:
            return left
        else:
            return str(left)+','+str(right)

def combine_catalogs(catalogs, output_cat):
    cats = []
    for cat in catalogs:
        cats.append(Table.read(cat))
    with enable_merge_strategies(MergeNumbersAsList):
        full_cat = vstack(cats)

    print('Writing full catalog to '+output_cat)
    full_cat.write(output_cat, format='fits', overwrite=True)

    return output_cat

def main():
    parser = new_argument_parser()
    args = parser.parse_args()

    input_cats = args.input_cats
    output_cat = args.output_cat

    combine_catalogs(input_cats, output_cat)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("input_cats", nargs='+',
                        help="""Pointing catalogs made by PyBDSF, to be combined.""")
    parser.add_argument('output_cat',
                        help="""Name of the full output catalog""")

    return parser

if __name__ == '__main__':
    main()