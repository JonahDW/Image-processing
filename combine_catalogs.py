from argparse import ArgumentParser
from astropy.table import Table, vstack, hstack, join

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
            return None

def vstack_catalogs(catalogs, output_cat):
    cats = []
    for cat in catalogs:
        cats.append(Table.read(cat))
    with enable_merge_strategies(MergeNumbersAsList):
        full_cat = vstack(cats)

    print('Writing full catalog to '+output_cat)
    full_cat.write(output_cat, format='fits', overwrite=True)

    return output_cat

def hstack_catalogs(catalogs, output_cat):
    cats = []
    for cat in catalogs:
        cats.append(Table.read(cat))
    with enable_merge_strategies(MergeNumbersAsList):
        full_cat = hstack(cats)

    print('Writing full catalog to '+output_cat)
    full_cat.write(output_cat, format='fits', overwrite=True)

    return output_cat

def join_catalogs(catalogs, output_cat, join_type, keys):
    left = Table.read(catalogs[0])
    right = Table.read(catalogs[1])

    full_cat = join(left, right, keys=keys, join_type=join_type)

    print('Writing full catalog to '+output_cat)
    full_cat.write(output_cat, format='fits', overwrite=True)

    return output_cat

def main():
    parser = new_argument_parser()
    args = parser.parse_args()

    input_cats = args.input_cats
    output_cat = args.output_cat
    mode = args.mode
    keys = args.keys

    print('Combining catalogs: '+','.join(input_cats))

    if mode == 'vstack':
        vstack_catalogs(input_cats, output_cat)
    if mode == 'hstack':
        hstack_catalogs(input_cats, output_cat)
    if 'join' in mode:
        if len(mode.split(' ')) > 1:
            join_type = mode.split(' ')[0]
        else:
            join_type = 'inner'
        join_catalogs(input_cats, output_cat, join_type, keys)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("input_cats", nargs='+',
                        help="""Pointing catalogs made by PyBDSF, to be combined.""")
    parser.add_argument('output_cat',
                        help="""Name of the full output catalog""")
    parser.add_argument('-m', '--mode', default='vstack',
                        help="""How to combine tables. Supports vstack, hstack, or
                                different join types (inner join, outer join, etc.)
                                default = vstack""")
    parser.add_argument('-k', '--keys', nargs='+', default='Source_id',
                        help="""Key column(s) for join""")

    return parser

if __name__ == '__main__':
    main()