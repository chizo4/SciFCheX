'''
--------------------------------------------------------------
FILE:
    pipeline/eval/tools/label.py

INFO:
    The script implements tools associated with processing
    dataset labels, e.g. mapping them to integer values.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    03/2024
--------------------------------------------------------------
'''

from enum import Enum

class Label(Enum):
    '''
    -------------------------
    Label enum class to map dataset labels into integer values.
    -------------------------
    '''
    SUPPORTS = 2
    NEI      = 1
    REFUTES  = 0

def map_label(label: str, include_nei=True) -> int:
    '''
    Map a dataset label into its integer equivalence.

        Parameters:
        -------------------------
        label : str
            Label name to be mapped.
        include_nei : boolean, default=True
            Boolean indication to include NEI label.

        Returns:
        -------------------------
        res_label_int : int
            Input label mapped into an integer equivalence.
    '''
    label_map = {
        'SUPPORT': Label.SUPPORTS,
        'NOT_ENOUGH_INFO':  Label.NEI,
        'CONTRADICT': Label.REFUTES
    }
    res_label_int = label_map[label]
    if (not include_nei) and (res_label_int is Label.NEI):
        raise ValueError('NEI was provided in a setting where it is not allowed.')
    return res_label_int
