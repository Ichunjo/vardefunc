"""Helper functions for the main functions in the script."""
from string import ascii_lowercase
import vapoursynth as vs

core = vs.core


def get_sample_type(clip: vs.VideoNode) -> vs.SampleType:
    """[summary]

    Args:
        clip (vs.VideoNode): [description]

    Returns:
        [type]: [description]
    """
    return clip.format.sample_type


def load_operators_expr() -> str:
    """[summary]

    Returns:
        str: [description]
    """
    abcd = list(ascii_lowercase)
    return abcd[-3:] + abcd[:-3]


def max_expr(n: int) -> str:
    """Dynamic max string to be integrated in std.Expr.

    Args:
        n (int): Number of elements.

    Returns:
        str: Expression.
    """
    return 'x y max ' + ' max '.join(
        load_operators_expr()[i] for i in range(2, n)
    ) + ' max'
