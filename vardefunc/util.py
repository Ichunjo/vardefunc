"""Helper functions for the main functions in the script."""
from string import ascii_lowercase
import vapoursynth as vs

core = vs.core


def get_sample_type(clip: vs.VideoNode) -> vs.SampleType:
    """Returns the sample type of a VideoNode as an SampleType."""
    return clip.format.sample_type


def load_operators_expr() -> List[str]:
    """Returns clip loads operators for std.Expr as a list of string."""
    abcd = list(ascii_lowercase)
    return abcd[-3:] + abcd[:-3]
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
def copy_docstring_from(source: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator intended to copy the docstring from an other function

    Args:
        source (Callable[..., Any]): Source function.

    Returns:
        Callable[..., Any]: Function decorated
    """
    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        func.__doc__ = source.__doc__
        return func
    return wrapper
