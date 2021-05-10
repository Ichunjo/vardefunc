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

def mae_expr(gray_only: bool = True) -> str:
    """Mean Absolute Error string to be integrated in std.Expr.

    Args:
        gray_only (bool, optional):
            If both actual observation and prediction are one plane each.
            Defaults to True.

    Returns:
        str: Expression.
    """
    return 'x y - abs' if gray_only else 'x a - abs y b - abs max z c - abs max'


def rmse_expr(gray_only: bool = True) -> str:
    """Root Mean Squared Error string to be integrated in std.Expr.

    Args:
        gray_only (bool, optional):
            If both actual observation and prediction are one plane each.
            Defaults to True.

    Returns:
        str: Expression.
    """
    return 'x y - dup * sqrt' if gray_only else 'x a - dup * sqrt y b - dup * sqrt max z c - dup * sqrt max'


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
