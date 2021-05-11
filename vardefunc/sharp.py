"""Sharpening functions"""
import math
import vapoursynth as vs
core = vs.core


def z4USM(clip: vs.VideoNode, radius: int = 1, strength: float = 100.0) -> vs.VideoNode:
    """Zastin unsharp mask.

    Args:
        clip (vs.VideoNode): Source clip.
        radius (int, optional): Radius setting, it could be 1 or 2. Defaults to 1.
        strength (float, optional): Sharpening strength. Defaults to 100.0.

    Returns:
        vs.VideoNode: Sharpened clip.
    """
    if radius not in (1, 2):
        raise vs.Error('z4USM: "radius" must be 1 or 2')

    strength = max(1e-6, min(math.log2(3) * strength / 100, math.log2(3)))
    weight = 0.5 ** strength / ((1 - 0.5 ** strength) / 2)

    if clip.format.sample_type == 0:
        all_matrices = [[x] for x in range(1, 1024)]
        for x in range(1023):
            while len(all_matrices[x]) < radius * 2 + 1:
                all_matrices[x].append(all_matrices[x][-1] / weight)
        error = [sum([abs(x - round(x)) for x in matrix[1:]]) for matrix in all_matrices]
        matrix = [round(x) for x in all_matrices[error.index(min(error))]]
    else:
        matrix = [1]
        while len(matrix) < radius * 2 + 1:
            matrix.append(matrix[-1] / weight)

    matrix = [
        matrix[x] for x in [(2, 1, 2, 1, 0, 1, 2, 1, 2),
                            (4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4)]
                           [radius - 1]]

    return clip.std.MergeDiff(clip.std.MakeDiff(clip.std.Convolution(matrix)))
