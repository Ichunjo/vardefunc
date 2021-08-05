import vapoursynth as vs
from vardefunc.ocr import OCR
from vsutil import get_y

core = vs.core

# Import your clip
SOURCE = core.std.BlankClip(format=vs.YUV410P8)


def ocring() -> None:
    clip = SOURCE

    ocr = OCR(get_y(clip), (1900, 125, 70), coord_alt=(1500, 125, 70))
    ocr.preview_cropped.set_output(0)
    ocr.preview_cleaned.set_output(1)

    ocr.launch(datapath=r'C:\Users\Varde\AppData\Roaming\VapourSynth\plugins64\tessdata', language='fra+eng')
    ocr.write_ass(
        'output.ass',
        [('_', '-'), ('…', '...'), ('‘', "'"), ('’', "'"), (" '", "'"),
         ('—', '-'), ('- ', '– '), ('0u', 'Ou'), ('Gomme', 'Comme'), ('A ', 'À '),
         ('II', 'Il'), ('ees', 'ces'), ('@', 'O'), ('oe', 'œ'), ('téte', 'tête')]
    )


if __name__ == '__main__':
    ocring()
else:
    ocring()
