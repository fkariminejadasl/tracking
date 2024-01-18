import shutil
from pathlib import Path

import motmetrics as mm
import numpy as np


def mot_txt_to_zip(txt_file: Path):
    """
    Convert a .txt file to a .zip archive containing a modified copy of the .txt file.

    This function creates a directory named 'gt' in the same directory as the input file,
    writes a predefined string into a 'labels.txt' file in this directory, copies the
    original .txt file into this directory, and then zips the directory. The original
    'gt' directory is removed after zipping.

    Parameters
    ----------
    txt_file : Path
        The path of the .txt file to be converted to a .zip archive.

    Examples
    --------
    >>> p = Path("/home/user/data/file.txt")
    >>> mot_txt_to_zip(p)
    """
    track_path = txt_file.parent / "gt"
    track_path.mkdir(parents=True, exist_ok=True)

    with open(track_path / "labels.txt", "w") as wf:
        wf.write("fish")

    track_file = track_path / "gt.txt"
    shutil.copy(txt_file, track_file)

    shutil.make_archive(txt_file.with_suffix(""), "zip", txt_file.parent, "gt")
    shutil.rmtree(track_path)


def mot_zip_to_txt(zip_file: Path):
    """
    Extract a .zip archive and retrieve a specific .txt file from it.

    This function extracts a .zip file into a directory with the same name (without
    the .zip extension), retrieves a .txt file located in a specific subdirectory
    inside it ('gt/gt.txt'), copies this .txt file to the parent directory of the
    .zip file with a name matching the .zip file, and then removes the extracted
    directory.

    Parameters
    ----------
    zip_file : Path
        The path of the .zip archive to be converted back to a .txt file.

    Examples
    --------
    >>> p = Path("/home/user/data/file.zip")
    >>> mot_zip_to_txt(p)
    """
    shutil.unpack_archive(zip_file, zip_file.parent / zip_file.stem, "zip")
    track_file = zip_file.parent / zip_file.stem / "gt/gt.txt"
    txt_file = zip_file.parent / f"{zip_file.stem}.txt"
    shutil.copy(track_file, txt_file)
    shutil.rmtree(zip_file.parent / zip_file.stem)


def mot_to_challenge(mot_file):
    """
    Parameters
    ----------
    mot_file: Path
        The path of the .txt mot file
    """
    pass


def transform_to_mot(input_file: str, output_file: str):
    """
    Transform each line of a text file according to specific rules.

    The transformation rules are as follows:
    1. Swap the first and second columns and add 1 to their values.
    2. Ignore the third column.
    3. Take the fourth and fifth columns, add 1 to their values.
    4. Ignore columns six through nine.
    5. Use columns ten through twelve as they are.
    6. Ignore any remaining columns.
    7. Append '1, 1.0' to the end of each line.

    Parameters
    ----------
    input_file : str
        The path to the input text file.
    output_file : str
        The path where the transformed output text file will be saved.

    Examples
    --------
    >>> input_path = 'path/to/input.txt'
    >>> output_path = 'path/to/output.txt'
    >>> transform_to_mot(input_path, output_path)
    """

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.strip().split(",")

            # Applying the specified transformations
            transformed = [
                str(int(parts[1]) + 1),  # Swap and increment first column
                str(int(parts[0]) + 1),  # Swap and increment second column
                str(int(parts[3]) + 1),  # Increment fourth column
                str(int(parts[4]) + 1),  # Increment fifth column
                parts[9],  # Use tenth column
                parts[10],  # Use eleventh column
                parts[11],  # Use twelfth column
                "1",
                "1.0",  # Append '1, 1.0'
            ]

            modified_line = ",".join(transformed) + "\n"
            outfile.write(modified_line)


def mot_to_challenge(mot_file, output_file):
    """
    Modify a text file by removing the last three columns of each line and
    replacing them with four columns, each containing -1.

    N.B.
    The mot file format is from MOT20: https://arxiv.org/pdf/2003.09003.pdf
        frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

    The output file is from mot challenge:
    https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md
        frame, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z

    mot_file.txt

    Parameters
    ----------
    mot_file : str
        The path to the mot .txt file
    output_file : str
        The path where the modified output text file will be saved.

    Examples
    --------
    >>> input_file = 'path/to/input.txt'
    >>> output_file = 'path/to/output.txt'
    >>> mot_to_challenge(input_file, output_file)
    """

    with open(mot_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.strip().split(",")
            if int(parts[-3]) != 1:
                modified_line = ",".join(parts[:-2] + ["-1", "-1", "-1"]) + "\n"
            else:
                modified_line = ",".join(parts[:-3] + ["-1", "-1", "-1", "-1"]) + "\n"
            outfile.write(modified_line)


def motMetricsEnhancedCalculator(gtSource, tSource):
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=",")

    # load tracking output
    t = np.loadtxt(tSource, delimiter=",")

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:, 0] == frame, 1:6]  # select all detections in gt
        t_dets = t[t[:, 0] == frame, 1:6]  # select all detections in t

        C = mm.distances.iou_matrix(
            gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5
        )  # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(
            gt_dets[:, 0].astype("int").tolist(),
            t_dets[:, 0].astype("int").tolist(),
            C,
        )

    mh = mm.metrics.create()

    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "idf1",
            "idp",
            "idr",
            "recall",
            "precision",
            "num_objects",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
            "mota",
            "motp",
        ],
        name="acc",
    )

    strsummary = mm.io.render_summary(
        summary,
        # formatters={'mota' : '{:.2%}'.format},
        namemap={
            "idf1": "IDF1",
            "idp": "IDP",
            "idr": "IDR",
            "recall": "Rcll",
            "precision": "Prcn",
            "num_objects": "GT",
            "mostly_tracked": "MT",
            "partially_tracked": "PT",
            "mostly_lost": "ML",
            "num_false_positives": "FP",
            "num_misses": "FN",
            "num_switches": "IDsw",
            "num_fragmentations": "FM",
            "mota": "MOTA",
            "motp": "MOTP",
        },
    )
    print(strsummary)


gt_file = "/home/fatemeh/Downloads/fish/mot_data/challenge/gt/8_1.txt"
pr_file = "/home/fatemeh/Downloads/fish/mot_data/challenge/pred/8_1_ms_exp1.txt"
motMetricsEnhancedCalculator(gt_file, pr_file)
motMetricsEnhancedCalculator(gt_file, gt_file)
print("")
main_path = Path("/home/fatemeh/Downloads/fish/mot_data")
for p in main_path.glob("mots/*.txt"):
    mot_to_challenge(p, main_path / f"challenge/gt/{p.name}")
