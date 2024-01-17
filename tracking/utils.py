# import shutil
# from pathlib import Path

# def mot_txt_to_zip(txt_file: Path):
#     track_path = txt_file.parent / "gt"
#     track_path.mkdir(parents=True, exist_ok=True)

#     with open(track_path / "labels.txt", "w") as wf:
#         wf.write("fish")

#     track_file = track_path / "gt.txt"
#     shutil.copy(txt_file, track_file)

#     shutil.make_archive(txt_file.with_suffix(""), "zip", txt_file.parent, "gt")
#     shutil.rmtree(track_path)


# def mot_zip_to_txt(zip_file: Path):
#     shutil.unpack_archive(zip_file, zip_file.parent / zip_file.stem, "zip")
#     track_file = zip_file.parent / zip_file.stem / "gt/gt.txt"
#     txt_file = zip_file.parent/ f"{zip_file.stem}.txt"
#     shutil.copy(track_file, txt_file)
#     shutil.rmtree(zip_file.parent  / zip_file.stem)

import shutil
from pathlib import Path


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
