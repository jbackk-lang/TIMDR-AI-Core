import io
import zipfile

import numpy as np

from activation_probe_uci_har import adjacent_pairs, load_train_only, split_subjects


def test_adjacent_pairs_stay_within_subject_and_activity():
    subjects = np.array([1, 1, 1, 2, 2, 2])
    labels = np.array([1, 1, 2, 2, 2, 3])
    assert adjacent_pairs(subjects, labels, {1, 2}).tolist() == [1, 4]
    assert adjacent_pairs(subjects, labels, {2}).tolist() == [4]


def test_subject_split_is_deterministic_and_disjoint():
    subjects = np.repeat(np.arange(1, 22), 2)
    fit, cal = split_subjects(subjects)
    assert len(cal) == 5
    assert fit.isdisjoint(cal)
    assert fit | cal == set(range(1, 22))
    assert split_subjects(subjects) == (fit, cal)


def test_loader_only_opens_train_members(tmp_path, monkeypatch):
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        X = np.zeros((6, 561))
        x_text = "\n".join(" ".join(map(str, row)) for row in X)
        inner.writestr("UCI HAR Dataset/train/X_train.txt", x_text)
        inner.writestr("UCI HAR Dataset/train/y_train.txt", "1\n2\n3\n4\n5\n6\n")
        inner.writestr("UCI HAR Dataset/train/subject_train.txt", "1\n1\n1\n2\n2\n2\n")
        inner.writestr("UCI HAR Dataset/test/X_test.txt", "DO NOT OPEN")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as outer:
        outer.writestr("UCI HAR Dataset.zip", inner_buffer.getvalue())

    opened = []
    original_open = zipfile.ZipFile.open

    def guarded_open(self, name, *args, **kwargs):
        opened.append(name)
        assert "test/" not in str(name).lower()
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded_open)
    X, y, subject = load_train_only(archive)
    assert X.shape == (6, 561)
    assert y.tolist() == [1, 2, 3, 4, 5, 6]
    assert subject.tolist() == [1, 1, 1, 2, 2, 2]
    assert len(opened) == 4  # nested archive + three train members
