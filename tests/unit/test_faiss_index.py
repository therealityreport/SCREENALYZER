import numpy as np

from screenalyzer.reid.faiss_index import FaceIndex


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def test_add_and_search():
    index = FaceIndex(metric="ip", normalize=True)
    index.add("A", _vec(1.0, 0.0, 0.0))
    index.add("B", _vec(0.0, 1.0, 0.0))

    query = _vec(0.9, 0.1, 0.0)
    results = index.search(query, k=1)

    assert results
    label, score = results[0]
    assert label == "A"
    assert score > 0.8


def test_update_centroid_and_sigma():
    index = FaceIndex(metric="ip", normalize=True)
    index.add("A", _vec(1.0, 0.0, 0.0))
    index.update_centroid("A", _vec(0.8, 0.2, 0.0), alpha=0.5)
    index.update_centroid("A", _vec(0.7, 0.3, 0.0), alpha=0.5)

    stats = index.get_label_stats("A")
    assert stats is not None
    assert stats.count >= 2
    assert stats.sigma >= 0.0

    centroid = index.centroids["A"]
    assert centroid.shape == (3,)
    assert centroid[0] > centroid[1]


def test_persistence_round_trip(tmp_path):
    index = FaceIndex(metric="ip", normalize=True)
    index.add("A", _vec(1.0, 0.0, 0.0))
    index.add("B", _vec(0.0, 1.0, 0.0))
    index.update_centroid("A", _vec(0.9, 0.1, 0.0), alpha=0.3)

    faiss_path = tmp_path / "faces.faiss"
    centroids_path = tmp_path / "centroids.json"
    index.save(str(faiss_path), str(centroids_path))

    restored = FaceIndex(metric="ip", normalize=True)
    restored.load(str(faiss_path), str(centroids_path))

    results = restored.search(_vec(0.95, 0.05, 0.0), k=1)
    assert results[0][0] == "A"

    assert sorted(index.labels) == sorted(restored.labels)
