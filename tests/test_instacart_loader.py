"""Gates for the Instacart frequency loader, on tiny hand-made files."""

import zipfile

from src.demand.orders import (
    FREQUENCY_CACHE,
    build_demand_model,
    demand_skew_summary,
    load_instacart_frequencies,
    synthetic_frequencies,
)

HEADER = "order_id,product_id,add_to_cart_order,reordered\n"
# product 7 x3, product 2 x2, product 9 x1
ROWS = ["1,7,1,0", "1,2,2,0", "2,7,1,1", "3,7,1,1", "3,2,2,1", "4,9,1,0"]


def _write_raw(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "order_products__prior.csv").write_text(HEADER + "\n".join(ROWS) + "\n")


def test_missing_data_returns_none(tmp_path):
    assert load_instacart_frequencies(str(tmp_path / "nothing")) is None


def test_counts_become_a_normalised_distribution(tmp_path):
    _write_raw(tmp_path)
    f = load_instacart_frequencies(str(tmp_path))
    assert f == {7: 3 / 6, 2: 2 / 6, 9: 1 / 6}


def test_top_n_keeps_the_most_ordered(tmp_path):
    _write_raw(tmp_path)
    f = load_instacart_frequencies(str(tmp_path), top_n=2)
    assert set(f) == {7, 2}
    assert abs(sum(f.values()) - 1.0) < 1e-12


def test_summary_file_is_written_and_used_without_the_raw_file(tmp_path):
    _write_raw(tmp_path)
    load_instacart_frequencies(str(tmp_path))
    cache = tmp_path / FREQUENCY_CACHE
    assert cache.exists()
    (tmp_path / "order_products__prior.csv").unlink()     # raw gone ...
    other = tmp_path.parent / (tmp_path.name + "_copy")    # ... fresh process key
    other.mkdir()
    (other / FREQUENCY_CACHE).write_text(cache.read_text())
    assert load_instacart_frequencies(str(other)) == {7: 3 / 6, 2: 2 / 6, 9: 1 / 6}


def test_reads_kaggles_zip_directly(tmp_path):
    with zipfile.ZipFile(tmp_path / "order_products__prior.csv.zip", "w") as z:
        z.writestr("order_products__prior.csv", HEADER + "\n".join(ROWS) + "\n")
    assert load_instacart_frequencies(str(tmp_path)) == {7: 3 / 6, 2: 2 / 6, 9: 1 / 6}


def test_reads_kaggles_whole_dataset_archive(tmp_path):
    """The dataset page ships one archive.zip with every table in it."""
    with zipfile.ZipFile(tmp_path / "archive.zip", "w") as z:
        z.writestr("products.csv", "product_id,product_name\n7,Banana\n")
        z.writestr("order_products__prior.csv", HEADER + "\n".join(ROWS) + "\n")
    (tmp_path / "notes.zip").write_text("not a zip file")
    with zipfile.ZipFile(tmp_path / "other.zip", "w") as z:
        z.writestr("readme.txt", "nothing useful")
    assert load_instacart_frequencies(str(tmp_path)) == {7: 3 / 6, 2: 2 / 6, 9: 1 / 6}


def test_demand_model_records_instacart_as_its_source(tmp_path):
    _write_raw(tmp_path)
    shelves = [(x, 0) for x in range(5)]
    m = build_demand_model(shelves, [(0, 1)], skew="real", data_dir=str(tmp_path))
    assert m.source == "instacart"
    assert abs(sum(m.shelf_weights.values()) - 1.0) < 1e-9


def test_skew_summary_recovers_a_known_zipf_exponent():
    s = demand_skew_summary(synthetic_frequencies(n_products=1000, exponent=1.0))
    assert abs(s["zipf_exponent"] - 1.0) < 1e-6
    flat = demand_skew_summary({i: 1.0 for i in range(100)})
    assert abs(flat["share_top_20pct"] - 0.2) < 1e-9
    assert abs(flat["zipf_exponent"]) < 1e-9
