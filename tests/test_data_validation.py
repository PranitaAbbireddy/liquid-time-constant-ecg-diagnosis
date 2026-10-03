from src.data.validation import validate_ecg_csv
from src.utils.paths import resolve_path


def test_mitbih_train_valid():
    issues = validate_ecg_csv(resolve_path("mitbih/mitbih_train.csv"))
    assert issues == []
