import tempfile
import unittest
from pathlib import Path

from gru_ddos_detection.persistence import raw_snapshot
from gru_ddos_detection.source_files import find_source_csvs


class SourceFilesTest(unittest.TestCase):
    def test_day_selection_excludes_nested_csvs_and_orders_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "01-12/TFTP.csv",
                "01-12/Syn.csv",
                "01-12/DrDoS_DNS.csv",
                "01-12/Data_Augmentation/Samples/DrDoS_DNS_data_augmented.csv",
                "01-12/Dataset_Description/preprocessing_summary.csv",
                "01-12/Feature_Analysis/PCA/PCA_Results.csv",
                "01-12/Stacking/Cache_Results/cache.csv",
                "03-11/Portmap.csv",
                "03-11/MSSQL.csv",
                "03-11/LDAP.csv",
                "03-11/some_nested_directory/unrelated.csv",
                "unrelated.csv",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Label\n", encoding="utf-8")

            first_day = [path.relative_to(root).as_posix() for path in find_source_csvs(root, "01-12")]
            second_day = [path.relative_to(root).as_posix() for path in find_source_csvs(root, "03-11")]
            both = [path.relative_to(root).as_posix() for path in find_source_csvs(root, "both")]
            self.assertEqual(first_day, ["01-12/DrDoS_DNS.csv", "01-12/Syn.csv", "01-12/TFTP.csv"])
            self.assertEqual(second_day, ["03-11/LDAP.csv", "03-11/MSSQL.csv", "03-11/Portmap.csv"])
            self.assertEqual(both, first_day + second_day)
            self.assertEqual(list(raw_snapshot(root, "01-12")), first_day)
            self.assertEqual(list(raw_snapshot(root, "both")), both)
            self.assertEqual(raw_snapshot(root, "missing-day"), {})


if __name__ == "__main__":
    unittest.main()
