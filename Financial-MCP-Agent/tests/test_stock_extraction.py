import unittest

from src.utils.stock_extraction import clean_company_name, rule_extract_stock_info


class CompanyNameCleaningTests(unittest.TestCase):
    def test_removes_colloquial_look_prefix(self):
        self.assertEqual(clean_company_name("看宇树科技"), "宇树科技")
        self.assertEqual(clean_company_name("看看阿里巴巴"), "阿里巴巴")

    def test_rule_extraction_does_not_treat_look_as_company_name(self):
        company_name, stock_code = rule_extract_stock_info("看宇树科技这只股票")

        self.assertEqual(company_name, "宇树科技")
        self.assertIsNone(stock_code)


if __name__ == "__main__":
    unittest.main()
