import unittest

from fedp0.contracts import contract_symbol, parse_contract


class ContractTest(unittest.TestCase):
    def test_symbols(self):
        self.assertEqual(contract_symbol(2026, 12), "ZQZ26")
        self.assertEqual(contract_symbol(2027, 1), "ZQF27")
        self.assertEqual(contract_symbol(2027, 3), "ZQH27")
        self.assertEqual(contract_symbol(2027, 4), "ZQJ27")
        self.assertEqual(contract_symbol(2027, 5), "ZQK27")

    def test_roundtrip_all_months(self):
        for month in range(1, 13):
            symbol = contract_symbol(2031, month)
            self.assertEqual(parse_contract(symbol), (2031, month))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            contract_symbol(2026, 13)
        with self.assertRaises(ValueError):
            contract_symbol(1999, 1)
        with self.assertRaises(ValueError):
            parse_contract("ESZ26")
        with self.assertRaises(ValueError):
            parse_contract("ZQZ")


if __name__ == "__main__":
    unittest.main()
