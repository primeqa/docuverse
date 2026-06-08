import unittest
from docuverse.engines.search_data import SAPProccesor


class TestSAPProccesor(unittest.TestCase):

    def test_process_product_id(self):
        proccesor = SAPProccesor()
        result = proccesor.process_product_id(["field1", "field2", "field3"], "Uniform Product", "sap")
        self.assertEqual(result, "Uniform Product")

        result = proccesor.process_product_id([], "Uniform Product", "sap")
        self.assertEqual(result, "Uniform Product")

        result = proccesor.process_product_id(["field1", "field2", "field3"], "Uniform Product", "other")
        self.assertEqual(result, "")

    def test_get_course_product_id(self):
        proccesor = SAPProccesor()
        # Uppercase substrings — get_course_product_id matches case-sensitively.
        result = proccesor.get_course_product_id("SUCCESS_FACTORS_S4HANA")
        self.assertEqual(result, "S4")

    def test_process_url(self):
        proccesor = SAPProccesor()
        doc_url = "https://example.com/some_document.html?locale=en-US"
        data_type = "sap"
        result = proccesor.process_url(doc_url, data_type)
        self.assertEqual(result, ("https://example.com/some_document",
                                  ["https:", "", "example.com", "some_document"]))

        data_type = "other"
        result = proccesor.process_url(doc_url, data_type)
        self.assertEqual(result, ("", ["", "", "", "", "", ""]))

    def test_fix_title(self):
        proccesor = SAPProccesor()
        # Single space before the pipe — fix_title strips the SAP suffix
        # then collapses any remaining double spaces.
        title = "Sample Title | SAP Help Portal"
        result = proccesor.fix_title(title)
        self.assertEqual(result, "Sample Title")

    def test_find_document_id(self):
        proccesor = SAPProccesor()
        # find_document_id walks ['document_id', 'docid', 'id'] in order and
        # returns the first match — document_id is the most specific key.
        args = {"docid": "121", "id": "0", "document_id": "-1"}
        self.assertEqual(proccesor.find_document_id(args), "-1")

        args = {"id": "0", "document_id": "-1"}
        self.assertEqual(proccesor.find_document_id(args), "-1")

        args = {"docid": "121", "id": "0"}
        self.assertEqual(proccesor.find_document_id(args), "121")

        args = {"id": "0"}
        self.assertEqual(proccesor.find_document_id(args), "0")

        args = {}
        self.assertEqual(proccesor.find_document_id(args), "")


if __name__ == "__main__":
    unittest.main()
