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
        result = proccesor.get_course_product_id("Success_Factors_S4Hana")
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
        title = "Sample Title  | SAP Help Portal"
        result = proccesor.fix_title(title)
        self.assertEqual(result, "Sample Title")

    def test_find_document_id(self):
        proccesor = SAPProccesor()
        args = {"docid": "121", "id": "0", "document_id": "-1"}
        result = proccesor.find_document_id(args)
        self.assertEqual(result, "121")

        args = {"id": "0", "document_id": "-1"}
        result = proccesor.find_document_id(args)
        self.assertEqual(result, "0")

        args = {"document_id": "-1"}
        result = proccesor.find_document_id(args)
        self.assertEqual(result, "-1")

        args = {}
        result = proccesor.find_document_id(args)
        self.assertEqual(result, "")

if __name__ == "__main__":
    unittest.main()