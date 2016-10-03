# Copyright 2016 MongoDB, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Elastic2 DocManager."""
import base64
import sys
import time

sys.path[0:0] = [""]

from mongo_connector.command_helper import CommandHelper
from mongo_connector.doc_managers.elastic2_doc_manager import DocManager
from mongo_connector.test_utils import MockGridFSFile, TESTARGS

from tests import unittest, elastic_pair
from tests.test_elastic2 import ElasticsearchTestCase


class TestElasticDocManager(ElasticsearchTestCase):
    """Unit tests for the Elastic DocManager."""

    def test_update(self):
        """Test the update method."""
        doc_id = 1
        doc = {"_id": doc_id, "a": 1, "b": 2}
        self.elastic_doc.upsert(doc, *TESTARGS)
        # $set only
        update_spec = {"$set": {"a": 1, "b": 2}}
        doc = self.elastic_doc.update(doc_id, update_spec, *TESTARGS)
        self.assertEqual(doc, {"_id": '1', "a": 1, "b": 2})
        # $unset only
        update_spec = {"$unset": {"a": True}}
        doc = self.elastic_doc.update(doc_id, update_spec, *TESTARGS)
        self.assertEqual(doc, {"_id": '1', "b": 2})
        # mixed $set/$unset
        update_spec = {"$unset": {"b": True}, "$set": {"c": 3}}
        doc = self.elastic_doc.update(doc_id, update_spec, *TESTARGS)
        self.assertEqual(doc, {"_id": '1', "c": 3})

    def test_update_child_doc(self):
        """Test update child document."""
        doc_id = 1
        doc = {"_id": doc_id, "a": 1, "b": 2, "parent_id": 3}
        self.elastic_doc.upsert(doc, *self.PARENT_CHILD_TEST_ARGS)
        # $set only
        update_spec = {"$set": {"a": 1, "b": 2}}
        doc = self.elastic_doc.update(doc_id, update_spec, *self.PARENT_CHILD_TEST_ARGS)
        self.assertEqual(doc, {"_id": '1', "a": 1, "b": 2})
        # $unset only
        update_spec = {"$unset": {"a": True}}
        doc = self.elastic_doc.update(doc_id, update_spec, *self.PARENT_CHILD_TEST_ARGS)
        self.assertEqual(doc, {"_id": '1', "b": 2})
        # mixed $set/$unset
        update_spec = {"$unset": {"b": True}, "$set": {"c": 3}}
        doc = self.elastic_doc.update(doc_id, update_spec, *self.PARENT_CHILD_TEST_ARGS)
        self.assertEqual(doc, {"_id": '1', "c": 3})

    def test_upsert(self):
        """Test the upsert method."""
        docc = {'_id': '1', 'name': 'John'}
        self.elastic_doc.upsert(docc, *TESTARGS)
        res = self.elastic_conn.search(
            index="test", doc_type='test',
            body={"query": {"match_all": {}}}
        )["hits"]["hits"]
        for doc in res:
            self.assertEqual(doc['_id'], '1')
            self.assertEqual(doc['_source']['name'], 'John')

    def test_upsert_with_parent_id(self):
        """Test the upsert method with parent_id provided."""
        docc = {'_id': '1', 'name': 'John', 'parent_id': '2'}
        self.elastic_doc.upsert(docc, *self.PARENT_CHILD_TEST_ARGS)
        for doc in self._search(doc_type=self.PARENT_CHILD_TEST_TYPE):
            self.assertEqual(doc['_id'], '1')
            self.assertEqual(doc['name'], 'John')
            self.assertEqual(doc['_parent'], '2')
            self.assertNotIn('parent_id', doc)

    def test_bulk_upsert(self):
        """Test the bulk_upsert method."""
        self.elastic_doc.bulk_upsert([], *TESTARGS)

        docs = ({"_id": i} for i in range(1000))
        self.elastic_doc.bulk_upsert(docs, *TESTARGS)
        self.elastic_doc.commit()
        returned_ids = sorted(int(doc["_id"]) for doc in self._search())
        self.assertEqual(self._count(), 1000)
        self.assertEqual(len(returned_ids), 1000)
        for i, r in enumerate(returned_ids):
            self.assertEqual(r, i)

        docs = ({"_id": i, "weight": 2*i} for i in range(1000))
        self.elastic_doc.bulk_upsert(docs, *TESTARGS)

        returned_ids = sorted(
            int(doc["weight"]) for doc in self._search())
        self.assertEqual(len(returned_ids), 1000)
        for i, r in enumerate(returned_ids):
            self.assertEqual(r, 2*i)

    def test_bulk_upsert_with_parent_id(self):
        """Test the bulk_upsert method with parent_id provided."""
        self.elastic_doc.bulk_upsert([], *TESTARGS)

        docs = ({"_id": i, "parent_id": i * 2} for i in range(1000))
        self.elastic_doc.bulk_upsert(docs, *self.PARENT_CHILD_TEST_ARGS)
        self.elastic_doc.commit()
        returned_docs = sorted([(int(doc["_id"]), int(doc["_parent"]))
                                for doc in self._search(doc_type=self.PARENT_CHILD_TEST_TYPE)],
                                key = lambda x: x[0])
        self.assertEqual(self._count(), 1000)
        self.assertEqual(len(returned_docs), 1000)
        for i, r in enumerate(returned_docs):
            self.assertEqual(r[0], i)
            self.assertEqual(r[1], i * 2)

    def test_remove(self):
        """Test the remove method."""
        docc = {'_id': '1', 'name': 'John'}
        self.elastic_doc.upsert(docc, *TESTARGS)
        res = self.elastic_conn.search(
            index="test", doc_type='test',
            body={"query": {"match_all": {}}}
        )["hits"]["hits"]
        res = [x["_source"] for x in res]
        self.assertEqual(len(res), 1)

        self.elastic_doc.remove(docc['_id'], *TESTARGS)
        res = self.elastic_conn.search(
            index="test", doc_type='test',
            body={"query": {"match_all": {}}}
        )["hits"]["hits"]
        res = [x["_source"] for x in res]
        self.assertEqual(len(res), 0)

    def test_remove_child_doc(self):
        """Test remove child document."""
        docc = {'_id': '1', 'name': 'John', 'parent_id': '2'}
        self.elastic_doc.upsert(docc, *self.PARENT_CHILD_TEST_ARGS)
        res = [doc for doc in self._search(doc_type=self.PARENT_CHILD_TEST_TYPE)]
        self.assertEqual(len(res), 1)

        self.elastic_doc.remove(docc['_id'], *self.PARENT_CHILD_TEST_ARGS)
        res = [doc for doc in self._search(doc_type=self.PARENT_CHILD_TEST_TYPE)]
        self.assertEqual(len(res), 0)

    def test_insert_file(self):
        """Ensure we can properly insert a file into ElasticSearch
        """
        test_data = ' '.join(str(x) for x in range(100000)).encode('utf8')
        docc = {
            '_id': 'test_id',
            'filename': 'test_filename',
            'upload_date': 5,
            'md5': 'test_md5'
        }
        self.elastic_doc.insert_file(
            MockGridFSFile(docc, test_data), *TESTARGS)
        res = self._search()
        for doc in res:
            self.assertEqual(doc['_id'], docc['_id'])
            self.assertEqual(doc['filename'], docc['filename'])
            self.assertEqual(base64.b64decode(doc['content']),
                             test_data.strip())

    def test_remove_file(self):
        test_data = b'hello world'
        docc = {
            '_id': 'test_id',
            '_ts': 10,
            'ns': 'test.test',
            'filename': 'test_filename',
            'upload_date': 5,
            'md5': 'test_md5'
        }

        self.elastic_doc.insert_file(
            MockGridFSFile(docc, test_data), *TESTARGS)
        res = list(self._search())
        self.assertEqual(len(res), 1)

        self.elastic_doc.remove('test_id', *TESTARGS)
        res = list(self._search())
        self.assertEqual(len(res), 0)

    def test_search(self):
        """Test the search method.

        Make sure we can retrieve documents last modified within a time range.
        """
        docc = {'_id': '1', 'name': 'John'}
        self.elastic_doc.upsert(docc, 'test.test', 5767301236327972865)
        docc2 = {'_id': '2', 'name': 'John Paul'}
        self.elastic_doc.upsert(docc2, 'test.test', 5767301236327972866)
        docc3 = {'_id': '3', 'name': 'Paul'}
        self.elastic_doc.upsert(docc3, 'test.test', 5767301236327972870)
        search = list(self.elastic_doc.search(5767301236327972865,
                                              5767301236327972866))
        self.assertEqual(len(search), 2)
        result_ids = [result.get("_id") for result in search]
        self.assertIn('1', result_ids)
        self.assertIn('2', result_ids)

    def test_elastic_commit(self):
        """Test the auto_commit_interval attribute."""
        docc = {'_id': '3', 'name': 'Waldo'}
        docman = DocManager(elastic_pair)
        # test cases:
        # -1 = no autocommit
        # 0 = commit immediately
        # x > 0 = commit within x seconds
        for autocommit_interval in [None, 0, 1, 2]:
            docman.auto_commit_interval = autocommit_interval
            docman.upsert(docc, *TESTARGS)
            if autocommit_interval is None:
                docman.commit()
            else:
                # Allow just a little extra time
                time.sleep(autocommit_interval + 1)
            results = list(self._search())
            self.assertEqual(len(results), 1,
                             "should commit document with "
                             "auto_commit_interval = %s" % str(
                                 autocommit_interval))
            self.assertEqual(results[0]["name"], "Waldo")
            self._remove()
        docman.stop()

    def test_get_last_doc(self):
        """Test the get_last_doc method.

        Make sure we can retrieve the document most recently modified from ES.
        """
        base = self.elastic_doc.get_last_doc()
        ts = base.get("_ts", 0) if base else 0
        docc = {'_id': '4', 'name': 'Hare'}
        self.elastic_doc.upsert(docc, 'test.test', ts + 3)
        docc = {'_id': '5', 'name': 'Tortoise'}
        self.elastic_doc.upsert(docc, 'test.test', ts + 2)
        docc = {'_id': '6', 'name': 'Mr T.'}
        self.elastic_doc.upsert(docc, 'test.test', ts + 1)

        self.assertEqual(
            self.elastic_doc.elastic.count(index="test")['count'], 3)
        doc = self.elastic_doc.get_last_doc()
        self.assertEqual(doc['_id'], '4')

        docc = {'_id': '6', 'name': 'HareTwin'}
        self.elastic_doc.upsert(docc, 'test.test', ts + 4)
        doc = self.elastic_doc.get_last_doc()
        self.assertEqual(doc['_id'], '6')
        self.assertEqual(
            self.elastic_doc.elastic.count(index="test")['count'], 3)

    def test_commands(self):
        cmd_args = ('test.$cmd', 1)
        self.elastic_doc.command_helper = CommandHelper()

        self.elastic_doc.handle_command({'create': 'test2'}, *cmd_args)
        time.sleep(1)
        self.assertIn('test2', self._mappings('test'))

        docs = [
            {"_id": 0, "name": "ted"},
            {"_id": 1, "name": "marsha"},
            {"_id": 2, "name": "nikolas"}
        ]
        self.elastic_doc.upsert(docs[0], 'test.test2', 1)
        self.elastic_doc.upsert(docs[1], 'test.test2', 1)
        self.elastic_doc.upsert(docs[2], 'test.test2', 1)
        res = list(self.elastic_doc._stream_search(
            index="test", doc_type='test2',
            body={"query": {"match_all": {}}}
        ))
        for d in docs:
            self.assertTrue(d in res)

        self.elastic_doc.handle_command({'drop': 'test2'}, *cmd_args)
        time.sleep(3)
        res = list(self.elastic_doc._stream_search(
            index="test", doc_type='test2',
            body={"query": {"match_all": {}}})
        )
        self.assertEqual(0, len(res))

        self.elastic_doc.handle_command({'create': 'test2'}, *cmd_args)
        self.elastic_doc.handle_command({'create': 'test3'}, *cmd_args)
        time.sleep(1)
        self.elastic_doc.handle_command({'dropDatabase': 1}, *cmd_args)
        time.sleep(1)
        self.assertNotIn('test', self._indices())
        self.assertNotIn('test2', self._mappings())
        self.assertNotIn('test3', self._mappings())


if __name__ == '__main__':
    unittest.main()
