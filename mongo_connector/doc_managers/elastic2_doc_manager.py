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

"""Elasticsearch implementation of the DocManager interface.

Receives documents from an OplogThread and takes the appropriate actions on
Elasticsearch.
"""
import base64
import logging
import threading
import time
import warnings

import bson.json_util

try:
    import elasticsearch
except ImportError:
    raise ImportError(
        "Error: elasticsearch (https://pypi.python.org/pypi/elasticsearch) "
        "version 2.x or 5.x is not installed.\n"
        "Install with:\n"
        "  pip install elastic-doc-manager[elastic2]\n"
        "or:\n"
        "  pip install elastic-doc-manager[elastic5]\n"
    )

from elasticsearch import Elasticsearch, exceptions as es_exceptions, connection as es_connection
from elasticsearch.helpers import bulk, scan, streaming_bulk, BulkIndexError

from mongo_connector import errors
from mongo_connector.compat import u
from mongo_connector.constants import (DEFAULT_COMMIT_INTERVAL,
                                       DEFAULT_MAX_BULK)
from mongo_connector.util import exception_wrapper, retry_until_ok
from mongo_connector.doc_managers.doc_manager_base import DocManagerBase
from mongo_connector.doc_managers.formatters import DefaultDocumentFormatter

_HAS_AWS = True
try:
    from boto3 import session
    from requests_aws_sign import AWSV4Sign
except ImportError:
    _HAS_AWS = False

wrap_exceptions = exception_wrapper({
    BulkIndexError: errors.OperationFailed,
    es_exceptions.ConnectionError: errors.ConnectionFailed,
    es_exceptions.TransportError: errors.OperationFailed,
    es_exceptions.NotFoundError: errors.OperationFailed,
    es_exceptions.RequestError: errors.OperationFailed})

LOG = logging.getLogger(__name__)

DEFAULT_SEND_INTERVAL = 5
"""The default interval in seconds to send buffered operations."""

DEFAULT_AWS_REGION = 'us-east-1'

__version__ = '0.3.0'
"""Elasticsearch 2.X DocManager version."""


def convert_aws_args(aws_args):
    """Convert old style options into arguments to boto3.session.Session."""
    if not isinstance(aws_args, dict):
        raise errors.InvalidConfiguration(
            'Elastic DocManager config option "aws" must be a dict')
    old_session_kwargs = dict(region='region_name',
                              access_id='aws_access_key_id',
                              secret_key='aws_secret_access_key')
    new_kwargs = {}
    for arg in aws_args:
        if arg in old_session_kwargs:
            new_kwargs[old_session_kwargs[arg]] = aws_args[arg]
        else:
            new_kwargs[arg] = aws_args[arg]
    return new_kwargs


def create_aws_auth(aws_args):
    try:
        aws_session = session.Session(**convert_aws_args(aws_args))
    except TypeError as exc:
        raise errors.InvalidConfiguration(
            'Elastic DocManager unknown aws config option: %s' % (exc,))
    return AWSV4Sign(aws_session.get_credentials(),
                     aws_session.region_name or DEFAULT_AWS_REGION,
                     'es')


class AutoCommiter(threading.Thread):
    """Thread that periodically sends buffered operations to Elastic.

    :Parameters:
      - `docman`: The Elasticsearch DocManager.
      - `send_interval`: Number of seconds to wait before sending buffered
        operations to Elasticsearch. Set to None or 0 to disable.
      - `commit_interval`: Number of seconds to wait before committing
        buffered operations to Elasticsearch. Set to None or 0 to disable.
      - `sleep_interval`: Number of seconds to sleep.
    """
    def __init__(self, docman, send_interval, commit_interval,
                 sleep_interval=1):
        super(AutoCommiter, self).__init__()
        self._docman = docman
        # Change `None` intervals to 0
        self._send_interval = send_interval if send_interval else 0
        self._commit_interval = commit_interval if commit_interval else 0
        self._should_auto_send = self._send_interval > 0
        self._should_auto_commit = self._commit_interval > 0
        self._sleep_interval = max(sleep_interval, 1)
        self._stopped = False
        self.daemon = True

    def join(self, timeout=None):
        self._stopped = True
        super(AutoCommiter, self).join(timeout=timeout)

    def run(self):
        """Periodically sends buffered operations and/or commit.
        """
        if not self._should_auto_commit and not self._should_auto_send:
            return
        last_send, last_commit = 0, 0
        while not self._stopped:
            if self._should_auto_commit:
                if last_commit > self._commit_interval:
                    self._docman.commit()
                    # commit also sends so reset both
                    last_send, last_commit = 0, 0
                    # Give a chance to exit the loop
                    if self._stopped:
                        break

            if self._should_auto_send:
                if last_send > self._send_interval:
                    self._docman.send_buffered_operations()
                    last_send = 0
            time.sleep(self._sleep_interval)
            last_send += self._sleep_interval
            last_commit += self._sleep_interval


class DocManager(DocManagerBase):
    """Elasticsearch implementation of the DocManager interface.

    Receives documents from an OplogThread and takes the appropriate actions on
    Elasticsearch.
    """

    def __init__(self, url, auto_commit_interval=DEFAULT_COMMIT_INTERVAL,
                 unique_key='_id', chunk_size=450,
                 meta_index_name="mongodb_meta", meta_type="mongodb_meta",
                 attachment_field="content",
                 **kwargs):
        client_options = kwargs.get('clientOptions', {})
        if 'aws' in kwargs:
            if not _HAS_AWS:
                raise errors.InvalidConfiguration(
                    'aws extras must be installed to sign Elasticsearch '
                    'requests. Install with: '
                    'pip install elastic2-doc-manager[aws]')
            client_options['http_auth'] = create_aws_auth(kwargs['aws'])
            client_options['use_ssl'] = True
            client_options['verify_certs'] = True
            client_options['connection_class'] = \
                es_connection.RequestsHttpConnection
        if type(url) is not list:
            url = [url]
        self.elastic = Elasticsearch(hosts=url, **client_options)

        self._formatter = DefaultDocumentFormatter()
        self.BulkBuffer = BulkBuffer(self)
        self.routing = kwargs.get('routing', {})

        # As bulk operation can be done in another thread
        # lock is needed to prevent access to BulkBuffer
        # while commiting documents to Elasticsearch
        # It is because BulkBuffer might get outdated
        # docs from Elasticsearch if bulk is still ongoing
        self.lock = threading.Lock()

        self.auto_commit_interval = auto_commit_interval
        self.auto_send_interval = kwargs.get('autoSendInterval',
                                             DEFAULT_SEND_INTERVAL)
        self.meta_index_name = meta_index_name
        self.meta_type = meta_type
        self.unique_key = unique_key
        self.chunk_size = chunk_size
        self.routing = kwargs.get('routing', {})
        if self.auto_commit_interval not in [None, 0]:
            self.run_auto_commit()
        self._formatter = DefaultDocumentFormatter()

        self.has_attachment_mapping = False
        self.attachment_field = attachment_field
        self.auto_commiter = AutoCommiter(self, self.auto_send_interval,
                                          self.auto_commit_interval)
        self.auto_commiter.start()

    def _index_and_mapping(self, namespace):
        """Helper method for getting the index and type from a namespace."""
        index, doc_type = namespace.split('.', 1)
        return index.lower(), doc_type

    def is_float(self, str):
        try:
            if str is not None:
                float(str)
            else:
                return False
        except ValueError:
            return False
        return True

    def _get_parent_id(self, doc_type, doc):
        """Get parent ID from doc"""
        if doc_type in self.routing:
            if '_parent' in doc:
                return doc.pop('_parent')

            parent_field = self.routing[doc_type].get('parentField')
            if not parent_field:
                return None

            parent_id = doc.get(parent_field) if parent_field in doc else None
            formated_parent_id = self._formatter.transform_value(parent_id)

            if formated_parent_id is None:
                return formated_parent_id

            if self.is_float(formated_parent_id):
                return int(float(formated_parent_id))
            else:
                return str(formated_parent_id)

    def _get_routing_value(self, doc_type, doc):
        """Get routing value from doc"""
        if doc_type in self.routing:
            if '_routing' in doc:
                return doc.pop('_routing')

            routing_field = self.routing[doc_type].get('routingField')
            if not routing_field:
                return None

            routing_id = doc.get(routing_field) if routing_field in doc else None
            formated_routing_id = self._formatter.transform_value(routing_id)

            if formated_routing_id is None:
                return formated_routing_id

            if self.is_float(routing_id):
                return int(float(formated_routing_id))
            else:
                return str(formated_routing_id)

    def _search_doc_by_id(self, index, doc_type, doc_id):
        """Search document in Elasticsearch by _id"""
        result = self.elastic.search(index=index, doc_type=doc_type,
                                     body={
                                         'query': {
                                             'ids': {
                                                 'type': doc_type,
                                                 'values': [u(doc_id)]
                                             }
                                         }
                                     })
        if result['hits']['total'] == 1:
            return result['hits']['hits'][0]
        else:
            return None

    def stop(self):
        """Stop the auto-commit thread."""
        self.auto_commiter.join()
        self.auto_commit_interval = 0
        # Commit any remaining docs from buffer
        self.commit()

    def apply_update(self, doc, update_spec):
        if "$set" not in update_spec and "$unset" not in update_spec:
            # Don't try to add ns and _ts fields back in from doc
            return update_spec
        return super(DocManager, self).apply_update(doc, update_spec)

    @wrap_exceptions
    def handle_command(self, doc, namespace, timestamp):
        # Flush buffer before handle command
        self.commit()
        db = namespace.split('.', 1)[0]
        if doc.get('dropDatabase'):
            dbs = self.command_helper.map_db(db)
            for _db in dbs:
                self.elastic.indices.delete(index=_db.lower())

        if doc.get('renameCollection'):
            dbs = self.command_helper.map_db(db)
            if len(dbs) > 0:
                raise errors.OperationFailed(
                    "elastic_doc_manager does not support renaming a mapping.")


        if doc.get('create'):
            db, coll = self.command_helper.map_collection(db, doc['create'])
            if db and coll:
                self.elastic.indices.put_mapping(
                    index=db.lower(), doc_type=coll,
                    body={
                        "_source": {"enabled": True}
                    })

        if doc.get('drop'):
            db, coll = self.command_helper.map_collection(db, doc['drop'])
            if db and coll:
                # This will delete the items in coll, but not get rid of the
                # mapping.
                warnings.warn("Deleting all documents of type %s on index %s."
                              "The mapping definition will persist and must be"
                              "removed manually." % (coll, db))
                responses = streaming_bulk(
                    self.elastic,
                    (dict(result, _op_type='delete') for result in scan(
                        self.elastic, index=db.lower(), doc_type=coll)))
                for ok, resp in responses:
                    if not ok:
                        LOG.error(
                            "Error occurred while deleting ElasticSearch docum"
                            "ent during handling of 'drop' command: %r" % resp)

    @wrap_exceptions
    def update(self, document_id, update_spec, namespace, timestamp):
        """Apply updates given in update_spec to the document whose id
        matches that of doc.
        """
        LOG.error("U - %s - %s - %s", document_id, namespace, timestamp)

        index, doc_type = self._index_and_mapping(namespace)

        with self.lock:
            # Check if document source is stored in local buffer
            document = self.BulkBuffer.get_from_sources(index,
                                                        doc_type,
                                                        u(document_id))
        if document:
            # Document source collected from local buffer
            # Perform apply_update on it and then it will be
            # ready for commiting to Elasticsearch
            updated = self.apply_update(document, update_spec)
            # _id is immutable in MongoDB, so won't have changed in update
            updated['_id'] = document_id
            if '_parent' in document:
                updated['_parent'] = document['_parent']
            if '_routing' in document:
                updated['_routing'] = document['_routing']
            self.upsert(updated, namespace, timestamp)
        else:
            # Document source needs to be retrieved from Elasticsearch
            # before performing update. Pass update_spec to upsert function
            updated = {"_id": document_id}
            self.upsert(updated, namespace, timestamp, update_spec)
        # upsert() strips metadata, so only _id + fields in _source still here
        return updated

    @wrap_exceptions
    def upsert(self, doc, namespace, timestamp, update_spec=None):
        """Insert a document into Elasticsearch."""
        index, doc_type = self._index_and_mapping(namespace)
        # No need to duplicate '_id' in source document
        doc_id = u(doc.pop("_id"))

        if self.is_float(doc_id):
            doc_id = int(float(doc_id))
        else:
            doc_id = str(doc_id)

        LOG.error("I - %s - %s - %s", doc_id, namespace, timestamp)

        metadata = {
            'ns': namespace,
            '_ts': timestamp
        }

        parent_id = self._get_parent_id(doc_type, doc)
        routing_id = self._get_routing_value(doc_type, doc)

        # Index the source document, using lowercase namespace as index name.
        action = {
            '_op_type': 'index',
            '_index': index,
            '_type': doc_type,
            '_id': doc_id,
            '_source': self._formatter.format_document(doc)
        }
        # Index document metadata with original namespace (mixed upper/lower).
        meta_action = {
            '_op_type': 'index',
            '_index': self.meta_index_name,
            '_type': self.meta_type,
            '_id': doc_id,
            '_source': bson.json_util.dumps(metadata)
        }

        if parent_id is not None:
            action["_parent"] = parent_id

        if routing_id is not None:
            action["_routing"] = routing_id

        self.index(action, meta_action, doc, update_spec)

        # Leave _id, since it's part of the original document
        doc['_id'] = doc_id

    @wrap_exceptions
    def bulk_upsert(self, docs, namespace, timestamp):

        """Insert multiple documents into Elasticsearch."""
        def docs_to_upsert():
            doc = None
            for doc in docs:
                # Remove metadata and redundant _id
                index, doc_type = self._index_and_mapping(namespace)
                doc_id = u(doc.pop("_id"))
                document_action = {
                    '_index': index,
                    '_type': doc_type,
                    '_id': doc_id,
                    '_source': self._formatter.format_document(doc)
                }
                document_meta = {
                    '_index': self.meta_index_name,
                    '_type': self.meta_type,
                    '_id': doc_id,
                    '_source': {
                        'ns': namespace,
                        '_ts': timestamp
                    }
                }

                parent_id = self._get_parent_id(doc_type, doc)
                routing_id = self._get_routing_value(doc_type, doc)
                if parent_id is not None:
                    document_action["_parent"] = parent_id

                if routing_id is not None:
                    document_action["_routing"] = routing_id

                if parent_id is not None or routing_id is not None:
                    document_action["_source"] = self._formatter.format_document(doc)

                yield document_action
                yield document_meta
            if doc is None:
                raise errors.EmptyDocsError(
                    "Cannot upsert an empty sequence of "
                    "documents into Elastic Search")
        try:
            kw = {}
            if self.chunk_size > 0:
                kw['chunk_size'] = self.chunk_size

            responses = streaming_bulk(client=self.elastic,
                                       actions=docs_to_upsert(),
                                       **kw)

            for ok, resp in responses:
                if not ok:
                    LOG.error(
                        "Could not bulk-upsert document "
                        "into ElasticSearch: %r" % resp)
            if self.auto_commit_interval == 0:
                self.commit()
        except errors.EmptyDocsError:
            # This can happen when mongo-connector starts up, there is no
            # config file, but nothing to dump
            pass

    @wrap_exceptions
    def insert_file(self, f, namespace, timestamp):
        doc = f.get_metadata()
        doc_id = str(doc.pop('_id'))
        index, doc_type = self._index_and_mapping(namespace)

        # make sure that elasticsearch treats it like a file
        if not self.has_attachment_mapping:
            body = {
                "properties": {
                    self.attachment_field: {"type": "attachment"}
                }
            }
            self.elastic.indices.put_mapping(index=index,
                                             doc_type=doc_type,
                                             body=body)
            self.has_attachment_mapping = True

        metadata = {
            'ns': namespace,
            '_ts': timestamp,
        }

        doc = self._formatter.format_document(doc)
        doc[self.attachment_field] = base64.b64encode(f.read()).decode()
        additional_args = self._build_additional_args(doc_type, doc)

        action = {
            '_op_type': 'index',
            '_index': index,
            '_type': doc_type,
            '_id': doc_id,
            '_source': doc
        }
        meta_action = {
            '_op_type': 'index',
            '_index': self.meta_index_name,
            '_type': self.meta_type,
            '_id': doc_id,
            '_source': bson.json_util.dumps(metadata)
        }

        self.index(action, meta_action, **additional_args)

    def _build_additional_args(self, doc_type, doc):
        additional_args = {}
        parent_id = self._get_parent_id(doc_type, doc)
        if parent_id is not None:
            additional_args['parent'] = parent_id
        routing_id = self._get_routing_value(doc_type, doc)
        if routing_id is not None:
            additional_args['routing'] = routing_id

        return additional_args

    @wrap_exceptions
    def remove(self, document_id, namespace, timestamp):
        LOG.error("R - %s - %s - %s", document_id, namespace,timestamp)
        """Remove a document from Elasticsearch."""
        index, doc_type = self._index_and_mapping(namespace)

        doc_id = u(document_id)
        if self.is_float(doc_id):
            doc_id = int(float(doc_id))
        else:
            doc_id = str(doc_id)

        action = {
            '_op_type': 'delete',
            '_index': index,
            '_type': doc_type,
            '_id': doc_id
        }

        meta_action = {
            '_op_type': 'delete',
            '_index': self.meta_index_name,
            '_type': self.meta_type,
            '_id': doc_id
        }

        self.index(action, meta_action)

    @wrap_exceptions
    def _stream_search(self, *args, **kwargs):
        """Helper method for iterating over ES search results."""
        for hit in scan(self.elastic, query=kwargs.pop('body', None),
                        scroll='10m', **kwargs):
            hit['_source']['_id'] = hit['_id']
            if '_parent' in hit:
                hit['_source']['_parent'] = hit['_parent']
            if '_routing' in hit:
                hit['_source']['_routing'] = hit['_routing']

            yield hit['_source']

    def search(self, start_ts, end_ts):
        """Query Elasticsearch for documents in a time range.

        This method is used to find documents that may be in conflict during
        a rollback event in MongoDB.
        """
        return self._stream_search(
            index=self.meta_index_name,
            body={
                "query": {
                    "range": {
                        "_ts": {"gte": start_ts, "lte": end_ts}
                    }
                }
            })

    def index(self, action, meta_action, doc_source=None, update_spec=None):
        with self.lock:
            self.BulkBuffer.add_upsert(action, meta_action, doc_source, update_spec)

        # Divide by two to account for meta actions
        if len(self.BulkBuffer.action_buffer) / 2 >= self.chunk_size or self.auto_commit_interval == 0:
            self.commit()

    def send_buffered_operations(self):
        """Send buffered operations to Elasticsearch.

        This method is periodically called by the AutoCommitThread.
        """
        with self.lock:
            try:
                action_buffer = self.BulkBuffer.get_buffer()
                if action_buffer:
                    successes, errors = bulk(self.elastic, action_buffer)
                    LOG.debug("Bulk request finished, successfully sent %d "
                              "operations", successes)
                    if errors:
                        LOG.error(
                            "Bulk request finished with errors: %r", errors)
            except es_exceptions.ElasticsearchException:
                LOG.exception("Bulk request failed with exception")

    def commit(self):
        """Send buffered requests and refresh all indexes."""
        self.send_buffered_operations()
        retry_until_ok(self.elastic.indices.refresh, index="")

    @wrap_exceptions
    def get_last_doc(self):
        """Get the most recently modified document from Elasticsearch.

        This method is used to help define a time window within which documents
        may be in conflict after a MongoDB rollback.
        """
        try:
            result = self.elastic.search(
                index=self.meta_index_name,
                body={
                    "query": {"match_all": {}},
                    "sort": [{"_ts": "desc"}],
                },
                size=1
            )["hits"]["hits"]
            for r in result:
                r['_source']['_id'] = r['_id']
                return r['_source']
        except es_exceptions.RequestError:
            # no documents so ES returns 400 because of undefined _ts mapping
            return None


class BulkBuffer(object):

    def __init__(self, docman):

        # Parent object
        self.docman = docman

        # Action buffer for bulk indexing
        self.action_buffer = []

        # Docs to update
        # Dict stores all documents for which firstly
        # source has to be retrieved from Elasticsearch
        # and then apply_update needs to be performed
        # Format: [ (doc, update_spec, action_buffer_index, get_from_ES) ]
        self.doc_to_update = []

        # Below dictionary contains ids of documents
        # which need to be retrieved from Elasticsearch
        # It prevents from getting same document multiple times from ES
        # Format: {"_index": {"_type": {"_id": True}}}
        self.doc_to_get = {}

        # Dictionary of sources
        # Format: {"_index": {"_type": {"_id": {"_source": actual_source}}}}
        self.sources = {}

    def add_upsert(self, action, meta_action, doc_source, update_spec):
        """
        Function which stores sources for "insert" actions
        and decide if for "update" action has to add docs to
        get source buffer
        """

        # Whenever update_spec is provided to this method
        # it means that doc source needs to be retrieved
        # from Elasticsearch. It means also that source
        # is not stored in local buffer
        if update_spec:
            id_in_action_buffer = self.is_doc_in_action_buffer(action)
            if id_in_action_buffer > -1:
                if "$set" in update_spec:
                    fields_to_set = update_spec['$set']
                    if fields_to_set is not None:
                        for field in fields_to_set:
                            self.action_buffer[id_in_action_buffer]['_source'][field] = fields_to_set[field]
                if "$unset" in update_spec:
                    fields_to_unset = update_spec['$unset']
                    for field in fields_to_unset:
                        del self.action_buffer[id_in_action_buffer]['_source'][field]
            else:
                self.bulk_index(action, meta_action)
                # -1 -> to get latest index number
                # -1 -> to get action instead of meta_action
                # Update document based on source retrieved from ES
                self.add_doc_to_update(action, update_spec, len(self.action_buffer) - 2)
        else:
            # Insert and update operations provide source
            # Store it in local buffer and use for comming updates
            # inside same buffer
            # add_to_sources will not be called for delete operation
            # as it does not provide doc_source
            if doc_source:
                self.add_to_sources(action, doc_source)
            self.bulk_index(action, meta_action)

    def is_doc_in_action_buffer(self, action):
        for idx, doc in enumerate(self.action_buffer):
            if doc['_type'] == action['_type'] and doc['_id'] == action['_id']:
                return idx
        return -1

    def add_doc_to_update(self, action, update_spec, action_buffer_index):
        """
        Prepare document for update based on Elasticsearch response.
        Set flag if document needs to be retrieved from Elasticsearch
        """

        doc = {'_index': action['_index'],
               '_type': action['_type'],
               '_id': action['_id']}

        # If get_from_ES == True -> get document's source from Elasticsearch
        get_from_ES = self.should_get_id(action)
        self.doc_to_update.append((doc, update_spec, action_buffer_index, get_from_ES))

    def should_get_id(self, action):
        """
        Mark document to retrieve its source from Elasticsearch.
        Returns:
            True - if marking document for the first time in this bulk
            False - if document has been already marked
        """
        mapping_ids = self.doc_to_get.setdefault(
            action['_index'], {}).setdefault(action['_type'], set())

        if action['_id'] in mapping_ids:
            # There is an update on this id already
            return False
        else:
            mapping_ids.add(action['_id'])
            return True

    def get_docs_sources_from_ES(self, doc_to_update):
        """Get document sources using _search elasticsearch API"""
        # docs_to_query = [doc for doc, _, _, get_from_ES in doc_to_update if get_from_ES]
        docs_to_query = [doc for doc, _, _, get_from_ES in doc_to_update]

        if docs_to_query:
            es_query = self.build_query(docs_to_query)
            es_found_docs = self.execute_query(es_query, docs_to_query[0]['_index'])
            return es_found_docs['hits']['hits']
        else:
            return []

    def get_docs_to_update_sources_from_ES(self, docs_to_delete):
        """Get document sources using _search elasticsearch API"""
        if docs_to_delete:
            es_query = self.build_query(docs_to_delete)
            es_found_docs = self.execute_query(es_query, docs_to_delete[0]['_index'])
            return es_found_docs['hits']['hits']
        else:
            return []

    def build_query(self, docs_to_query):
        es_subquery = []
        for doc in docs_to_query:
            es_subquery.append(self.build_search_subquery(doc))
        query = {
                    "from" : 0,
                    "size" : DEFAULT_MAX_BULK,
                    "query": {
                        "bool": {
                            "should": [
                                es_subquery
                            ]
                        }
                    }
                }

        return query

    def execute_query(self, query_body, index):
        documents = self.docman.elastic.search(index=index,
                                        body=query_body)
        return documents

    def build_search_subquery(self, doc):
        return {
                "bool": {
                    "must": [
                        {"match": {"_id": str(doc['_id'])}},
                        {"match": {"_type": str(doc['_type'])}}
                    ]
                }
            }

    def find_in_ES_fetched(self, lst, value):
        for doc in lst:
            if u(doc[u'_type']) == u(value['_type']) and u(doc['_id']) == u(value['_id']):
                return doc
        return None

    @wrap_exceptions
    def update_documents_from_buffer_for_delete(self):
        """Update buffered sources based on response from Elasticsearch"""

        docs_to_query_ES_for_info = []
        #Map of documents having parent/routing information
        dict_idType_routing = {}

        # Get from buffer all documents needing a routing value or if having routing values put them into a map
        # to be used by other documents not having these informations
        for doc in self.action_buffer:
            parent = None
            routing = None
            if doc['_type'] in self.docman.routing:
                if '_parent' in doc:
                    parent = doc['_parent']
                if '_routing' in doc:
                    routing = doc['_routing']

                id_key = str(doc['_id']) + str(u(doc['_type']))
                if parent is not None or routing is not None:
                    dict_idType_routing[id_key] = doc
                else:
                    if dict_idType_routing[id_key] is None:
                        docs_to_query_ES_for_info.append(doc)

        # For all documents be deleted get sources from ES
        ES_documents = self.get_docs_to_update_sources_from_ES(docs_to_query_ES_for_info)
        if ES_documents is not None:
            for ES_doc in ES_documents:
                routing = None
                parent = None

                # For each document from ES get informations about parent and routing if these exist
                if ES_doc is not None :
                    if '_routing' in ES_doc:
                        routing = ES_doc['_routing']
                    if '_parent' in ES_doc:
                        parent = ES_doc['_parent']

                    if parent is not None or routing is not None:
                        dict_idType_routing[str(doc['_id']) + str(u(doc['_type']))] = ES_doc

        for idx, doc in enumerate(self.action_buffer):
            parent = None
            routing = None
            if doc['_type'] in self.docman.routing:
                if '_parent' in doc:
                    parent = doc['_parent']
                if '_routing' in doc:
                    routing = doc['_routing']

                if parent is None and routing is None:
                    dict_key = str(marked_doc['_id']) + str(u(marked_doc['_type']))
                    if dict_key in dict_idType_routing:
                        parentInfo = None
                        routingInfo = None
                        doc_routing_info = dict_idType_routing[dict_key]
                        if '_parent' in doc_routing_info:
                            parentInfo = doc_routing_info['_parent']
                        if '_routing' in doc_routing_info:
                            routingInfo = doc_routing_info['_routing']

                        if parentInfo is not None:
                            self.action_buffer[idx]['_parent'] = parentInfo
                        if routingInfo is not None:
                            self.action_buffer[idx]['_routing'] = routingInfo

    @wrap_exceptions
    def update_sources_reloaded(self):
        """Update local sources based on response from Elasticsearch"""
        ES_documents = self.get_docs_sources_from_ES(self.doc_to_update)

        for doc, update_spec, action_buffer_index, get_from_ES in self.doc_to_update:
            routing = None
            parent = None

            # Update source based on response from ES
            ES_doc = self.find_in_ES_fetched(ES_documents, doc)
            if ES_doc is not None:
                source = ES_doc['_source']
                if '_routing' in ES_doc:
                    routing = ES_doc['_routing']
                if '_parent' in ES_doc:
                    parent = ES_doc['_parent']

            updated = self.docman.apply_update(source, update_spec)

            # Remove _id field from source
            if '_id' in updated:
                del updated['_id']

            # Everytime update locally stored sources to keep them up-to-date
            self.add_to_sources(doc, updated)

            self.action_buffer[action_buffer_index]['_source'] = self.docman._formatter.format_document(updated)
            if routing is not None:
                self.action_buffer[action_buffer_index]['_routing'] = routing
                doc['_routing'] = routing
            if parent is not None:
                self.action_buffer[action_buffer_index]['_parent'] = parent
                doc['_parent'] = parent
            self.action_buffer[action_buffer_index]['_id'] = str(self.action_buffer[action_buffer_index]['_id'])

        # Remove empty actions if there were errors
        self.action_buffer = [each_action for each_action in self.action_buffer if each_action]

    def find_list_index(self, lst, value):
        match_index = []
        for idx, doc in enumerate(lst):
            if doc['_type'] == value['_type'] and str(doc['_id']) == str(value['_id']):
                match_index.append(idx)
        return match_index

    def reset_action(self, action_buffer_index):
        """Reset specific action as update failed"""
        self.action_buffer[action_buffer_index] = {}
        self.action_buffer[action_buffer_index + 1] = {}

    def add_to_sources(self, action, doc_source):
        """Store sources locally"""
        mapping = self.sources.setdefault(action['_index'], {}).setdefault(action['_type'], {})
        mapping[action['_id']] = doc_source

    def get_from_sources(self, index, doc_type, document_id):
        """Get source stored locally"""
        return self.sources.get(index, {}).get(doc_type, {}).get(document_id, {})

    def bulk_index(self, action, meta_action):
        self.action_buffer.append(action)
        self.action_buffer.append(meta_action)

    def clean_up(self):
        """Do clean-up before returning buffer"""
        self.action_buffer = []
        self.sources = {}
        self.doc_to_get = {}
        self.doc_to_update = []

    def get_buffer(self):
        """Get buffer which needs to be bulked to elasticsearch"""

        # Get sources for documents which are in Elasticsearch
        # and they are not in local buffer

        if self.doc_to_update:
            self.update_sources_reloaded()

        self.update_documents_from_buffer_for_delete()

        ES_buffer = self.action_buffer
        self.clean_up()
        return ES_buffer
