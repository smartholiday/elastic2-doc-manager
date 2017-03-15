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
import warnings

from threading import Timer

import bson.json_util

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
    from requests_aws_sign import AWSV4Sign
    from boto3 import session as aws_session
except ImportError:
    _HAS_AWS = False

wrap_exceptions = exception_wrapper({
    BulkIndexError: errors.OperationFailed,
    es_exceptions.ConnectionError: errors.ConnectionFailed,
    es_exceptions.TransportError: errors.OperationFailed,
    es_exceptions.NotFoundError: errors.OperationFailed,
    es_exceptions.RequestError: errors.OperationFailed})

LOG = logging.getLogger(__name__)


class DocManager(DocManagerBase):
    """Elasticsearch implementation of the DocManager interface.

    Receives documents from an OplogThread and takes the appropriate actions on
    Elasticsearch.
    """

    def __init__(self, url, auto_commit_interval=DEFAULT_COMMIT_INTERVAL,
                 unique_key='_id', chunk_size=DEFAULT_MAX_BULK,
                 meta_index_name="mongodb_meta", meta_type="mongodb_meta",
                 attachment_field="content", **kwargs):
        aws = kwargs.get('aws', {'access_id': '', 'secret_key': '', 'region': 'us-east-1'})
        client_options = kwargs.get('clientOptions', {})
        if 'aws' in kwargs:
            if _HAS_AWS is False:
                raise ConfigurationError('aws extras must be installed to sign Elasticsearch requests')
            aws_args = kwargs.get('aws', {'region': 'us-east-1'})
            aws = aws_session.Session()
            if 'access_id' in aws_args and 'secret_key' in aws_args:
                aws = aws_session.Session(
                    aws_access_key_id = aws_args['access_id'],
                    aws_secret_access_key = aws_args['secret_key'])
            credentials = aws.get_credentials()
            region = aws.region_name or aws_args['region']
            aws_auth = AWSV4Sign(credentials, region, 'es')
            client_options['http_auth'] = aws_auth
            client_options['use_ssl'] = True
            client_options['verify_certs'] = True
            client_options['connection_class'] = es_connection.RequestsHttpConnection
        self.elastic = Elasticsearch(
            hosts=[url], **client_options)
        self.auto_commit_interval = auto_commit_interval
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

    def _index_and_mapping(self, namespace):
        """Helper method for getting the index and type from a namespace."""
        index, doc_type = namespace.split('.', 1)
        return index.lower(), doc_type

    def _get_parent_id(self, doc_type, doc):
        """Get parent ID from doc"""
        if doc_type in self.routing:
            if '_parent' in doc:
                return doc.pop('_parent')

            parent_field = self.routing[doc_type].get('parentField')
            if not parent_field:
                return None

            parent_id = doc.get(parent_field) if parent_field in doc else None
            return self._formatter.transform_value(parent_id)

    def _get_routing_value(self, doc_type, doc):
        """Get routing value from doc"""
        if doc_type in self.routing:
            if '_routing' in doc:
                return doc.pop('_routing')

            routing_field = self.routing[doc_type].get('routingField')
            if not routing_field:
                return None

            routing_id = doc.get(routing_field) if routing_field in doc else None
            return self._formatter.transform_value(routing_id)

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
        self.auto_commit_interval = None

    def apply_update(self, doc, update_spec):
        if "$set" not in update_spec and "$unset" not in update_spec:
            # Don't try to add ns and _ts fields back in from doc
            return update_spec
        return super(DocManager, self).apply_update(doc, update_spec)

    @wrap_exceptions
    def handle_command(self, doc, namespace, timestamp):
        db = namespace.split('.', 1)[0]
        if doc.get('dropDatabase'):
            dbs = self.command_helper.map_db(db)
            for _db in dbs:
                self.elastic.indices.delete(index=_db.lower())

        if doc.get('renameCollection'):
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
        self.commit()
        index, doc_type = self._index_and_mapping(namespace)

        if doc_type in self.routing and ('parentField' in self.routing[doc_type] or 'routingField' in self.routing[doc_type]):
            # We can't use get() here and have to do a full search instead.
            # This is due to the fact that Elasticsearch needs the parent ID to
            # know where to route the get request. We might not have the parent
            # ID available in our update request though
            document = self._search_doc_by_id(index, doc_type, document_id)
            if document is None:
                LOG.error('Could not find document with ID "%s" in Elasticsearch to apply update', u(document_id))
                return None
        else:
            document = self.elastic.get(index=index, doc_type=doc_type,
                                        id = u(document_id))

        updated = self.apply_update(document['_source'], update_spec)
        # _id is immutable in MongoDB, so won't have changed in update
        updated['_id'] = document['_id']
        if '_parent' in document:
            updated['_parent'] = document['_parent']
        if '_routing' in document:
            updated['_routing'] = document['_routing']

        self.upsert(updated, namespace, timestamp)
        # upsert() strips metadata, so only _id + fields in _source still here
        return updated

    @wrap_exceptions
    def upsert(self, doc, namespace, timestamp):
        """Insert a document into Elasticsearch."""
        index, doc_type = self._index_and_mapping(namespace)
        # No need to duplicate '_id' in source document
        doc_id = u(doc.pop("_id"))
        metadata = {
            "ns": namespace,
            "_ts": timestamp
        }

        additional_args = self._build_additional_args(doc_type, doc)
        doc = self._formatter.format_document(doc)
        self.elastic.index(index=index, doc_type=doc_type,
                            body=doc, id=doc_id,
                            refresh=(self.auto_commit_interval == 0),
                            **additional_args)

        # Index document metadata with original namespace (mixed upper/lower).
        self.elastic.index(index=self.meta_index_name, doc_type=self.meta_type,
                           body=bson.json_util.dumps(metadata), id=doc_id,
                           refresh=(self.auto_commit_interval == 0))
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
                    "_index": index,
                    "_type": doc_type,
                    "_id": doc_id,
                    "_source": self._formatter.format_document(doc)
                }
                document_meta = {
                    "_index": self.meta_index_name,
                    "_type": self.meta_type,
                    "_id": doc_id,
                    "_source": {
                        "ns": namespace,
                        "_ts": timestamp
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
        additional_args= self._build_additional_args(doc_type, doc)

        self.elastic.index(index=index, doc_type=doc_type,
                           body=doc, id=doc_id,
                           refresh=(self.auto_commit_interval == 0),
                           **additional_args)

        self.elastic.index(index=self.meta_index_name, doc_type=self.meta_type,
                           body=bson.json_util.dumps(metadata), id=doc_id,
                           refresh=(self.auto_commit_interval == 0))

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
        """Remove a document from Elasticsearch."""
        index, doc_type = self._index_and_mapping(namespace)

        # We can't use delete() directly here and have to do a full search first.
        # This is due to the fact that Elasticsearch needs the parent ID to
        # know where to route the delete request. We might not have the parent
        # ID available in our remove request though.
        document = self._search_doc_by_id(index, doc_type, document_id)
        if document is None:
            LOG.error('Could not find document with ID "%s" in Elasticsearch to apply remove', u(document_id))
            return
        additional_args = self._build_additional_args(doc_type, document)

        self.elastic.delete(index=index, doc_type=doc_type,
                            id=u(document_id),
                            refresh=(self.auto_commit_interval == 0), **additional_args)

        self.elastic.delete(index=self.meta_index_name, doc_type=self.meta_type,
                            id=u(document_id),
                            refresh=(self.auto_commit_interval == 0))

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
                    "filtered": {
                        "filter": {
                            "range": {
                                "_ts": {"gte": start_ts, "lte": end_ts}
                            }
                        }
                    }
                }
            })

    def commit(self):
        """Refresh all Elasticsearch indexes."""
        retry_until_ok(self.elastic.indices.refresh, index="")

    def run_auto_commit(self):
        """Periodically commit to the Elastic server."""
        self.elastic.indices.refresh()
        if self.auto_commit_interval not in [None, 0]:
            Timer(self.auto_commit_interval, self.run_auto_commit).start()

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
